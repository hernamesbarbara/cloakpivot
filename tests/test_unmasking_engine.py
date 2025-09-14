"""Tests for unmasking functionality using CloakEngine."""

import json
from datetime import datetime

import pytest
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem

from cloakpivot.engine import CloakEngine, MaskResult
from cloakpivot.core.anchors import AnchorEntry
from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.unmasking.anchor_resolver import AnchorResolver, ResolvedAnchor
from cloakpivot.unmasking.cloakmap_loader import CloakMapLoader, CloakMapLoadError
from cloakpivot.unmasking.document_unmasker import DocumentUnmasker
from cloakpivot.unmasking.engine import UnmaskingEngine, UnmaskingResult


class TestCloakMapLoader:
    """Test CloakMapLoader functionality."""

    def test_load_valid_cloakmap_file(self, tmp_path):
        """Test loading a valid CloakMap from file."""
        # Create a test CloakMap
        anchor = AnchorEntry(
            node_id="test_node",
            start=0,
            end=5,
            entity_type="TEST",
            confidence=0.95,
            masked_value="[TEST]",
            replacement_id="repl_123",
            original_checksum="a" * 64,
            checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="template",
            timestamp=datetime.utcnow(),
        )

        cloakmap = CloakMap(
            version="1.0",
            doc_id="test_document",
            doc_hash="b" * 64,
            anchors=[anchor],
            policy_snapshot={"default_strategy": "template"},
            created_at=datetime.utcnow(),
        )

        # Save to file
        cloakmap_file = tmp_path / "test.cloakmap"
        cloakmap.save_to_file(cloakmap_file)

        # Load using loader
        loader = CloakMapLoader()
        loaded_cloakmap = loader.load(cloakmap_file)

        # Verify loaded content
        assert loaded_cloakmap.version == "1.0"
        assert loaded_cloakmap.doc_id == "test_document"
        assert loaded_cloakmap.doc_hash == "b" * 64
        assert len(loaded_cloakmap.anchors) == 1
        assert loaded_cloakmap.anchors[0].replacement_id == "repl_123"

    def test_load_nonexistent_file(self):
        """Test loading from a nonexistent file."""
        loader = CloakMapLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.cloakmap")

    def test_load_invalid_json(self, tmp_path):
        """Test loading from a file with invalid JSON."""
        invalid_file = tmp_path / "invalid.cloakmap"
        invalid_file.write_text("{ invalid json content")

        loader = CloakMapLoader()
        with pytest.raises(CloakMapLoadError, match="Invalid JSON format"):
            loader.load(invalid_file)

    def test_load_missing_required_fields(self, tmp_path):
        """Test loading from a file missing required fields."""
        incomplete_data = {"version": "1.0", "doc_id": "test"}
        incomplete_file = tmp_path / "incomplete.cloakmap"
        incomplete_file.write_text(json.dumps(incomplete_data))

        loader = CloakMapLoader()
        with pytest.raises(CloakMapLoadError, match="Missing required fields"):
            loader.load(incomplete_file)

    def test_validate_file(self, tmp_path):
        """Test file validation functionality."""
        # Create a valid CloakMap
        anchor = AnchorEntry(
            node_id="test_node",
            start=0,
            end=5,
            entity_type="TEST",
            confidence=0.95,
            masked_value="[TEST]",
            replacement_id="repl_123",
            original_checksum="a" * 64,
            checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="template",
        )

        cloakmap = CloakMap(
            version="1.0",
            doc_id="test_document",
            doc_hash="b" * 64,
            anchors=[anchor],
        )

        cloakmap_file = tmp_path / "test.cloakmap"
        cloakmap.save_to_file(cloakmap_file)

        loader = CloakMapLoader()
        validation_result = loader.validate_file(cloakmap_file)

        assert validation_result["valid"] is True
        assert validation_result["file_info"]["exists"] is True
        assert validation_result["cloakmap_info"]["version"] == "1.0"
        assert validation_result["cloakmap_info"]["anchor_count"] == 1


class TestAnchorResolver:
    """Test AnchorResolver functionality."""

    def create_test_document(self) -> DoclingDocument:
        """Create a test document with masked content."""
        doc = DoclingDocument(name="test_doc")

        # Create a text item with masked content
        text_item = TextItem(
            text="Hello [PHONE] and [EMAIL] in text.",
            self_ref="#/texts/0",
            label="text",
            orig="Hello [PHONE] and [EMAIL] in text.",
        )

        doc.texts = [text_item]
        return doc

    def create_test_anchors(self) -> list[AnchorEntry]:
        """Create test anchors for the masked content."""
        return [
            AnchorEntry(
                node_id="#/texts/0",
                start=6,
                end=13,  # "[PHONE]"
                entity_type="PHONE_NUMBER",
                confidence=0.95,
                masked_value="[PHONE]",
                replacement_id="repl_phone",
                original_checksum="a" * 64,
                checksum_salt="dGVzdA==",  # base64 encoded "test"
                strategy_used="template",
            ),
            AnchorEntry(
                node_id="#/texts/0",
                start=18,
                end=25,  # "[EMAIL]"
                entity_type="EMAIL_ADDRESS",
                confidence=0.90,
                masked_value="[EMAIL]",
                replacement_id="repl_email",
                original_checksum="b" * 64,
                checksum_salt="dGVzdA==",  # base64 encoded "test"
                strategy_used="template",
            ),
        ]

    def test_resolve_exact_match_anchors(self):
        """Test resolving anchors with exact position matches."""
        resolver = AnchorResolver()
        document = self.create_test_document()
        anchors = self.create_test_anchors()

        results = resolver.resolve_anchors(document, anchors)

        assert len(results["resolved"]) == 2
        assert len(results["failed"]) == 0
        assert results["stats"]["success_rate"] == 100.0

        # Check first resolved anchor
        resolved_phone = results["resolved"][0]
        assert resolved_phone.anchor.entity_type == "PHONE_NUMBER"
        assert resolved_phone.found_text == "[PHONE]"
        assert resolved_phone.confidence == 1.0
        assert resolved_phone.position_delta == 0

    def test_resolve_with_position_drift(self):
        """Test resolving anchors with slight position drift."""
        resolver = AnchorResolver()
        document = self.create_test_document()

        # Create anchors with slightly wrong positions
        anchors = [
            AnchorEntry(
                node_id="#/texts/0",
                start=5,  # Off by 1
                end=12,  # Off by 1
                entity_type="PHONE_NUMBER",
                confidence=0.95,
                masked_value="[PHONE]",
                replacement_id="repl_phone",
                original_checksum="a" * 64,
                checksum_salt="dGVzdA==",  # base64 encoded "test"
                strategy_used="template",
            ),
        ]

        results = resolver.resolve_anchors(document, anchors)

        # Should still resolve due to fuzzy matching
        assert len(results["resolved"]) == 1
        resolved = results["resolved"][0]
        assert resolved.confidence < 1.0  # Reduced confidence due to drift
        assert resolved.position_delta == 1  # 1 character drift

    def test_resolve_nonexistent_node(self):
        """Test handling of anchors for nonexistent nodes."""
        resolver = AnchorResolver()
        document = self.create_test_document()

        anchors = [
            AnchorEntry(
                node_id="#/texts/999",  # Nonexistent node
                start=0,
                end=5,
                entity_type="TEST",
                confidence=0.95,
                masked_value="[TEST]",
                replacement_id="repl_test",
                original_checksum="a" * 64,
                checksum_salt="dGVzdA==",  # base64 encoded "test"
                strategy_used="template",
            ),
        ]

        results = resolver.resolve_anchors(document, anchors)

        assert len(results["resolved"]) == 0
        assert len(results["failed"]) == 1
        assert not results["failed"][0].node_found


class TestDocumentUnmasker:
    """Test DocumentUnmasker functionality."""

    def create_test_resolved_anchor(self) -> ResolvedAnchor:
        """Create a test resolved anchor."""
        anchor = AnchorEntry(
            node_id="#/texts/0",
            start=6,
            end=13,
            entity_type="PHONE_NUMBER",
            confidence=0.95,
            masked_value="[PHONE]",
            replacement_id="repl_phone",
            original_checksum="a" * 64,
            checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="template",
        )

        text_item = TextItem(
            text="Hello [PHONE] world",
            self_ref="#/texts/0",
            label="text",
            orig="Hello [PHONE] world",
        )

        return ResolvedAnchor(
            anchor=anchor,
            node_item=text_item,
            found_position=(6, 13),
            found_text="[PHONE]",
            position_delta=0,
            confidence=1.0,
        )

    def test_apply_unmasking_to_text_node(self):
        """Test applying unmasking to a text node."""
        unmasker = DocumentUnmasker()

        # Create document
        doc = DoclingDocument(name="test_doc")
        text_item = TextItem(
            text="Hello [PHONE] world",
            self_ref="#/texts/0",
            label="text",
            orig="Hello [PHONE] world",
        )
        doc.texts = [text_item]

        # Create resolved anchor that points to the actual text_item
        anchor = AnchorEntry(
            node_id="#/texts/0",
            start=6,
            end=13,
            entity_type="PHONE_NUMBER",
            confidence=0.95,
            masked_value="[PHONE]",
            replacement_id="repl_phone",
            original_checksum="a" * 64,
            checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="template",
        )

        resolved_anchor = ResolvedAnchor(
            anchor=anchor,
            node_item=text_item,  # Use the actual text_item from the document
            found_position=(6, 13),
            found_text="[PHONE]",
            position_delta=0,
            confidence=1.0,
        )

        # Create CloakMap
        cloakmap = CloakMap(
            version="1.0",
            doc_id="test_doc",
            doc_hash="a" * 64,
            anchors=[resolved_anchor.anchor],
        )

        # Apply unmasking
        stats = unmasker.apply_unmasking(doc, [resolved_anchor], cloakmap)

        # Verify results
        assert stats["restored_anchors"] == 1
        assert stats["failed_restorations"] == 0
        assert stats["success_rate"] == 100.0

        # Verify text was changed (placeholder content)
        assert text_item.text != "Hello [PHONE] world"
        assert "[PHONE]" not in text_item.text

    def test_apply_unmasking_empty_anchors(self):
        """Test applying unmasking with no anchors."""
        unmasker = DocumentUnmasker()

        doc = DoclingDocument(name="test_doc")
        cloakmap = CloakMap(
            version="1.0", doc_id="test_doc", doc_hash="a" * 64, anchors=[]
        )

        stats = unmasker.apply_unmasking(doc, [], cloakmap)

        assert stats["total_anchors"] == 0
        assert stats["success_rate"] == 100.0


class TestUnmaskingWithCloakEngine:
    """Test unmasking functionality via CloakEngine."""

    @pytest.fixture
    def simple_unmasking_document(self) -> DoclingDocument:
        """Create a simple document with PII for unmasking tests."""
        doc = DoclingDocument(name="unmask_test")
        text_item = TextItem(
            text="Contact us at 555-123-4567 or john@example.com.",
            self_ref="#/texts/0",
            label="text",
            orig="Contact us at 555-123-4567 or john@example.com.",
        )
        doc.texts = [text_item]
        return doc

    def test_round_trip_masking_and_unmasking(self, simple_unmasking_document):
        """Test that masking and unmasking preserves original content."""
        engine = CloakEngine()
        original_text = simple_unmasking_document.texts[0].text

        # Mask the document
        mask_result = engine.mask_document(simple_unmasking_document)

        # Verify masking worked
        assert mask_result.document.texts[0].text != original_text
        assert "555-123-4567" not in mask_result.document.texts[0].text
        assert "john@example.com" not in mask_result.document.texts[0].text

        # Unmask the document
        unmasked_doc = engine.unmask_document(mask_result.document, mask_result.cloakmap)

        # Verify original content restored
        assert unmasked_doc.texts[0].text == original_text

    def test_unmask_with_custom_policy(self, simple_unmasking_document):
        """Test unmasking works with custom masking policies."""
        # Create custom policy with partial masking
        policy = MaskingPolicy(
            per_entity={
                "PHONE_NUMBER": Strategy(
                    kind=StrategyKind.PARTIAL,
                    parameters={"visible_chars": 4, "position": "end"}
                ),
                "EMAIL_ADDRESS": Strategy(
                    kind=StrategyKind.TEMPLATE,
                    parameters={"template": "[EMAIL]"}
                )
            }
        )

        engine = CloakEngine(default_policy=policy)
        original_text = simple_unmasking_document.texts[0].text

        # Mask and unmask
        mask_result = engine.mask_document(simple_unmasking_document)
        unmasked_doc = engine.unmask_document(mask_result.document, mask_result.cloakmap)

        # Verify restoration
        assert unmasked_doc.texts[0].text == original_text

    def test_unmask_from_cloakmap_file(self, simple_unmasking_document, tmp_path):
        """Test unmasking using CloakMap from file."""
        engine = CloakEngine()

        # Mask the document
        mask_result = engine.mask_document(simple_unmasking_document)

        # Save CloakMap to file
        cloakmap_file = tmp_path / "test.cloakmap"
        mask_result.cloakmap.save_to_file(cloakmap_file)

        # Create new engine and unmask using file path
        new_engine = CloakEngine()
        unmasked_doc = new_engine.unmask_document(mask_result.document, cloakmap_file)

        # Verify restoration
        assert unmasked_doc.texts[0].text == simple_unmasking_document.texts[0].text

    def test_unmask_empty_document(self):
        """Test unmasking an empty document."""
        doc = DoclingDocument(name="empty_doc")
        doc.texts = []

        engine = CloakEngine()

        # Mask empty document
        mask_result = engine.mask_document(doc)

        # Unmask empty document
        unmasked_doc = engine.unmask_document(mask_result.document, mask_result.cloakmap)

        # Should return empty document unchanged
        assert len(unmasked_doc.texts) == 0

    def test_unmask_no_pii_document(self):
        """Test unmasking document with no PII."""
        doc = DoclingDocument(name="no_pii")
        text_item = TextItem(
            text="This is a simple text with no personal information.",
            self_ref="#/texts/0",
            label="text",
            orig="This is a simple text with no personal information."
        )
        doc.texts = [text_item]

        engine = CloakEngine()
        original_text = doc.texts[0].text

        # Mask and unmask
        mask_result = engine.mask_document(doc)
        unmasked_doc = engine.unmask_document(mask_result.document, mask_result.cloakmap)

        # Should be identical since no PII was found
        assert unmasked_doc.texts[0].text == original_text

    def test_unmask_multiple_sections(self):
        """Test unmasking document with multiple text sections."""
        doc = DoclingDocument(name="multi_section")
        doc.texts = [
            TextItem(
                text="Section 1: Call 555-123-4567",
                self_ref="#/texts/0",
                label="text",
                orig="Section 1: Call 555-123-4567"
            ),
            TextItem(
                text="Section 2: Email john@example.com",
                self_ref="#/texts/1",
                label="text",
                orig="Section 2: Email john@example.com"
            )
        ]

        engine = CloakEngine()
        original_texts = [t.text for t in doc.texts]

        # Mask and unmask
        mask_result = engine.mask_document(doc)
        unmasked_doc = engine.unmask_document(mask_result.document, mask_result.cloakmap)

        # Verify all sections restored
        for i, original_text in enumerate(original_texts):
            assert unmasked_doc.texts[i].text == original_text

    def test_unmask_preserves_document_structure(self, simple_unmasking_document):
        """Test that unmasking preserves document structure and metadata."""
        engine = CloakEngine()

        # Mask the document
        mask_result = engine.mask_document(simple_unmasking_document)

        # Unmask the document
        unmasked_doc = engine.unmask_document(mask_result.document, mask_result.cloakmap)

        # Verify structure preserved
        assert unmasked_doc.name == simple_unmasking_document.name
        assert len(unmasked_doc.texts) == len(simple_unmasking_document.texts)
        assert unmasked_doc.texts[0].self_ref == simple_unmasking_document.texts[0].self_ref
        assert unmasked_doc.texts[0].label == simple_unmasking_document.texts[0].label


class TestUnmaskingIntegration:
    """Integration tests for the complete unmasking workflow using CloakEngine."""

    def test_round_trip_workflow_with_cloakengine(self):
        """Test a complete round-trip workflow with CloakEngine."""
        # Create original document with PII
        doc = DoclingDocument(name="integration_test")
        text_item = TextItem(
            text="Call me at 555-123-4567 or email user@example.com.",
            self_ref="#/texts/0",
            label="text",
            orig="Call me at 555-123-4567 or email user@example.com.",
        )
        doc.texts = [text_item]

        # Use CloakEngine for masking and unmasking
        engine = CloakEngine()
        original_text = doc.texts[0].text

        # Perform masking
        mask_result = engine.mask_document(doc)

        # Verify masking worked
        masked_text = mask_result.document.texts[0].text
        assert masked_text != original_text
        assert "555-123-4567" not in masked_text
        assert "user@example.com" not in masked_text
        assert len(mask_result.cloakmap.anchors) >= 2

        # Perform unmasking
        unmasked_doc = engine.unmask_document(mask_result.document, mask_result.cloakmap)

        # Verify complete restoration
        assert unmasked_doc.texts[0].text == original_text

    def test_workflow_with_multiple_entity_types(self):
        """Test workflow with various entity types."""
        doc = DoclingDocument(name="multi_entity_test")
        text_item = TextItem(
            text="John Doe (SSN: 123-45-6789) lives at 123 Main St, phone: 555-987-6543.",
            self_ref="#/texts/0",
            label="text",
            orig="John Doe (SSN: 123-45-6789) lives at 123 Main St, phone: 555-987-6543.",
        )
        doc.texts = [text_item]

        engine = CloakEngine()
        original_text = doc.texts[0].text

        # Mask with default policy
        mask_result = engine.mask_document(doc)

        # Verify multiple entities were found and masked
        assert mask_result.entities_found > 0
        assert mask_result.entities_masked > 0
        masked_text = mask_result.document.texts[0].text
        assert masked_text != original_text

        # Unmask and verify restoration
        unmasked_doc = engine.unmask_document(mask_result.document, mask_result.cloakmap)
        assert unmasked_doc.texts[0].text == original_text

    def test_selective_entity_masking_workflow(self):
        """Test masking only specific entity types and unmasking."""
        doc = DoclingDocument(name="selective_test")
        text_item = TextItem(
            text="Contact John at 555-123-4567 or john@example.com",
            self_ref="#/texts/0",
            label="text",
            orig="Contact John at 555-123-4567 or john@example.com",
        )
        doc.texts = [text_item]

        engine = CloakEngine()
        original_text = doc.texts[0].text

        # Mask only phone numbers
        mask_result = engine.mask_document(doc, entities=['PHONE_NUMBER'])

        # Verify selective masking
        masked_text = mask_result.document.texts[0].text
        assert "555-123-4567" not in masked_text  # Phone should be masked
        assert "john@example.com" in masked_text  # Email should NOT be masked

        # Unmask and verify
        unmasked_doc = engine.unmask_document(mask_result.document, mask_result.cloakmap)
        assert unmasked_doc.texts[0].text == original_text

    def test_error_recovery_in_workflow(self):
        """Test that CloakEngine handles edge cases gracefully."""
        # Test with empty document
        empty_doc = DoclingDocument(name="empty")
        empty_doc.texts = []

        engine = CloakEngine()

        # Should handle empty document
        mask_result = engine.mask_document(empty_doc)
        assert mask_result.entities_found == 0
        assert mask_result.entities_masked == 0

        unmasked_doc = engine.unmask_document(mask_result.document, mask_result.cloakmap)
        assert len(unmasked_doc.texts) == 0

        # Test with no PII document
        no_pii_doc = DoclingDocument(name="no_pii")
        no_pii_doc.texts = [
            TextItem(
                text="This is a regular sentence with no PII.",
                self_ref="#/texts/0",
                label="text",
                orig="This is a regular sentence with no PII."
            )
        ]

        mask_result = engine.mask_document(no_pii_doc)
        unmasked_doc = engine.unmask_document(mask_result.document, mask_result.cloakmap)

        # Should be identical
        assert unmasked_doc.texts[0].text == no_pii_doc.texts[0].text
