"""Tests for the UnmaskingEngine and related components."""

import json
from datetime import datetime

import pytest
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem

from cloakpivot.core.anchors import AnchorEntry
from cloakpivot.core.cloakmap import CloakMap
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
            orig="Hello [PHONE] and [EMAIL] in text."
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
                end=12,   # Off by 1
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
            orig="Hello [PHONE] world"
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
            orig="Hello [PHONE] world"
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
        cloakmap = CloakMap(version="1.0", doc_id="test_doc", doc_hash="a" * 64, anchors=[])

        stats = unmasker.apply_unmasking(doc, [], cloakmap)

        assert stats["total_anchors"] == 0
        assert stats["success_rate"] == 100.0


class TestUnmaskingEngine:
    """Test UnmaskingEngine integration."""

    def create_test_setup(self) -> tuple[DoclingDocument, CloakMap]:
        """Create a test document and CloakMap."""
        # Create masked document
        doc = DoclingDocument(name="test_document")
        text_item = TextItem(
            text="Contact us at [PHONE] or [EMAIL].",
            self_ref="#/texts/0",
            label="text",
            orig="Contact us at [PHONE] or [EMAIL]."
        )
        doc.texts = [text_item]

        # Create anchors
        anchors = [
            AnchorEntry(
                node_id="#/texts/0",
                start=14,
                end=21,  # "[PHONE]"
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
                start=25,
                end=32,  # "[EMAIL]"
                entity_type="EMAIL_ADDRESS",
                confidence=0.90,
                masked_value="[EMAIL]",
                replacement_id="repl_email",
                original_checksum="b" * 64,
                checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="template",
            ),
        ]

        # Create CloakMap
        cloakmap = CloakMap(
            version="1.0",
            doc_id="test_document",
            doc_hash="c" * 64,
            anchors=anchors,
            policy_snapshot={"default_strategy": "template"},
        )

        return doc, cloakmap

    def test_unmask_document_success(self):
        """Test successful document unmasking."""
        engine = UnmaskingEngine()
        doc, cloakmap = self.create_test_setup()

        result = engine.unmask_document(doc, cloakmap, verify_integrity=False)

        assert isinstance(result, UnmaskingResult)
        assert result.cloakmap == cloakmap
        assert result.stats["success_rate"] == 100.0

        # Verify document was modified
        original_text = doc.texts[0].text
        restored_text = result.restored_document.texts[0].text
        assert original_text != restored_text
        assert "[PHONE]" not in restored_text
        assert "[EMAIL]" not in restored_text

    def test_unmask_document_with_integrity_verification(self):
        """Test document unmasking with integrity verification."""
        engine = UnmaskingEngine()
        doc, cloakmap = self.create_test_setup()

        result = engine.unmask_document(doc, cloakmap, verify_integrity=True)

        assert result.integrity_report is not None
        assert "valid" in result.integrity_report
        assert "stats" in result.integrity_report

    def test_unmask_document_invalid_input(self):
        """Test unmasking with invalid inputs."""
        engine = UnmaskingEngine()
        doc, cloakmap = self.create_test_setup()

        with pytest.raises(ValueError, match="document must be a DoclingDocument"):
            engine.unmask_document("not_a_document", cloakmap)

        with pytest.raises(FileNotFoundError, match="CloakMap file not found"):
            engine.unmask_document(doc, "not_a_cloakmap")

        with pytest.raises(ValueError, match="cloakmap must be a CloakMap"):
            engine.unmask_document(doc, 123)  # Invalid type

    def test_unmask_from_cloakmap_file(self, tmp_path):
        """Test unmasking using CloakMap from file."""
        engine = UnmaskingEngine()
        doc, cloakmap = self.create_test_setup()

        # Save CloakMap to file
        cloakmap_file = tmp_path / "test.cloakmap"
        cloakmap.save_to_file(cloakmap_file)

        # Unmask using file path
        result = engine.unmask_document(doc, cloakmap_file)

        assert isinstance(result, UnmaskingResult)
        assert result.cloakmap.version == cloakmap.version
        assert result.stats["success_rate"] == 100.0

    def test_unmask_document_no_anchors(self):
        """Test unmasking a document with no anchors."""
        engine = UnmaskingEngine()

        doc = DoclingDocument(name="test_document")
        text_item = TextItem(
            text="No masked content here.",
            self_ref="#/texts/0",
            label="text",
            orig="No masked content here."
        )
        doc.texts = [text_item]

        cloakmap = CloakMap(
            version="1.0",
            doc_id="test_document",
            doc_hash="d" * 64,
            anchors=[],
        )

        # Should not raise an exception, but return document unchanged
        result = engine.unmask_document(doc, cloakmap)
            
        # Should return the document unchanged
        assert result.restored_document is not None
        assert result.stats["total_anchors_processed"] == 0
        assert result.stats["successful_restorations"] == 0
        assert result.stats["failed_restorations"] == 0


class TestUnmaskingIntegration:
    """Integration tests for the complete unmasking workflow."""

    def test_round_trip_placeholder_workflow(self):
        """Test a complete round-trip workflow with placeholder content."""
        # This test demonstrates the unmasking workflow
        # In a real system, this would use actual secure content restoration

        # Create original-style document structure
        doc = DoclingDocument(name="integration_test")
        text_item = TextItem(
            text="Call me at [PHONE] or email [EMAIL].",
            self_ref="#/texts/0",
            label="text",
            orig="Call me at [PHONE] or email [EMAIL]."
        )
        doc.texts = [text_item]

        # Create realistic anchors
        anchors = [
            AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=11,
                end=18,
                entity_type="PHONE_NUMBER",
                confidence=0.95,
                original_text="555-1234",  # This would be the original content
                masked_value="[PHONE]",
                strategy_used="template",
            ),
            AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=28,
                end=35,
                entity_type="EMAIL_ADDRESS",
                confidence=0.90,
                original_text="user@example.com",  # This would be the original content
                masked_value="[EMAIL]",
                strategy_used="template",
            ),
        ]

        # Create CloakMap
        cloakmap = CloakMap(
            version="1.0",
            doc_id="integration_test",
            doc_hash="integration_hash",
            anchors=anchors,
            policy_snapshot={"default_strategy": "template"},
        )

        # Perform unmasking
        engine = UnmaskingEngine()
        result = engine.unmask_document(doc, cloakmap, verify_integrity=True)

        # Verify results
        assert result.stats["success_rate"] == 100.0
        assert len(result.cloakmap.anchors) == 2

        # Verify text was restored (with placeholder content)
        restored_text = result.restored_document.texts[0].text
        assert "[PHONE]" not in restored_text
        assert "[EMAIL]" not in restored_text

        # Should contain placeholder content
        assert "555-0123" in restored_text or "555" in restored_text  # Phone placeholder
        assert "example.com" in restored_text or "@" in restored_text  # Email placeholder

    def test_error_handling_workflow(self):
        """Test error handling in the unmasking workflow."""
        # Create document with mismatched content
        doc = DoclingDocument(name="error_test")
        text_item = TextItem(
            text="This has different content than expected.",
            self_ref="#/texts/0",
            label="text",
            orig="This has different content than expected."
        )
        doc.texts = [text_item]

        # Create anchors that won't match the document
        anchors = [
            AnchorEntry(
                node_id="#/texts/0",
                start=50,  # Beyond text length
                end=60,
                entity_type="PHONE_NUMBER",
                confidence=0.95,
                masked_value="[PHONE]",
                replacement_id="repl_phone",
                original_checksum="a" * 64,
                checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="template",
            ),
        ]

        cloakmap = CloakMap(
            version="1.0",
            doc_id="error_test",
            doc_hash="error_hash",
            anchors=anchors,
        )

        engine = UnmaskingEngine()
        result = engine.unmask_document(doc, cloakmap)

        # Should handle errors gracefully
        assert result.stats["success_rate"] < 100.0
        assert result.stats["failed_anchors"] > 0
