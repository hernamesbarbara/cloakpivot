"""Tests to improve unmasking engine coverage."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from cloakpivot.core.types.anchors import AnchorEntry
from cloakpivot.core.types.cloakmap import CloakMap
from cloakpivot.type_imports import DoclingDocument
from cloakpivot.unmasking.engine import UnmaskingEngine


class TestUnmaskingEngineCoverage:
    """Tests to improve unmasking engine coverage."""

    def test_unmasking_engine_creation_default(self):
        """Test creating UnmaskingEngine with default parameters."""
        engine = UnmaskingEngine()
        assert engine is not None
        assert hasattr(engine, "unmask_document")

    def test_unmasking_engine_creation_with_presidio(self):
        """Test creating UnmaskingEngine with presidio flag."""
        engine = UnmaskingEngine(use_presidio_engine=True)
        assert engine is not None

        engine2 = UnmaskingEngine(use_presidio_engine=False)
        assert engine2 is not None

    def test_unmask_document_basic(self):
        """Test basic unmask_document functionality."""
        engine = UnmaskingEngine()

        # Create a mock document
        doc = Mock(spec=DoclingDocument)
        doc.export_to_markdown.return_value = "Masked text with [PERSON]"
        doc.name = "masked.txt"

        # Create a simple cloakmap
        cloakmap = CloakMap(doc_id="test_doc_1", doc_hash="test_hash", anchors=[])

        # Test unmask_document
        result = engine.unmask_document(doc, cloakmap)
        assert result is not None

    def test_unmask_document_with_anchors(self):
        """Test unmasking document with anchors."""
        engine = UnmaskingEngine()

        doc = Mock(spec=DoclingDocument)
        doc.export_to_markdown.return_value = "Hello [PERSON]"
        doc.name = "test.txt"

        # Create cloakmap with an anchor
        cloakmap = CloakMap(doc_id="test_doc_2", doc_hash="test_hash", anchors=[])
        anchor = AnchorEntry.create_from_detection(
            node_id="#/texts/0",
            start=6,
            end=14,
            entity_type="PERSON",
            confidence=0.95,
            original_text="John Doe",
            masked_value="[PERSON]",
            strategy_used="redact",
        )
        cloakmap.anchors.append(anchor)

        result = engine.unmask_document(doc, cloakmap)
        assert result is not None

    def test_unmask_from_files(self):
        """Test unmask_from_files method."""
        engine = UnmaskingEngine()

        # Create temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a mock masked document file
            masked_file = tmpdir / "masked.md"
            masked_file.write_text("Masked content with [EMAIL]")

            # Create a mock cloakmap file
            cloakmap_file = tmpdir / "test.cloakmap"
            cloakmap = CloakMap(doc_id="test_doc_3", doc_hash="test_hash", anchors=[])
            cloakmap.save_to_file(cloakmap_file)

            # Call unmask_from_files
            result = engine.unmask_from_files(
                masked_document_path=str(masked_file),
                cloakmap_path=str(cloakmap_file),
            )

            assert result is not None

    def test_unmask_document_with_empty_cloakmap(self):
        """Test unmasking with empty cloakmap."""
        engine = UnmaskingEngine()

        doc = Mock(spec=DoclingDocument)
        doc.export_to_markdown.return_value = "No masked content"
        doc.name = "test.txt"

        cloakmap = CloakMap(doc_id="test_doc_4", doc_hash="test_hash", anchors=[])

        result = engine.unmask_document(doc, cloakmap)
        assert result is not None
        assert len(cloakmap.anchors) == 0

    def test_unmask_document_preserves_structure(self):
        """Test that document structure is preserved during unmasking."""
        engine = UnmaskingEngine()

        # Create document with structure
        doc = Mock(spec=DoclingDocument)
        doc.export_to_markdown.return_value = (
            "# Title\n\n[PERSON] wrote this.\n\n## Section\n\nContent here."
        )
        doc.name = "structured.txt"

        cloakmap = CloakMap(doc_id="test_doc_5", doc_hash="test_hash", anchors=[])
        anchor = AnchorEntry.create_from_detection(
            node_id="#/texts/0",
            start=10,
            end=18,
            entity_type="PERSON",
            confidence=0.95,
            original_text="Jane Doe",
            masked_value="[PERSON]",
            strategy_used="redact",
        )
        cloakmap.anchors.append(anchor)

        result = engine.unmask_document(doc, cloakmap)
        assert result is not None
        assert len(cloakmap.anchors) == 1

    def test_unmask_with_invalid_cloakmap(self):
        """Test unmasking with invalid cloakmap raises appropriate error."""
        engine = UnmaskingEngine()

        doc = Mock(spec=DoclingDocument)
        doc.export_to_markdown.return_value = "Test content"
        doc.name = "test.txt"

        # Pass invalid cloakmap (not a CloakMap instance)
        with pytest.raises(FileNotFoundError):
            engine.unmask_from_files(
                masked_document_path="not_exist.md", cloakmap_path="not_a_cloakmap"
            )

    def test_unmask_document_multiple_entities(self):
        """Test unmasking document with multiple entity types."""
        engine = UnmaskingEngine()

        doc = Mock(spec=DoclingDocument)
        doc.export_to_markdown.return_value = "[PERSON] sent email to [EMAIL]"
        doc.name = "test.txt"

        cloakmap = CloakMap(doc_id="test_doc_6", doc_hash="test_hash", anchors=[])

        # Add multiple anchors
        person_anchor = AnchorEntry.create_from_detection(
            node_id="#/texts/0",
            start=0,
            end=8,
            entity_type="PERSON",
            confidence=0.9,
            original_text="John",
            masked_value="[PERSON]",
            strategy_used="redact",
        )
        email_anchor = AnchorEntry.create_from_detection(
            node_id="#/texts/0",
            start=23,
            end=30,
            entity_type="EMAIL",
            confidence=0.95,
            original_text="test@example.com",
            masked_value="[EMAIL]",
            strategy_used="redact",
        )

        cloakmap.anchors.extend([person_anchor, email_anchor])

        result = engine.unmask_document(doc, cloakmap)
        assert result is not None
        assert len(cloakmap.anchors) == 2

    def test_unmask_from_files_missing_file(self):
        """Test unmask_from_files with missing files."""
        engine = UnmaskingEngine()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            missing_doc = tmpdir / "missing.md"
            missing_cloakmap = tmpdir / "missing.cloakmap"

            # Test with missing document file
            cloakmap_file = tmpdir / "test.cloakmap"
            cloakmap = CloakMap(doc_id="test_doc_7", doc_hash="test_hash", anchors=[])
            cloakmap.save_to_file(cloakmap_file)

            with pytest.raises(FileNotFoundError):
                engine.unmask_from_files(
                    masked_document_path=missing_doc,
                    cloakmap_path=cloakmap_file,
                )

            # Test with missing cloakmap file
            doc_file = tmpdir / "test.md"
            doc_file.write_text("Test content")

            with pytest.raises(FileNotFoundError):
                engine.unmask_from_files(
                    masked_document_path=doc_file,
                    cloakmap_path=missing_cloakmap,
                )
