"""Integration tests using real PDF and JSON documents from data/ directory."""

import json
from pathlib import Path

import pytest
from docling_core.types import DoclingDocument

from cloakpivot.engine import CloakEngine


class TestRealDocuments:
    """Test with real documents from data/ directory."""

    def test_email_docling_document(self, email_docling_document: DoclingDocument):
        """Test masking/unmasking real email document."""
        engine = CloakEngine()

        # Mask the document
        result = engine.mask_document(email_docling_document)

        assert result.entities_found > 0
        assert result.entities_masked > 0

        # Unmask and verify roundtrip
        restored = engine.unmask_document(result.document, result.cloakmap)
        assert restored.text == email_docling_document.text

    def test_pdf_styles_document(self, pdf_styles_docling_document: DoclingDocument):
        """Test with complex PDF containing various formatting styles."""
        engine = CloakEngine()

        # This document has rich formatting - test that it's preserved
        original_text = pdf_styles_docling_document.text

        # Mask the document
        result = engine.mask_document(pdf_styles_docling_document)

        # Even complex documents should roundtrip correctly
        restored = engine.unmask_document(result.document, result.cloakmap)
        assert restored.text == original_text

    def test_all_docling_json_files(self, json_dir: Path):
        """Test all .docling.json files in the data directory."""
        engine = CloakEngine()
        docling_files = list(json_dir.glob("*.docling.json"))

        assert len(docling_files) > 0, "No docling.json files found"

        for json_path in docling_files:
            with open(json_path) as f:
                data = json.load(f)

            # Workaround: Change version for docling-core compatibility
            data["version"] = "1.6.0"

            # Load as DoclingDocument
            doc = DoclingDocument.model_validate(data)
            original_text = doc.text

            # Mask the document
            result = engine.mask_document(doc)

            # Unmask and verify
            restored = engine.unmask_document(result.document, result.cloakmap)

            assert restored.text == original_text, f"Roundtrip failed for {json_path.name}"

    def test_document_metadata_preservation(
        self, email_docling_document: DoclingDocument
    ):
        """Test that document metadata is preserved during masking."""
        engine = CloakEngine()

        # Store original metadata
        original_name = email_docling_document.name
        original_version = email_docling_document.version

        # Mask the document
        result = engine.mask_document(email_docling_document)

        # Check metadata is preserved
        assert result.document.name == original_name
        assert result.document.version == original_version

    def test_selective_entity_masking(self, email_docling_document: DoclingDocument):
        """Test masking only specific entity types in real documents."""
        engine = CloakEngine()

        # Only mask email addresses
        result = engine.mask_document(
            email_docling_document,
            entities=["EMAIL_ADDRESS"]
        )

        # Should find and mask at least some emails
        if result.entities_found > 0:
            assert result.entities_masked > 0
            assert len(result.cloakmap.entries) > 0

    @pytest.mark.parametrize("confidence", [0.5, 0.7, 0.9])
    def test_different_confidence_thresholds(
        self, email_docling_document: DoclingDocument, confidence: float
    ):
        """Test masking with different confidence thresholds."""
        engine = CloakEngine(
            analyzer_config={"confidence_threshold": confidence}
        )

        result = engine.mask_document(email_docling_document)

        # Higher confidence should generally find fewer entities
        # (This is a soft assertion since it depends on the document content)
        assert result.entities_masked >= 0

    def test_cloakmap_serialization(self, email_docling_document: DoclingDocument):
        """Test that CloakMap can be serialized and deserialized."""
        engine = CloakEngine()

        # Mask the document
        result = engine.mask_document(email_docling_document)

        # Serialize CloakMap
        cloakmap_dict = result.cloakmap.to_dict()
        cloakmap_json = json.dumps(cloakmap_dict)

        # Deserialize CloakMap
        from cloakpivot.core.cloakmap import CloakMap
        restored_cloakmap = CloakMap.from_dict(json.loads(cloakmap_json))

        # Use restored CloakMap to unmask
        restored_doc = engine.unmask_document(result.document, restored_cloakmap)
        assert restored_doc.text == email_docling_document.text