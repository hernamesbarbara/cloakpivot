"""Integration tests for masking functionality using CloakEngine."""

import json
import tempfile
from pathlib import Path

import pytest
from docling_core.types import DoclingDocument
from presidio_analyzer import AnalyzerEngine

from cloakpivot.engine import CloakEngine
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import (
    EMAIL_TEMPLATE,
    PHONE_TEMPLATE,
    Strategy,
    StrategyKind,
)
from cloakpivot.document.processor import DocumentProcessor


class TestMaskingIntegration:
    """Integration tests for the complete masking pipeline."""

    @pytest.fixture
    def sample_docling_json(self) -> str:
        """Create a sample docling.json document with PII content."""
        document_data = {
            "schema_name": "DoclingDocument",
            "version": "1.4.0",
            "name": "test_document_with_pii",
            "origin": {
                "mimetype": "application/json",
                "filename": "test.json",
                "binary_hash": 123456789,
            },
            "furniture": {
                "self_ref": "#/furniture",
                "children": [],
                "content_layer": "furniture",
                "name": "_root_",
                "label": "unspecified",
            },
            "body": {
                "self_ref": "#/body",
                "children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}],
                "content_layer": "body",
                "name": "_root_",
                "label": "unspecified",
            },
            "texts": [
                {
                    "self_ref": "#/texts/0",
                    "parent": {"$ref": "#/body"},
                    "children": [],
                    "content_layer": "body",
                    "label": "text",
                    "prov": [
                        {
                            "page_no": 1,
                            "bbox": {
                                "l": 0,
                                "t": 0,
                                "r": 100,
                                "b": 20,
                                "coord_origin": "BOTTOMLEFT",
                            },
                            "charspan": [0, 47],
                        }
                    ],
                    "orig": "Contact John at 555-123-4567 for more information.",
                    "text": "Contact John at 555-123-4567 for more information.",
                },
                {
                    "self_ref": "#/texts/1",
                    "parent": {"$ref": "#/body"},
                    "children": [],
                    "content_layer": "body",
                    "label": "text",
                    "prov": [
                        {
                            "page_no": 1,
                            "bbox": {
                                "l": 0,
                                "t": 25,
                                "r": 100,
                                "b": 45,
                                "coord_origin": "BOTTOMLEFT",
                            },
                            "charspan": [0, 47],
                        }
                    ],
                    "orig": "Email support at help@company.com for assistance.",
                    "text": "Email support at help@company.com for assistance.",
                },
            ],
            "tables": [],
            "key_value_items": [],
            "form_items": [],
            "pictures": [],
            "groups": [],
            "pages": {},
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".docling.json", delete=False
        ) as f:
            json.dump(document_data, f, indent=2)
            return f.name

    def test_full_masking_pipeline_with_cloakengine(self, sample_docling_json):
        """Test complete pipeline using CloakEngine."""
        # Step 1: Load document using DocumentProcessor
        processor = DocumentProcessor()
        document = processor.load_document(sample_docling_json)

        assert isinstance(document, DoclingDocument)
        assert len(document.texts) == 2

        # Step 2: Set up masking policy
        policy = MaskingPolicy(
            per_entity={"PHONE_NUMBER": PHONE_TEMPLATE, "EMAIL_ADDRESS": EMAIL_TEMPLATE}
        )

        # Step 3: Use CloakEngine for detection and masking
        engine = CloakEngine(default_policy=policy)
        result = engine.mask_document(document)

        # Step 4: Verify results
        assert result.document is not None
        assert result.cloakmap is not None
        assert result.entities_found > 0
        assert result.entities_masked > 0

        # Check that PII was masked
        masked_text_0 = result.document.texts[0].text
        masked_text_1 = result.document.texts[1].text

        # Phone number should be masked in first text
        assert "555-123-4567" not in masked_text_0
        assert "[PHONE]" in masked_text_0 or "[REDACTED]" in masked_text_0

        # Email should be masked in second text
        assert "help@company.com" not in masked_text_1
        assert "[EMAIL]" in masked_text_1 or "[REDACTED]" in masked_text_1

        # Check CloakMap has entries
        assert len(result.cloakmap.anchors) >= 2

        # Verify CloakMap structure
        assert result.cloakmap.version == "1.0"
        assert result.cloakmap.doc_id == "test_document_with_pii"

        # Verify no original PII in CloakMap JSON
        cloakmap_json = result.cloakmap.to_json()
        assert "555-123-4567" not in cloakmap_json
        assert "help@company.com" not in cloakmap_json

        # Cleanup
        Path(sample_docling_json).unlink()

    def test_masking_with_cloakengine_and_custom_entities(self):
        """Test CloakEngine with selective entity masking."""
        # Create a simple document
        document_data = {
            "schema_name": "DoclingDocument",
            "version": "1.4.0",
            "name": "presidio_test",
            "origin": {
                "mimetype": "application/json",
                "filename": "test.json",
                "binary_hash": 987654321,
            },
            "furniture": {
                "self_ref": "#/furniture",
                "children": [],
                "content_layer": "furniture",
                "name": "_root_",
                "label": "unspecified",
            },
            "body": {
                "self_ref": "#/body",
                "children": [{"$ref": "#/texts/0"}],
                "content_layer": "body",
                "name": "_root_",
                "label": "unspecified",
            },
            "texts": [
                {
                    "self_ref": "#/texts/0",
                    "parent": {"$ref": "#/body"},
                    "children": [],
                    "content_layer": "body",
                    "label": "text",
                    "prov": [
                        {
                            "page_no": 1,
                            "bbox": {
                                "l": 0,
                                "t": 0,
                                "r": 100,
                                "b": 20,
                                "coord_origin": "BOTTOMLEFT",
                            },
                            "charspan": [0, 51],
                        }
                    ],
                    "orig": "My SSN is 456-78-9012 and phone is 555-987-6543.",
                    "text": "My SSN is 456-78-9012 and phone is 555-987-6543.",
                }
            ],
            "tables": [],
            "key_value_items": [],
            "form_items": [],
            "pictures": [],
            "groups": [],
            "pages": {},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".docling.json", delete=False
        ) as f:
            json.dump(document_data, f, indent=2)
            temp_path = f.name

        try:
            # Load document
            processor = DocumentProcessor()
            document = processor.load_document(temp_path)

            # Set up policy with custom strategies for specific entity types
            policy = MaskingPolicy(
                per_entity={
                    "PHONE_NUMBER": PHONE_TEMPLATE,
                    "US_SSN": Strategy(
                        StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end"}
                    ),
                },
                default_strategy=Strategy(
                    StrategyKind.TEMPLATE, {"template": "[REDACTED]"}
                )
            )

            # Use CloakEngine for masking
            engine = CloakEngine(default_policy=policy)
            result = engine.mask_document(document)

            # Verify masking worked
            masked_text = result.document.texts[0].text

            # Should not contain original PII
            assert "456-78-9012" not in masked_text
            assert "555-987-6543" not in masked_text

            # Should have found and masked entities
            assert result.entities_found > 0
            assert result.entities_masked > 0
            assert len(result.cloakmap.anchors) > 0

            # Verify CloakMap integrity
            assert result.cloakmap.doc_id == "presidio_test"
            assert all(anchor.confidence > 0 for anchor in result.cloakmap.anchors)

            # Test round-trip
            unmasked_doc = engine.unmask_document(result.document, result.cloakmap)
            assert unmasked_doc.texts[0].text == document.texts[0].text

        finally:
            Path(temp_path).unlink()

    def test_empty_document_handling(self):
        """Test handling of documents with no text content."""
        document_data = {
            "schema_name": "DoclingDocument",
            "version": "1.4.0",
            "name": "empty_doc",
            "origin": {
                "mimetype": "application/json",
                "filename": "empty.json",
                "binary_hash": 123456789,
            },
            "furniture": {
                "self_ref": "#/furniture",
                "children": [],
                "content_layer": "furniture",
                "name": "_root_",
                "label": "unspecified",
            },
            "body": {
                "self_ref": "#/body",
                "children": [],
                "content_layer": "body",
                "name": "_root_",
                "label": "unspecified",
            },
            "texts": [],
            "tables": [],
            "key_value_items": [],
            "form_items": [],
            "pictures": [],
            "groups": [],
            "pages": {},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".docling.json", delete=False
        ) as f:
            json.dump(document_data, f, indent=2)
            temp_path = f.name

        try:
            processor = DocumentProcessor()
            document = processor.load_document(temp_path)

            # Use CloakEngine to mask empty document
            engine = CloakEngine()
            result = engine.mask_document(document)

            # Should succeed with empty results
            assert result.document is not None
            assert result.cloakmap is not None
            assert result.entities_found == 0
            assert result.entities_masked == 0
            assert len(result.cloakmap.anchors) == 0

            # Test round-trip on empty document
            unmasked_doc = engine.unmask_document(result.document, result.cloakmap)
            assert len(unmasked_doc.texts) == 0

        finally:
            Path(temp_path).unlink()
