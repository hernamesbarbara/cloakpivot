"""Integration tests for MaskingEngine with existing Presidio and DocPivot modules."""

import json
import tempfile
from pathlib import Path

import pytest
from docling_core.types import DoclingDocument
from presidio_analyzer import AnalyzerEngine, RecognizerResult

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import (
    EMAIL_TEMPLATE,
    PHONE_TEMPLATE,
    Strategy,
    StrategyKind,
)
from cloakpivot.document.extractor import TextExtractor
from cloakpivot.document.processor import DocumentProcessor
from cloakpivot.masking.engine import MaskingEngine


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
                "binary_hash": 123456789
            },
            "furniture": {
                "self_ref": "#/furniture",
                "children": [],
                "content_layer": "furniture",
                "name": "_root_",
                "label": "unspecified"
            },
            "body": {
                "self_ref": "#/body",
                "children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}],
                "content_layer": "body",
                "name": "_root_",
                "label": "unspecified"
            },
            "texts": [
                {
                    "self_ref": "#/texts/0",
                    "parent": {"$ref": "#/body"},
                    "children": [],
                    "content_layer": "body",
                    "label": "text",
                    "prov": [{"page_no": 1, "bbox": {"l": 0, "t": 0, "r": 100, "b": 20, "coord_origin": "BOTTOMLEFT"}, "charspan": [0, 47]}],
                    "orig": "Contact John at 555-123-4567 for more information.",
                    "text": "Contact John at 555-123-4567 for more information."
                },
                {
                    "self_ref": "#/texts/1",
                    "parent": {"$ref": "#/body"},
                    "children": [],
                    "content_layer": "body",
                    "label": "text",
                    "prov": [{"page_no": 1, "bbox": {"l": 0, "t": 25, "r": 100, "b": 45, "coord_origin": "BOTTOMLEFT"}, "charspan": [0, 47]}],
                    "orig": "Email support at help@company.com for assistance.",
                    "text": "Email support at help@company.com for assistance."
                }
            ],
            "tables": [],
            "key_value_items": [],
            "form_items": [],
            "pictures": [],
            "groups": [],
            "pages": {}
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.docling.json', delete=False) as f:
            json.dump(document_data, f, indent=2)
            return f.name

    def test_full_masking_pipeline_with_docpivot_loading(self, sample_docling_json):
        """Test complete pipeline: DocPivot loading -> text extraction -> PII detection -> masking."""
        # Step 1: Load document using DocumentProcessor
        processor = DocumentProcessor()
        document = processor.load_document(sample_docling_json)

        assert isinstance(document, DoclingDocument)
        assert len(document.texts) == 2

        # Step 2: Extract text segments
        extractor = TextExtractor()
        text_segments = extractor.extract_text_segments(document)

        assert len(text_segments) >= 2

        # Step 3: Simulate PII detection (instead of calling Presidio directly)
        # This simulates what PIIAnalyzer would return
        [
            RecognizerResult(entity_type="PHONE_NUMBER", start=16, end=28, score=0.95),  # "555-123-4567"
            RecognizerResult(entity_type="EMAIL_ADDRESS", start=17, end=33, score=0.90)  # "help@company.com" in second segment
        ]

        # Step 4: Set up masking policy
        policy = MaskingPolicy(
            per_entity={
                "PHONE_NUMBER": PHONE_TEMPLATE,
                "EMAIL_ADDRESS": EMAIL_TEMPLATE
            }
        )

        # Step 5: Apply masking
        masking_engine = MaskingEngine()

        # Adjust entities to account for segment offsets
        # First entity is in first segment (offset 0)
        # Second entity is in second segment, need to adjust its position
        extractor.extract_full_text(document)
        second_segment = text_segments[1]

        adjusted_entities = [
            RecognizerResult(entity_type="PHONE_NUMBER", start=16, end=28, score=0.95),  # First segment
            RecognizerResult(entity_type="EMAIL_ADDRESS", start=second_segment.start_offset + 17,
                           end=second_segment.start_offset + 33, score=0.90)  # Second segment
        ]

        result = masking_engine.mask_document(
            document=document,
            entities=adjusted_entities,
            policy=policy,
            text_segments=text_segments
        )

        # Step 6: Verify results
        assert result.masked_document is not None
        assert result.cloakmap is not None

        # Check that PII was masked
        masked_text_0 = result.masked_document.texts[0].text
        masked_text_1 = result.masked_document.texts[1].text

        assert "[PHONE]" in masked_text_0
        assert "555-123-4567" not in masked_text_0
        assert "[EMAIL]" in masked_text_1
        assert "help@company.com" not in masked_text_1

        # Check CloakMap has correct entries
        assert len(result.cloakmap.anchors) == 2

        phone_anchor = next(a for a in result.cloakmap.anchors if a.entity_type == "PHONE_NUMBER")
        email_anchor = next(a for a in result.cloakmap.anchors if a.entity_type == "EMAIL_ADDRESS")

        assert phone_anchor.masked_value == "[PHONE]"
        assert email_anchor.masked_value == "[EMAIL]"

        # Verify no original PII in CloakMap
        cloakmap_json = result.cloakmap.to_json()
        assert "555-123-4567" not in cloakmap_json
        assert "help@company.com" not in cloakmap_json

        # Cleanup
        Path(sample_docling_json).unlink()

    def test_masking_with_presidio_analyzer(self):
        """Test integration with actual Presidio AnalyzerEngine."""
        # Create a simple document
        document_data = {
            "schema_name": "DoclingDocument",
            "version": "1.4.0",
            "name": "presidio_test",
            "origin": {"mimetype": "application/json", "filename": "test.json", "binary_hash": 987654321},
            "furniture": {"self_ref": "#/furniture", "children": [], "content_layer": "furniture", "name": "_root_", "label": "unspecified"},
            "body": {"self_ref": "#/body", "children": [{"$ref": "#/texts/0"}], "content_layer": "body", "name": "_root_", "label": "unspecified"},
            "texts": [{
                "self_ref": "#/texts/0",
                "parent": {"$ref": "#/body"},
                "children": [],
                "content_layer": "body",
                "label": "text",
                "prov": [{"page_no": 1, "bbox": {"l": 0, "t": 0, "r": 100, "b": 20, "coord_origin": "BOTTOMLEFT"}, "charspan": [0, 51]}],
                "orig": "My SSN is 456-78-9012 and phone is 555-987-6543.",
                "text": "My SSN is 456-78-9012 and phone is 555-987-6543."
            }],
            "tables": [], "key_value_items": [], "form_items": [], "pictures": [], "groups": [], "pages": {}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.docling.json', delete=False) as f:
            json.dump(document_data, f, indent=2)
            temp_path = f.name

        try:
            # Load document
            processor = DocumentProcessor()
            document = processor.load_document(temp_path)

            # Extract text
            extractor = TextExtractor()
            text_segments = extractor.extract_text_segments(document)
            full_text = extractor.extract_full_text(document)

            # Use Presidio analyzer
            analyzer = AnalyzerEngine()
            presidio_results = analyzer.analyze(text=full_text, language='en')

            # Convert to our format - simple position mapping for single segment
            detected_entities = [
                RecognizerResult(
                    entity_type=result.entity_type,
                    start=result.start,
                    end=result.end,
                    score=result.score
                )
                for result in presidio_results
            ]

            # Set up policy with strategies for detected entity types
            entity_strategies = {}
            for entity in detected_entities:
                if entity.entity_type in ["PHONE_NUMBER"]:
                    entity_strategies[entity.entity_type] = PHONE_TEMPLATE
                elif entity.entity_type in ["US_SSN"]:
                    entity_strategies[entity.entity_type] = Strategy(
                        StrategyKind.PARTIAL,
                        {"visible_chars": 4, "position": "end"}
                    )
                else:
                    entity_strategies[entity.entity_type] = Strategy(
                        StrategyKind.TEMPLATE,
                        {"template": f"[{entity.entity_type}]"}
                    )

            policy = MaskingPolicy(per_entity=entity_strategies)

            # Apply masking
            masking_engine = MaskingEngine()
            result = masking_engine.mask_document(
                document=document,
                entities=detected_entities,
                policy=policy,
                text_segments=text_segments
            )

            # Verify masking worked
            masked_text = result.masked_document.texts[0].text

            # Should not contain original PII
            assert "456-78-9012" not in masked_text
            assert "555-987-6543" not in masked_text

            # Should contain masked values
            # (Exact format depends on Presidio detection and our strategies)
            assert len(result.cloakmap.anchors) > 0

            # Verify CloakMap integrity
            assert result.cloakmap.doc_id == "presidio_test"
            assert all(anchor.confidence > 0 for anchor in result.cloakmap.anchors)

        finally:
            Path(temp_path).unlink()

    def test_empty_document_handling(self):
        """Test handling of documents with no text content."""
        document_data = {
            "schema_name": "DoclingDocument",
            "version": "1.4.0",
            "name": "empty_doc",
            "origin": {"mimetype": "application/json", "filename": "empty.json", "binary_hash": 123456789},
            "furniture": {"self_ref": "#/furniture", "children": [], "content_layer": "furniture", "name": "_root_", "label": "unspecified"},
            "body": {"self_ref": "#/body", "children": [], "content_layer": "body", "name": "_root_", "label": "unspecified"},
            "texts": [],
            "tables": [], "key_value_items": [], "form_items": [], "pictures": [], "groups": [], "pages": {}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.docling.json', delete=False) as f:
            json.dump(document_data, f, indent=2)
            temp_path = f.name

        try:
            processor = DocumentProcessor()
            document = processor.load_document(temp_path)

            extractor = TextExtractor()
            text_segments = extractor.extract_text_segments(document)

            # No text segments expected
            assert len(text_segments) == 0

            # Masking empty entities should work fine
            masking_engine = MaskingEngine()
            policy = MaskingPolicy()

            result = masking_engine.mask_document(
                document=document,
                entities=[],  # No entities to mask
                policy=policy,
                text_segments=text_segments
            )

            # Should succeed with empty results
            assert result.masked_document is not None
            assert result.cloakmap is not None
            assert len(result.cloakmap.anchors) == 0

        finally:
            Path(temp_path).unlink()
