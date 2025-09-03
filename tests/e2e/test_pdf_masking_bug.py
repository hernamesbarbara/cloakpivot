#!/usr/bin/env python3
"""
End-to-End test for PDF masking bug discovered in examples/pdf_masking_with_localstorage.py

Bug Description:
- Some PERSON entities are detected by Presidio but NOT masked in the output
- Example: "Cameron MacIntyre" is detected with 0.85 confidence but remains unmasked
- Some names like "inness robins" are not detected at all (NER model limitation)

This test reproduces the minimal case to help diagnose the issue.
"""

from pathlib import Path

import pytest
from docling.document_converter import DocumentConverter
from presidio_analyzer import AnalyzerEngine

from cloakpivot import (
    MaskingEngine,
    MaskingPolicy,
    Strategy,
    StrategyKind,
    TextExtractor,
)


@pytest.mark.e2e
class TestPDFMaskingBug:
    """Test cases for unmasked PII bug in PDF processing."""

    def setup_method(self):
        """Set up test dependencies."""
        self.converter = DocumentConverter()
        self.analyzer = AnalyzerEngine()
        self.extractor = TextExtractor()
        self.masking_engine = MaskingEngine(resolve_conflicts=True)

    def test_minimal_masking_case(self):
        """
        Minimal test case reproducing the unmasked PERSON entity bug.

        Expected: All detected PERSON entities should be masked
        Actual: Some PERSON entities remain unmasked despite being detected
        """
        # Create a minimal DoclingDocument-like structure with test data
        test_texts = [
            "From: Cameron MacIntyre <cameron@example.com>",
            "To: Julian Margaret <jmargaret@company.com>",
            "Subject: Re: FYI emailing with from inness robins",
            "Hi Cameron MacIntyre, this is a test message.",
        ]

        # Detect entities in each text
        all_detections = []
        for i, text in enumerate(test_texts):
            entities = self.analyzer.analyze(text=text, language="en")
            print(f"\nText {i}: {text}")
            for entity in entities:
                entity_text = text[entity.start:entity.end]
                print(f"  Detected: {entity.entity_type} = '{entity_text}' (score: {entity.score:.2f})")
                all_detections.append({
                    "text_index": i,
                    "text": text,
                    "entity_type": entity.entity_type,
                    "entity_text": entity_text,
                    "start": entity.start,
                    "end": entity.end,
                    "score": entity.score
                })

        # Check specific expectations
        person_detections = [d for d in all_detections if d["entity_type"] == "PERSON"]

        # Assert we detected the expected PERSON entities
        person_names = [d["entity_text"] for d in person_detections]
        assert "Cameron MacIntyre" in person_names, "Should detect 'Cameron MacIntyre' as PERSON"
        assert "Julian Margaret" in person_names, "Should detect 'Julian Margaret' as PERSON"

        # Note: "inness robins" is NOT detected - this is a known NER limitation
        assert "inness robins" not in person_names, "Known issue: 'inness robins' not detected"

        # Print summary for debugging
        print("\n" + "=" * 60)
        print("PERSON entities detected:")
        for detection in person_detections:
            print(f"  - '{detection['entity_text']}' in text {detection['text_index']} (score: {detection['score']:.2f})")

    def test_full_pdf_masking_workflow(self):
        """
        Test the complete PDF masking workflow to reproduce the bug.

        This test processes the actual email.pdf file and checks if all
        detected PERSON entities are properly masked.
        """
        pdf_path = Path("data/pdf/email.pdf")

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")

        # Convert PDF to DoclingDocument
        dl_doc = self.converter.convert(pdf_path).document

        # Extract text segments
        text_segments = self.extractor.extract_text_segments(dl_doc)

        # Detect all entities
        all_entities = []
        entity_types_found = set()

        for segment in text_segments:
            segment_entities = self.analyzer.analyze(text=segment.text, language="en")

            # Adjust positions to global coordinates
            for entity in segment_entities:
                from presidio_analyzer import RecognizerResult
                adjusted_entity = RecognizerResult(
                    entity_type=entity.entity_type,
                    start=entity.start + segment.start_offset,
                    end=entity.end + segment.start_offset,
                    score=entity.score,
                    analysis_explanation=entity.analysis_explanation,
                )
                all_entities.append(adjusted_entity)
                entity_types_found.add(entity.entity_type)

        # Create masking policy
        policy = MaskingPolicy(
            per_entity={
                "EMAIL_ADDRESS": Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL]"}),
                "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[NAME]"}),
                "PHONE_NUMBER": Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"}),
                "LOCATION": Strategy(StrategyKind.TEMPLATE, {"template": "[LOCATION]"}),
                "DATE_TIME": Strategy(StrategyKind.TEMPLATE, {"template": "[DATE]"}),
            },
            default_strategy=Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"})
        )

        # Mask the document with fixed conflict resolution config
        from cloakpivot.core.normalization import ConflictResolutionConfig
        conflict_config = ConflictResolutionConfig(
            merge_threshold_chars=0  # Don't group adjacent entities (bug fix)
        )

        # Create masking engine with the fix
        self.masking_engine = MaskingEngine(
            resolve_conflicts=True,
            conflict_resolution_config=conflict_config
        )

        # Mask the document
        mask_result = self.masking_engine.mask_document(
            document=dl_doc,
            entities=all_entities,
            policy=policy,
            text_segments=text_segments
        )

        # Check masking results
        print("\n" + "=" * 60)
        print("Masking Results:")
        print(f"Total entities detected: {len(all_entities)}")
        print(f"Total anchors created: {len(mask_result.cloakmap.anchors)}")
        print(f"Entity types found: {entity_types_found}")

        # Count PERSON entities
        person_entities = [e for e in all_entities if e.entity_type == "PERSON"]
        person_anchors = [a for a in mask_result.cloakmap.anchors if a.entity_type == "PERSON"]

        print(f"\nPERSON entities detected: {len(person_entities)}")
        print(f"PERSON anchors created: {len(person_anchors)}")

        # Check specific text items for proper masking
        original_texts = dl_doc.texts
        masked_texts = mask_result.masked_document.texts

        # Check the "From" line (text index 1)
        from_line_original = original_texts[1].text if len(original_texts) > 1 else ""
        from_line_masked = masked_texts[1].text if len(masked_texts) > 1 else ""

        print("\nFrom line check:")
        print(f"  Original: {from_line_original}")
        print(f"  Masked:   {from_line_masked}")

        # Assert that Cameron MacIntyre was masked
        if "Cameron MacIntyre" in from_line_original:
            assert "Cameron MacIntyre" not in from_line_masked, \
                "BUG: 'Cameron MacIntyre' was detected but not masked!"
            assert "[NAME]" in from_line_masked or "Cameron MacIntyre" not in from_line_masked, \
                "PERSON entity should be replaced with [NAME]"

        # Check the Subject line (text index 3)
        subject_line_original = original_texts[3].text if len(original_texts) > 3 else ""
        subject_line_masked = masked_texts[3].text if len(masked_texts) > 3 else ""

        print("\nSubject line check:")
        print(f"  Original: {subject_line_original}")
        print(f"  Masked:   {subject_line_masked}")

        # Note: "inness robins" is not detected, so it won't be masked
        # This is a known NER limitation, not a masking bug

        # Final assertion: number of PERSON anchors should match PERSON entities
        assert len(person_anchors) == len(person_entities), \
            f"BUG: Mismatch between detected PERSON entities ({len(person_entities)}) " \
            f"and created anchors ({len(person_anchors)})"


if __name__ == "__main__":
    # Run the test directly for debugging
    test = TestPDFMaskingBug()
    test.setup_method()

    print("Running minimal masking case test...")
    test.test_minimal_masking_case()

    print("\n" + "=" * 60)
    print("Running full PDF masking workflow test...")
    test.test_full_pdf_masking_workflow()
