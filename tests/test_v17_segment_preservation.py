"""Test that v1.7.0 documents preserve segment structure after masking."""

import pytest
from docling_core.types.doc.document import DoclingDocument, TextItem, DocItemLabel
from presidio_analyzer import RecognizerResult

from cloakpivot import (
    TextExtractor,
    MaskingEngine,
    MaskingPolicy,
    Strategy,
    StrategyKind
)


def test_v17_multiple_segments_preserved():
    """Test that masking preserves v1.7.0 document structure with multiple segments."""
    
    # Create a v1.7.0-style document with multiple segments
    doc = DoclingDocument(name="test_v17")
    doc.version = "1.7.0"
    
    # Add multiple text segments (simulating the bug report scenario)
    test_segments = [
        "---------- Forwarded message ----------",
        "From: Cameron MacIntyre <cameron@example.com>",
        "Date: Tuesday, September 2 2025 at 12:19 PM EDT",
        "Subject: Meeting confirmation",
        "To: Alice Johnson <alice@company.com>",
        "The meeting is scheduled for tomorrow.",
        "Please confirm your attendance.",
        "Best regards,",
        "Cameron",
    ]
    
    for i, text in enumerate(test_segments):
        text_item = TextItem(
            text=text,
            self_ref=f"#/texts/{i}",
            label=DocItemLabel.TEXT,
            orig=text
        )
        doc.texts.append(text_item)
    
    # Extract text segments
    extractor = TextExtractor()
    segments = extractor.extract_text_segments(doc)
    full_text = extractor.extract_full_text(doc)
    
    # Find actual positions in the full text
    cameron_start = full_text.find("Cameron MacIntyre")
    cameron_end = cameron_start + len("Cameron MacIntyre")
    
    cameron_email_start = full_text.find("cameron@example.com")
    cameron_email_end = cameron_email_start + len("cameron@example.com")
    
    date_start = full_text.find("Tuesday, September 2 2025 at 12:19 PM EDT")
    date_end = date_start + len("Tuesday, September 2 2025 at 12:19 PM EDT")
    
    alice_start = full_text.find("Alice Johnson")
    alice_end = alice_start + len("Alice Johnson")
    
    alice_email_start = full_text.find("alice@company.com")
    alice_email_end = alice_email_start + len("alice@company.com")
    
    # Create mock entities
    entities = [
        RecognizerResult(
            entity_type="PERSON",
            start=cameron_start,
            end=cameron_end,
            score=0.95
        ),
        RecognizerResult(
            entity_type="EMAIL_ADDRESS",
            start=cameron_email_start,
            end=cameron_email_end,
            score=0.99
        ),
        RecognizerResult(
            entity_type="DATE_TIME",
            start=date_start,
            end=date_end,
            score=0.90
        ),
        RecognizerResult(
            entity_type="PERSON",
            start=alice_start,
            end=alice_end,
            score=0.93
        ),
        RecognizerResult(
            entity_type="EMAIL_ADDRESS",
            start=alice_email_start,
            end=alice_email_end,
            score=0.99
        ),
    ]
    
    # Create masking policy
    policy = MaskingPolicy(
        per_entity={
            "EMAIL_ADDRESS": Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL]"}),
            "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[NAME]"}),
            "DATE_TIME": Strategy(StrategyKind.TEMPLATE, {"template": "[DATE]"}),
        },
        default_strategy=Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"})
    )
    
    # Apply masking
    engine = MaskingEngine(use_presidio_engine=True)
    result = engine.mask_document(
        document=doc,
        entities=entities,
        policy=policy,
        text_segments=segments
    )
    
    # Verify structure is preserved
    assert len(result.masked_document.texts) == len(doc.texts), \
        f"Expected {len(doc.texts)} segments, got {len(result.masked_document.texts)}"
    
    # Check that masks were applied correctly
    assert "[NAME]" in result.masked_document.texts[1].text
    assert "[EMAIL]" in result.masked_document.texts[1].text
    assert "[DATE]" in result.masked_document.texts[2].text
    assert "[NAME]" in result.masked_document.texts[4].text
    assert "[EMAIL]" in result.masked_document.texts[4].text
    
    # Verify no corruption (overlapping masks)
    for i, text_segment in enumerate(result.masked_document.texts):
        # Check for corrupted masks like those in the bug report
        assert "[D[NAME]" not in text_segment.text, \
            f"Found corrupted mask in segment {i}: {text_segment.text}"
        assert "[EMAIL]TE]" not in text_segment.text, \
            f"Found corrupted mask in segment {i}: {text_segment.text}"
        assert "NA[EMAIL]TE]" not in text_segment.text, \
            f"Found corrupted mask in segment {i}: {text_segment.text}"
    
    # Verify specific segments have expected content
    assert result.masked_document.texts[0].text == "---------- Forwarded message ----------"
    assert result.masked_document.texts[1].text == "From: [NAME] <[EMAIL]>"
    assert result.masked_document.texts[2].text == "Date: [DATE]"
    assert result.masked_document.texts[3].text == "Subject: Meeting confirmation"
    assert result.masked_document.texts[4].text == "To: [NAME] <[EMAIL]>"
    assert result.masked_document.texts[5].text == "The meeting is scheduled for tomorrow."


def test_empty_segments_handled():
    """Test that documents with empty segments are handled correctly."""
    
    doc = DoclingDocument(name="test_empty")
    doc.version = "1.7.0"
    
    # Add segments including empty ones
    test_segments = ["First segment", "", "Third segment", ""]
    
    for i, text in enumerate(test_segments):
        text_item = TextItem(
            text=text,
            self_ref=f"#/texts/{i}",
            label=DocItemLabel.TEXT,
            orig=text
        )
        doc.texts.append(text_item)
    
    # Extract and mask (no entities in this test)
    extractor = TextExtractor()
    segments = extractor.extract_text_segments(doc)
    
    policy = MaskingPolicy(
        default_strategy=Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"})
    )
    
    engine = MaskingEngine(use_presidio_engine=True)
    result = engine.mask_document(
        document=doc,
        entities=[],
        policy=policy,
        text_segments=segments
    )
    
    # All segments should be preserved, even empty ones
    assert len(result.masked_document.texts) == len(doc.texts)