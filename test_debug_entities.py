#!/usr/bin/env python3
"""Debug entity detection."""

from cloakpivot.engine import CloakEngine
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem

# Create a document with SSN
doc = DoclingDocument(name="test_doc.txt")
text_item = TextItem(
    text="SSN: 123-45-6789, Phone: 555-123-4567",
    self_ref="#/texts/0",
    label="text",
    orig="SSN: 123-45-6789, Phone: 555-123-4567"
)
doc.texts = [text_item]

# Create engine and detect entities
engine = CloakEngine()

# Extract text and detect entities with detailed output
from presidio_analyzer import AnalyzerEngine
analyzer = AnalyzerEngine()

# Detect all entities
entities_detected = analyzer.analyze(
    text=text_item.text,
    language="en"
)

print("Detected entities:")
for entity in entities_detected:
    print(f"  - {entity.entity_type}: '{text_item.text[entity.start:entity.end]}' (confidence: {entity.score})")

# Now try masking
result = engine.mask_document(doc)
print(f"\nMasking result:")
print(f"Original: {text_item.text}")
print(f"Masked:   {result.document.texts[0].text}")
print(f"Entities found: {result.entities_found}")