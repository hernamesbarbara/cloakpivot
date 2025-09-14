"""Test examples from documentation and specification."""

import pytest
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem

from cloakpivot.engine import CloakEngine
from cloakpivot.defaults import get_default_policy, get_conservative_policy


class TestSpecificationExamples:
    """Test examples from the specification document."""

    def test_simple_one_line_masking(self):
        """Test the simplest use case from the spec."""
        # From specification: Users can mask a document in 1-2 lines of code
        doc = DoclingDocument(name="test.txt")
        doc.texts = [TextItem(
            text="Contact John Doe at john.doe@example.com",
            self_ref="#/texts/0",
            label="text",
            orig="Contact John Doe at john.doe@example.com"
        )]

        engine = CloakEngine()
        result = engine.mask_document(doc)

        assert result.entities_found > 0
        assert "john.doe@example.com" not in result.document.texts[0].text

    def test_auto_detection_example(self):
        """Test auto-detection of common PII types."""
        # From specification: Auto-detect all common PII
        doc = DoclingDocument(name="test.txt")
        doc.texts = [TextItem(
            text="John Smith (SSN: 123-45-6789) lives at 123 Main St. "
                 "Email: john@example.com, Phone: 555-123-4567",
            self_ref="#/texts/0",
            label="text",
            orig="John Smith (SSN: 123-45-6789) lives at 123 Main St. "
                 "Email: john@example.com, Phone: 555-123-4567"
        )]

        engine = CloakEngine()
        result = engine.mask_document(doc)  # No entities specified = auto-detect

        masked_text = result.document.texts[0].text
        # Should detect and mask various PII types
        assert "john@example.com" not in masked_text
        assert "555-123-4567" not in masked_text
        # Note: SSN detection depends on Presidio configuration

    def test_specific_entities_only(self):
        """Test detecting specific entity types only."""
        # From specification: Detect specific entities only
        doc = DoclingDocument(name="test.txt")
        doc.texts = [TextItem(
            text="Contact Jane at jane@example.com or visit example.com",
            self_ref="#/texts/0",
            label="text",
            orig="Contact Jane at jane@example.com or visit example.com"
        )]

        engine = CloakEngine()
        result = engine.mask_document(doc, entities=['EMAIL_ADDRESS'])

        masked_text = result.document.texts[0].text
        # Only email should be masked
        assert "jane@example.com" not in masked_text
        # URL might not be masked since we only asked for EMAIL_ADDRESS
        # Name "Jane" should not be masked

    def test_custom_policy_example(self):
        """Test using a custom masking policy."""
        # From specification: Use custom policy
        doc = DoclingDocument(name="test.txt")
        doc.texts = [TextItem(
            text="Email: admin@company.com, ID: 12345",
            self_ref="#/texts/0",
            label="text",
            orig="Email: admin@company.com, ID: 12345"
        )]

        custom_policy = get_conservative_policy()
        engine = CloakEngine()
        result = engine.mask_document(doc, policy=custom_policy)

        # Conservative policy should mask aggressively
        assert "admin@company.com" not in result.document.texts[0].text

    def test_builder_pattern_example(self):
        """Test builder pattern configuration from spec."""
        # From specification: Advanced configuration via builder
        doc = DoclingDocument(name="test.txt")
        doc.texts = [TextItem(
            text="Contact support at support@example.com",
            self_ref="#/texts/0",
            label="text",
            orig="Contact support at support@example.com"
        )]

        engine = CloakEngine.builder()\
            .with_confidence_threshold(0.9)\
            .build()

        result = engine.mask_document(doc)
        assert result is not None
        assert "support@example.com" not in result.document.texts[0].text

    def test_round_trip_example(self):
        """Test mask/unmask round trip from specification."""
        doc = DoclingDocument(name="test.txt")
        original_text = "My email is test@example.com and phone is 555-1234"
        doc.texts = [TextItem(
            text=original_text,
            self_ref="#/texts/0",
            label="text",
            orig=original_text
        )]

        # Simple one-line masking
        engine = CloakEngine()
        masked_result = engine.mask_document(doc)

        # Verify masking worked
        assert masked_result.entities_found > 0
        masked_text = masked_result.document.texts[0].text
        assert "test@example.com" not in masked_text

        # Unmask when needed
        original = engine.unmask_document(
            masked_result.document,
            masked_result.cloakmap
        )

        # Verify round-trip preservation
        assert original.texts[0].text == original_text


class TestDocumentationExamples:
    """Test examples that would appear in documentation."""

    def test_basic_usage(self):
        """Test basic usage example for documentation."""
        from cloakpivot import CloakEngine

        # Create a document with PII
        doc = DoclingDocument(name="employee.txt")
        doc.texts = [TextItem(
            text="Employee: Alice Johnson, Email: alice@company.com",
            self_ref="#/texts/0",
            label="text",
            orig="Employee: Alice Johnson, Email: alice@company.com"
        )]

        # Mask PII with one line
        engine = CloakEngine()
        result = engine.mask_document(doc)

        print(f"Found {result.entities_found} PII entities")
        print(f"Masked text: {result.document.texts[0].text}")

        # Should mask the PII
        assert "alice@company.com" not in result.document.texts[0].text
        assert result.entities_found >= 1

    def test_policy_presets(self):
        """Test using policy presets."""
        from cloakpivot import CloakEngine
        from cloakpivot.defaults import get_policy_preset

        doc = DoclingDocument(name="data.txt")
        doc.texts = [TextItem(
            text="User data: john@example.com, ID: 12345, Phone: 555-0123",
            self_ref="#/texts/0",
            label="text",
            orig="User data: john@example.com, ID: 12345, Phone: 555-0123"
        )]

        # Use permissive policy preset
        permissive_policy = get_policy_preset("permissive")
        engine = CloakEngine(default_policy=permissive_policy)
        result = engine.mask_document(doc)

        # Should mask based on permissive policy
        assert result is not None

    def test_confidence_threshold_example(self):
        """Test confidence threshold configuration."""
        from cloakpivot import CloakEngine

        doc = DoclingDocument(name="test.txt")
        doc.texts = [TextItem(
            text="Contact: maybe-email@test or definite@example.com",
            self_ref="#/texts/0",
            label="text",
            orig="Contact: maybe-email@test or definite@example.com"
        )]

        # High confidence = fewer false positives
        high_conf_engine = CloakEngine(
            analyzer_config={'confidence_threshold': 0.95}
        )

        # Low confidence = catch more entities
        low_conf_engine = CloakEngine(
            analyzer_config={'confidence_threshold': 0.5}
        )

        high_result = high_conf_engine.mask_document(doc)
        low_result = low_conf_engine.mask_document(doc)

        # Low confidence might find more entities
        assert low_result.entities_found >= high_result.entities_found

    def test_selective_masking(self):
        """Test masking only specific PII types."""
        from cloakpivot import CloakEngine

        doc = DoclingDocument(name="mixed.txt")
        doc.texts = [TextItem(
            text="Name: Bob Smith, Email: bob@example.com, Date: 2024-01-15",
            self_ref="#/texts/0",
            label="text",
            orig="Name: Bob Smith, Email: bob@example.com, Date: 2024-01-15"
        )]

        engine = CloakEngine()

        # Mask only emails and phone numbers
        result = engine.mask_document(
            doc,
            entities=['EMAIL_ADDRESS', 'PHONE_NUMBER']
        )

        masked_text = result.document.texts[0].text
        # Email should be masked
        assert "bob@example.com" not in masked_text
        # Name might not be masked (not in entity list)
        # Date should not be masked

    def test_multiple_sections(self):
        """Test processing documents with multiple sections."""
        from cloakpivot import CloakEngine

        doc = DoclingDocument(name="report.txt")
        doc.texts = [
            TextItem(
                text="Section 1: Contact admin@example.com for help",
                self_ref="#/texts/0",
                label="text",
                orig="Section 1: Contact admin@example.com for help"
            ),
            TextItem(
                text="Section 2: Call support at 1-800-555-1234",
                self_ref="#/texts/1",
                label="text",
                orig="Section 2: Call support at 1-800-555-1234"
            ),
            TextItem(
                text="Section 3: Visit our office at 123 Main Street",
                self_ref="#/texts/2",
                label="text",
                orig="Section 3: Visit our office at 123 Main Street"
            )
        ]

        engine = CloakEngine()
        result = engine.mask_document(doc)

        # Should process all sections
        assert len(result.document.texts) == 3
        assert result.entities_found > 0

        # Check specific sections were masked
        assert "admin@example.com" not in result.document.texts[0].text
        assert "1-800-555-1234" not in result.document.texts[1].text