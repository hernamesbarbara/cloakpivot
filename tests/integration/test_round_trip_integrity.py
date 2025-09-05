"""Round-trip integrity validation tests for Presidio integration."""

import hashlib

import pytest
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TableItem, TextItem
from presidio_analyzer import AnalyzerEngine

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.document.extractor import TextSegment
from cloakpivot.masking.engine import MaskingEngine
from cloakpivot.masking.presidio_adapter import PresidioMaskingAdapter
from cloakpivot.unmasking.engine import UnmaskingEngine
from cloakpivot.unmasking.presidio_adapter import PresidioUnmaskingAdapter


class TestRoundTripIntegrity:
    """Validate perfect round-trip restoration."""

    @pytest.fixture
    def analyzer_engine(self):
        """Create a Presidio analyzer engine for entity detection."""
        return AnalyzerEngine()

    def _create_text_segments(self, document: DoclingDocument) -> list[TextSegment]:
        """Helper to create text segments from document."""
        return [
            TextSegment(
                node_id="#/texts/0",
                text=document._main_text,
                start_offset=0,
                end_offset=len(document._main_text),
                node_type="TextItem"
            )
        ]

    def _analyze_document(self, document: DoclingDocument, analyzer: AnalyzerEngine) -> list:
        """Helper to analyze document and get entities."""
        return analyzer.analyze(text=document._main_text, language="en")

    @pytest.fixture
    def complex_document(self) -> DoclingDocument:
        """Create a complex document with various content types."""
        doc = DoclingDocument(name="complex_test.txt")

        # Main text content
        doc._main_text = """
        === EMPLOYEE RECORD ===

        Name: Robert James Anderson
        Employee ID: EMP-2024-78945
        Department: Information Technology

        Contact Information:
        - Phone: (555) 234-5678
        - Mobile: +1-555-987-6543
        - Email: r.anderson@techcorp.com
        - Personal: robert.anderson.personal@gmail.com

        Personal Details:
        - DOB: March 15, 1985
        - SSN: 123-45-6789
        - Driver's License: D123-4567-8901

        Financial Information:
        - Salary: $145,000
        - Bank Account: ****7890 (Wells Fargo)
        - Credit Card: 4532-1234-5678-9012
        - 401k Account: 401K-RA-2024

        Medical Information:
        - Insurance ID: HLTH-123456
        - Blood Type: O+
        - Allergies: Penicillin
        - Medications: Lisinopril 10mg

        Security Clearance:
        - Level: Top Secret (TS-2024-0123)
        - Granted: January 15, 2024
        - Expires: January 15, 2029

        Performance Reviews:
        - 2023: Exceeds Expectations (4.5/5.0)
        - 2022: Meets Expectations (3.8/5.0)

        Notes: Eligible for senior position promotion in Q2 2024.
        Schedule review meeting with HR by April 30, 2024.
        """

        # Add text items
        text_item = TextItem(
            text="Employee Record for Robert James Anderson",
            prov=[{"page_no": 1, "bbox": {"l": 0, "t": 0, "r": 100, "b": 20}}]
        )
        doc.texts.append(text_item)

        # Add table data
        table_data = [
            ["Field", "Value", "Status"],
            ["Employee ID", "EMP-2024-78945", "Active"],
            ["Email", "r.anderson@techcorp.com", "Verified"],
            ["Phone", "(555) 234-5678", "Primary"],
            ["SSN", "123-45-6789", "Secured"],
        ]

        table_item = TableItem(
            data=table_data,
            prov=[{"page_no": 1, "bbox": {"l": 0, "t": 100, "r": 300, "b": 200}}]
        )
        doc.tables.append(table_item)

        # Add key-value items
        doc.key_value_items.extend([
            {"key": "Name", "value": "Robert James Anderson"},
            {"key": "Department", "value": "Information Technology"},
            {"key": "Salary", "value": "$145,000"},
        ])

        return doc

    @pytest.fixture
    def all_strategies_policy(self) -> MaskingPolicy:
        """Create a policy using all strategy types."""
        return MaskingPolicy(
            per_entity={
                "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[PERSON_{}]"}),
                "EMAIL_ADDRESS": Strategy(StrategyKind.REDACT, {"value": "[EMAIL]"}),
                "PHONE_NUMBER": Strategy(StrategyKind.PARTIAL, {"prefix": 6, "suffix": 0}),
                "US_SSN": Strategy(StrategyKind.REDACT, {"char": "#"}),
                "CREDIT_CARD": Strategy(StrategyKind.HASH, {"char_set": "0-9"}),
                "DATE_TIME": Strategy(StrategyKind.CUSTOM, {}),
                "LOCATION": Strategy(StrategyKind.REDACT, {"value": "[LOCATION]"}),
                "US_DRIVER_LICENSE": Strategy(StrategyKind.REDACT, {"char": "X"}),
                "MEDICAL_LICENSE": Strategy(StrategyKind.TEMPLATE, {"template": "[ID_{}]"}),
                "US_BANK_NUMBER": Strategy(StrategyKind.PARTIAL, {"prefix": 0, "suffix": 4}),
            }
        )

    @pytest.mark.parametrize("engine_combo", [
        ("legacy", "legacy"),
        ("presidio", "presidio"),
        ("legacy", "presidio"),
        ("presidio", "legacy")
    ])
    def test_round_trip_integrity(self, engine_combo, complex_document, all_strategies_policy):
        """Test round-trip with all engine combinations."""
        mask_type, unmask_type = engine_combo

        # Setup masking engine
        if mask_type == "legacy":
            masking_engine = MaskingEngine(use_presidio_engine=False)
        else:
            masking_engine = MaskingEngine(use_presidio_engine=True)

        # Setup unmasking engine
        if unmask_type == "legacy":
            unmasking_engine = UnmaskingEngine(use_presidio_engine=False)
        else:
            unmasking_engine = UnmaskingEngine(use_presidio_engine=True)

        # Perform masking
        masked_result = masking_engine.mask_document(complex_document, all_strategies_policy)

        # Validate masking occurred
        assert masked_result.masked_document is not None
        assert masked_result.cloakmap is not None
        assert masked_result.masked_document._main_text != complex_document._main_text

        # Perform unmasking
        unmasked_doc = unmasking_engine.unmask_document(
            masked_result.masked_document,
            masked_result.cloakmap
        )

        # Validate perfect restoration
        assert unmasked_doc._main_text == complex_document._main_text

        # Validate text items restoration
        assert len(unmasked_doc.texts) == len(complex_document.texts)
        for i, text_item in enumerate(unmasked_doc.texts):
            assert text_item.text == complex_document.texts[i].text
            assert text_item.prov == complex_document.texts[i].prov

        # Validate table restoration
        assert len(unmasked_doc.tables) == len(complex_document.tables)
        for i, table_item in enumerate(unmasked_doc.tables):
            assert table_item.data == complex_document.tables[i].data
            assert table_item.prov == complex_document.tables[i].prov

        # Validate key-value items restoration
        assert unmasked_doc.key_value_items == complex_document.key_value_items

    def test_content_preservation(self, complex_document, all_strategies_policy):
        """Verify exact content restoration."""
        # Use Presidio for this test
        masking_engine = MaskingEngine(
            enable_custom_recognizer=False,
            presidio_adapter=PresidioMaskingAdapter()
        )

        unmasking_engine = UnmaskingEngine(
            enable_anchor_strategy=False,
            presidio_adapter=PresidioUnmaskingAdapter()
        )

        # Create content hash before masking
        original_hash = hashlib.sha256(complex_document._main_text.encode()).hexdigest()

        # Mask and unmask
        masked_result = masking_engine.mask_document(complex_document, all_strategies_policy)
        unmasked_doc = unmasking_engine.unmask_document(
            masked_result.masked_document,
            masked_result.cloakmap
        )

        # Create content hash after round-trip
        restored_hash = hashlib.sha256(unmasked_doc._main_text.encode()).hexdigest()

        # Verify exact match
        assert original_hash == restored_hash
        assert unmasked_doc._main_text == complex_document._main_text

        # Character-by-character comparison
        for i, (orig_char, restored_char) in enumerate(zip(complex_document._main_text, unmasked_doc._main_text)):
            assert orig_char == restored_char, f"Mismatch at position {i}: '{orig_char}' != '{restored_char}'"

    def test_document_structure_preservation(self, complex_document, all_strategies_policy):
        """Verify DoclingDocument structure preservation."""
        # Test with Presidio
        masking_engine = MaskingEngine(
            enable_custom_recognizer=False,
            presidio_adapter=PresidioMaskingAdapter()
        )

        unmasking_engine = UnmaskingEngine(
            enable_anchor_strategy=False,
            presidio_adapter=PresidioUnmaskingAdapter()
        )

        # Mask and unmask
        masked_result = masking_engine.mask_document(complex_document, all_strategies_policy)
        unmasked_doc = unmasking_engine.unmask_document(
            masked_result.masked_document,
            masked_result.cloakmap
        )

        # Verify document attributes
        assert unmasked_doc.name == complex_document.name
        assert isinstance(unmasked_doc, type(complex_document))

        # Verify all document sections
        assert hasattr(unmasked_doc, '_main_text')
        assert hasattr(unmasked_doc, 'texts')
        assert hasattr(unmasked_doc, 'tables')
        assert hasattr(unmasked_doc, 'key_value_items')

        # Verify section lengths
        assert len(unmasked_doc.texts) == len(complex_document.texts)
        assert len(unmasked_doc.tables) == len(complex_document.tables)
        assert len(unmasked_doc.key_value_items) == len(complex_document.key_value_items)

    def test_metadata_preservation(self, complex_document, all_strategies_policy):
        """Verify all metadata is preserved correctly."""
        # Use mixed engines for this test
        masking_engine = MaskingEngine(
            enable_custom_recognizer=True,
            presidio_adapter=None
        )

        unmasking_engine = UnmaskingEngine(
            enable_anchor_strategy=False,
            presidio_adapter=PresidioUnmaskingAdapter()
        )

        # Mask the document
        masked_result = masking_engine.mask_document(complex_document, all_strategies_policy)

        # Verify CloakMap metadata
        assert masked_result.cloakmap is not None
        assert hasattr(masked_result.cloakmap, 'version')
        assert hasattr(masked_result.cloakmap, 'transformations')
        assert len(masked_result.cloakmap.transformations) > 0

        # Each transformation should have required fields
        for transform in masked_result.cloakmap.transformations:
            assert 'start' in transform
            assert 'end' in transform
            assert 'entity_type' in transform
            assert 'original_text' in transform
            assert 'new_text' in transform
            assert 'strategy' in transform

        # Unmask and verify restoration
        unmasked_doc = unmasking_engine.unmask_document(
            masked_result.masked_document,
            masked_result.cloakmap
        )

        # Verify metadata restoration
        assert unmasked_doc._main_text == complex_document._main_text

    def test_whitespace_preservation(self):
        """Test that whitespace is preserved exactly."""
        doc = DoclingDocument(name="whitespace.txt")
        doc._main_text = """
        Line with spaces:     John Smith
        Line with tabs:	Jane	Doe
        Multiple newlines:


        Email: john@example.com

        Mixed whitespace:	  Bob   Johnson
        """

        policy = MaskingPolicy(
            per_entity={
                "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[PERSON]"}),
                "EMAIL_ADDRESS": Strategy(StrategyKind.REDACT, {"value": "[EMAIL]"}),
            }
        )

        # Test with Presidio
        masking_engine = MaskingEngine(
            enable_custom_recognizer=False,
            presidio_adapter=PresidioMaskingAdapter()
        )

        unmasking_engine = UnmaskingEngine(
            enable_anchor_strategy=False,
            presidio_adapter=PresidioUnmaskingAdapter()
        )

        masked_result = masking_engine.mask_document(doc, policy)
        unmasked_doc = unmasking_engine.unmask_document(
            masked_result.masked_document,
            masked_result.cloakmap
        )

        # Verify exact whitespace preservation
        assert unmasked_doc._main_text == doc._main_text

        # Count whitespace characters
        original_spaces = doc._main_text.count(' ')
        original_tabs = doc._main_text.count('\t')
        original_newlines = doc._main_text.count('\n')

        restored_spaces = unmasked_doc._main_text.count(' ')
        restored_tabs = unmasked_doc._main_text.count('\t')
        restored_newlines = unmasked_doc._main_text.count('\n')

        assert original_spaces == restored_spaces
        assert original_tabs == restored_tabs
        assert original_newlines == restored_newlines

    def test_special_characters_preservation(self):
        """Test preservation of special and Unicode characters."""
        doc = DoclingDocument(name="special.txt")
        doc._main_text = """
        Special chars: @#$%^&*()_+-={}[]|\\:";'<>?,./
        Unicode: café, naïve, résumé, Zürich
        Emojis: 😀 😃 😄 🎉 🚀
        Math: α β γ δ ε ∑ ∏ ∫ ∞
        Currency: $100 €50 £75 ¥1000
        Quotes: "double" 'single' «guillemets» „German"

        Contact: François Müller at françois@société.fr
        Phone: +33 (0) 6 12-34-56-78
        """

        policy = MaskingPolicy(
            per_entity={
                "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[PERSON]"}),
                "EMAIL_ADDRESS": Strategy(StrategyKind.REDACT, {"value": "[EMAIL]"}),
                "PHONE_NUMBER": Strategy(StrategyKind.REDACT, {"value": "[PHONE]"}),
            }
        )

        # Test with both engine types
        for use_presidio in [True, False]:
            masking_engine = MaskingEngine(
                enable_custom_recognizer=not use_presidio,
                presidio_adapter=PresidioMaskingAdapter() if use_presidio else None
            )

            unmasking_engine = UnmaskingEngine(
                enable_anchor_strategy=not use_presidio,
                presidio_adapter=PresidioUnmaskingAdapter() if use_presidio else None
            )

            masked_result = masking_engine.mask_document(doc, policy)
            unmasked_doc = unmasking_engine.unmask_document(
                masked_result.masked_document,
                masked_result.cloakmap
            )

            # Verify exact preservation
            assert unmasked_doc._main_text == doc._main_text

            # Verify specific special characters
            assert "café" in unmasked_doc._main_text
            assert "résumé" in unmasked_doc._main_text
            assert "😀" in unmasked_doc._main_text
            assert "∑" in unmasked_doc._main_text
            assert "€50" in unmasked_doc._main_text
            assert "«guillemets»" in unmasked_doc._main_text

    def test_boundary_cases(self):
        """Test edge cases at text boundaries."""
        test_cases = [
            # Empty document
            ("", MaskingPolicy(per_entity={})),

            # Single character
            ("A", MaskingPolicy(per_entity={})),

            # Only whitespace
            ("   \t\n  ", MaskingPolicy(per_entity={})),

            # Entity at start
            ("john@example.com is the email", MaskingPolicy(
                per_entity={"EMAIL_ADDRESS": Strategy(StrategyKind.REDACT, {"value": "[EMAIL]"})}
            )),

            # Entity at end
            ("The email is john@example.com", MaskingPolicy(
                per_entity={"EMAIL_ADDRESS": Strategy(StrategyKind.REDACT, {"value": "[EMAIL]"})}
            )),

            # Back-to-back entities
            ("555-1234john@example.com", MaskingPolicy(
                per_entity={
                    "PHONE_NUMBER": Strategy(StrategyKind.REDACT, {"value": "[PHONE]"}),
                    "EMAIL_ADDRESS": Strategy(StrategyKind.REDACT, {"value": "[EMAIL]"}),
                }
            )),
        ]

        for text, policy in test_cases:
            doc = DoclingDocument(name="boundary.txt")
            doc._main_text = text

            # Test with Presidio
            masking_engine = MaskingEngine(
                enable_custom_recognizer=False,
                presidio_adapter=PresidioMaskingAdapter()
            )

            unmasking_engine = UnmaskingEngine(
                enable_anchor_strategy=False,
                presidio_adapter=PresidioUnmaskingAdapter()
            )

            masked_result = masking_engine.mask_document(doc, policy)
            unmasked_doc = unmasking_engine.unmask_document(
                masked_result.masked_document,
                masked_result.cloakmap
            )

            assert unmasked_doc._main_text == doc._main_text, f"Failed for text: '{text}'"

    def test_large_text_integrity(self):
        """Test integrity with very large text blocks."""
        # Generate large text
        large_text_parts = []
        for i in range(1000):
            large_text_parts.append(f"""
            Record {i:05d}:
            Name: Person_{i:05d} Smith
            Email: person{i}@example.com
            Phone: 555-{i:04d}
            SSN: {i%1000:03d}-45-{6789+i:04d}
            Address: {i} Main Street, City {i%100}
            """)

        doc = DoclingDocument(name="large.txt")
        doc._main_text = "\n".join(large_text_parts)

        policy = MaskingPolicy(
            per_entity={
                "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[P]"}),
                "EMAIL_ADDRESS": Strategy(StrategyKind.REDACT, {"value": "[E]"}),
                "PHONE_NUMBER": Strategy(StrategyKind.REDACT, {"value": "[PH]"}),
                "US_SSN": Strategy(StrategyKind.REDACT, {"char": "#"}),
                "LOCATION": Strategy(StrategyKind.REDACT, {"value": "[L]"}),
            }
        )

        # Test with Presidio (more challenging due to entity detection)
        masking_engine = MaskingEngine(
            enable_custom_recognizer=False,
            presidio_adapter=PresidioMaskingAdapter()
        )

        unmasking_engine = UnmaskingEngine(
            enable_anchor_strategy=False,
            presidio_adapter=PresidioUnmaskingAdapter()
        )

        masked_result = masking_engine.mask_document(doc, policy)
        unmasked_doc = unmasking_engine.unmask_document(
            masked_result.masked_document,
            masked_result.cloakmap
        )

        # Verify integrity
        assert len(unmasked_doc._main_text) == len(doc._main_text)
        assert unmasked_doc._main_text == doc._main_text
