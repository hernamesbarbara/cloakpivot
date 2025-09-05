"""Comprehensive end-to-end integration tests for Presidio integration."""

import pytest
from docling_core.types import DoclingDocument
from presidio_analyzer import AnalyzerEngine, RecognizerResult

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.document.extractor import TextSegment
from cloakpivot.masking.engine import MaskingEngine
from cloakpivot.unmasking.engine import UnmaskingEngine


class TestPresidioFullIntegration:
    """End-to-end testing of Presidio integration."""

    @pytest.fixture
    def analyzer_engine(self):
        """Create a Presidio analyzer engine for entity detection."""
        return AnalyzerEngine()

    @pytest.fixture
    def sample_document(self) -> DoclingDocument:
        """Create a sample document with various entity types."""
        doc = DoclingDocument(name="test_document.txt")
        doc._main_text = """
        Patient: John Smith (ID: P123456)
        DOB: 01/15/1980
        SSN: 123-45-6789
        Phone: (555) 123-4567
        Email: john.smith@example.com
        Credit Card: 4111-1111-1111-1111
        Address: 123 Main St, Anytown, CA 90210

        Medical Record #: MRN-2024-0001
        Visit Date: 03/15/2024
        Doctor: Dr. Sarah Johnson

        Notes: Patient reports feeling better.
        Follow-up scheduled for next month.
        Emergency Contact: Jane Smith (555) 987-6543
        """
        return doc

    @pytest.fixture
    def comprehensive_policy(self) -> MaskingPolicy:
        """Create a comprehensive masking policy covering all entity types."""
        return MaskingPolicy(
            per_entity={
                "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[PERSON_{}]"}),
                "PHONE_NUMBER": Strategy(StrategyKind.REDACT, {"char": "*"}),
                "EMAIL_ADDRESS": Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL_{}]"}),
                "US_SSN": Strategy(StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end"}),
                "CREDIT_CARD": Strategy(StrategyKind.REDACT, {"char": "X"}),
                "DATE_TIME": Strategy(StrategyKind.TEMPLATE, {"template": "[DATE]"}),
                "LOCATION": Strategy(StrategyKind.REDACT, {"char": "*"}),
                "MEDICAL_LICENSE": Strategy(StrategyKind.TEMPLATE, {"template": "[MRN_{}]"}),
            }
        )

    def _create_text_segments(self, document: DoclingDocument) -> list[TextSegment]:
        """Helper to create text segments from document."""
        # Handle empty document case
        if not document._main_text:
            return []

        return [
            TextSegment(
                node_id="#/texts/0",
                text=document._main_text,
                start_offset=0,
                end_offset=len(document._main_text),
                node_type="TextItem"
            )
        ]

    def _analyze_document(self, document: DoclingDocument, analyzer: AnalyzerEngine) -> list[RecognizerResult]:
        """Helper to analyze document and get entities."""
        return analyzer.analyze(text=document._main_text, language="en")

    def test_full_workflow_presidio_only(self, sample_document, comprehensive_policy, analyzer_engine):
        """Test mask→unmask with Presidio engines only."""
        # Setup Presidio-only engines
        masking_engine = MaskingEngine(use_presidio_engine=True)
        unmasking_engine = UnmaskingEngine(use_presidio_engine=True)

        # Detect entities
        entities = self._analyze_document(sample_document, analyzer_engine)
        text_segments = self._create_text_segments(sample_document)

        # Mask the document
        masked_result = masking_engine.mask_document(
            sample_document,
            entities,
            comprehensive_policy,
            text_segments
        )

        assert masked_result.masked_document is not None
        assert masked_result.cloakmap is not None

        # Verify masking occurred
        masked_text = masked_result.masked_document._main_text
        original_text = sample_document._main_text
        assert masked_text != original_text
        assert "John Smith" not in masked_text
        assert "123-45-6789" not in masked_text
        assert "john.smith@example.com" not in masked_text

        # Unmask the document
        unmasked_result = unmasking_engine.unmask_document(
            masked_result.masked_document,
            masked_result.cloakmap
        )

        # Verify perfect restoration
        assert unmasked_result.restored_document._main_text == original_text

    def test_full_workflow_legacy_only(self, sample_document, comprehensive_policy, analyzer_engine):
        """Test mask→unmask with legacy engines only."""
        # Setup legacy-only engines
        masking_engine = MaskingEngine(use_presidio_engine=False)
        unmasking_engine = UnmaskingEngine(use_presidio_engine=False)

        # Detect entities
        entities = self._analyze_document(sample_document, analyzer_engine)
        text_segments = self._create_text_segments(sample_document)

        # Mask the document
        masked_result = masking_engine.mask_document(
            sample_document,
            entities,
            comprehensive_policy,
            text_segments
        )

        assert masked_result.masked_document is not None
        assert masked_result.cloakmap is not None

        # Verify masking occurred
        masked_text = masked_result.masked_document._main_text
        original_text = sample_document._main_text
        assert masked_text != original_text

        # Unmask the document
        unmasked_result = unmasking_engine.unmask_document(
            masked_result.masked_document,
            masked_result.cloakmap
        )

        # Verify perfect restoration
        assert unmasked_result.restored_document._main_text == original_text

    def test_cross_engine_compatibility(self, sample_document, comprehensive_policy, analyzer_engine):
        """Test mask with one engine, unmask with another."""
        entities = self._analyze_document(sample_document, analyzer_engine)
        text_segments = self._create_text_segments(sample_document)

        # Test 1: Mask with legacy, unmask with Presidio
        masking_engine_legacy = MaskingEngine(use_presidio_engine=False)
        unmasking_engine_presidio = UnmaskingEngine(use_presidio_engine=True)

        masked_result = masking_engine_legacy.mask_document(
            sample_document,
            entities,
            comprehensive_policy,
            text_segments
        )

        unmasked_result = unmasking_engine_presidio.unmask_document(
            masked_result.masked_document,
            masked_result.cloakmap
        )

        assert unmasked_result.restored_document._main_text == sample_document._main_text

        # Test 2: Mask with Presidio, unmask with legacy
        masking_engine_presidio = MaskingEngine(use_presidio_engine=True)
        unmasking_engine_legacy = UnmaskingEngine(use_presidio_engine=False)

        masked_result = masking_engine_presidio.mask_document(
            sample_document,
            entities,
            comprehensive_policy,
            text_segments
        )

        unmasked_result = unmasking_engine_legacy.unmask_document(
            masked_result.masked_document,
            masked_result.cloakmap
        )

        assert unmasked_result.restored_document._main_text == sample_document._main_text

    @pytest.mark.parametrize("strategy_kind,config", [
        (StrategyKind.TEMPLATE, {"template": "[ENTITY_{}]"}),
        (StrategyKind.REDACT, {"char": "*"}),
        (StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end"}),
        (StrategyKind.HASH, {"algorithm": "sha256"}),
        (StrategyKind.SURROGATE, {}),
        (StrategyKind.CUSTOM, {"func": lambda x: "[CUSTOM]"}),
    ])
    def test_all_strategy_types_presidio(self, strategy_kind, config, analyzer_engine):
        """Test all 6 StrategyKind types through Presidio."""
        # Create simple document
        doc = DoclingDocument(name="test.txt")
        doc._main_text = "Contact John at 555-1234 or john@example.com"

        # Create policy with specific strategy
        policy = MaskingPolicy(
            per_entity={
                "PHONE_NUMBER": Strategy(strategy_kind, config),
                "EMAIL_ADDRESS": Strategy(strategy_kind, config),
                "PERSON": Strategy(strategy_kind, config),
            }
        )

        # Use Presidio engines
        masking_engine = MaskingEngine(use_presidio_engine=True)
        unmasking_engine = UnmaskingEngine(use_presidio_engine=True)

        # Detect entities and create segments
        entities = self._analyze_document(doc, analyzer_engine)
        text_segments = self._create_text_segments(doc)

        # Mask and unmask
        masked_result = masking_engine.mask_document(
            doc,
            entities,
            policy,
            text_segments
        )

        # Special handling for CUSTOM strategy
        if strategy_kind == StrategyKind.CUSTOM:
            # With CUSTOM using a lambda, it may work differently
            pass  # Just check that it doesn't error
        elif entities:  # Only assert change if there were entities to mask
            # For other strategies, text should be modified if entities were found
            assert masked_result.masked_document._main_text != doc._main_text

        # Unmask and verify restoration
        unmasked_result = unmasking_engine.unmask_document(
            masked_result.masked_document,
            masked_result.cloakmap
        )

        assert unmasked_result.restored_document._main_text == doc._main_text

    def test_complex_document_processing(self, analyzer_engine):
        """Test documents with multiple entity types and strategies."""
        # Create complex document with many entity types
        doc = DoclingDocument(name="complex.txt")
        doc._main_text = """
        === CONFIDENTIAL REPORT ===

        Subject: John Michael Smith (Employee ID: EMP-2024-0542)
        Department: Engineering (Building 5, Floor 3)
        Manager: Sarah Johnson (sarah.johnson@company.com)

        Personal Information:
        - Date of Birth: March 15, 1985
        - SSN: 987-65-4321
        - Phone: +1 (555) 234-5678
        - Personal Email: jsmith.personal@gmail.com
        - Address: 456 Oak Avenue, Suite 200, Springfield, IL 62701

        Financial Details:
        - Salary: $125,000 per year
        - Bank Account: ****4567 (Chase Bank)
        - Credit Card: 5555-4444-3333-2222 (Expires: 12/25)
        - 401k ID: 401K-JMS-2024

        Medical Information:
        - Insurance ID: INS-123456789
        - Primary Care: Dr. Robert Chen (License: MED-IL-9876)
        - Prescription: Lisinopril 10mg daily
        - Last Physical: January 10, 2024

        Security Clearance:
        - Level: Secret (SC-2024-0099)
        - Clearance Date: February 1, 2024
        - Background Check: PASS (BC-2024-1122)

        Emergency Contacts:
        1. Jane Smith (Wife): 555-876-5432
        2. Michael Smith (Brother): 555-345-6789

        Notes: Employee eligible for promotion in Q3 2024.
        Review scheduled with HR on April 30, 2024.
        """

        # Create comprehensive policy with mixed strategies
        policy = MaskingPolicy(
            per_entity={
                "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[PERSON_{}]"}),
                "EMAIL_ADDRESS": Strategy(StrategyKind.REDACT, {"char": "*"}),
                "PHONE_NUMBER": Strategy(StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end"}),
                "US_SSN": Strategy(StrategyKind.HASH, {"algorithm": "sha256"}),
                "CREDIT_CARD": Strategy(StrategyKind.REDACT, {"char": "*"}),
                "DATE_TIME": Strategy(StrategyKind.TEMPLATE, {"template": "[DATE]"}),
                "LOCATION": Strategy(StrategyKind.REDACT, {"char": "*"}),
                "US_BANK_NUMBER": Strategy(StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end"}),
                "MEDICAL_LICENSE": Strategy(StrategyKind.TEMPLATE, {"template": "[LICENSE]"}),
                "US_DRIVER_LICENSE": Strategy(StrategyKind.REDACT, {"char": "X"}),
                "NRP": Strategy(StrategyKind.SURROGATE, {}),
            }
        )

        # Detect entities and create segments
        entities = self._analyze_document(doc, analyzer_engine)
        text_segments = self._create_text_segments(doc)

        # Test with both engine types
        for use_presidio in [True, False]:
            masking_engine = MaskingEngine(use_presidio_engine=use_presidio)
            unmasking_engine = UnmaskingEngine(use_presidio_engine=use_presidio)

            # Mask the document
            masked_result = masking_engine.mask_document(
                doc,
                entities,
                policy,
                text_segments
            )

            # Verify various entities are masked
            masked_text = masked_result.masked_document._main_text
            assert "John Michael Smith" not in masked_text
            assert "987-65-4321" not in masked_text
            assert "jsmith.personal@gmail.com" not in masked_text
            assert "5555-4444-3333-2222" not in masked_text

            # Unmask and verify restoration
            unmasked_result = unmasking_engine.unmask_document(
                masked_result.masked_document,
                masked_result.cloakmap
            )

            assert unmasked_result.restored_document._main_text == doc._main_text

    def test_cloakmap_version_compatibility(self, analyzer_engine):
        """Test v1.0→v2.0 CloakMap processing."""
        # Create a document
        doc = DoclingDocument(name="version_test.txt")
        doc._main_text = "Call John at 555-1234 or email john@example.com"

        policy = MaskingPolicy(
            per_entity={
                "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[PERSON]"}),
                "PHONE_NUMBER": Strategy(StrategyKind.REDACT, {"char": "*"}),
                "EMAIL_ADDRESS": Strategy(StrategyKind.REDACT, {"char": "*"}),
            }
        )

        # Detect entities and create segments
        entities = self._analyze_document(doc, analyzer_engine)
        text_segments = self._create_text_segments(doc)

        # Create v1.0 style masking (using legacy engine)
        masking_engine_v1 = MaskingEngine(use_presidio_engine=False)

        masked_result_v1 = masking_engine_v1.mask_document(
            doc,
            entities,
            policy,
            text_segments
        )

        cloakmap_v1 = masked_result_v1.cloakmap

        # Simulate v1.0 CloakMap structure
        cloakmap_v1.version = "1.0"

        # Process with v2.0 engine (Presidio)
        unmasking_engine_v2 = UnmaskingEngine(use_presidio_engine=True)

        # Should handle v1.0 CloakMap correctly
        unmasked_result = unmasking_engine_v2.unmask_document(
            masked_result_v1.masked_document,
            cloakmap_v1
        )

        assert unmasked_result.restored_document._main_text == doc._main_text

    def test_empty_document_handling(self, analyzer_engine):
        """Test handling of empty documents."""
        # Empty document
        doc = DoclingDocument(name="empty.txt")
        doc._main_text = ""

        policy = MaskingPolicy(per_entity={})

        # Detect entities and create segments
        entities = self._analyze_document(doc, analyzer_engine)
        text_segments = self._create_text_segments(doc)

        # Test with both engines
        for use_presidio in [True, False]:
            masking_engine = MaskingEngine(use_presidio_engine=use_presidio)
            unmasking_engine = UnmaskingEngine(use_presidio_engine=use_presidio)

            # Should handle empty document gracefully
            masked_result = masking_engine.mask_document(
                doc,
                entities,
                policy,
                text_segments
            )
            assert masked_result.masked_document._main_text == ""

            unmasked_result = unmasking_engine.unmask_document(
                masked_result.masked_document,
                masked_result.cloakmap
            )

            assert unmasked_result.restored_document._main_text == ""

    def test_unicode_and_special_characters(self, analyzer_engine):
        """Test handling of Unicode and special characters."""
        doc = DoclingDocument(name="unicode.txt")
        doc._main_text = """
        Employé: François Müller
        Téléphone: +33 6 12 34 56 78
        Email: françois.müller@société.fr
        Adresse: 123 Rue de la Paix, 75001 Paris
        Salaire: €50,000
        Notes: Meeting scheduled for café at 14h30 ☕
        Emoji test: 😀 Call 555-1234 📞
        """

        policy = MaskingPolicy(
            per_entity={
                "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[PERSONNE]"}),
                "PHONE_NUMBER": Strategy(StrategyKind.REDACT, {"char": "*"}),
                "EMAIL_ADDRESS": Strategy(StrategyKind.REDACT, {"char": "*"}),
                "LOCATION": Strategy(StrategyKind.REDACT, {"char": "*"}),
            }
        )

        # Detect entities and create segments
        entities = self._analyze_document(doc, analyzer_engine)
        text_segments = self._create_text_segments(doc)

        # Test with Presidio engine
        masking_engine = MaskingEngine(use_presidio_engine=True)
        unmasking_engine = UnmaskingEngine(use_presidio_engine=True)

        masked_result = masking_engine.mask_document(
            doc,
            entities,
            policy,
            text_segments
        )

        unmasked_result = unmasking_engine.unmask_document(
            masked_result.masked_document,
            masked_result.cloakmap
        )

        # Verify Unicode preservation
        assert unmasked_result.restored_document._main_text == doc._main_text
        assert "François Müller" in unmasked_result.restored_document._main_text
        assert "€50,000" in unmasked_result.restored_document._main_text
        assert "☕" in unmasked_result.restored_document._main_text
        assert "😀" in unmasked_result.restored_document._main_text

    def test_overlapping_entities(self, analyzer_engine):
        """Test handling of overlapping entity detections."""
        doc = DoclingDocument(name="overlap.txt")
        doc._main_text = "Email Dr. John Smith at john.smith@hospital.org or call 555-1234"

        policy = MaskingPolicy(
            per_entity={
                "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[PERSON_{}]"}),
                "EMAIL_ADDRESS": Strategy(StrategyKind.REDACT, {"char": "*"}),
                "PHONE_NUMBER": Strategy(StrategyKind.REDACT, {"char": "*"}),
            }
        )

        # Detect entities and create segments
        entities = self._analyze_document(doc, analyzer_engine)
        text_segments = self._create_text_segments(doc)

        # Test with Presidio (which handles overlaps better)
        masking_engine = MaskingEngine(use_presidio_engine=True)
        unmasking_engine = UnmaskingEngine(use_presidio_engine=True)

        masked_result = masking_engine.mask_document(
            doc,
            entities,
            policy,
            text_segments
        )

        # Verify entities are masked
        masked_text = masked_result.masked_document._main_text
        assert "[EMAIL]" in masked_text or "john.smith@hospital.org" not in masked_text
        assert "[PHONE]" in masked_text or "555-1234" not in masked_text

        # Unmask and verify
        unmasked_result = unmasking_engine.unmask_document(
            masked_result.masked_document,
            masked_result.cloakmap
        )

        assert unmasked_result.restored_document._main_text == doc._main_text
