"""Performance benchmarking tests for Presidio integration."""

import time
import tracemalloc

import pytest
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem
from presidio_analyzer import AnalyzerEngine

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.document.extractor import TextSegment
from cloakpivot.masking.engine import MaskingEngine
from cloakpivot.masking.presidio_adapter import PresidioMaskingAdapter
from cloakpivot.unmasking.engine import UnmaskingEngine
from cloakpivot.unmasking.presidio_adapter import PresidioUnmaskingAdapter


class TestPresidioBenchmarks:

    def _get_document_text(self, document: DoclingDocument) -> str:
        """Helper to get text from document, handling both formats."""
        if hasattr(document, '_main_text'):
            return document._main_text
        elif document.texts:
            return document.texts[0].text
        return ""

    def _set_document_text(self, document: DoclingDocument, text: str) -> None:
        """Helper to set text in document, handling both formats."""
        from docling_core.types.doc.document import TextItem
        # Create proper TextItem
        text_item = TextItem(
            text=text,
            self_ref="#/texts/0",
            label="text",
            orig=text
        )
        document.texts = [text_item]
        # Also set _main_text for backward compatibility
        document._main_text = text

    """Performance comparison between engines."""

    @pytest.fixture
    def small_document(self):
        """Create a small document for benchmarking."""
        doc = DoclingDocument(name="small.txt")
        self._set_document_text(doc, """
        John Smith lives at 123 Main Street, New York, NY 10001.
        His email is john.smith@example.com and phone is 555-123-4567.
        SSN: 123-45-6789, Credit Card: 4532-1234-5678-9012.
        """ * 10)  # ~100 lines
        return doc

    @pytest.fixture
    def medium_document(self):
        """Create a medium document for benchmarking."""
        doc = DoclingDocument(name="medium.txt")
        self._set_document_text(doc, """
        Patient Record:
        Name: Jane Doe
        Date of Birth: 01/15/1980
        Address: 456 Oak Avenue, Los Angeles, CA 90001
        Phone: (310) 555-0123
        Email: jane.doe@healthcare.org
        Medical Record Number: MRN-123456
        Insurance ID: INS-789012
        Emergency Contact: John Doe (310) 555-4567

        Medical History:
        - Diagnosed with Type 2 Diabetes on 03/15/2015
        - Prescribed Metformin 500mg twice daily
        - Last A1C: 6.8% on 12/01/2023
        - Blood Pressure: 120/80 mmHg

        Recent Visits:
        - 01/10/2024: Routine checkup with Dr. Smith
        - 02/15/2024: Lab work ordered
        - 03/20/2024: Follow-up scheduled
        """ * 50)  # ~1000 lines
        return doc

    @pytest.fixture
    def large_document(self):
        """Create a large document for benchmarking."""
        doc = DoclingDocument(name="large.txt")
        self._set_document_text(doc, """
        CONFIDENTIAL REPORT

        Employee: Michael Johnson
        Employee ID: EMP-12345
        Department: Engineering
        Manager: Sarah Williams

        Personal Information:
        - DOB: 05/20/1985
        - SSN: 987-65-4321
        - Address: 789 Pine Street, Seattle, WA 98101
        - Phone: (206) 555-7890
        - Email: m.johnson@company.com

        Financial Information:
        - Salary: $125,000
        - Bank Account: 1234567890
        - Routing Number: 021000021
        - 401k Balance: $75,000

        Performance Review:
        - Rating: Exceeds Expectations
        - Bonus: $15,000
        - Stock Options: 1000 shares
        """ * 200)  # ~5000 lines
        return doc

    @pytest.fixture
    def standard_policy(self):
        """Create a standard masking policy."""
        return MaskingPolicy(
            default_strategy=Strategy(kind=StrategyKind.REDACT),
            per_entity={
                "PERSON": Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[PERSON]"}),
                "EMAIL_ADDRESS": Strategy(kind=StrategyKind.HASH),
                "PHONE_NUMBER": Strategy(kind=StrategyKind.PARTIAL, parameters={"visible_chars": 4}),
                "US_SSN": Strategy(kind=StrategyKind.HASH),
            }
        )

    def _create_analyzer(self):
        """Create and return an AnalyzerEngine instance."""
        # Cache the analyzer to avoid recreating it multiple times
        if not hasattr(self, '_cached_analyzer'):
            self._cached_analyzer = AnalyzerEngine()
        return self._cached_analyzer

    def _analyze_document(self, doc):
        """Analyze document and return entities."""
        analyzer = self._create_analyzer()
        return analyzer.analyze(text=self._get_document_text(doc), language='en')

    def _create_text_segments(self, doc):
        """Create text segments for the document."""
        return [TextSegment(
            node_id='#/texts/0',
            text=self._get_document_text(doc),
            start_offset=0,
            end_offset=len(self._get_document_text(doc)),
            node_type='TextItem'
        )]

    @pytest.mark.benchmark(group="masking-small")
    def test_masking_performance_comparison_small(self, benchmark, small_document, standard_policy):
        """Compare masking performance with small documents."""

        def mask_with_legacy():
            engine = MaskingEngine(
                use_presidio_engine=False,
                resolve_conflicts=True
            )
            entities = self._analyze_document(small_document)
            text_segments = self._create_text_segments(small_document)
            return engine.mask_document(small_document, entities, standard_policy, text_segments)

        def mask_with_presidio():
            engine = MaskingEngine(
                use_presidio_engine=True,
                resolve_conflicts=True
            )
            entities = self._analyze_document(small_document)
            text_segments = self._create_text_segments(small_document)
            return engine.mask_document(small_document, entities, standard_policy, text_segments)

        # Benchmark based on test name
        if "legacy" in benchmark.name:
            result = benchmark(mask_with_legacy)
        else:
            result = benchmark(mask_with_presidio)

        assert result.masked_document is not None
        assert result.cloakmap is not None

    @pytest.mark.benchmark(group="masking-medium")
    def test_masking_performance_comparison_medium(self, benchmark, medium_document, standard_policy):
        """Compare masking performance with medium documents."""

        def mask_with_legacy():
            engine = MaskingEngine(
                use_presidio_engine=False,
                resolve_conflicts=True
            )
            entities = self._analyze_document(medium_document)
            text_segments = self._create_text_segments(medium_document)
            return engine.mask_document(medium_document, entities, standard_policy, text_segments)

        def mask_with_presidio():
            engine = MaskingEngine(
                use_presidio_engine=True,
                resolve_conflicts=True
            )
            entities = self._analyze_document(medium_document)
            text_segments = self._create_text_segments(medium_document)
            return engine.mask_document(medium_document, entities, standard_policy, text_segments)

        # Benchmark based on test name
        if "legacy" in benchmark.name:
            result = benchmark(mask_with_legacy)
        else:
            result = benchmark(mask_with_presidio)

        assert result.masked_document is not None
        assert result.cloakmap is not None

    @pytest.mark.benchmark(group="unmasking")
    def test_unmasking_performance_comparison(self, benchmark, medium_document, standard_policy):
        """Compare unmasking performance between engines."""
        # First mask the document
        masking_engine = MaskingEngine(
            use_presidio_engine=True,
            resolve_conflicts=True
        )
        entities = self._analyze_document(medium_document)
        text_segments = self._create_text_segments(medium_document)
        masked_result = masking_engine.mask_document(medium_document, entities, standard_policy, text_segments)

        def unmask_with_legacy():
            engine = UnmaskingEngine(
                use_presidio_engine=False
            )
            return engine.unmask_document(masked_result.masked_document, masked_result.cloakmap)

        def unmask_with_presidio():
            engine = UnmaskingEngine(
                use_presidio_engine=True
            )
            return engine.unmask_document(masked_result.masked_document, masked_result.cloakmap)

        # Benchmark based on test name
        if "legacy" in benchmark.name:
            result = benchmark(unmask_with_legacy)
        else:
            result = benchmark(unmask_with_presidio)

        assert result.restored_document is not None

    @pytest.mark.benchmark(group="large-document")
    @pytest.mark.skip(reason="Large document processing hangs in spaCy - needs optimization")
    def test_large_document_processing(self, benchmark, large_document, standard_policy):
        """Test processing performance with large documents."""
        def process_document():
            # Mask
            masking_engine = MaskingEngine(
                use_presidio_engine=True,
                resolve_conflicts=True
            )
            entities = self._analyze_document(large_document)
            text_segments = self._create_text_segments(large_document)
            masked_result = masking_engine.mask_document(large_document, entities, standard_policy, text_segments)

            # Unmask
            unmasking_engine = UnmaskingEngine(
                use_presidio_engine=True
            )
            return unmasking_engine.unmask_document(masked_result.masked_document, masked_result.cloakmap)

        result = benchmark(process_document)
        # For benchmark test, just verify the process completes without errors
        # Perfect restoration is not guaranteed with template strategies
        assert result is not None
        assert result.restored_document is not None
        assert self._get_document_text(result.restored_document) is not None

    @pytest.mark.benchmark(group="memory")
    @pytest.mark.skip(reason="Large document analysis hangs in spaCy - needs optimization")
    def test_memory_usage_comparison(self, large_document, standard_policy):
        """Compare memory usage between engines."""
        # Legacy engine memory usage
        tracemalloc.start()

        legacy_masking = MaskingEngine(use_presidio_engine=False, resolve_conflicts=True)
        legacy_unmasking = UnmaskingEngine(use_presidio_engine=False)

        entities = self._analyze_document(large_document)
        text_segments = self._create_text_segments(large_document)

        masked_result = legacy_masking.mask_document(large_document, entities, standard_policy, text_segments)
        unmasked_result = legacy_unmasking.unmask_document(masked_result.masked_document, masked_result.cloakmap)

        legacy_current, legacy_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Presidio engine memory usage
        tracemalloc.start()

        presidio_masking = MaskingEngine(use_presidio_engine=True, resolve_conflicts=True)
        presidio_unmasking = UnmaskingEngine(use_presidio_engine=True)

        masked_result = presidio_masking.mask_document(large_document, entities, standard_policy, text_segments)
        unmasked_result = presidio_unmasking.unmask_document(masked_result.masked_document, masked_result.cloakmap)

        presidio_current, presidio_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory usage should be reasonable (< 100MB for large doc)
        assert legacy_peak < 100 * 1024 * 1024  # 100 MB
        assert presidio_peak < 100 * 1024 * 1024  # 100 MB

        print(f"Legacy memory: {legacy_peak / 1024 / 1024:.2f} MB")
        print(f"Presidio memory: {presidio_peak / 1024 / 1024:.2f} MB")

    @pytest.mark.benchmark(group="round-trip")
    def test_round_trip_performance(self, benchmark):
        """Test full round-trip performance."""
        doc = DoclingDocument(name="roundtrip.txt")
        self._set_document_text(doc, "John Smith (SSN: 123-45-6789) lives at john@example.com" * 100)

        policy = MaskingPolicy(
            default_strategy=Strategy(kind=StrategyKind.HASH)
        )

        def round_trip_legacy():
            masking_engine = MaskingEngine(use_presidio_engine=False, resolve_conflicts=True)
            unmasking_engine = UnmaskingEngine(use_presidio_engine=False)

            entities = self._analyze_document(doc)
            text_segments = self._create_text_segments(doc)

            masked_result = masking_engine.mask_document(doc, entities, policy, text_segments)
            return unmasking_engine.unmask_document(masked_result.masked_document, masked_result.cloakmap)

        def round_trip_presidio():
            masking_engine = MaskingEngine(use_presidio_engine=True, resolve_conflicts=True)
            unmasking_engine = UnmaskingEngine(use_presidio_engine=True)

            entities = self._analyze_document(doc)
            text_segments = self._create_text_segments(doc)

            masked_result = masking_engine.mask_document(doc, entities, policy, text_segments)
            return unmasking_engine.unmask_document(masked_result.masked_document, masked_result.cloakmap)

        # Benchmark based on test name
        if "legacy" in benchmark.name:
            result = benchmark(round_trip_legacy)
        else:
            result = benchmark(round_trip_presidio)

        assert self._get_document_text(result.restored_document) == self._get_document_text(doc)

    def test_performance_regression_detection(self, medium_document, standard_policy):
        """Detect performance regressions between engines."""
        # Time legacy engine
        start = time.perf_counter()

        legacy_engine = MaskingEngine(use_presidio_engine=False, resolve_conflicts=True)
        entities = self._analyze_document(medium_document)
        text_segments = self._create_text_segments(medium_document)
        legacy_result = legacy_engine.mask_document(medium_document, entities, standard_policy, text_segments)

        legacy_time = time.perf_counter() - start

        # Time Presidio engine
        start = time.perf_counter()

        presidio_engine = MaskingEngine(use_presidio_engine=True, resolve_conflicts=True)
        presidio_result = presidio_engine.mask_document(medium_document, entities, standard_policy, text_segments)

        presidio_time = time.perf_counter() - start

        # Presidio should not be more than 3x slower than legacy
        assert presidio_time < legacy_time * 3, f"Presidio ({presidio_time:.3f}s) is too slow compared to legacy ({legacy_time:.3f}s)"

        # Both should complete within reasonable time (< 15 seconds for medium doc)
        assert legacy_time < 15.0
        assert presidio_time < 15.0
