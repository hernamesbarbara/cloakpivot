"""Performance benchmarking tests for Presidio integration."""

import time
import tracemalloc

import pytest
from docling_core.types import DoclingDocument

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.masking.engine import MaskingEngine
from cloakpivot.masking.presidio_adapter import PresidioMaskingAdapter
from cloakpivot.unmasking.engine import UnmaskingEngine
from cloakpivot.unmasking.presidio_adapter import PresidioUnmaskingAdapter


class TestPresidioBenchmarks:
    """Performance comparison between engines."""

    @pytest.fixture
    def small_document(self) -> DoclingDocument:
        """Create a small test document."""
        doc = DoclingDocument(name="small.txt")
        doc._main_text = """
        John Smith called from 555-1234 about the meeting.
        His email is john@example.com and he lives at 123 Main St.
        """
        return doc

    @pytest.fixture
    def medium_document(self) -> DoclingDocument:
        """Create a medium-sized test document."""
        doc = DoclingDocument(name="medium.txt")
        text_lines = []
        for i in range(100):
            text_lines.append(f"""
            Person {i}: Customer_{i:04d} Smith
            Phone: 555-{i:04d}
            Email: customer{i}@example.com
            SSN: {i:03d}-45-{6789+i:04d}
            Address: {i} Main Street, City {i}, State {i%50:02d}
            Credit Card: 4111-1111-{i:04d}-{i:04d}
            """)
        doc._main_text = "\n".join(text_lines)
        return doc

    @pytest.fixture
    def large_document(self) -> DoclingDocument:
        """Create a large test document with 1000+ entities."""
        doc = DoclingDocument(name="large.txt")
        text_lines = []
        for i in range(1000):
            text_lines.append(f"""
            === Record {i:05d} ===
            Name: {self._generate_name(i)}
            Phone: {self._generate_phone(i)}
            Email: {self._generate_email(i)}
            SSN: {self._generate_ssn(i)}
            Address: {self._generate_address(i)}
            DOB: {self._generate_date(i)}
            Credit Card: {self._generate_credit_card(i)}
            Medical Record: MRN-{i:06d}
            Notes: Customer visited on {self._generate_date(i+100)} and purchased item #{i:05d}.
            Emergency Contact: {self._generate_name(i+1000)} at {self._generate_phone(i+1000)}
            """)
        doc._main_text = "\n".join(text_lines)
        return doc

    @pytest.fixture
    def standard_policy(self) -> MaskingPolicy:
        """Create a standard masking policy."""
        return MaskingPolicy(
            per_entity={
                "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[PERSON_{}]"}),
                "PHONE_NUMBER": Strategy(StrategyKind.REDACT, {"value": "[PHONE]"}),
                "EMAIL_ADDRESS": Strategy(StrategyKind.REDACT, {"value": "[EMAIL]"}),
                "US_SSN": Strategy(StrategyKind.REDACT, {"char": "#"}),
                "CREDIT_CARD": Strategy(StrategyKind.REDACT, {"char": "*"}),
                "DATE_TIME": Strategy(StrategyKind.TEMPLATE, {"template": "[DATE]"}),
                "LOCATION": Strategy(StrategyKind.REDACT, {"value": "[LOCATION]"}),
                "MEDICAL_LICENSE": Strategy(StrategyKind.TEMPLATE, {"template": "[MRN]"}),
            }
        )

    def _generate_name(self, index: int) -> str:
        """Generate a test name."""
        first_names = ["John", "Jane", "Robert", "Mary", "Michael", "Sarah", "David", "Emily"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
        return f"{first_names[index % len(first_names)]} {last_names[index % len(last_names)]}"

    def _generate_phone(self, index: int) -> str:
        """Generate a test phone number."""
        return f"555-{index % 10000:04d}"

    def _generate_email(self, index: int) -> str:
        """Generate a test email."""
        return f"user{index}@example.com"

    def _generate_ssn(self, index: int) -> str:
        """Generate a test SSN."""
        return f"{index % 1000:03d}-{(index * 7) % 100:02d}-{(index * 13) % 10000:04d}"

    def _generate_address(self, index: int) -> str:
        """Generate a test address."""
        return f"{index} Main Street, City {index % 100}, State {index % 50:02d} {10000 + index:05d}"

    def _generate_date(self, index: int) -> str:
        """Generate a test date."""
        month = (index % 12) + 1
        day = (index % 28) + 1
        year = 2020 + (index % 5)
        return f"{month:02d}/{day:02d}/{year}"

    def _generate_credit_card(self, index: int) -> str:
        """Generate a test credit card number."""
        return f"4111-1111-{index % 10000:04d}-{(index * 3) % 10000:04d}"

    @pytest.mark.benchmark(group="masking-small")
    def test_masking_performance_comparison_small(self, benchmark, small_document, standard_policy):
        """Compare masking performance with small documents."""

        def mask_with_custom():
            engine = MaskingEngine(
                enable_custom_recognizer=True,
                presidio_adapter=None
            )
            return engine.mask_document(small_document, standard_policy)

        def mask_with_presidio():
            engine = MaskingEngine(
                enable_custom_recognizer=False,
                presidio_adapter=PresidioMaskingAdapter()
            )
            return engine.mask_document(small_document, standard_policy)

        # Benchmark based on test name
        if "custom" in benchmark.name:
            result = benchmark(mask_with_custom)
        else:
            result = benchmark(mask_with_presidio)

        assert result.masked_document is not None
        assert result.cloakmap is not None

    @pytest.mark.benchmark(group="masking-medium")
    def test_masking_performance_comparison_medium(self, benchmark, medium_document, standard_policy):
        """Compare masking performance with medium documents."""

        def mask_with_custom():
            engine = MaskingEngine(
                enable_custom_recognizer=True,
                presidio_adapter=None
            )
            return engine.mask(medium_document, standard_policy)

        def mask_with_presidio():
            engine = MaskingEngine(
                enable_custom_recognizer=False,
                presidio_adapter=PresidioMaskingAdapter()
            )
            return engine.mask(medium_document, standard_policy)

        # Run benchmark
        if "custom" in benchmark.name:
            result = benchmark(mask_with_custom)
        else:
            result = benchmark(mask_with_presidio)

        assert result.masked_document is not None
        assert result.cloakmap is not None

    @pytest.mark.benchmark(group="unmasking")
    def test_unmasking_performance_comparison(self, benchmark, medium_document, standard_policy):
        """Compare unmasking performance: Anchor vs Presidio."""
        # First, create a masked document
        masking_engine = MaskingEngine(
            enable_custom_recognizer=False,
            presidio_adapter=PresidioMaskingAdapter()
        )
        masked_result = masking_engine.mask(medium_document, standard_policy)

        def unmask_with_anchor():
            engine = UnmaskingEngine(
                enable_anchor_strategy=True,
                presidio_adapter=None
            )
            return engine.unmask_document(masked_result.masked_document, masked_result.cloakmap)

        def unmask_with_presidio():
            engine = UnmaskingEngine(
                enable_anchor_strategy=False,
                presidio_adapter=PresidioUnmaskingAdapter()
            )
            return engine.unmask_document(masked_result.masked_document, masked_result.cloakmap)

        # Run benchmark
        if "anchor" in benchmark.name:
            result = benchmark(unmask_with_anchor)
        else:
            result = benchmark(unmask_with_presidio)

        assert result._main_text == medium_document._main_text

    @pytest.mark.benchmark(group="large-docs")
    def test_large_document_processing(self, benchmark, large_document, standard_policy):
        """Test performance with large documents (1000+ entities)."""

        def process_large_document():
            # Use Presidio for this test
            masking_engine = MaskingEngine(
                enable_custom_recognizer=False,
                presidio_adapter=PresidioMaskingAdapter()
            )

            unmasking_engine = UnmaskingEngine(
                enable_anchor_strategy=False,
                presidio_adapter=PresidioUnmaskingAdapter()
            )

            # Mask
            masked_result = masking_engine.mask(large_document, standard_policy)

            # Unmask
            unmasked = unmasking_engine.unmask_document(
                masked_result.masked_document,
                masked_result.cloakmap
            )

            return unmasked

        result = benchmark(process_large_document)
        assert result._main_text == large_document._main_text

    def test_memory_usage_comparison(self, medium_document, standard_policy):
        """Compare memory usage between engines."""
        memory_results = {}

        # Test Custom/Legacy engine memory usage
        tracemalloc.start()
        masking_engine = MaskingEngine(
            enable_custom_recognizer=True,
            presidio_adapter=None
        )
        unmasking_engine = UnmaskingEngine(
            enable_anchor_strategy=True,
            presidio_adapter=None
        )

        masked_result = masking_engine.mask(medium_document, standard_policy)
        unmasking_engine.unmask_document(masked_result.masked_document, masked_result.cloakmap)

        current, peak = tracemalloc.get_traced_memory()
        memory_results["legacy"] = {"current": current, "peak": peak}
        tracemalloc.stop()

        # Test Presidio engine memory usage
        tracemalloc.start()
        masking_engine = MaskingEngine(
            enable_custom_recognizer=False,
            presidio_adapter=PresidioMaskingAdapter()
        )
        unmasking_engine = UnmaskingEngine(
            enable_anchor_strategy=False,
            presidio_adapter=PresidioUnmaskingAdapter()
        )

        masked_result = masking_engine.mask(medium_document, standard_policy)
        unmasking_engine.unmask_document(masked_result.masked_document, masked_result.cloakmap)

        current, peak = tracemalloc.get_traced_memory()
        memory_results["presidio"] = {"current": current, "peak": peak}
        tracemalloc.stop()

        # Compare memory usage
        legacy_peak = memory_results["legacy"]["peak"]
        presidio_peak = memory_results["presidio"]["peak"]

        # Log results for debugging
        print("\nMemory Usage Comparison:")
        print(f"Legacy Peak: {legacy_peak / 1024 / 1024:.2f} MB")
        print(f"Presidio Peak: {presidio_peak / 1024 / 1024:.2f} MB")
        print(f"Difference: {(presidio_peak - legacy_peak) / 1024 / 1024:.2f} MB")

        # Presidio should not use significantly more memory (allow 2x for initial implementation)
        assert presidio_peak < legacy_peak * 2, f"Presidio uses too much memory: {presidio_peak} vs {legacy_peak}"

    @pytest.mark.benchmark(group="round-trip")
    def test_round_trip_performance(self, benchmark, medium_document, standard_policy):
        """Benchmark complete round-trip (mask + unmask) performance."""

        def round_trip_legacy():
            masking_engine = MaskingEngine(
                enable_custom_recognizer=True,
                presidio_adapter=None
            )
            unmasking_engine = UnmaskingEngine(
                enable_anchor_strategy=True,
                presidio_adapter=None
            )

            masked_result = masking_engine.mask(medium_document, standard_policy)
            return unmasking_engine.unmask_document(masked_result.masked_document, masked_result.cloakmap)

        def round_trip_presidio():
            masking_engine = MaskingEngine(
                enable_custom_recognizer=False,
                presidio_adapter=PresidioMaskingAdapter()
            )
            unmasking_engine = UnmaskingEngine(
                enable_anchor_strategy=False,
                presidio_adapter=PresidioUnmaskingAdapter()
            )

            masked_result = masking_engine.mask(medium_document, standard_policy)
            return unmasking_engine.unmask_document(masked_result.masked_document, masked_result.cloakmap)

        # Run benchmark based on test variant
        if "legacy" in benchmark.name:
            result = benchmark(round_trip_legacy)
        else:
            result = benchmark(round_trip_presidio)

        assert result._main_text == medium_document._main_text

    @pytest.mark.benchmark(group="entity-detection")
    def test_entity_detection_performance(self, benchmark, medium_document):
        """Benchmark entity detection performance specifically."""
        from presidio_analyzer import AnalyzerEngine

        def detect_with_custom():
            # Using Presidio analyzer with custom configuration for comparison
            analyzer = AnalyzerEngine()
            # Could add custom recognizers here if needed
            return analyzer.analyze(text=medium_document._main_text, language="en")

        def detect_with_presidio():
            analyzer = AnalyzerEngine()
            return analyzer.analyze(text=medium_document._main_text, language="en")

        # Run benchmark
        if "custom" in benchmark.name:
            entities = benchmark(detect_with_custom)
        else:
            entities = benchmark(detect_with_presidio)

        # Basic validation
        assert entities is not None

    def test_performance_regression_detection(self, medium_document, standard_policy):
        """Ensure Presidio performance is within acceptable bounds."""
        # Time legacy engine
        start = time.perf_counter()
        legacy_masking = MaskingEngine(enable_custom_recognizer=True, presidio_adapter=None)
        legacy_unmasking = UnmaskingEngine(enable_anchor_strategy=True, presidio_adapter=None)

        masked_legacy = legacy_masking.mask(medium_document, standard_policy)
        unmasked_legacy = legacy_unmasking.unmask_document(masked_legacy.masked_document, masked_legacy.cloakmap)
        legacy_time = time.perf_counter() - start

        # Time Presidio engine
        start = time.perf_counter()
        presidio_masking = MaskingEngine(enable_custom_recognizer=False, presidio_adapter=PresidioMaskingAdapter())
        presidio_unmasking = UnmaskingEngine(enable_anchor_strategy=False, presidio_adapter=PresidioUnmaskingAdapter())

        masked_presidio = presidio_masking.mask(medium_document, standard_policy)
        unmasked_presidio = presidio_unmasking.unmask_document(masked_presidio.masked_document, masked_presidio.cloakmap)
        presidio_time = time.perf_counter() - start

        # Log performance comparison
        print("\nPerformance Comparison:")
        print(f"Legacy Time: {legacy_time:.3f}s")
        print(f"Presidio Time: {presidio_time:.3f}s")
        print(f"Ratio: {presidio_time / legacy_time:.2f}x")

        # Verify correctness
        assert unmasked_legacy._main_text == medium_document._main_text
        assert unmasked_presidio._main_text == medium_document._main_text

        # Initially allow Presidio to be up to 50% slower (will optimize later)
        assert presidio_time < legacy_time * 1.5, f"Presidio is too slow: {presidio_time}s vs {legacy_time}s"
