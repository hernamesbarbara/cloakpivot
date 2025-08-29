"""Property-based tests for masking functionality with performance optimizations."""

import pytest
from hypothesis import assume, given, strategies as st
from presidio_analyzer import AnalyzerEngine

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from tests.utils.masking_helpers import (
    FastRegexDetector,
    create_multi_section_document,
    create_simple_document,
    mask_document_with_detection,
)


# Constrained generators for fast property testing
def constrained_text_strategy(max_size: int = 200) -> st.SearchStrategy[str]:
    """Generate text with constrained size and safe characters.
    
    Avoids surrogate characters and control characters that slow down
    Presidio analysis and Hypothesis shrinking.
    """
    return st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd", "Zs"),
            whitelist_characters=" .-@",
        ),
        min_size=1,
        max_size=max_size
    )


def pii_text_strategy() -> st.SearchStrategy[str]:
    """Generate text with embedded PII patterns for testing.
    
    Uses deterministic patterns that will be detected by FastRegexDetector
    for predictable unit-level testing.
    """
    return st.one_of([
        st.just("Call me at 555-123-4567 for more info"),
        st.just("Email john.doe@example.com for details"),
        st.just("SSN: 123-45-6789"),
        st.just("Contact 555-987-6543 or admin@test.org"),
    ])


def light_policy_strategy() -> st.SearchStrategy[MaskingPolicy]:
    """Generate lightweight masking policies.
    
    Prefers template strategies over computationally expensive ones
    like SURROGATE or HASH.
    """
    template_strategy = Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"})
    partial_strategy = Strategy(StrategyKind.PARTIAL, {
        "visible_chars": 4,
        "position": "end",
        "mask_char": "*"
    })
    
    return st.one_of([
        st.just(MaskingPolicy(default_strategy=template_strategy)),
        st.just(MaskingPolicy(default_strategy=partial_strategy)),
    ])


class TestPropertyBasedMaskingFast:
    """Fast property-based tests using optimized generators and shared analyzer."""

    @given(
        text=constrained_text_strategy(max_size=100),
        policy=light_policy_strategy()
    )
    def test_masking_preserves_document_structure(
        self, text: str, policy: MaskingPolicy, shared_analyzer: AnalyzerEngine
    ):
        """Property: Masking preserves document structure regardless of content."""
        assume(len(text.strip()) > 0)  # Skip empty strings
        
        document = create_simple_document(text)
        original_text_count = len(document.texts)
        
        result = mask_document_with_detection(
            document, policy, analyzer=shared_analyzer, resolve_conflicts=False
        )
        
        # Structure should be preserved
        assert len(result.masked_document.texts) == original_text_count
        assert result.cloakmap is not None

    @given(
        sections=st.lists(constrained_text_strategy(max_size=50), min_size=1, max_size=3),
        policy=light_policy_strategy()
    )
    def test_multi_section_masking_consistency(
        self, sections: list[str], policy: MaskingPolicy, shared_analyzer: AnalyzerEngine
    ):
        """Property: Multi-section documents are masked consistently."""
        # Filter out empty sections
        non_empty_sections = [s for s in sections if s.strip()]
        assume(len(non_empty_sections) > 0)
        
        document = create_multi_section_document(non_empty_sections)
        original_section_count = len(document.texts)
        
        result = mask_document_with_detection(
            document, policy, analyzer=shared_analyzer, resolve_conflicts=False
        )
        
        # All sections should be preserved
        assert len(result.masked_document.texts) == original_section_count
        
        # Each section should have corresponding masked content
        for i, original_section in enumerate(document.texts):
            masked_section = result.masked_document.texts[i]
            assert masked_section.self_ref == original_section.self_ref
            assert len(masked_section.text) > 0  # Should not be empty after masking

    @given(
        pii_text=pii_text_strategy(),
        policy=light_policy_strategy()
    )
    def test_pii_detection_and_masking_with_deterministic_input(
        self, pii_text: str, policy: MaskingPolicy
    ):
        """Property: PII is consistently detected and masked with deterministic patterns.
        
        Uses FastRegexDetector for predictable, fast unit-level testing.
        """
        document = create_simple_document(pii_text)
        
        # Use fast detector for unit-level testing
        fast_detector = FastRegexDetector()
        
        # Manually perform detection and masking for controlled testing
        from cloakpivot.document.extractor import TextExtractor
        from cloakpivot.masking.engine import MaskingEngine
        
        extractor = TextExtractor()
        segments = extractor.extract_text_segments(document)
        
        all_entities = []
        for segment in segments:
            entities = fast_detector.analyze_text(segment.text)
            all_entities.extend(entities)
        
        if all_entities:  # Only test if entities were detected
            engine = MaskingEngine(resolve_conflicts=False)
            result = engine.mask_document(document, all_entities, policy, segments)
            
            # PII should be masked (no original PII in final text)
            masked_text = result.masked_document.texts[0].text
            assert masked_text != pii_text  # Text should be changed
            
            # CloakMap should contain anchors for detected entities
            assert len(result.cloakmap.anchors) == len(all_entities)
            
            # No original PII should remain in cloakmap JSON
            cloakmap_json = result.cloakmap.to_json()
            if "555-123-4567" in pii_text:
                assert "555-123-4567" not in cloakmap_json
            if "john.doe@example.com" in pii_text:
                assert "john.doe@example.com" not in cloakmap_json

    @given(text=constrained_text_strategy(max_size=50))
    def test_masking_with_no_entities_is_identity(
        self, text: str, shared_analyzer: AnalyzerEngine
    ):
        """Property: Masking text with no detected entities should leave it unchanged."""
        assume(len(text.strip()) > 0)
        
        document = create_simple_document(text)
        policy = MaskingPolicy(
            default_strategy=Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"})
        )
        
        result = mask_document_with_detection(
            document, policy, analyzer=shared_analyzer, resolve_conflicts=False
        )
        
        # If no entities detected, text should be unchanged
        if len(result.cloakmap.anchors) == 0:
            assert result.masked_document.texts[0].text == document.texts[0].text

    @given(
        text=constrained_text_strategy(max_size=80),
        policy=light_policy_strategy()
    )
    def test_cloakmap_integrity(
        self, text: str, policy: MaskingPolicy, shared_analyzer: AnalyzerEngine
    ):
        """Property: CloakMap maintains integrity invariants."""
        assume(len(text.strip()) > 0)
        
        document = create_simple_document(text)
        
        result = mask_document_with_detection(
            document, policy, analyzer=shared_analyzer, resolve_conflicts=False
        )
        
        cloakmap = result.cloakmap
        
        # CloakMap invariants
        assert cloakmap.doc_id is not None
        assert len(cloakmap.doc_id) > 0
        
        # All anchors should have valid properties
        for anchor in cloakmap.anchors:
            assert anchor.node_id.startswith("#/")
            assert anchor.start >= 0
            assert anchor.end > anchor.start
            assert len(anchor.replacement_id) > 0
            assert len(anchor.original_checksum) == 64  # SHA-256 hex
            assert anchor.confidence >= 0.0
            assert anchor.confidence <= 1.0


@pytest.mark.slow
class TestPropertyBasedMaskingSlow:
    """Comprehensive property-based tests with broader generators and full Presidio."""

    @given(
        text=st.text(max_size=500),  # Larger, less constrained text
        policy=st.one_of([
            light_policy_strategy(),
            # Add more expensive strategies for comprehensive testing
            st.just(MaskingPolicy(
                default_strategy=Strategy(StrategyKind.SURROGATE, {})
            )),
        ])
    )
    def test_comprehensive_masking_with_full_presidio(
        self, text: str, policy: MaskingPolicy, shared_analyzer: AnalyzerEngine
    ):
        """Comprehensive property test using full Presidio and broader generators."""
        assume(len(text.strip()) > 2)
        
        document = create_simple_document(text)
        
        result = mask_document_with_detection(
            document, policy, analyzer=shared_analyzer, 
            resolve_conflicts=True,  # Enable conflict resolution for comprehensive testing
            timing_log=True  # Enable timing logs for slow examples
        )
        
        # Basic invariants should hold even with complex inputs
        assert result.masked_document is not None
        assert result.cloakmap is not None
        assert len(result.masked_document.texts) == len(document.texts)

    @given(
        sections=st.lists(st.text(max_size=200), min_size=2, max_size=8),
        policy=light_policy_strategy()
    )
    def test_large_multi_section_document_performance(
        self, sections: list[str], policy: MaskingPolicy, shared_analyzer: AnalyzerEngine
    ):
        """Test performance with larger multi-section documents."""
        non_empty_sections = [s for s in sections if len(s.strip()) > 1]
        assume(len(non_empty_sections) >= 2)
        
        document = create_multi_section_document(non_empty_sections)
        
        import time
        start_time = time.perf_counter()
        
        result = mask_document_with_detection(
            document, policy, analyzer=shared_analyzer, 
            resolve_conflicts=False,
            timing_log=True
        )
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        # Performance assertion: should complete within reasonable time
        # This helps identify performance regressions
        assert processing_time < 10.0, f"Processing took too long: {processing_time:.2f}s"
        
        # Functional assertions
        assert len(result.masked_document.texts) == len(non_empty_sections)
        assert result.cloakmap is not None


@pytest.mark.performance
class TestMaskingPerformanceBenchmarks:
    """Performance-focused tests for identifying bottlenecks."""

    def test_analyzer_reuse_performance_benefit(self, shared_analyzer: AnalyzerEngine):
        """Benchmark the performance benefit of analyzer reuse."""
        import time
        
        test_text = "Contact John Smith at 555-123-4567 for more information"
        document = create_simple_document(test_text)
        policy = MaskingPolicy(
            default_strategy=Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"})
        )
        
        # Test with shared analyzer
        start_time = time.perf_counter()
        for _ in range(10):
            mask_document_with_detection(document, policy, analyzer=shared_analyzer)
        shared_time = time.perf_counter() - start_time
        
        # Test with per-call analyzer creation
        start_time = time.perf_counter()
        for _ in range(10):
            mask_document_with_detection(document, policy, analyzer=None)
        individual_time = time.perf_counter() - start_time
        
        # Shared analyzer should be significantly faster
        assert shared_time < individual_time
        performance_improvement = individual_time / shared_time
        print(f"Performance improvement with shared analyzer: {performance_improvement:.2f}x")

    def test_conflict_resolution_performance_impact(self, shared_analyzer: AnalyzerEngine):
        """Measure the performance impact of conflict resolution."""
        import time
        
        # Create text with potential overlapping entities
        test_text = "John Smith's phone is 555-123-4567 and another number 555-987-6543"
        document = create_simple_document(test_text)
        policy = MaskingPolicy(
            default_strategy=Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"})
        )
        
        # Test without conflict resolution
        start_time = time.perf_counter()
        for _ in range(20):
            mask_document_with_detection(
                document, policy, analyzer=shared_analyzer, resolve_conflicts=False
            )
        no_conflicts_time = time.perf_counter() - start_time
        
        # Test with conflict resolution
        start_time = time.perf_counter()
        for _ in range(20):
            mask_document_with_detection(
                document, policy, analyzer=shared_analyzer, resolve_conflicts=True
            )
        with_conflicts_time = time.perf_counter() - start_time
        
        # Report the performance difference
        overhead_ratio = with_conflicts_time / no_conflicts_time
        print(f"Conflict resolution overhead: {overhead_ratio:.2f}x")
        
        # Conflict resolution should complete within reasonable time
        assert with_conflicts_time < 5.0  # Should not take more than 5 seconds for 20 iterations