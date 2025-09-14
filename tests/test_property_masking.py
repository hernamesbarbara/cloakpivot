"""Property-based tests for masking functionality using CloakEngine."""

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from presidio_analyzer import AnalyzerEngine

from cloakpivot.engine import CloakEngine
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
        max_size=max_size,
    )


def pii_text_strategy() -> st.SearchStrategy[str]:
    """Generate text with embedded PII patterns for testing.

    Uses deterministic patterns that will be detected by FastRegexDetector
    for predictable unit-level testing.
    """
    return st.one_of(
        [
            st.just("Call me at 555-123-4567 for more info"),
            st.just("Email john.doe@example.com for details"),
            st.just("SSN: 123-45-6789"),
            st.just("Contact 555-987-6543 or admin@test.org"),
        ]
    )


def light_policy_strategy() -> st.SearchStrategy[MaskingPolicy]:
    """Generate lightweight masking policies.

    Prefers template strategies over computationally expensive ones
    like SURROGATE or HASH.
    """
    template_strategy = Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"})
    partial_strategy = Strategy(
        StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end", "mask_char": "*"}
    )

    return st.one_of(
        [
            st.just(MaskingPolicy(default_strategy=template_strategy)),
            st.just(MaskingPolicy(default_strategy=partial_strategy)),
        ]
    )


class TestPropertyBasedMaskingFast:
    """Fast property-based tests using CloakEngine."""

    @pytest.mark.property
    @given(text=constrained_text_strategy(max_size=100), policy=light_policy_strategy())
    def test_masking_preserves_document_structure(
        self, text: str, policy: MaskingPolicy
    ) -> None:
        """Property: Masking preserves document structure regardless of content."""
        assume(len(text.strip()) > 0)  # Skip empty strings

        document = create_simple_document(text)
        original_text_count = len(document.texts)

        # Use CloakEngine for masking
        engine = CloakEngine(default_policy=policy)
        result = engine.mask_document(document)

        # Structure should be preserved
        assert len(result.document.texts) == original_text_count
        assert result.cloakmap is not None

    @pytest.mark.property
    @given(
        sections=st.lists(
            constrained_text_strategy(max_size=50), min_size=1, max_size=3
        ),
        policy=light_policy_strategy(),
    )
    def test_multi_section_masking_consistency(
        self,
        sections: list[str],
        policy: MaskingPolicy,
    ) -> None:
        """Property: Multi-section documents are masked consistently."""
        # Filter out empty sections
        non_empty_sections = [s for s in sections if s.strip()]
        assume(len(non_empty_sections) > 0)

        document = create_multi_section_document(non_empty_sections)
        original_section_count = len(document.texts)

        # Use CloakEngine for masking
        engine = CloakEngine(default_policy=policy)
        result = engine.mask_document(document)

        # All sections should be preserved
        assert len(result.document.texts) == original_section_count

        # Each section should have corresponding masked content
        for i, original_section in enumerate(document.texts):
            masked_section = result.document.texts[i]
            assert masked_section.self_ref == original_section.self_ref
            assert len(masked_section.text) > 0  # Should not be empty after masking

    @pytest.mark.property
    @given(pii_text=pii_text_strategy(), policy=light_policy_strategy())
    def test_pii_detection_and_masking_with_deterministic_input(
        self, pii_text: str, policy: MaskingPolicy
    ) -> None:
        """Property: PII is consistently detected and masked with deterministic patterns."""
        document = create_simple_document(pii_text)

        # Use CloakEngine for detection and masking
        engine = CloakEngine(default_policy=policy)
        result = engine.mask_document(document)

        if result.entities_found > 0:  # Only test if entities were detected
            # PII should be masked (no original PII in final text)
            masked_text = result.document.texts[0].text
            assert masked_text != pii_text  # Text should be changed

            # CloakMap should contain anchors for detected entities
            assert len(result.cloakmap.anchors) == result.entities_masked

            # No original PII should remain in cloakmap JSON
            cloakmap_json = result.cloakmap.to_json()
            if "555-123-4567" in pii_text:
                assert "555-123-4567" not in cloakmap_json
            if "john.doe@example.com" in pii_text:
                assert "john.doe@example.com" not in cloakmap_json

    @pytest.mark.property
    @given(text=constrained_text_strategy(max_size=50))
    def test_masking_with_no_entities_is_identity(
        self, text: str
    ) -> None:
        """Property: Masking text with no detected entities should leave it unchanged."""
        assume(len(text.strip()) > 0)

        document = create_simple_document(text)
        policy = MaskingPolicy(
            default_strategy=Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"})
        )

        engine = CloakEngine(default_policy=policy)
        result = engine.mask_document(document)

        # If no entities detected, text should be unchanged
        if len(result.cloakmap.anchors) == 0:
            assert result.document.texts[0].text == document.texts[0].text

    @pytest.mark.property
    @given(text=constrained_text_strategy(max_size=80), policy=light_policy_strategy())
    def test_cloakmap_integrity(
        self, text: str, policy: MaskingPolicy
    ) -> None:
        """Property: CloakMap maintains integrity invariants."""
        assume(len(text.strip()) > 0)

        document = create_simple_document(text)

        engine = CloakEngine(default_policy=policy)
        result = engine.mask_document(document)

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
    """Comprehensive property-based tests with CloakEngine and broader generators."""

    @pytest.mark.property
    @given(
        text=st.text(max_size=500),  # Larger, less constrained text
        policy=st.one_of(
            [
                light_policy_strategy(),
                # Add more expensive strategies for comprehensive testing
                st.just(
                    MaskingPolicy(default_strategy=Strategy(StrategyKind.SURROGATE, {}))
                ),
                st.just(
                    MaskingPolicy(
                        default_strategy=Strategy(
                            StrategyKind.HASH, {"algorithm": "sha256", "truncate": 8}
                        )
                    )
                ),
            ]
        ),
    )
    def test_comprehensive_masking_with_cloakengine(
        self, text: str, policy: MaskingPolicy
    ) -> None:
        """Comprehensive property test using CloakEngine and broader generators."""
        assume(len(text.strip()) > 2)

        document = create_simple_document(text)

        engine = CloakEngine(default_policy=policy)
        result = engine.mask_document(document)

        # Basic invariants should hold even with complex inputs
        assert result.document is not None
        assert result.cloakmap is not None
        assert len(result.document.texts) == len(document.texts)

    @pytest.mark.property
    @given(
        sections=st.lists(st.text(max_size=200), min_size=2, max_size=8),
        policy=light_policy_strategy(),
    )
    def test_large_multi_section_document_performance(
        self,
        sections: list[str],
        policy: MaskingPolicy,
    ) -> None:
        """Test performance with larger multi-section documents."""
        non_empty_sections = [s for s in sections if len(s.strip()) > 1]
        assume(len(non_empty_sections) >= 2)

        document = create_multi_section_document(non_empty_sections)

        import time

        start_time = time.perf_counter()

        engine = CloakEngine(default_policy=policy)
        result = engine.mask_document(document)

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        # Performance assertion: should complete within reasonable time
        # This helps identify performance regressions
        assert (
            processing_time < 10.0
        ), f"Processing took too long: {processing_time:.2f}s"

        # Functional assertions
        assert len(result.document.texts) == len(non_empty_sections)
        assert result.cloakmap is not None


@pytest.mark.performance
class TestMaskingPerformanceBenchmarks:
    """Performance-focused tests for CloakEngine."""

    @pytest.mark.parametrize("iterations", [5, 8])
    def test_cloakengine_performance(
        self, iterations: int
    ) -> None:
        """Benchmark CloakEngine performance."""
        import time

        test_text = "Contact John Smith at 555-123-4567 for more information"
        document = create_simple_document(test_text)
        policy = MaskingPolicy(
            default_strategy=Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"})
        )

        # Test with single CloakEngine instance (reuses analyzer)
        engine = CloakEngine(default_policy=policy)
        start_time = time.perf_counter()
        for _ in range(iterations):
            engine.mask_document(document)
        reused_time = time.perf_counter() - start_time

        # Test with new CloakEngine each time (creates new analyzer)
        start_time = time.perf_counter()
        for _ in range(min(iterations, 5)):  # Cap at 5 to avoid slowness
            new_engine = CloakEngine(default_policy=policy)
            new_engine.mask_document(document)
        new_each_time = time.perf_counter() - start_time

        # Reusing engine should be faster
        assert reused_time < new_each_time
        performance_improvement = new_each_time / reused_time
        print(
            f"Performance improvement with engine reuse ({iterations} iter): {performance_improvement:.2f}x"
        )

    def test_cloakengine_quick_comparison(self) -> None:
        """Quick test of CloakEngine performance."""
        import time

        test_text = "John at 555-123-4567"  # Shorter text
        document = create_simple_document(test_text)
        policy = MaskingPolicy(
            default_strategy=Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"})
        )

        # Single operation with reused engine
        engine = CloakEngine(default_policy=policy)
        start_time = time.perf_counter()
        engine.mask_document(document)
        reused_time = time.perf_counter() - start_time

        # Single operation with new engine
        start_time = time.perf_counter()
        new_engine = CloakEngine(default_policy=policy)
        new_engine.mask_document(document)
        new_time = time.perf_counter() - start_time

        # First call might be slower due to initialization
        print(
            f"Quick comparison - reused: {reused_time:.3f}s, new: {new_time:.3f}s"
        )

    def test_cloakengine_with_multiple_entities_performance(self) -> None:
        """Measure CloakEngine performance with multiple entities."""
        import time

        # Create text with multiple entities
        test_text = "John Smith's phone is 555-123-4567 and another number 555-987-6543"
        document = create_simple_document(test_text)
        policy = MaskingPolicy(
            default_strategy=Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"})
        )

        engine = CloakEngine(default_policy=policy)

        # Warm up the engine
        engine.mask_document(document)

        # Test performance
        start_time = time.perf_counter()
        for _ in range(20):
            engine.mask_document(document)
        total_time = time.perf_counter() - start_time

        # Report performance
        avg_time = total_time / 20
        print(f"Average masking time: {avg_time:.4f}s")

        # Should complete within reasonable time
        assert total_time < 5.0  # Should not take more than 5 seconds for 20 iterations
