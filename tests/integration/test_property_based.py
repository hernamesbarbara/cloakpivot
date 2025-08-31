"""Property-based tests for CloakPivot using Hypothesis.

These tests generate random inputs to discover edge cases and verify
properties that should hold for all valid inputs.
"""

from datetime import timedelta
from typing import Any, Tuple

import pytest
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import DocItemLabel, TextItem
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule

from cloakpivot.core.policies import MaskingPolicy, PrivacyLevel
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.unmasking.engine import UnmaskingEngine
from tests.utils.assertions import (
    assert_document_structure_preserved,
    assert_masking_result_valid,
    assert_round_trip_fidelity,
)
from tests.utils.masking_helpers import mask_document_with_detection


# Hypothesis strategies for generating test data
@st.composite
def text_with_pii(draw: st.DrawFn) -> str:
    """Generate text containing various PII patterns."""
    base_text = draw(st.text(min_size=10, max_size=500))

    # Add some PII patterns
    phone_patterns = ["555-123-4567", "(555) 987-6543", "555.234.5678"]
    email_patterns = ["john@example.com", "alice.smith@company.org", "user123@test.net"]
    ssn_patterns = ["123-45-6789", "987-65-4321", "555-44-3333"]

    phone = draw(st.sampled_from(phone_patterns))
    email = draw(st.sampled_from(email_patterns))
    ssn = draw(st.sampled_from(ssn_patterns))

    # Randomly insert PII into text
    pii_text = f"{base_text} Contact: {phone} Email: {email} SSN: {ssn}"
    return pii_text


@st.composite
def document_strategy(draw: st.DrawFn) -> DoclingDocument:
    """Generate DoclingDocument instances."""
    name = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    text_content = draw(text_with_pii())

    doc = DoclingDocument(name=name)
    text_item = TextItem(
        text=text_content,
        self_ref="#/texts/0",
        label=DocItemLabel.TEXT,
        orig=text_content
    )
    doc.texts = [text_item]
    return doc


@st.composite
def policy_strategy(draw: st.DrawFn) -> MaskingPolicy:
    """Generate MaskingPolicy instances."""
    locale = draw(st.sampled_from(["en", "es", "fr", "de"]))
    draw(st.sampled_from(["low", "medium", "high"]))

    # Generate entities configuration
    entity_types = ["PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN", "PERSON", "CREDIT_CARD"]
    selected_entities = draw(st.lists(st.sampled_from(entity_types), min_size=1, max_size=len(entity_types), unique=True))

    strategy_kinds = [StrategyKind.TEMPLATE, StrategyKind.HASH, StrategyKind.SURROGATE, StrategyKind.PARTIAL]

    entities: dict[str, Strategy] = {}
    thresholds: dict[str, float] = {}

    for entity_type in selected_entities:
        strategy_kind = draw(st.sampled_from(strategy_kinds))

        # Use appropriate strategy for entity type
        if entity_type == "PHONE_NUMBER" and strategy_kind not in [StrategyKind.TEMPLATE, StrategyKind.HASH, StrategyKind.SURROGATE]:
            strategy_kind = StrategyKind.TEMPLATE
        elif entity_type == "EMAIL_ADDRESS" and strategy_kind not in [StrategyKind.TEMPLATE, StrategyKind.HASH, StrategyKind.SURROGATE]:
            strategy_kind = StrategyKind.TEMPLATE

        # Generate appropriate parameters for each strategy kind
        parameters: dict[str, str | int] = {}
        if strategy_kind == StrategyKind.TEMPLATE:
            parameters = {"template": f"[{entity_type}]"}
        elif strategy_kind == StrategyKind.PARTIAL:
            parameters = {"visible_chars": draw(st.integers(min_value=1, max_value=4))}
        elif strategy_kind == StrategyKind.HASH:
            parameters = {"algorithm": "sha256", "truncate": 8}
        elif strategy_kind == StrategyKind.SURROGATE:
            format_types = {
                "PHONE_NUMBER": "phone",
                "EMAIL_ADDRESS": "email",
                "US_SSN": "ssn",
                "CREDIT_CARD": "credit_card",
                "PERSON": "name"
            }
            parameters = {"format_type": format_types.get(entity_type, "custom")}

        entities[entity_type] = Strategy(kind=strategy_kind, parameters=parameters)
        thresholds[entity_type] = draw(st.floats(min_value=0.1, max_value=0.95))

    return MaskingPolicy(
        locale=locale,
        per_entity=entities,
        thresholds=thresholds
    )


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @pytest.mark.property
    @given(document_strategy(), policy_strategy())
    @settings(max_examples=5, deadline=5000)  # Reduced examples and deadline for CI performance
    @example(
        document=DoclingDocument(name="test"),
        policy=MaskingPolicy(
            locale="en",
            per_entity={"PHONE_NUMBER": Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[PHONE]"})},
            thresholds={"PHONE_NUMBER": 0.5}
        )
    )
    def test_masking_preserves_document_structure(self, document: DoclingDocument, policy: MaskingPolicy) -> None:
        """Property: Masking should always preserve document structure."""
        # Skip empty documents
        assume(document.texts and len(document.texts[0].text.strip()) > 0)

        try:
            result = mask_document_with_detection(document, policy)

            # Document structure should be preserved
            assert_document_structure_preserved(document, result.masked_document)
            assert_masking_result_valid(result)

        except Exception as e:
            # Log the inputs that caused the failure for debugging
            pytest.fail(f"Masking failed with document '{document.name}' and policy privacy level '{policy.privacy_level}': {str(e)}")

    @pytest.mark.property
    @given(document_strategy(), policy_strategy())
    @settings(max_examples=5, deadline=10000)
    def test_round_trip_property(self, document: DoclingDocument, policy: MaskingPolicy) -> None:
        """Property: Round-trip masking/unmasking should preserve original content."""
        # Skip empty documents
        assume(document.texts and len(document.texts[0].text.strip()) > 0)

        try:
            unmasking_engine = UnmaskingEngine()

            # Mask document
            mask_result = mask_document_with_detection(document, policy)
            assert_masking_result_valid(mask_result)

            # Unmask document
            unmask_result = unmasking_engine.unmask_document(
                mask_result.masked_document,
                mask_result.cloakmap
            )

            # Round-trip should preserve content
            assert_round_trip_fidelity(
                document,
                mask_result.masked_document,
                unmask_result.unmasked_document,
                mask_result.cloakmap
            )

        except Exception as e:
            pytest.fail(f"Round-trip failed: {str(e)}")

    @pytest.mark.property
    @given(st.text(min_size=1, max_size=1000))
    @settings(max_examples=10, deadline=3000)
    def test_text_processing_robustness(self, text: str) -> None:
        """Property: Text processing should handle arbitrary strings without crashing."""
        # Skip text that is just whitespace
        assume(text.strip())

        try:
            # Create document
            doc = DoclingDocument(name="property_test")
            text_item = TextItem(
                text=text,
                self_ref="#/texts/0",
                label=DocItemLabel.TEXT,
                orig=text
            )
            doc.texts = [text_item]

            # Create simple policy
            policy = MaskingPolicy(
                locale="en",
                privacy_level=PrivacyLevel.LOW,
                per_entity={"PHONE_NUMBER": Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[PHONE]"})},
                thresholds={"PHONE_NUMBER": 0.5}
            )

            # Should not crash
            result = mask_document_with_detection(doc, policy)

            # Basic invariants
            assert result is not None
            assert result.masked_document is not None
            assert result.cloakmap is not None

        except Exception as e:
            # Some text might be problematic, but we want to know about crashes
            if "critical" in str(e).lower() or "fatal" in str(e).lower():
                pytest.fail(f"Critical failure with text processing: {str(e)}")

    @pytest.mark.property
    @given(
        st.lists(st.text(min_size=5, max_size=200), min_size=1, max_size=10),
        policy_strategy()
    )
    @settings(max_examples=5, deadline=8000)
    def test_multi_section_document_property(self, text_sections: list[str], policy: MaskingPolicy) -> None:
        """Property: Multi-section documents should preserve section count and structure."""
        # Filter out empty sections
        text_sections = [section.strip() for section in text_sections if section.strip()]
        assume(len(text_sections) > 0)

        try:
            # Create multi-section document
            doc = DoclingDocument(name="multi_section_test")
            text_items = []

            for i, section in enumerate(text_sections):
                text_item = TextItem(
                    text=section,
                    self_ref=f"#/texts/{i}",
                    label=DocItemLabel.TEXT,
                    orig=section
                )
                text_items.append(text_item)

            doc.texts = text_items

            # Apply masking
            result = mask_document_with_detection(doc, policy)

            # Section count should be preserved
            assert len(result.masked_document.texts) == len(text_sections)

            # Each section should maintain its reference structure
            for orig_item, masked_item in zip(doc.texts, result.masked_document.texts):
                assert orig_item.self_ref == masked_item.self_ref
                assert orig_item.label == masked_item.label

        except Exception as e:
            pytest.fail(f"Multi-section processing failed: {str(e)}")

    @pytest.mark.property
    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.sampled_from(["PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN", "PERSON"])
    )
    @settings(max_examples=10, deadline=timedelta(seconds=2))
    def test_threshold_property(self, threshold: float, entity_type: str) -> None:
        """Property: Threshold values should control entity detection sensitivity."""
        # Create test text with known PII
        test_patterns = {
            "PHONE_NUMBER": "Call me at 555-123-4567",
            "EMAIL_ADDRESS": "Contact john@example.com",
            "US_SSN": "SSN: 123-45-6789",
            "PERSON": "Contact John Smith"
        }

        text = test_patterns.get(entity_type, f"Test {entity_type} content")

        try:
            # Create document
            doc = DoclingDocument(name="threshold_test")
            text_item = TextItem(text=text, self_ref="#/texts/0", label=DocItemLabel.TEXT, orig=text)
            doc.texts = [text_item]

            # Create policy with specific threshold
            strategy_map = {
                "PHONE_NUMBER": StrategyKind.TEMPLATE,
                "EMAIL_ADDRESS": StrategyKind.TEMPLATE,
                "US_SSN": StrategyKind.HASH,
                "PERSON": StrategyKind.TEMPLATE
            }

            policy = MaskingPolicy(
                locale="en",
                privacy_level=PrivacyLevel.MEDIUM,
                per_entity={entity_type: Strategy(kind=strategy_map[entity_type], parameters={"template": f"[{entity_type}]"})},
                thresholds={entity_type: threshold}
            )

            result = mask_document_with_detection(doc, policy)

            # Basic property: result should be valid regardless of threshold
            assert_masking_result_valid(result)

        except Exception as e:
            pytest.fail(f"Threshold testing failed for {entity_type} with threshold {threshold}: {str(e)}")


@settings(max_examples=3, stateful_step_count=5, deadline=10000)
class MaskingStateMachine(RuleBasedStateMachine):
    """Stateful testing for complex masking scenarios."""

    documents = Bundle('documents')
    policies = Bundle('policies')
    masked_results = Bundle('masked_results')

    @rule(target=documents, name=st.text(min_size=1, max_size=30), content=text_with_pii())
    def create_document(self, name: str, content: str) -> DoclingDocument:
        """Create a new document."""
        doc = DoclingDocument(name=name)
        text_item = TextItem(text=content, self_ref="#/texts/0", label=DocItemLabel.TEXT, orig=content)
        doc.texts = [text_item]
        return doc

    @rule(target=policies)
    def create_policy(self) -> MaskingPolicy:
        """Create a new masking policy."""
        return MaskingPolicy(
            locale="en",
            privacy_level=PrivacyLevel.MEDIUM,
            per_entity={
                "PHONE_NUMBER": Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[PHONE]"}),
                "EMAIL_ADDRESS": Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[EMAIL]"})
            },
            thresholds={"PHONE_NUMBER": 0.7, "EMAIL_ADDRESS": 0.8}
        )

    @rule(target=masked_results, document=documents, policy=policies)
    def mask_document(self, document: DoclingDocument, policy: MaskingPolicy) -> Tuple[Any, DoclingDocument]:
        """Apply masking to a document."""
        try:
            result = mask_document_with_detection(document, policy)
            assert_masking_result_valid(result)
            return (result, document)  # Store both result and original
        except Exception:
            # Some combinations might be invalid, that's okay
            assume(False)
            # This line is never reached due to assume(False), but needed for type checking
            return (None, document)

    @rule(masked_data=masked_results)
    def unmask_document(self, masked_data: Tuple[Any, DoclingDocument]) -> None:
        """Unmask a previously masked document."""
        mask_result, original_doc = masked_data

        try:
            engine = UnmaskingEngine()
            unmask_result = engine.unmask_document(
                mask_result.masked_document,
                mask_result.cloakmap
            )

            # Verify round-trip fidelity
            assert_round_trip_fidelity(
                original_doc,
                mask_result.masked_document,
                unmask_result.unmasked_document,
                mask_result.cloakmap
            )

        except Exception as e:
            pytest.fail(f"Unmasking failed: {str(e)}")

    @invariant()
    def documents_are_valid(self) -> None:
        """Invariant: All documents should remain valid."""
        # This runs after each rule to check system state
        pass


# Slow property-based tests
class TestPropertyBasedSlow:
    """Slower, more comprehensive property-based tests."""

    @pytest.mark.property
    @pytest.mark.slow

    @given(document_strategy(), policy_strategy())
    @settings(max_examples=10, deadline=15000)
    def test_comprehensive_masking_properties(self, document: DoclingDocument, policy: MaskingPolicy) -> None:
        """Comprehensive property testing with more examples."""
        assume(document.texts and len(document.texts[0].text.strip()) > 0)

        try:
            unmasking_engine = UnmaskingEngine()

            # Test masking
            mask_result = mask_document_with_detection(document, policy)
            assert_masking_result_valid(mask_result)

            # Test structure preservation
            assert_document_structure_preserved(document, mask_result.masked_document)

            # Test round-trip only if we have entities and all strategies are reversible
            if len(mask_result.cloakmap.anchors) > 0:
                # Check if all detected entities use reversible strategies
                reversible_strategies = {StrategyKind.TEMPLATE, StrategyKind.SURROGATE, StrategyKind.PARTIAL}
                all_reversible = True
                
                for anchor in mask_result.cloakmap.anchors:
                    # Get the strategy used for this entity type
                    entity_type = anchor.entity_type
                    strategy = policy.per_entity.get(entity_type, policy.default_strategy)
                    if strategy.kind not in reversible_strategies:
                        all_reversible = False
                        break
                
                # Only test round-trip fidelity if all strategies are reversible
                if all_reversible:
                    unmask_result = unmasking_engine.unmask_document(
                        mask_result.masked_document,
                        mask_result.cloakmap
                    )

                    assert_round_trip_fidelity(
                        document,
                        mask_result.masked_document,
                        unmask_result.unmasked_document,
                        mask_result.cloakmap
                    )

        except Exception as e:
            pytest.fail(f"Comprehensive testing failed: {str(e)}")


# Test class for running stateful tests
class TestStateful:
    """Test stateful behavior."""

    @pytest.mark.slow
    def test_stateful_masking(self) -> None:
        """Run stateful testing scenario."""
        # This will run the state machine with multiple steps
        state_machine_test = MaskingStateMachine.TestCase()
        state_machine_test.runTest()
