"""Tests for the StrategyToOperatorMapper class."""

import pytest
from presidio_anonymizer.entities import OperatorConfig

from cloakpivot.core.presidio_mapper import StrategyToOperatorMapper
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind


class TestStrategyToOperatorMapper:
    """Test suite for StrategyToOperatorMapper class."""

    @pytest.fixture
    def mapper(self) -> StrategyToOperatorMapper:
        """Create a mapper instance for testing."""
        return StrategyToOperatorMapper()

    def test_mapper_initialization(self, mapper: StrategyToOperatorMapper) -> None:
        """Test that mapper initializes correctly."""
        assert mapper is not None
        assert hasattr(mapper, '_strategy_mapping')
        assert len(mapper._strategy_mapping) == 6  # All StrategyKind values

    def test_redact_strategy_basic(self, mapper: StrategyToOperatorMapper) -> None:
        """Test basic REDACT strategy mapping."""
        strategy = Strategy(StrategyKind.REDACT)
        result = mapper.strategy_to_operator(strategy)
        
        assert isinstance(result, OperatorConfig)
        assert result.operator_name == "redact"
        assert result.params["redact_char"] == "*"

    def test_redact_strategy_custom_char(self, mapper: StrategyToOperatorMapper) -> None:
        """Test REDACT strategy with custom redact character."""
        strategy = Strategy(StrategyKind.REDACT, {"redact_char": "#"})
        result = mapper.strategy_to_operator(strategy)
        
        assert result.operator_name == "redact"
        assert result.params["redact_char"] == "#"

    def test_redact_strategy_preserve_length(self, mapper: StrategyToOperatorMapper) -> None:
        """Test REDACT strategy with preserve_length parameter."""
        strategy = Strategy(StrategyKind.REDACT, {"preserve_length": False})
        result = mapper.strategy_to_operator(strategy)
        
        # Should still work even though preserve_length is not directly supported
        assert result.operator_name == "redact"
        assert "redact_char" in result.params

    def test_template_strategy_basic(self, mapper: StrategyToOperatorMapper) -> None:
        """Test basic TEMPLATE strategy mapping."""
        strategy = Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"})
        result = mapper.strategy_to_operator(strategy)
        
        assert result.operator_name == "replace"
        assert result.params["new_value"] == "[PHONE]"

    def test_template_strategy_auto_generate(self, mapper: StrategyToOperatorMapper) -> None:
        """Test TEMPLATE strategy with auto_generate."""
        strategy = Strategy(StrategyKind.TEMPLATE, {"auto_generate": True})
        result = mapper.strategy_to_operator(strategy)
        
        assert result.operator_name == "replace"
        assert "new_value" in result.params
        assert result.params["new_value"] == "[MASKED]"

    def test_template_strategy_default(self, mapper: StrategyToOperatorMapper) -> None:
        """Test TEMPLATE strategy with auto_generate as fallback."""
        strategy = Strategy(StrategyKind.TEMPLATE, {"template": "[DEFAULT]"})
        result = mapper.strategy_to_operator(strategy)
        
        assert result.operator_name == "replace"
        assert result.params["new_value"] == "[DEFAULT]"

    def test_hash_strategy_basic(self, mapper: StrategyToOperatorMapper) -> None:
        """Test basic HASH strategy mapping."""
        strategy = Strategy(StrategyKind.HASH)
        result = mapper.strategy_to_operator(strategy)
        
        assert result.operator_name == "hash"
        assert result.params["hash_type"] == "sha256"

    def test_hash_strategy_custom_algorithm(self, mapper: StrategyToOperatorMapper) -> None:
        """Test HASH strategy with custom algorithm."""
        strategy = Strategy(StrategyKind.HASH, {"algorithm": "md5"})
        result = mapper.strategy_to_operator(strategy)
        
        assert result.operator_name == "hash"
        assert result.params["hash_type"] == "md5"

    def test_hash_strategy_with_salt(self, mapper: StrategyToOperatorMapper) -> None:
        """Test HASH strategy with salt parameter."""
        strategy = Strategy(StrategyKind.HASH, {"algorithm": "sha256", "salt": "my-salt"})
        result = mapper.strategy_to_operator(strategy)
        
        assert result.operator_name == "hash"
        assert result.params["hash_type"] == "sha256"
        assert result.params["salt"] == "my-salt"

    def test_hash_strategy_per_entity_salt(self, mapper: StrategyToOperatorMapper) -> None:
        """Test HASH strategy with per_entity_salt parameter."""
        strategy = Strategy(StrategyKind.HASH, {
            "per_entity_salt": {
                "PHONE_NUMBER": "phone_salt",
                "EMAIL_ADDRESS": "email_salt",
                "default": "default_salt"
            }
        })
        result = mapper.strategy_to_operator(strategy)
        
        assert result.operator_name == "hash"
        assert result.params["salt"] == "default_salt"

    def test_partial_strategy_basic(self, mapper: StrategyToOperatorMapper) -> None:
        """Test basic PARTIAL strategy mapping."""
        strategy = Strategy(StrategyKind.PARTIAL, {"visible_chars": 4})
        result = mapper.strategy_to_operator(strategy)
        
        assert result.operator_name == "mask"
        assert result.params["chars_to_mask"] == 4
        assert result.params["from_end"] == True
        assert result.params["masking_char"] == "*"

    def test_partial_strategy_start_position(self, mapper: StrategyToOperatorMapper) -> None:
        """Test PARTIAL strategy with start position."""
        strategy = Strategy(StrategyKind.PARTIAL, {
            "visible_chars": 3,
            "position": "start",
            "mask_char": "#"
        })
        result = mapper.strategy_to_operator(strategy)
        
        assert result.operator_name == "mask"
        assert result.params["chars_to_mask"] == -1
        assert result.params["from_end"] == False
        assert result.params["masking_char"] == "#"

    def test_partial_strategy_end_position(self, mapper: StrategyToOperatorMapper) -> None:
        """Test PARTIAL strategy with end position."""
        strategy = Strategy(StrategyKind.PARTIAL, {
            "visible_chars": 4,
            "position": "end",
            "mask_char": "X"
        })
        result = mapper.strategy_to_operator(strategy)
        
        assert result.operator_name == "mask"
        assert result.params["chars_to_mask"] == 4
        assert result.params["from_end"] == True
        assert result.params["masking_char"] == "X"

    def test_partial_strategy_unsupported_positions(self, mapper: StrategyToOperatorMapper) -> None:
        """Test PARTIAL strategy with unsupported positions."""
        for position in ["middle", "random"]:
            strategy = Strategy(StrategyKind.PARTIAL, {
                "visible_chars": 2,
                "position": position
            })
            result = mapper.strategy_to_operator(strategy)
            
            # Should fallback to end position behavior
            assert result.operator_name == "mask"
            assert result.params["chars_to_mask"] == 2
            assert result.params["from_end"] == True

    def test_surrogate_strategy_phone(self, mapper: StrategyToOperatorMapper) -> None:
        """Test SURROGATE strategy with phone format."""
        strategy = Strategy(StrategyKind.SURROGATE, {"format_type": "phone"})
        result = mapper.strategy_to_operator(strategy)
        
        assert result.operator_name == "replace"
        assert result.params["new_value"] == "(555) 123-4567"

    def test_surrogate_strategy_email(self, mapper: StrategyToOperatorMapper) -> None:
        """Test SURROGATE strategy with email format."""
        strategy = Strategy(StrategyKind.SURROGATE, {"format_type": "email"})
        result = mapper.strategy_to_operator(strategy)
        
        assert result.operator_name == "replace"
        assert result.params["new_value"] == "user@example.com"

    def test_surrogate_strategy_dictionary(self, mapper: StrategyToOperatorMapper) -> None:
        """Test SURROGATE strategy with custom dictionary."""
        strategy = Strategy(StrategyKind.SURROGATE, {
            "dictionary": ["replacement1", "replacement2", "replacement3"]
        })
        result = mapper.strategy_to_operator(strategy)
        
        assert result.operator_name == "replace"
        assert result.params["new_value"] == "replacement1"

    def test_surrogate_strategy_pattern(self, mapper: StrategyToOperatorMapper) -> None:
        """Test SURROGATE strategy with custom pattern."""
        strategy = Strategy(StrategyKind.SURROGATE, {"pattern": "\\d{3}-\\d{2}-\\d{4}"})
        result = mapper.strategy_to_operator(strategy)
        
        assert result.operator_name == "replace"
        assert result.params["new_value"] == "[PATTERN:\\d{3}-\\d{2}-\\d{4}]"

    def test_custom_strategy_valid_callback(self, mapper: StrategyToOperatorMapper) -> None:
        """Test CUSTOM strategy with valid callback."""
        def test_callback(original_text: str, entity_type: str, confidence: float) -> str:
            return f"CUSTOM:{original_text}"
        
        strategy = Strategy(StrategyKind.CUSTOM, {"callback": test_callback})
        result = mapper.strategy_to_operator(strategy)
        
        assert result.operator_name == "custom"
        assert result.params["lambda"] == test_callback

    def test_custom_strategy_missing_callback(self, mapper: StrategyToOperatorMapper) -> None:
        """Test CUSTOM strategy with missing callback parameter should not be creatable."""
        # Strategy validation should prevent creation of CUSTOM strategy without callback
        with pytest.raises(ValueError, match="Custom strategy requires 'callback' parameter"):
            Strategy(StrategyKind.CUSTOM, {})

    def test_custom_strategy_invalid_callback(self, mapper: StrategyToOperatorMapper) -> None:
        """Test CUSTOM strategy with non-callable callback - this should fail at Strategy creation."""
        # Strategy validation should prevent invalid callbacks
        with pytest.raises(ValueError, match="Callback must be a callable function"):
            Strategy(StrategyKind.CUSTOM, {"callback": "not_callable"})

    def test_policy_to_operators_basic(self, mapper: StrategyToOperatorMapper) -> None:
        """Test basic policy to operators conversion."""
        policy = MaskingPolicy(
            per_entity={
                "PHONE_NUMBER": Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"}),
                "EMAIL_ADDRESS": Strategy(StrategyKind.PARTIAL, {"visible_chars": 3, "position": "start"})
            }
        )
        
        result = mapper.policy_to_operators(policy)
        
        assert len(result) == 2
        assert "PHONE_NUMBER" in result
        assert "EMAIL_ADDRESS" in result
        
        assert result["PHONE_NUMBER"].operator_name == "replace"
        assert result["PHONE_NUMBER"].params["new_value"] == "[PHONE]"
        
        assert result["EMAIL_ADDRESS"].operator_name == "mask"
        assert result["EMAIL_ADDRESS"].params["from_end"] == False

    def test_policy_to_operators_empty_policy(self, mapper: StrategyToOperatorMapper) -> None:
        """Test policy to operators with empty per_entity."""
        policy = MaskingPolicy()
        result = mapper.policy_to_operators(policy)
        
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_policy_to_operators_with_failures(self, mapper: StrategyToOperatorMapper) -> None:
        """Test policy to operators when mapping function raises exception."""
        def good_callback(original_text: str, entity_type: str, confidence: float) -> str:
            return "good"
        
        policy = MaskingPolicy(
            per_entity={
                "GOOD_ENTITY": Strategy(StrategyKind.REDACT),
                "BAD_ENTITY": Strategy(StrategyKind.CUSTOM, {"callback": good_callback})
            }
        )
        
        # Mock the strategy_to_operator method to fail for CUSTOM strategy
        original_strategy_to_operator = mapper.strategy_to_operator
        def failing_strategy_to_operator(strategy):
            if strategy.kind == StrategyKind.CUSTOM:
                raise ValueError("Intentional mapping failure")
            return original_strategy_to_operator(strategy)
        
        mapper.strategy_to_operator = failing_strategy_to_operator
        
        result = mapper.policy_to_operators(policy)
        
        assert len(result) == 2
        assert result["GOOD_ENTITY"].operator_name == "redact"
        assert result["BAD_ENTITY"].operator_name == "redact"  # Fallback to default
        
        # Restore original method
        mapper.strategy_to_operator = original_strategy_to_operator

    def test_edge_case_empty_parameters(self, mapper: StrategyToOperatorMapper) -> None:
        """Test strategies with None or empty parameters."""
        strategy = Strategy(StrategyKind.HASH, None)
        result = mapper.strategy_to_operator(strategy)
        
        assert result.operator_name == "hash"
        assert result.params["hash_type"] == "sha256"

    def test_edge_case_invalid_strategy_kind(self, mapper: StrategyToOperatorMapper) -> None:
        """Test handling of invalid strategy kind."""
        # Create a mock strategy with an invalid kind
        class MockStrategy:
            def __init__(self):
                self.kind = "invalid_kind"
                self.parameters = {}
        
        mock_strategy = MockStrategy()
        result = mapper.strategy_to_operator(mock_strategy)
        
        # Should fallback to redact
        assert result.operator_name == "redact"
        assert result.params["redact_char"] == "*"

    def test_mapping_exception_handling(self, mapper: StrategyToOperatorMapper, monkeypatch) -> None:
        """Test that exceptions in mapping functions are handled gracefully."""
        def failing_map_redact(strategy):
            raise ValueError("Intentional test failure")
        
        # Patch the redact mapping function to fail
        monkeypatch.setattr(mapper, '_map_redact_strategy', failing_map_redact)
        
        strategy = Strategy(StrategyKind.REDACT)
        result = mapper.strategy_to_operator(strategy)
        
        # Should fallback to default redact
        assert result.operator_name == "redact"
        assert result.params["redact_char"] == "*"

    def test_all_strategy_kinds_supported(self, mapper: StrategyToOperatorMapper) -> None:
        """Test that all StrategyKind values can be mapped."""
        def test_callback(original_text: str, entity_type: str, confidence: float) -> str:
            return "test"
        
        # Create valid strategies for each kind
        strategy_configs = {
            StrategyKind.REDACT: {},
            StrategyKind.TEMPLATE: {"template": "[TEST]"},
            StrategyKind.HASH: {},
            StrategyKind.PARTIAL: {"visible_chars": 4},
            StrategyKind.SURROGATE: {"format_type": "custom"},
            StrategyKind.CUSTOM: {"callback": test_callback}
        }
        
        for strategy_kind in StrategyKind:
            strategy = Strategy(strategy_kind, strategy_configs[strategy_kind])
            result = mapper.strategy_to_operator(strategy)
            
            assert isinstance(result, OperatorConfig)
            assert result.operator_name is not None
            assert isinstance(result.params, dict)

    def test_parameter_validation_preserved(self, mapper: StrategyToOperatorMapper) -> None:
        """Test that Strategy parameter validation is preserved."""
        with pytest.raises(ValueError):
            # This should fail due to Strategy validation
            Strategy(StrategyKind.PARTIAL, {"visible_chars": "not_an_int"})

    def test_complex_policy_scenario(self, mapper: StrategyToOperatorMapper) -> None:
        """Test complex policy with multiple entity types and strategies."""
        policy = MaskingPolicy(
            default_strategy=Strategy(StrategyKind.REDACT, {"redact_char": "X"}),
            per_entity={
                "PHONE_NUMBER": Strategy(StrategyKind.PARTIAL, {
                    "visible_chars": 4, 
                    "position": "end",
                    "mask_char": "*"
                }),
                "EMAIL_ADDRESS": Strategy(StrategyKind.HASH, {
                    "algorithm": "sha256",
                    "salt": "email_salt"
                }),
                "CREDIT_CARD": Strategy(StrategyKind.TEMPLATE, {"template": "[CC]"}),
                "PERSON": Strategy(StrategyKind.SURROGATE, {"format_type": "name"}),
            }
        )
        
        result = mapper.policy_to_operators(policy)
        
        assert len(result) == 4
        
        # Verify each mapping
        assert result["PHONE_NUMBER"].operator_name == "mask"
        assert result["PHONE_NUMBER"].params["chars_to_mask"] == 4
        
        assert result["EMAIL_ADDRESS"].operator_name == "hash"
        assert result["EMAIL_ADDRESS"].params["salt"] == "email_salt"
        
        assert result["CREDIT_CARD"].operator_name == "replace"
        assert result["CREDIT_CARD"].params["new_value"] == "[CC]"
        
        assert result["PERSON"].operator_name == "replace"
        assert result["PERSON"].params["new_value"] == "John Doe"