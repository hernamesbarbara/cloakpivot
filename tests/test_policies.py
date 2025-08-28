"""Tests for the policy system."""

import pytest
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.core.policies import (
    MaskingPolicy,
    CONSERVATIVE_POLICY,
    TEMPLATE_POLICY,
    PARTIAL_POLICY
)


class TestMaskingPolicy:
    """Test the MaskingPolicy dataclass."""
    
    def test_basic_creation(self) -> None:
        """Test basic policy creation with defaults."""
        policy = MaskingPolicy()
        
        assert policy.default_strategy.kind == StrategyKind.REDACT
        assert policy.per_entity == {}
        assert policy.thresholds == {}
        assert policy.locale == "en"
        assert policy.seed is None
        assert policy.custom_callbacks is None
        assert policy.allow_list == set()
        assert policy.deny_list == set()
        assert policy.context_rules == {}
        assert policy.min_entity_length == 1
    
    def test_creation_with_parameters(self) -> None:
        """Test policy creation with custom parameters."""
        phone_strategy = Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"})
        
        policy = MaskingPolicy(
            default_strategy=phone_strategy,
            per_entity={"EMAIL_ADDRESS": Strategy(StrategyKind.REDACT)},
            thresholds={"PHONE_NUMBER": 0.9},
            locale="fr",
            seed="test-seed",
            allow_list={"allowed@example.com"},
            deny_list={"secret@example.com"},
            min_entity_length=3
        )
        
        assert policy.default_strategy == phone_strategy
        assert policy.per_entity["EMAIL_ADDRESS"].kind == StrategyKind.REDACT
        assert policy.thresholds["PHONE_NUMBER"] == 0.9
        assert policy.locale == "fr"
        assert policy.seed == "test-seed"
        assert "allowed@example.com" in policy.allow_list
        assert "secret@example.com" in policy.deny_list
        assert policy.min_entity_length == 3
    
    def test_frozen_immutability(self) -> None:
        """Test that policy instances are immutable."""
        policy = MaskingPolicy()
        with pytest.raises(Exception):  # FrozenInstanceError
            policy.locale = "fr"  # type: ignore
    
    def test_threshold_validation(self) -> None:
        """Test confidence threshold validation."""
        # Valid thresholds
        MaskingPolicy(thresholds={"PHONE": 0.0, "EMAIL": 1.0, "SSN": 0.5})
        
        # Invalid threshold type
        with pytest.raises(ValueError, match="Threshold for PHONE must be a number"):
            MaskingPolicy(thresholds={"PHONE": "invalid"})
        
        # Out of range threshold
        with pytest.raises(ValueError, match="Threshold for PHONE must be between 0.0 and 1.0"):
            MaskingPolicy(thresholds={"PHONE": -0.1})
        
        with pytest.raises(ValueError, match="Threshold for EMAIL must be between 0.0 and 1.0"):
            MaskingPolicy(thresholds={"EMAIL": 1.1})
    
    def test_locale_validation(self) -> None:
        """Test locale format validation."""
        # Valid locales
        MaskingPolicy(locale="en")
        MaskingPolicy(locale="fr")
        MaskingPolicy(locale="en-US")
        MaskingPolicy(locale="pt-BR")
        
        # Invalid locale formats
        with pytest.raises(ValueError, match="Locale must follow format"):
            MaskingPolicy(locale="english")
        
        with pytest.raises(ValueError, match="Locale must follow format"):
            MaskingPolicy(locale="en-us")  # lowercase country code
        
        with pytest.raises(ValueError, match="Locale must be a string"):
            MaskingPolicy(locale=123)  # type: ignore
    
    def test_seed_validation(self) -> None:
        """Test seed validation."""
        # Valid seeds
        MaskingPolicy(seed="test-seed")
        MaskingPolicy(seed=None)
        
        # Invalid seed
        with pytest.raises(ValueError, match="Seed must be a string or None"):
            MaskingPolicy(seed=123)  # type: ignore
    
    def test_entity_length_validation(self) -> None:
        """Test minimum entity length validation."""
        # Valid lengths
        MaskingPolicy(min_entity_length=0)
        MaskingPolicy(min_entity_length=5)
        
        # Invalid lengths
        with pytest.raises(ValueError, match="min_entity_length must be a non-negative integer"):
            MaskingPolicy(min_entity_length=-1)
        
        with pytest.raises(ValueError, match="min_entity_length must be a non-negative integer"):
            MaskingPolicy(min_entity_length="invalid")  # type: ignore
    
    def test_context_rules_validation(self) -> None:
        """Test context rules validation."""
        # Valid context rules
        MaskingPolicy(context_rules={
            "heading": {"enabled": False},
            "table": {"threshold": 0.9},
            "footer": {"strategy": Strategy(StrategyKind.REDACT)}
        })
        
        # Invalid context type
        with pytest.raises(ValueError, match="Unknown context type 'invalid'"):
            MaskingPolicy(context_rules={"invalid": {}})
        
        # Invalid rule structure
        with pytest.raises(ValueError, match="Context rules for 'heading' must be a dictionary"):
            MaskingPolicy(context_rules={"heading": "invalid"})
        
        # Invalid rule key
        with pytest.raises(ValueError, match="Unknown rule key 'invalid_key'"):
            MaskingPolicy(context_rules={"heading": {"invalid_key": True}})
    
    def test_callback_validation(self) -> None:
        """Test custom callback validation."""
        def valid_callback(original_text: str, entity_type: str, confidence: float) -> str:
            return "masked"
        
        def invalid_callback(text: str) -> str:  # Wrong signature
            return "masked"
        
        # Valid callbacks
        MaskingPolicy(custom_callbacks={"PHONE": valid_callback})
        
        # Invalid callback type
        with pytest.raises(ValueError, match="custom_callbacks must be a dictionary or None"):
            MaskingPolicy(custom_callbacks="invalid")  # type: ignore
        
        # Non-callable callback
        with pytest.raises(ValueError, match="Callback for 'PHONE' must be callable"):
            MaskingPolicy(custom_callbacks={"PHONE": "not_callable"})
        
        # Invalid callback signature
        with pytest.raises(ValueError, match="Callback for 'PHONE' missing parameters"):
            MaskingPolicy(custom_callbacks={"PHONE": invalid_callback})
    
    def test_get_strategy_for_entity(self) -> None:
        """Test strategy retrieval for entities."""
        phone_strategy = Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"})
        default_strategy = Strategy(StrategyKind.REDACT)
        
        policy = MaskingPolicy(
            default_strategy=default_strategy,
            per_entity={"PHONE_NUMBER": phone_strategy}
        )
        
        # Entity-specific strategy
        assert policy.get_strategy_for_entity("PHONE_NUMBER") == phone_strategy
        
        # Default strategy for unknown entity
        assert policy.get_strategy_for_entity("EMAIL_ADDRESS") == default_strategy
    
    def test_get_strategy_with_context(self) -> None:
        """Test strategy retrieval with context rules."""
        default_strategy = Strategy(StrategyKind.REDACT)
        context_strategy = Strategy(StrategyKind.TEMPLATE, {"template": "[HEADING]"})
        
        policy = MaskingPolicy(
            default_strategy=default_strategy,
            context_rules={
                "heading": {"strategy": context_strategy}
            }
        )
        
        # Context-specific strategy
        assert policy.get_strategy_for_entity("PHONE", "heading") == context_strategy
        
        # Default strategy for different context
        assert policy.get_strategy_for_entity("PHONE", "paragraph") == default_strategy
        
        # Disabled context returns redact strategy (no-op behavior handled at higher level)
        disabled_policy = MaskingPolicy(
            default_strategy=default_strategy,
            context_rules={"heading": {"enabled": False}}
        )
        disabled_strategy = disabled_policy.get_strategy_for_entity("PHONE", "heading")
        assert disabled_strategy.kind == StrategyKind.REDACT
    
    def test_get_threshold_for_entity(self) -> None:
        """Test threshold retrieval for entities."""
        policy = MaskingPolicy(
            thresholds={"PHONE_NUMBER": 0.8},
            context_rules={
                "heading": {"threshold": 0.9}
            }
        )
        
        # Entity-specific threshold
        assert policy.get_threshold_for_entity("PHONE_NUMBER") == 0.8
        
        # Context-specific threshold
        assert policy.get_threshold_for_entity("EMAIL", "heading") == 0.9
        
        # Default threshold
        assert policy.get_threshold_for_entity("EMAIL") == 0.5
    
    def test_should_mask_entity(self) -> None:
        """Test entity masking decision logic."""
        policy = MaskingPolicy(
            thresholds={"PHONE": 0.8},
            allow_list={"allowed@example.com"},
            deny_list={"secret@example.com"},
            min_entity_length=3
        )
        
        # Should mask based on confidence
        assert policy.should_mask_entity("555-1234", "PHONE", 0.9) == True
        assert policy.should_mask_entity("555-1234", "PHONE", 0.7) == False
        
        # Allow list check (never mask)
        assert policy.should_mask_entity("allowed@example.com", "EMAIL", 1.0) == False
        
        # Deny list check (always mask)
        assert policy.should_mask_entity("secret@example.com", "EMAIL", 0.1) == True
        
        # Minimum length check
        assert policy.should_mask_entity("ab", "NAME", 1.0) == False
        assert policy.should_mask_entity("abc", "NAME", 1.0) == True
    
    def test_get_custom_callback(self) -> None:
        """Test custom callback retrieval."""
        def test_callback(original_text: str, entity_type: str, confidence: float) -> str:
            return "custom"
        
        policy = MaskingPolicy(custom_callbacks={"PHONE": test_callback})
        
        # Has callback
        assert policy.get_custom_callback("PHONE") == test_callback
        
        # No callback
        assert policy.get_custom_callback("EMAIL") is None
        
        # No callbacks at all
        policy_no_callbacks = MaskingPolicy()
        assert policy_no_callbacks.get_custom_callback("PHONE") is None
    
    def test_with_entity_strategy(self) -> None:
        """Test creating new policy with additional entity strategy."""
        original = MaskingPolicy()
        phone_strategy = Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"})
        
        updated = original.with_entity_strategy("PHONE_NUMBER", phone_strategy)
        
        # Original unchanged
        assert "PHONE_NUMBER" not in original.per_entity
        
        # Updated has new strategy
        assert updated.per_entity["PHONE_NUMBER"] == phone_strategy
    
    def test_with_threshold(self) -> None:
        """Test creating new policy with updated threshold."""
        original = MaskingPolicy()
        updated = original.with_threshold("PHONE", 0.9)
        
        # Original unchanged
        assert "PHONE" not in original.thresholds
        
        # Updated has new threshold
        assert updated.thresholds["PHONE"] == 0.9
    
    def test_to_dict(self) -> None:
        """Test policy serialization to dictionary."""
        phone_strategy = Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"})
        policy = MaskingPolicy(
            per_entity={"PHONE": phone_strategy},
            thresholds={"EMAIL": 0.8},
            locale="fr",
            seed="test",
            allow_list={"allowed"},
            deny_list={"denied"},
            min_entity_length=2
        )
        
        data = policy.to_dict()
        
        assert data["locale"] == "fr"
        assert data["seed"] == "test"
        assert data["thresholds"]["EMAIL"] == 0.8
        assert data["per_entity"]["PHONE"]["kind"] == "template"
        assert data["per_entity"]["PHONE"]["parameters"]["template"] == "[PHONE]"
        assert "allowed" in data["allow_list"]
        assert "denied" in data["deny_list"]
        assert data["min_entity_length"] == 2
    
    def test_from_dict(self) -> None:
        """Test policy deserialization from dictionary."""
        data = {
            "default_strategy": {"kind": "redact", "parameters": {}},
            "per_entity": {
                "PHONE": {"kind": "template", "parameters": {"template": "[PHONE]"}}
            },
            "thresholds": {"EMAIL": 0.8},
            "locale": "fr",
            "seed": "test",
            "allow_list": ["allowed"],
            "deny_list": ["denied"],
            "min_entity_length": 2
        }
        
        policy = MaskingPolicy.from_dict(data)
        
        assert policy.locale == "fr"
        assert policy.seed == "test"
        assert policy.thresholds["EMAIL"] == 0.8
        assert policy.per_entity["PHONE"].kind == StrategyKind.TEMPLATE
        assert policy.per_entity["PHONE"].get_parameter("template") == "[PHONE]"
        assert "allowed" in policy.allow_list
        assert "denied" in policy.deny_list
        assert policy.min_entity_length == 2


class TestPredefinedPolicies:
    """Test predefined policy constants."""
    
    def test_conservative_policy(self) -> None:
        """Test CONSERVATIVE_POLICY configuration."""
        policy = CONSERVATIVE_POLICY
        
        assert policy.default_strategy.kind == StrategyKind.REDACT
        assert policy.thresholds["PHONE_NUMBER"] == 0.9
        assert policy.thresholds["EMAIL_ADDRESS"] == 0.9
        assert policy.thresholds["CREDIT_CARD"] == 0.8
    
    def test_template_policy(self) -> None:
        """Test TEMPLATE_POLICY configuration."""
        policy = TEMPLATE_POLICY
        
        phone_strategy = policy.per_entity["PHONE_NUMBER"]
        assert phone_strategy.kind == StrategyKind.TEMPLATE
        assert phone_strategy.get_parameter("template") == "[PHONE]"
        
        email_strategy = policy.per_entity["EMAIL_ADDRESS"]
        assert email_strategy.kind == StrategyKind.TEMPLATE
        assert email_strategy.get_parameter("template") == "[EMAIL]"
    
    def test_partial_policy(self) -> None:
        """Test PARTIAL_POLICY configuration."""
        policy = PARTIAL_POLICY
        
        phone_strategy = policy.per_entity["PHONE_NUMBER"]
        assert phone_strategy.kind == StrategyKind.PARTIAL
        assert phone_strategy.get_parameter("visible_chars") == 4
        assert phone_strategy.get_parameter("position") == "end"
        
        email_strategy = policy.per_entity["EMAIL_ADDRESS"]
        assert email_strategy.kind == StrategyKind.PARTIAL
        assert email_strategy.get_parameter("visible_chars") == 3
        assert email_strategy.get_parameter("position") == "start"