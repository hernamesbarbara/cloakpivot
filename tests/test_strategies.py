"""Tests for the strategy system."""

import pytest
from cloakpivot.core.strategies import (
    StrategyKind,
    Strategy,
    DEFAULT_REDACT,
    PHONE_TEMPLATE,
    EMAIL_TEMPLATE,
    SSN_PARTIAL,
    HASH_SHA256
)


class TestStrategyKind:
    """Test the StrategyKind enum."""
    
    def test_enum_values(self) -> None:
        """Test that enum has expected values."""
        assert StrategyKind.REDACT.value == "redact"
        assert StrategyKind.TEMPLATE.value == "template"
        assert StrategyKind.HASH.value == "hash"
        assert StrategyKind.SURROGATE.value == "surrogate"
        assert StrategyKind.PARTIAL.value == "partial"
        assert StrategyKind.CUSTOM.value == "custom"


class TestStrategy:
    """Test the Strategy dataclass."""
    
    def test_basic_creation(self) -> None:
        """Test basic strategy creation."""
        strategy = Strategy(StrategyKind.REDACT)
        assert strategy.kind == StrategyKind.REDACT
        assert strategy.parameters == {}
    
    def test_creation_with_parameters(self) -> None:
        """Test strategy creation with parameters."""
        params = {"template": "[PHONE]"}
        strategy = Strategy(StrategyKind.TEMPLATE, params)
        assert strategy.kind == StrategyKind.TEMPLATE
        assert strategy.parameters == params
    
    def test_frozen_immutability(self) -> None:
        """Test that strategy instances are immutable."""
        strategy = Strategy(StrategyKind.REDACT)
        with pytest.raises(Exception):  # FrozenInstanceError
            strategy.kind = StrategyKind.HASH  # type: ignore
    
    def test_redact_validation(self) -> None:
        """Test redact strategy parameter validation."""
        # Valid parameters
        Strategy(StrategyKind.REDACT, {"redact_char": "*"})
        Strategy(StrategyKind.REDACT, {"preserve_length": True})
        
        # Invalid parameters
        with pytest.raises(ValueError, match="redact_char must be a single character"):
            Strategy(StrategyKind.REDACT, {"redact_char": "**"})
        
        with pytest.raises(ValueError, match="preserve_length must be a boolean"):
            Strategy(StrategyKind.REDACT, {"preserve_length": "yes"})
    
    def test_template_validation(self) -> None:
        """Test template strategy parameter validation."""
        # Valid parameters
        Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"})
        
        # Missing template
        with pytest.raises(ValueError, match="Template strategy requires 'template' parameter"):
            Strategy(StrategyKind.TEMPLATE, {})
        
        # Invalid template type
        with pytest.raises(ValueError, match="Template must be a string"):
            Strategy(StrategyKind.TEMPLATE, {"template": 123})
    
    def test_hash_validation(self) -> None:
        """Test hash strategy parameter validation."""
        # Valid parameters
        Strategy(StrategyKind.HASH, {"algorithm": "sha256"})
        Strategy(StrategyKind.HASH, {"algorithm": "md5", "salt": "test"})
        
        # Invalid algorithm
        with pytest.raises(ValueError, match="Hash algorithm must be one of"):
            Strategy(StrategyKind.HASH, {"algorithm": "invalid"})
        
        # Invalid salt
        with pytest.raises(ValueError, match="Salt must be a string"):
            Strategy(StrategyKind.HASH, {"salt": 123})
        
        # Invalid truncate
        with pytest.raises(ValueError, match="Truncate must be a positive integer"):
            Strategy(StrategyKind.HASH, {"truncate": 0})
    
    def test_partial_validation(self) -> None:
        """Test partial strategy parameter validation."""
        # Valid parameters
        Strategy(StrategyKind.PARTIAL, {"visible_chars": 4})
        Strategy(StrategyKind.PARTIAL, {"visible_chars": 2, "position": "start"})
        
        # Missing visible_chars
        with pytest.raises(ValueError, match="Partial strategy requires 'visible_chars' parameter"):
            Strategy(StrategyKind.PARTIAL, {})
        
        # Invalid visible_chars
        with pytest.raises(ValueError, match="visible_chars must be a non-negative integer"):
            Strategy(StrategyKind.PARTIAL, {"visible_chars": -1})
        
        # Invalid position
        with pytest.raises(ValueError, match="Position must be 'start', 'end', 'middle', or 'random'"):
            Strategy(StrategyKind.PARTIAL, {"visible_chars": 4, "position": "invalid"})
    
    def test_custom_validation(self) -> None:
        """Test custom strategy parameter validation."""
        def test_callback(original_text: str, entity_type: str, confidence: float) -> str:
            return "masked"
        
        # Valid parameters
        Strategy(StrategyKind.CUSTOM, {"callback": test_callback})
        
        # Missing callback
        with pytest.raises(ValueError, match="Custom strategy requires 'callback' parameter"):
            Strategy(StrategyKind.CUSTOM, {})
        
        # Invalid callback
        with pytest.raises(ValueError, match="Callback must be a callable function"):
            Strategy(StrategyKind.CUSTOM, {"callback": "not_callable"})
    
    def test_get_parameter(self) -> None:
        """Test parameter retrieval."""
        strategy = Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"})
        
        assert strategy.get_parameter("template") == "[PHONE]"
        assert strategy.get_parameter("missing") is None
        assert strategy.get_parameter("missing", "default") == "default"
    
    def test_with_parameters(self) -> None:
        """Test creating new strategy with updated parameters."""
        original = Strategy(StrategyKind.HASH, {"algorithm": "sha256"})
        updated = original.with_parameters(salt="test", truncate=8)
        
        # Original unchanged
        assert original.parameters == {"algorithm": "sha256"}
        
        # Updated has merged parameters
        assert updated.parameters == {"algorithm": "sha256", "salt": "test", "truncate": 8}


class TestPredefinedStrategies:
    """Test predefined strategy constants."""
    
    def test_default_redact(self) -> None:
        """Test DEFAULT_REDACT strategy."""
        assert DEFAULT_REDACT.kind == StrategyKind.REDACT
        assert DEFAULT_REDACT.parameters == {}
    
    def test_phone_template(self) -> None:
        """Test PHONE_TEMPLATE strategy."""
        assert PHONE_TEMPLATE.kind == StrategyKind.TEMPLATE
        assert PHONE_TEMPLATE.parameters == {"template": "[PHONE]"}
    
    def test_email_template(self) -> None:
        """Test EMAIL_TEMPLATE strategy."""
        assert EMAIL_TEMPLATE.kind == StrategyKind.TEMPLATE
        assert EMAIL_TEMPLATE.parameters == {"template": "[EMAIL]"}
    
    def test_ssn_partial(self) -> None:
        """Test SSN_PARTIAL strategy."""
        assert SSN_PARTIAL.kind == StrategyKind.PARTIAL
        assert SSN_PARTIAL.parameters == {"visible_chars": 4, "position": "end"}
    
    def test_hash_sha256(self) -> None:
        """Test HASH_SHA256 strategy."""
        assert HASH_SHA256.kind == StrategyKind.HASH
        assert HASH_SHA256.parameters == {"algorithm": "sha256", "truncate": 8}