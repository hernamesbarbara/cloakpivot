"""Unit tests for cloakpivot.masking.applicator module."""

import hashlib
from unittest.mock import patch

import pytest

from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.masking.applicator import StrategyApplicator


class TestStrategyApplicator:
    """Test StrategyApplicator class."""

    def test_initialization_without_seed(self):
        """Test StrategyApplicator initialization without seed."""
        applicator = StrategyApplicator()
        assert applicator.seed is None
        assert applicator._random is not None
        assert applicator._surrogate_generator is not None

    def test_initialization_with_seed(self):
        """Test StrategyApplicator initialization with seed."""
        applicator = StrategyApplicator(seed="test-seed")
        assert applicator.seed == "test-seed"
        assert applicator._random is not None
        assert applicator._surrogate_generator is not None

    def test_apply_redact_strategy_preserve_length(self):
        """Test redact strategy with preserve_length=True."""
        applicator = StrategyApplicator()
        strategy = Strategy(StrategyKind.REDACT, {"redact_char": "*", "preserve_length": True})
        result = applicator.apply_strategy("test@example.com", "EMAIL", strategy, 0.95)
        assert result == "*" * len("test@example.com")

    def test_apply_redact_strategy_fixed_length(self):
        """Test redact strategy with fixed length."""
        applicator = StrategyApplicator()
        strategy = Strategy(
            StrategyKind.REDACT,
            {"redact_char": "#", "preserve_length": False, "redaction_length": 5},
        )
        result = applicator.apply_strategy("test@example.com", "EMAIL", strategy, 0.95)
        assert result == "#####"

    def test_apply_template_strategy_basic(self):
        """Test template strategy with basic template."""
        applicator = StrategyApplicator()
        strategy = Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL]"})
        result = applicator.apply_strategy("test@example.com", "EMAIL_ADDRESS", strategy, 0.95)
        assert result == "[EMAIL]"

    def test_apply_template_strategy_with_placeholders(self):
        """Test template strategy with placeholders."""
        applicator = StrategyApplicator()
        strategy = Strategy(StrategyKind.TEMPLATE, {"template": "[{entity_type}:{length}]"})
        result = applicator.apply_strategy("test@example.com", "EMAIL_ADDRESS", strategy, 0.95)
        assert result == "[EMAIL_ADDRESS:16]"

    def test_apply_template_strategy_auto_generate(self):
        """Test template strategy with auto_generate."""
        applicator = StrategyApplicator()
        strategy = Strategy(StrategyKind.TEMPLATE, {"auto_generate": True})
        result = applicator.apply_strategy("test@example.com", "EMAIL_ADDRESS", strategy, 0.95)
        assert "@" in result or result == "[EMAIL]"  # Should preserve format or use default

    def test_apply_template_strategy_no_template_error(self):
        """Test template strategy without template or auto_generate raises error."""
        with pytest.raises(ValueError, match="Template strategy requires"):
            Strategy(StrategyKind.TEMPLATE, {})

    def test_apply_hash_strategy_default(self):
        """Test hash strategy with default parameters."""
        applicator = StrategyApplicator()
        strategy = Strategy(StrategyKind.HASH)
        result = applicator.apply_strategy("test@example.com", "EMAIL", strategy, 0.95)

        # Result should be a SHA256 hex hash
        assert len(result) == 64  # SHA256 produces 64 hex characters
        assert all(c in "0123456789abcdef" for c in result)

    def test_apply_hash_strategy_with_salt(self):
        """Test hash strategy with salt."""
        applicator = StrategyApplicator()
        strategy = Strategy(StrategyKind.HASH, {"salt": "my-salt"})
        result1 = applicator.apply_strategy("test@example.com", "EMAIL", strategy, 0.95)
        result2 = applicator.apply_strategy("test@example.com", "EMAIL", strategy, 0.95)

        # Same input with same salt should produce same hash
        assert result1 == result2

    def test_apply_hash_strategy_truncated(self):
        """Test hash strategy with truncation."""
        applicator = StrategyApplicator()
        strategy = Strategy(StrategyKind.HASH, {"truncate": 10})
        result = applicator.apply_strategy("test@example.com", "EMAIL", strategy, 0.95)
        assert len(result) == 10

    def test_apply_hash_strategy_with_prefix(self):
        """Test hash strategy with prefix."""
        applicator = StrategyApplicator()
        strategy = Strategy(StrategyKind.HASH, {"prefix": "HASH:"})
        result = applicator.apply_strategy("test@example.com", "EMAIL", strategy, 0.95)
        assert result.startswith("HASH:")

    def test_apply_hash_strategy_different_algorithms(self):
        """Test hash strategy with different algorithms."""
        applicator = StrategyApplicator()

        # Test MD5
        strategy = Strategy(StrategyKind.HASH, {"algorithm": "md5"})
        result = applicator.apply_strategy("test", "EMAIL", strategy, 0.95)
        assert len(result) == 32  # MD5 produces 32 hex characters

        # Test SHA1
        strategy = Strategy(StrategyKind.HASH, {"algorithm": "sha1"})
        result = applicator.apply_strategy("test", "EMAIL", strategy, 0.95)
        assert len(result) == 40  # SHA1 produces 40 hex characters

        # Test SHA512
        strategy = Strategy(StrategyKind.HASH, {"algorithm": "sha512"})
        result = applicator.apply_strategy("test", "EMAIL", strategy, 0.95)
        assert len(result) == 128  # SHA512 produces 128 hex characters

    def test_apply_hash_strategy_invalid_algorithm(self):
        """Test hash strategy with invalid algorithm."""
        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            Strategy(StrategyKind.HASH, {"algorithm": "invalid"})

    def test_apply_partial_strategy_end(self):
        """Test partial strategy showing end characters."""
        applicator = StrategyApplicator()
        strategy = Strategy(
            StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end", "mask_char": "*"}
        )
        result = applicator.apply_strategy("1234567890", "PHONE", strategy, 0.95)
        assert result == "******7890"

    def test_apply_partial_strategy_start(self):
        """Test partial strategy showing start characters."""
        applicator = StrategyApplicator()
        strategy = Strategy(
            StrategyKind.PARTIAL, {"visible_chars": 3, "position": "start", "mask_char": "X"}
        )
        result = applicator.apply_strategy("1234567890", "PHONE", strategy, 0.95)
        assert result == "123XXXXXXX"

    def test_apply_partial_strategy_middle(self):
        """Test partial strategy showing middle characters."""
        applicator = StrategyApplicator()
        strategy = Strategy(
            StrategyKind.PARTIAL, {"visible_chars": 4, "position": "middle", "mask_char": "*"}
        )
        result = applicator.apply_strategy("1234567890", "PHONE", strategy, 0.95)
        # Should show 2 chars at start and 2 at end
        assert result[0:2] == "12"
        assert result[-2:] == "90"
        assert "*" in result

    def test_apply_partial_strategy_too_short(self):
        """Test partial strategy with text too short."""
        applicator = StrategyApplicator()
        strategy = Strategy(
            StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end", "min_length": 5}
        )
        result = applicator.apply_strategy("123", "NUMBER", strategy, 0.95)
        assert result == "***"  # Should mask completely

    def test_apply_partial_strategy_format_aware(self):
        """Test partial strategy with format awareness."""
        applicator = StrategyApplicator()
        strategy = Strategy(
            StrategyKind.PARTIAL,
            {
                "visible_chars": 4,
                "position": "end",
                "format_aware": True,
                "preserve_delimiters": True,
            },
        )
        result = applicator.apply_strategy("123-456-7890", "PHONE", strategy, 0.95)
        # Should preserve hyphens
        assert "-" in result
        assert result.endswith("7890") or "7890" in result

    def test_apply_surrogate_strategy_phone(self):
        """Test surrogate strategy for phone numbers."""
        applicator = StrategyApplicator(seed="test")
        strategy = Strategy(StrategyKind.SURROGATE)
        result = applicator.apply_strategy("555-123-4567", "PHONE_NUMBER", strategy, 0.95)

        # Should generate a phone-like pattern
        assert len(result) > 0
        # May contain digits and hyphens
        assert any(c.isdigit() for c in result)

    def test_apply_surrogate_strategy_email(self):
        """Test surrogate strategy for email addresses."""
        applicator = StrategyApplicator(seed="test")
        strategy = Strategy(StrategyKind.SURROGATE)
        result = applicator.apply_strategy("test@example.com", "EMAIL_ADDRESS", strategy, 0.95)

        # Should generate an email-like pattern
        assert len(result) > 0

    def test_apply_surrogate_strategy_with_pattern(self):
        """Test surrogate strategy with custom pattern."""
        applicator = StrategyApplicator()
        strategy = Strategy(StrategyKind.SURROGATE, {"pattern": "XXX-999"})

        with patch.object(
            applicator._surrogate_generator, "generate_from_pattern", return_value="ABC-123"
        ):
            result = applicator.apply_strategy("test", "CUSTOM", strategy, 0.95)
            assert result == "ABC-123"

    def test_apply_custom_strategy_success(self):
        """Test custom strategy with callback function."""

        def custom_callback(original_text, entity_type, confidence):
            return f"[{entity_type}:{len(original_text)}]"

        applicator = StrategyApplicator()
        strategy = Strategy(StrategyKind.CUSTOM, {"callback": custom_callback})
        result = applicator.apply_strategy("test@example.com", "EMAIL", strategy, 0.95)
        assert result == "[EMAIL:16]"

    def test_apply_custom_strategy_no_callback(self):
        """Test custom strategy without callback raises error."""
        with pytest.raises(ValueError, match="Custom strategy requires 'callback'"):
            Strategy(StrategyKind.CUSTOM, {})

    def test_apply_custom_strategy_callback_failure(self):
        """Test custom strategy with failing callback."""

        def failing_callback(original_text, entity_type, confidence):
            raise Exception("Callback failed")

        applicator = StrategyApplicator()
        strategy = Strategy(StrategyKind.CUSTOM, {"callback": failing_callback})
        result = applicator.apply_strategy("test", "EMAIL", strategy, 0.95)
        # Should fallback to redaction
        assert result == "****"

    def test_apply_strategy_with_fallback(self):
        """Test apply_strategy with fallback when primary fails."""
        applicator = StrategyApplicator()

        # Use a strategy that won't fail during creation
        def failing_callback(text, entity, confidence):
            raise ValueError("Callback error")

        strategy = Strategy(StrategyKind.CUSTOM, {"callback": failing_callback})

        # Should use fallback strategy
        result = applicator.apply_strategy("test@example.com", "EMAIL", strategy, 0.95)
        assert result is not None
        assert len(result) > 0

    def test_compose_strategies_single(self):
        """Test compose_strategies with single strategy."""
        applicator = StrategyApplicator()
        strategies = [Strategy(StrategyKind.REDACT)]
        result = applicator.compose_strategies("test", "EMAIL", strategies, 0.95)
        assert result == "****"

    def test_compose_strategies_multiple(self):
        """Test compose_strategies with multiple strategies."""
        applicator = StrategyApplicator()
        strategies = [
            Strategy(StrategyKind.REDACT, {"redact_char": "X"}),
            Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"}),
        ]
        result = applicator.compose_strategies("test@example.com", "EMAIL", strategies, 0.95)
        assert result == "[REDACTED]"

    def test_compose_strategies_empty_list(self):
        """Test compose_strategies with empty list raises error."""
        applicator = StrategyApplicator()
        with pytest.raises(ValueError, match="At least one strategy"):
            applicator.compose_strategies("test", "EMAIL", [], 0.95)

    def test_generate_phone_template(self):
        """Test phone template generation."""
        applicator = StrategyApplicator()

        # Test standard US format
        assert applicator._generate_phone_template("555-123-4567") == "XXX-XXX-XXXX"
        assert applicator._generate_phone_template("(555) 123-4567") == "(XXX) XXX-XXXX"
        assert applicator._generate_phone_template("+1 555 123 4567").startswith("+X ")

    def test_generate_ssn_template(self):
        """Test SSN template generation."""
        applicator = StrategyApplicator()

        assert applicator._generate_ssn_template("123-45-6789") == "XXX-XX-XXXX"
        assert applicator._generate_ssn_template("123456789") == "XXXXXXXXX"

    def test_generate_credit_card_template(self):
        """Test credit card template generation."""
        applicator = StrategyApplicator()

        assert (
            applicator._generate_credit_card_template("1234-5678-9012-3456")
            == "XXXX-XXXX-XXXX-XXXX"
        )
        assert (
            applicator._generate_credit_card_template("1234 5678 9012 3456")
            == "XXXX XXXX XXXX XXXX"
        )

    def test_generate_email_template(self):
        """Test email template generation."""
        applicator = StrategyApplicator()

        result = applicator._generate_email_template("test@example.com")
        assert "@" in result
        assert result == "xxxx@xxxxxxx.xxx"

    def test_generate_generic_template(self):
        """Test generic template generation."""
        applicator = StrategyApplicator()

        result = applicator._generate_generic_template("ABC123-XYZ")
        assert result == "XXX###-XXX"

    def test_detect_format_pattern(self):
        """Test format pattern detection."""
        applicator = StrategyApplicator()

        assert applicator._detect_format_pattern("ABC123") == "LLLDDD"
        assert applicator._detect_format_pattern("A-1 B") == "LPDSL"

    def test_get_hash_algorithm(self):
        """Test getting hash algorithm."""
        applicator = StrategyApplicator()

        assert isinstance(applicator._get_hash_algorithm("md5"), type(hashlib.md5()))
        assert isinstance(applicator._get_hash_algorithm("sha256"), type(hashlib.sha256()))

        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            applicator._get_hash_algorithm("invalid")

    def test_surrogate_quality_metrics(self):
        """Test getting surrogate quality metrics."""
        applicator = StrategyApplicator()

        with patch.object(
            applicator._surrogate_generator, "get_quality_metrics", return_value={"quality": "high"}
        ):
            metrics = applicator.get_surrogate_quality_metrics()
            assert metrics == {"quality": "high"}

    def test_reset_document_scope(self):
        """Test resetting document scope."""
        applicator = StrategyApplicator()

        with patch.object(applicator._surrogate_generator, "reset_document_scope") as mock_reset:
            applicator.reset_document_scope()
            mock_reset.assert_called_once()

    def test_deterministic_results_with_seed(self):
        """Test that same seed produces deterministic results."""
        applicator1 = StrategyApplicator(seed="test-seed")
        applicator2 = StrategyApplicator(seed="test-seed")

        strategy = Strategy(StrategyKind.HASH, {"salt": "test"})
        result1 = applicator1.apply_strategy("test@example.com", "EMAIL", strategy, 0.95)
        result2 = applicator2.apply_strategy("test@example.com", "EMAIL", strategy, 0.95)

        assert result1 == result2

    def test_different_results_without_seed(self):
        """Test that different applicators without seed may produce different results."""
        applicator1 = StrategyApplicator()
        applicator2 = StrategyApplicator()

        # For hash strategy, results should still be same (deterministic)
        strategy = Strategy(StrategyKind.HASH)
        result1 = applicator1.apply_strategy("test@example.com", "EMAIL", strategy, 0.95)
        result2 = applicator2.apply_strategy("test@example.com", "EMAIL", strategy, 0.95)

        assert result1 == result2  # Hash should be deterministic even without seed
