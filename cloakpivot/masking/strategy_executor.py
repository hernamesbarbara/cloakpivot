"""Strategy execution engine for masking operations."""

import logging
import random
import string
from typing import TYPE_CHECKING

from ..core.types.strategies import Strategy, StrategyKind
from .format_helpers import FormatPreserver
from .template_helpers import TemplateGenerator

if TYPE_CHECKING:
    from .applicator import StrategyApplicator

logger = logging.getLogger(__name__)


class StrategyExecutor:
    """
    Executes individual masking strategies for PII replacement.

    This class contains the implementation of all strategy types:
    - REDACT: Replace with redaction characters
    - TEMPLATE: Replace with fixed templates
    - HASH: Replace with hashed values
    - PARTIAL: Show partial content with masking
    - SURROGATE: Generate fake data
    - CUSTOM: Apply custom callbacks
    """

    def __init__(self, applicator: "StrategyApplicator") -> None:
        """
        Initialize strategy executor.

        Args:
            applicator: Parent strategy applicator instance
        """
        self.applicator = applicator
        self._random = applicator._random
        self._surrogate_generator = applicator._surrogate_generator
        self.seed = applicator.seed

    def execute_strategy(
        self,
        original_text: str,
        entity_type: str,
        strategy: Strategy,
        confidence: float,
    ) -> str:
        """
        Execute a single masking strategy.

        Args:
            original_text: The original PII text to mask
            entity_type: Type of entity
            strategy: The masking strategy to apply
            confidence: Detection confidence score

        Returns:
            str: The masked replacement text

        Raises:
            ValueError: If strategy parameters are invalid
            NotImplementedError: If strategy type is not supported
        """
        if strategy.kind == StrategyKind.REDACT:
            return self._apply_redact_strategy(original_text, strategy)
        if strategy.kind == StrategyKind.TEMPLATE:
            return self._apply_template_strategy(original_text, entity_type, strategy)
        if strategy.kind == StrategyKind.HASH:
            # Pass entity_type to hash strategy for per-entity salting
            hash_strategy = strategy.with_parameters(entity_type=entity_type)
            return self._apply_hash_strategy(original_text, hash_strategy)
        if strategy.kind == StrategyKind.PARTIAL:
            return self._apply_partial_strategy(original_text, strategy)
        if strategy.kind == StrategyKind.SURROGATE:
            return self._apply_surrogate_strategy(original_text, entity_type, strategy)
        if strategy.kind == StrategyKind.CUSTOM:
            return self._apply_custom_strategy(original_text, entity_type, confidence, strategy)

        raise NotImplementedError(f"Strategy {strategy.kind.value} not implemented")

    def _apply_redact_strategy(self, original_text: str, strategy: Strategy) -> str:
        """Apply redaction strategy - replace with redaction characters."""
        redact_char = str(strategy.get_parameter("redact_char", "*"))
        preserve_length = bool(strategy.get_parameter("preserve_length", True))

        if preserve_length:
            return str(redact_char) * len(original_text)

        # Fixed length redaction
        redaction_length = int(strategy.get_parameter("redaction_length", 8))
        return str(redact_char) * redaction_length

    def _apply_template_strategy(
        self, original_text: str, entity_type: str, strategy: Strategy
    ) -> str:
        """Apply template strategy - replace with format-preserving templates."""
        template = strategy.get_parameter("template")
        preserve_format = bool(strategy.get_parameter("preserve_format", False))
        auto_generate = bool(strategy.get_parameter("auto_generate", False))

        if not template and not auto_generate:
            raise ValueError(
                "Template strategy requires 'template' parameter or auto_generate=True"
            )

        # Auto-generate template based on format if requested
        if auto_generate or (preserve_format and template):
            format_template = TemplateGenerator.generate_format_template(original_text, entity_type)
            if auto_generate:
                template = format_template
            elif preserve_format:
                # Merge user template with format preservation
                template = TemplateGenerator.merge_template_with_format(
                    template, format_template, original_text
                )

        # Support placeholder substitution
        placeholders = {
            "entity_type": entity_type,
            "length": len(original_text),
            "original_format": TemplateGenerator.detect_format_pattern(original_text),
        }

        result = template or "[REDACTED]"
        for key, value in placeholders.items():
            result = result.replace(f"{{{key}}}", str(value))

        return result

    def _apply_hash_strategy(self, original_text: str, strategy: Strategy) -> str:
        """Apply hash strategy - replace with deterministic hashed values."""
        algorithm = str(strategy.get_parameter("algorithm", "sha256"))
        salt = str(strategy.get_parameter("salt", ""))
        per_entity_salt = strategy.get_parameter("per_entity_salt", None)
        entity_type = str(strategy.get_parameter("entity_type", "UNKNOWN"))
        truncate = strategy.get_parameter("truncate", None)
        prefix = str(strategy.get_parameter("prefix", ""))
        format_output = str(strategy.get_parameter("format_output", "hex"))
        consistent_length = bool(strategy.get_parameter("consistent_length", True))
        preserve_format_structure = bool(strategy.get_parameter("preserve_format_structure", False))

        # Build deterministic salt
        effective_salt = FormatPreserver.build_deterministic_salt(
            base_salt=salt,
            per_entity_salt=per_entity_salt,
            entity_type=entity_type,
            original_text=original_text,
        )

        # Combine text with salt for hashing
        content_to_hash = original_text + effective_salt

        # Compute hash
        hash_obj = FormatPreserver.get_hash_algorithm(algorithm)
        hash_obj.update(content_to_hash.encode("utf-8"))

        # Get hash result in requested format
        if format_output == "hex":
            hash_result = hash_obj.hexdigest()
        elif format_output == "base64":
            import base64

            hash_result = base64.b64encode(hash_obj.digest()).decode("ascii")
        elif format_output == "base32":
            import base64

            hash_result = base64.b32encode(hash_obj.digest()).decode("ascii")
        else:
            hash_result = hash_obj.hexdigest()

        # Apply length consistency if requested
        if consistent_length and truncate:
            hash_result = FormatPreserver.apply_consistent_truncation(
                hash_result, truncate, original_text, algorithm
            )
        elif truncate and isinstance(truncate, int) and truncate > 0:
            hash_result = hash_result[:truncate]

        # Apply format structure preservation if requested
        if preserve_format_structure:
            hash_result = FormatPreserver.preserve_format_in_hash(
                original_text, hash_result, prefix
            )

        return str(prefix) + str(hash_result)

    def _apply_partial_strategy(self, original_text: str, strategy: Strategy) -> str:
        """Apply partial strategy - show some chars, mask others with format awareness."""
        visible_chars = int(strategy.get_parameter("visible_chars", 4))
        position = str(strategy.get_parameter("position", "end"))
        mask_char = str(strategy.get_parameter("mask_char", "*"))
        min_length = int(strategy.get_parameter("min_length", 1))
        format_aware = bool(strategy.get_parameter("format_aware", True))
        preserve_delimiters = bool(strategy.get_parameter("preserve_delimiters", True))
        deterministic = bool(strategy.get_parameter("deterministic", True))

        if len(original_text) < min_length:
            # Text too short, mask completely
            return mask_char * len(original_text)

        # Apply format-aware partial masking if enabled
        if format_aware:
            return FormatPreserver.apply_format_aware_partial_masking(
                original_text,
                visible_chars,
                position,
                mask_char,
                preserve_delimiters,
                deterministic,
            )

        # Original basic partial masking logic
        if visible_chars >= len(original_text):
            # Would show everything, apply minimal masking
            if len(original_text) <= 2:
                return mask_char * len(original_text)
            return original_text[0] + mask_char * (len(original_text) - 2) + original_text[-1]

        # Apply partial masking based on position
        if position == "start":
            visible_part = original_text[:visible_chars]
            masked_part = mask_char * (len(original_text) - visible_chars)
            return visible_part + masked_part
        if position == "end":
            visible_part = original_text[-visible_chars:]
            masked_part = mask_char * (len(original_text) - visible_chars)
            return masked_part + visible_part
        if position == "middle":
            # Show chars at both ends
            chars_per_side = visible_chars // 2
            remaining = visible_chars % 2

            start_chars = chars_per_side + remaining
            end_chars = chars_per_side

            if start_chars + end_chars >= len(original_text):
                # Fallback to end position
                return self._apply_partial_strategy(
                    original_text, strategy.with_parameters(position="end")
                )

            start_part = original_text[:start_chars]
            end_part = original_text[-end_chars:] if end_chars > 0 else ""
            middle_length = len(original_text) - start_chars - end_chars
            masked_part = mask_char * middle_length

            return start_part + masked_part + end_part

        raise ValueError(f"Invalid position for partial strategy: {position}")

    def _apply_surrogate_strategy(
        self, original_text: str, entity_type: str, strategy: Strategy
    ) -> str:
        """Apply surrogate strategy - generate fake data in same format using enhanced generator."""
        # Check for custom pattern in strategy parameters
        pattern = strategy.get_parameter("pattern")
        if pattern:
            return self._surrogate_generator.generate_from_pattern(pattern)

        # Use the enhanced surrogate generator for format-preserving generation
        try:
            return self._surrogate_generator.generate_surrogate(original_text, entity_type)
        except Exception as e:
            logger.warning(f"Enhanced surrogate generation failed for {entity_type}: {e}")

            # Fallback to legacy generation methods for backward compatibility
            return self._apply_legacy_surrogate_strategy(original_text, entity_type, strategy)

    def _apply_custom_strategy(
        self,
        original_text: str,
        entity_type: str,
        confidence: float,
        strategy: Strategy,
    ) -> str:
        """Apply custom strategy - use callback function."""
        callback = strategy.get_parameter("callback")
        if not callback or not callable(callback):
            raise ValueError("Custom strategy requires a callable 'callback' parameter")

        try:
            result = callback(
                original_text=original_text,
                entity_type=entity_type,
                confidence=confidence,
            )

            if not isinstance(result, str):
                raise ValueError("Custom callback must return a string")

            return result
        except Exception as e:
            logger.error(f"Custom strategy callback failed: {e}")
            # Fallback to redaction
            return "*" * len(original_text)

    # Legacy surrogate methods for backward compatibility
    def _apply_legacy_surrogate_strategy(
        self, original_text: str, entity_type: str, strategy: Strategy
    ) -> str:
        """Legacy surrogate strategy implementation for backward compatibility."""
        format_type = strategy.get_parameter("format_type", entity_type.lower())
        seed = strategy.get_parameter("seed", self.seed)

        # Use seed for deterministic generation
        local_random = random.Random(seed + original_text if seed else None)

        if format_type in ["phone", "phone_number"]:
            return self._generate_surrogate_phone(local_random)
        if format_type in ["email", "email_address"]:
            return self._generate_surrogate_email(local_random)
        if format_type in ["ssn", "us_ssn"]:
            return self._generate_surrogate_ssn(local_random)
        if format_type == "credit_card":
            return self._generate_surrogate_credit_card(local_random)
        if format_type == "name":
            return self._generate_surrogate_name(local_random)
        if format_type == "custom":
            pattern = strategy.get_parameter("pattern")
            if pattern:
                return self._generate_from_pattern(pattern, local_random)

        # Fallback: generate alphanumeric string of same length
        chars = string.ascii_letters + string.digits
        return "".join(local_random.choice(chars) for _ in range(len(original_text)))

    def _generate_surrogate_phone(self, rng: random.Random) -> str:
        """Generate a surrogate phone number."""
        area_code = rng.randint(200, 999)  # Valid US area codes start from 200
        exchange = rng.randint(200, 999)  # Exchange codes start from 200
        number = rng.randint(0, 9999)  # Last 4 digits
        return f"{area_code:03d}-{exchange:03d}-{number:04d}"

    def _generate_surrogate_email(self, rng: random.Random) -> str:
        """Generate a surrogate email address."""
        users = ["user", "john", "jane", "test", "sample", "demo"]
        domains = ["example.com", "test.org", "sample.net", "demo.edu"]

        username = rng.choice(users) + str(rng.randint(1, 999))
        domain = rng.choice(domains)
        return f"{username}@{domain}"

    def _generate_surrogate_ssn(self, rng: random.Random) -> str:
        """Generate a surrogate SSN."""
        # Generate in XXX-XX-XXXX format, avoiding invalid ranges
        area = rng.randint(100, 899)  # Avoid 000, 666, 900+
        group = rng.randint(10, 99)  # Avoid 00
        serial = rng.randint(1000, 9999)  # Avoid 0000
        return f"{area:03d}-{group:02d}-{serial:04d}"

    def _generate_surrogate_credit_card(self, rng: random.Random) -> str:
        """Generate a surrogate credit card number."""
        # Simple 16-digit format
        digits = [str(rng.randint(0, 9)) for _ in range(16)]
        return "".join(digits)

    def _generate_surrogate_name(self, rng: random.Random) -> str:
        """Generate a surrogate person name."""
        first_names = ["John", "Jane", "Alex", "Sam", "Chris", "Jordan"]
        last_names = [
            "Smith",
            "Johnson",
            "Brown",
            "Davis",
            "Wilson",
            "Anderson",
        ]

        first = rng.choice(first_names)
        last = rng.choice(last_names)
        return f"{first} {last}"

    def _generate_from_pattern(self, pattern: str, rng: random.Random) -> str:
        """Generate text from a pattern string (simple implementation)."""
        # Simple pattern replacement:
        # X = random letter, 9 = random digit, ? = random alphanumeric
        result = ""
        for char in pattern:
            if char == "X":
                result += rng.choice(string.ascii_uppercase)
            elif char == "x":
                result += rng.choice(string.ascii_lowercase)
            elif char == "9":
                result += str(rng.randint(0, 9))
            elif char == "?":
                result += rng.choice(string.ascii_letters + string.digits)
            else:
                result += char
        return result
