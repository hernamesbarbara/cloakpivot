"""StrategyApplicator for generating masked replacement tokens."""

import hashlib
import logging
import random
import string
from typing import Any, Optional

from ..core.strategies import Strategy, StrategyKind
from ..core.surrogate import SurrogateGenerator

logger = logging.getLogger(__name__)


class StrategyApplicator:
    """
    Applies masking strategies to generate replacement tokens for detected PII.

    This class implements the core masking logic for different strategy types:
    - REDACT: Replace with redaction characters (*)
    - TEMPLATE: Replace with fixed templates like [PHONE], [EMAIL]
    - HASH: Replace with hashed values
    - PARTIAL: Show partial content with masking
    - SURROGATE: Generate fake data in same format
    - CUSTOM: Apply custom callback functions

    Examples:
        >>> applicator = StrategyApplicator()
        >>>
        >>> # Template strategy
        >>> result = applicator.apply_strategy(
        ...     "555-123-4567",
        ...     "PHONE_NUMBER",
        ...     Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"}),
        ...     0.95
        ... )
        >>> assert result == "[PHONE]"

        >>> # Partial strategy
        >>> result = applicator.apply_strategy(
        ...     "555-123-4567",
        ...     "PHONE_NUMBER",
        ...     Strategy(StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end"}),
        ...     0.95
        ... )
        >>> assert result == "*********4567"
    """

    def __init__(self, seed: Optional[str] = None) -> None:
        """
        Initialize the strategy applicator.

        Args:
            seed: Optional seed for deterministic random generation
        """
        self.seed = seed
        if seed:
            # Use seed for deterministic results
            self._random = random.Random(seed)
        else:
            self._random = random.Random()

        # Initialize enhanced surrogate generator
        self._surrogate_generator = SurrogateGenerator(seed=seed)

        logger.debug(f"StrategyApplicator initialized with seed: {seed}")

    def apply_strategy(
        self,
        original_text: str,
        entity_type: str,
        strategy: Strategy,
        confidence: float,
    ) -> str:
        """
        Apply a masking strategy to generate a replacement token with fallback support.

        Args:
            original_text: The original PII text to mask
            entity_type: Type of entity (e.g., 'PHONE_NUMBER', 'EMAIL_ADDRESS')
            strategy: The masking strategy to apply
            confidence: Detection confidence score

        Returns:
            str: The masked replacement text

        Raises:
            ValueError: If strategy parameters are invalid
            NotImplementedError: If strategy type is not supported
        """
        logger.debug(
            f"Applying {strategy.kind.value} strategy to {entity_type}"
        )

        try:
            # Try the primary strategy
            return self._apply_single_strategy(
                original_text, entity_type, strategy, confidence
            )
        except Exception as e:
            logger.warning(
                f"Strategy {strategy.kind.value} failed for {entity_type}: {e}"
            )

            # Try fallback strategies
            return self._apply_fallback_strategy(
                original_text, entity_type, strategy, confidence, str(e)
            )

    def _apply_single_strategy(
        self,
        original_text: str,
        entity_type: str,
        strategy: Strategy,
        confidence: float,
    ) -> str:
        """Apply a single strategy without fallback."""
        if strategy.kind == StrategyKind.REDACT:
            return self._apply_redact_strategy(original_text, strategy)
        elif strategy.kind == StrategyKind.TEMPLATE:
            return self._apply_template_strategy(
                original_text, entity_type, strategy
            )
        elif strategy.kind == StrategyKind.HASH:
            # Pass entity_type to hash strategy for per-entity salting
            hash_strategy = strategy.with_parameters(entity_type=entity_type)
            return self._apply_hash_strategy(original_text, hash_strategy)
        elif strategy.kind == StrategyKind.PARTIAL:
            return self._apply_partial_strategy(original_text, strategy)
        elif strategy.kind == StrategyKind.SURROGATE:
            return self._apply_surrogate_strategy(
                original_text, entity_type, strategy
            )
        elif strategy.kind == StrategyKind.CUSTOM:
            return self._apply_custom_strategy(
                original_text, entity_type, confidence, strategy
            )
        else:
            raise NotImplementedError(
                f"Strategy {strategy.kind.value} not implemented"
            )

    def _apply_fallback_strategy(
        self,
        original_text: str,
        entity_type: str,
        primary_strategy: Strategy,
        confidence: float,
        error_msg: str,
    ) -> str:
        """Apply fallback strategies when primary strategy fails."""
        # Define fallback chain
        fallback_strategies = [
            Strategy(StrategyKind.TEMPLATE, {"template": f"[{entity_type}]"}),
            Strategy(StrategyKind.REDACT, {"redact_char": "*", "preserve_length": True}),
            Strategy(StrategyKind.REDACT, {"redact_char": "*", "preserve_length": False})
        ]

        for fallback_strategy in fallback_strategies:
            try:
                logger.info(
                    f"Trying fallback {fallback_strategy.kind.value} for {entity_type}"
                )
                return self._apply_single_strategy(
                    original_text, entity_type, fallback_strategy, confidence
                )
            except Exception as fallback_error:
                logger.warning(
                    f"Fallback {fallback_strategy.kind.value} failed: {fallback_error}"
                )
                continue

        # Ultimate fallback - simple asterisk masking
        logger.error(
            f"All fallback strategies failed for {entity_type}, using ultimate fallback"
        )
        return "*" * max(1, len(original_text))

    def compose_strategies(
        self,
        original_text: str,
        entity_type: str,
        strategies: list[Strategy],
        confidence: float,
    ) -> str:
        """
        Compose multiple strategies in sequence or parallel.

        Args:
            original_text: The original PII text to mask
            entity_type: Type of entity
            strategies: List of strategies to compose
            confidence: Detection confidence score

        Returns:
            str: The result of composed strategies
        """
        if not strategies:
            raise ValueError("At least one strategy must be provided")

        if len(strategies) == 1:
            return self.apply_strategy(original_text, entity_type, strategies[0], confidence)

        # For now, implement sequential composition
        # Could be extended to support parallel composition with voting
        result = original_text
        for i, strategy in enumerate(strategies):
            try:
                result = self.apply_strategy(result, entity_type, strategy, confidence)
            except Exception as e:
                logger.warning(
                    f"Strategy {i+1}/{len(strategies)} ({strategy.kind.value}) failed: {e}"
                )
                # Continue with current result
                continue

        return result

    def _apply_redact_strategy(
        self, original_text: str, strategy: Strategy
    ) -> str:
        """Apply redaction strategy - replace with redaction characters."""
        redact_char = str(strategy.get_parameter("redact_char", "*"))
        preserve_length = bool(strategy.get_parameter("preserve_length", True))

        if preserve_length:
            return str(redact_char) * len(original_text)
        else:
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
            raise ValueError("Template strategy requires 'template' parameter or auto_generate=True")

        # Auto-generate template based on format if requested
        if auto_generate or (preserve_format and template):
            format_template = self._generate_format_template(original_text, entity_type)
            if auto_generate:
                template = format_template
            elif preserve_format:
                # Merge user template with format preservation
                template = self._merge_template_with_format(template, format_template, original_text)

        # Support placeholder substitution
        placeholders = {
            "entity_type": entity_type,
            "length": len(original_text),
            "original_format": self._detect_format_pattern(original_text),
        }

        result = template or "[REDACTED]"
        for key, value in placeholders.items():
            result = result.replace(f"{{{key}}}", str(value))

        return result

    def _generate_format_template(self, original_text: str, entity_type: str) -> str:
        """Generate a format-preserving template based on the original text structure."""
        if not original_text:
            return "[REDACTED]"

        # Entity-specific format templates
        entity_templates = {
            "PHONE_NUMBER": self._generate_phone_template,
            "US_SSN": self._generate_ssn_template,
            "CREDIT_CARD": self._generate_credit_card_template,
            "EMAIL_ADDRESS": self._generate_email_template,
        }

        # Try entity-specific template generation first
        if entity_type in entity_templates:
            return entity_templates[entity_type](original_text)

        # Generic format template generation
        return self._generate_generic_template(original_text)

    def _generate_phone_template(self, original_text: str) -> str:
        """Generate format-preserving template for phone numbers."""
        # Common phone patterns
        if len(original_text) == 10 and original_text.isdigit():
            return "XXX-XXX-XXXX"
        elif len(original_text) == 12 and original_text[3] == '-' and original_text[7] == '-':
            return "XXX-XXX-XXXX"
        elif len(original_text) == 14 and original_text.startswith('(') and ')' in original_text:
            return "(XXX) XXX-XXXX"
        elif '+' in original_text:
            # International format
            return "+X " + "X" * (len(original_text) - 3)
        else:
            # Generic phone template
            return "X" * len([c for c in original_text if c.isdigit()]) + "".join(c for c in original_text if not c.isdigit())

    def _generate_ssn_template(self, original_text: str) -> str:
        """Generate format-preserving template for SSN."""
        if len(original_text) == 11 and original_text[3] == '-' and original_text[6] == '-':
            return "XXX-XX-XXXX"
        elif len(original_text) == 9 and original_text.isdigit():
            return "XXXXXXXXX"
        else:
            # Preserve structure but mask digits
            result = ""
            for char in original_text:
                if char.isdigit():
                    result += "X"
                else:
                    result += char
            return result

    def _generate_credit_card_template(self, original_text: str) -> str:
        """Generate format-preserving template for credit cards."""
        if len(original_text) == 16 and original_text.isdigit():
            return "XXXX-XXXX-XXXX-XXXX"
        elif len(original_text) == 19 and original_text.count('-') == 3:
            return "XXXX-XXXX-XXXX-XXXX"
        elif len(original_text) == 19 and original_text.count(' ') == 3:
            return "XXXX XXXX XXXX XXXX"
        else:
            # Preserve structure but mask digits
            result = ""
            for char in original_text:
                if char.isdigit():
                    result += "X"
                else:
                    result += char
            return result

    def _generate_email_template(self, original_text: str) -> str:
        """Generate format-preserving template for email addresses."""
        if '@' not in original_text:
            return "[EMAIL]"

        username, domain = original_text.split('@', 1)

        # Preserve username length and domain structure
        username_template = "x" * len(username)

        # Preserve domain structure
        if '.' in domain:
            domain_parts = domain.split('.')
            domain_template = ".".join("x" * len(part) for part in domain_parts)
        else:
            domain_template = "x" * len(domain)

        return f"{username_template}@{domain_template}"

    def _generate_generic_template(self, original_text: str) -> str:
        """Generate a generic format-preserving template."""
        result = ""
        for char in original_text:
            if char.isalpha():
                result += "X"
            elif char.isdigit():
                result += "#"
            else:
                result += char
        return result

    def _detect_format_pattern(self, text: str) -> str:
        """Detect and return the format pattern of the input text."""
        pattern = ""
        for char in text:
            if char.isalpha():
                pattern += "L"  # Letter
            elif char.isdigit():
                pattern += "D"  # Digit
            elif char.isspace():
                pattern += "S"  # Space
            else:
                pattern += "P"  # Punctuation/Special
        return pattern

    def _merge_template_with_format(self, user_template: str, format_template: str, original_text: str) -> str:
        """Merge user-provided template with format preservation."""
        # If user template contains format placeholders, replace them
        if "{format}" in user_template:
            return user_template.replace("{format}", format_template)

        # If user template is simple (like [PHONE]), append format info
        if user_template.startswith("[") and user_template.endswith("]"):
            base_template = user_template[1:-1]
            return f"[{base_template}:{len(original_text)}]"

        # Otherwise return user template as-is
        return user_template

    def _apply_hash_strategy(
        self, original_text: str, strategy: Strategy
    ) -> str:
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
        effective_salt = self._build_deterministic_salt(
            base_salt=salt,
            per_entity_salt=per_entity_salt,
            entity_type=entity_type,
            original_text=original_text
        )

        # Combine text with salt for hashing
        content_to_hash = original_text + effective_salt

        # Compute hash
        hash_obj = self._get_hash_algorithm(algorithm)
        hash_obj.update(content_to_hash.encode("utf-8"))

        # Get hash result in requested format
        if format_output == "hex":
            hash_result = hash_obj.hexdigest()
        elif format_output == "base64":
            import base64
            hash_result = base64.b64encode(hash_obj.digest()).decode('ascii')
        elif format_output == "base32":
            import base64
            hash_result = base64.b32encode(hash_obj.digest()).decode('ascii')
        else:
            hash_result = hash_obj.hexdigest()

        # Apply length consistency if requested
        if consistent_length and truncate:
            hash_result = self._apply_consistent_truncation(
                hash_result, truncate, original_text, algorithm
            )
        elif truncate and isinstance(truncate, int) and truncate > 0:
            hash_result = hash_result[:truncate]

        # Apply format structure preservation if requested
        if preserve_format_structure:
            hash_result = self._preserve_format_in_hash(
                original_text, hash_result, prefix
            )

        return str(prefix) + str(hash_result)

    def _build_deterministic_salt(
        self, base_salt: str, per_entity_salt: dict, entity_type: str, original_text: str
    ) -> str:
        """Build a deterministic salt combining base, per-entity, and content-based salts."""
        salt_components = [base_salt or ""]

        # Add per-entity-type salt for security isolation
        if per_entity_salt and isinstance(per_entity_salt, dict):
            entity_salt = per_entity_salt.get(entity_type, per_entity_salt.get("default", ""))
            salt_components.append(str(entity_salt))

        # Add content-length-based component for additional entropy
        salt_components.append(f"len:{len(original_text)}")

        # Add entity type for separation
        salt_components.append(f"type:{entity_type}")

        return "|".join(salt_components)

    def _get_hash_algorithm(self, algorithm: str) -> Any:
        """Get hash algorithm object."""
        if algorithm == "md5":
            return hashlib.md5()
        elif algorithm == "sha1":
            return hashlib.sha1()
        elif algorithm == "sha256":
            return hashlib.sha256()
        elif algorithm == "sha384":
            return hashlib.sha384()
        elif algorithm == "sha512":
            return hashlib.sha512()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    def _apply_consistent_truncation(
        self, hash_result: str, truncate: int, original_text: str, algorithm: str
    ) -> str:
        """Apply truncation that's consistent for similar-length inputs."""
        if truncate >= len(hash_result):
            return hash_result

        # Use original text characteristics to determine truncation offset
        # This ensures similar inputs get similar hash patterns
        offset = hash(original_text + algorithm) % max(1, len(hash_result) - truncate)
        return hash_result[offset:offset + truncate]

    def _preserve_format_in_hash(self, original_text: str, hash_result: str, prefix: str) -> str:
        """Preserve format structure in hash output."""
        # Detect structural elements in original text
        delimiters = []
        delimiter_positions = []

        for i, char in enumerate(original_text):
            if char in ['-', '_', '.', '@', ' ', '(', ')', '+']:
                delimiters.append(char)
                delimiter_positions.append(i)

        if not delimiters:
            return hash_result

        # Try to maintain similar structure in hash
        result = list(hash_result)

        # Insert delimiters at proportional positions
        hash_len = len(hash_result)
        orig_len = len(original_text)

        for i, (delimiter, orig_pos) in enumerate(zip(delimiters, delimiter_positions)):
            # Calculate proportional position in hash
            if orig_len > 0:
                hash_pos = int((orig_pos / orig_len) * hash_len)
                hash_pos = min(hash_pos, len(result) - 1)

                # Insert delimiter if it makes sense
                if hash_pos < len(result) and result[hash_pos].isalnum():
                    result[hash_pos] = delimiter

        return ''.join(result)

    def _apply_partial_strategy(
        self, original_text: str, strategy: Strategy
    ) -> str:
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
            return self._apply_format_aware_partial_masking(
                original_text, visible_chars, position, mask_char,
                preserve_delimiters, deterministic
            )

        # Original basic partial masking logic
        if visible_chars >= len(original_text):
            # Would show everything, apply minimal masking
            if len(original_text) <= 2:
                return mask_char * len(original_text)
            else:
                return (
                    original_text[0]
                    + mask_char * (len(original_text) - 2)
                    + original_text[-1]
                )

        # Apply partial masking based on position
        if position == "start":
            visible_part = original_text[:visible_chars]
            masked_part = mask_char * (len(original_text) - visible_chars)
            return visible_part + masked_part
        elif position == "end":
            visible_part = original_text[-visible_chars:]
            masked_part = mask_char * (len(original_text) - visible_chars)
            return masked_part + visible_part
        elif position == "middle":
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
        else:
            raise ValueError(
                f"Invalid position for partial strategy: {position}"
            )

    def _apply_format_aware_partial_masking(
        self, original_text: str, visible_chars: int, position: str,
        mask_char: str, preserve_delimiters: bool, deterministic: bool
    ) -> str:
        """Apply format-aware partial masking that preserves delimiters and structure."""
        # Detect common delimiters and structural elements
        delimiters = {'-', '_', '.', '@', ' ', '(', ')', '+'}

        # Find delimiter positions
        delimiter_positions = []
        if preserve_delimiters:
            for i, char in enumerate(original_text):
                if char in delimiters:
                    delimiter_positions.append(i)

        # Extract maskable characters (non-delimiters)
        maskable_chars = []
        maskable_positions = []
        for i, char in enumerate(original_text):
            if not preserve_delimiters or char not in delimiters:
                maskable_chars.append(char)
                maskable_positions.append(i)

        if not maskable_chars:
            # All characters are delimiters, return as-is
            return original_text

        # Apply partial masking to maskable characters only
        visible_chars = min(visible_chars, len(maskable_chars))

        # Determine which characters to keep visible
        visible_indices = self._select_visible_characters(
            len(maskable_chars), visible_chars, position, deterministic, original_text
        )

        # Build result by preserving delimiters and masking non-visible characters
        result = list(original_text)
        for i, pos in enumerate(maskable_positions):
            if i not in visible_indices:
                result[pos] = mask_char

        return ''.join(result)

    def _select_visible_characters(
        self, total_chars: int, visible_chars: int, position: str,
        deterministic: bool, original_text: str
    ) -> set:
        """Select which character indices should remain visible."""
        if visible_chars >= total_chars:
            return set(range(total_chars))

        if position == "start":
            return set(range(visible_chars))
        elif position == "end":
            return set(range(total_chars - visible_chars, total_chars))
        elif position == "middle":
            # Show chars at both ends
            chars_per_side = visible_chars // 2
            remaining = visible_chars % 2

            start_chars = chars_per_side + remaining
            end_chars = chars_per_side

            if start_chars + end_chars >= total_chars:
                # Fallback to showing alternating characters
                return self._select_alternating_characters(
                    total_chars, visible_chars, deterministic, original_text
                )

            visible_indices: set[int] = set()
            visible_indices.update(range(start_chars))
            visible_indices.update(range(total_chars - end_chars, total_chars))
            return visible_indices
        elif position == "random":
            return self._select_random_characters(
                total_chars, visible_chars, deterministic, original_text
            )
        else:
            raise ValueError(f"Invalid position for partial strategy: {position}")

    def _select_alternating_characters(
        self, total_chars: int, visible_chars: int, deterministic: bool, original_text: str
    ) -> set:
        """Select alternating characters for visibility."""
        if not deterministic:
            # Non-deterministic alternating
            step = max(1, total_chars // visible_chars)
            return set(range(0, total_chars, step)[:visible_chars])

        # Deterministic alternating based on text content
        hash_seed = hash(original_text) % total_chars
        step = max(1, total_chars // visible_chars)
        visible_indices = set()

        for i in range(visible_chars):
            idx = (hash_seed + i * step) % total_chars
            visible_indices.add(idx)

        return visible_indices

    def _select_random_characters(
        self, total_chars: int, visible_chars: int, deterministic: bool, original_text: str
    ) -> set:
        """Select random characters for visibility."""
        if deterministic:
            # Use text content as seed for deterministic randomness
            local_random = random.Random(hash(original_text))
            indices = list(range(total_chars))
            local_random.shuffle(indices)
            return set(indices[:visible_chars])
        else:
            # Non-deterministic random selection
            indices = list(range(total_chars))
            self._random.shuffle(indices)
            return set(indices[:visible_chars])

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

    def _apply_legacy_surrogate_strategy(
        self, original_text: str, entity_type: str, strategy: Strategy
    ) -> str:
        """Legacy surrogate strategy implementation for backward compatibility."""
        format_type = strategy.get_parameter(
            "format_type", entity_type.lower()
        )
        seed = strategy.get_parameter("seed", self.seed)

        # Use seed for deterministic generation
        local_random = random.Random(seed + original_text if seed else None)

        if format_type in ["phone", "phone_number"]:
            return self._generate_surrogate_phone(local_random)
        elif format_type in ["email", "email_address"]:
            return self._generate_surrogate_email(local_random)
        elif format_type in ["ssn", "us_ssn"]:
            return self._generate_surrogate_ssn(local_random)
        elif format_type == "credit_card":
            return self._generate_surrogate_credit_card(local_random)
        elif format_type == "name":
            return self._generate_surrogate_name(local_random)
        elif format_type == "custom":
            pattern = strategy.get_parameter("pattern")
            if pattern:
                return self._generate_from_pattern(pattern, local_random)

        # Fallback: generate alphanumeric string of same length
        chars = string.ascii_letters + string.digits
        return "".join(
            local_random.choice(chars) for _ in range(len(original_text))
        )

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
            raise ValueError(
                "Custom strategy requires a callable 'callback' parameter"
            )

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

    def get_surrogate_quality_metrics(self):
        """Get quality metrics from the surrogate generator."""
        return self._surrogate_generator.get_quality_metrics()

    def reset_document_scope(self) -> None:
        """Reset document scope for new document processing."""
        self._surrogate_generator.reset_document_scope()
