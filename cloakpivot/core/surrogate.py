"""Format-preserving surrogate generation for realistic-looking replacement values."""

import hashlib
import logging
import random
import string
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FormatPattern:
    """
    Analyzed format pattern of original text for surrogate generation.

    Captures the structure and characteristics of the original data to enable
    format-preserving surrogate generation.
    """

    original_text: str
    original_length: int
    character_classes: (
        str  # String like "DDDSDDDSDDD" where D=digit, L=letter, S=separator, P=punct
    )
    digit_positions: list[int] = field(default_factory=list)
    letter_positions: list[int] = field(default_factory=list)
    separator_positions: dict[int, str] = field(default_factory=dict)
    special_chars: set[str] = field(default_factory=set)
    detected_format: Optional[str] = None

    @classmethod
    def analyze(cls, text: str) -> "FormatPattern":
        """
        Analyze text to extract format pattern for surrogate generation.

        Args:
            text: Original text to analyze

        Returns:
            FormatPattern: Analyzed pattern information
        """
        if not text:
            return cls(
                original_text=text,
                original_length=0,
                character_classes="",
                detected_format=None,
            )

        digit_positions = []
        letter_positions = []
        separator_positions = {}
        special_chars = set()
        character_classes = ""

        for i, char in enumerate(text):
            if char.isdigit():
                digit_positions.append(i)
                character_classes += "D"
            elif char.isalpha():
                letter_positions.append(i)
                character_classes += "L"
            elif char in ["-", "_", ".", " "]:
                separator_positions[i] = char
                character_classes += "S"
                special_chars.add(char)
            else:
                separator_positions[i] = char  # Treat other special chars as separators
                special_chars.add(char)
                character_classes += "P"

        # Detect common format types
        detected_format = cls._detect_format_type(
            text, character_classes, special_chars
        )

        return cls(
            original_text=text,
            original_length=len(text),
            character_classes=character_classes,
            digit_positions=digit_positions,
            letter_positions=letter_positions,
            separator_positions=separator_positions,
            special_chars=special_chars,
            detected_format=detected_format,
        )

    @staticmethod
    def _detect_format_type(
        text: str, character_classes: str, special_chars: set[str]
    ) -> Optional[str]:
        """Detect the likely format type of the text."""
        # Phone number patterns
        if (
            len(text) in [10, 12, 14]
            and character_classes.count("D") >= 10
            and "-" in special_chars
        ):
            return "phone"

        # SSN pattern
        if (
            len(text) == 11
            and character_classes == "DDDSDDSDDDD"
            and "-" in special_chars
        ):
            return "ssn"

        # Email pattern
        if "@" in special_chars and "." in special_chars:
            return "email"

        # Credit card patterns
        if (
            len(text) in [16, 19]
            and character_classes.count("D") >= 13
            and ("-" in special_chars or " " in special_chars)
        ):
            return "credit_card"

        return None


@dataclass
class SurrogateQualityMetrics:
    """
    Track quality metrics for surrogate generation.

    Monitors format preservation, uniqueness, validation success,
    and other quality indicators.
    """

    total_generated: int = 0
    format_preserved_count: int = 0
    unique_count: int = 0
    validation_success_count: int = 0
    collision_count: int = 0
    collision_examples: list[str] = field(default_factory=list)
    generation_failures: list[str] = field(default_factory=list)

    @property
    def format_preservation_rate(self) -> float:
        """Calculate format preservation success rate."""
        if self.total_generated == 0:
            return 0.0
        return self.format_preserved_count / self.total_generated

    @property
    def uniqueness_rate(self) -> float:
        """Calculate uniqueness success rate."""
        if self.total_generated == 0:
            return 0.0
        return self.unique_count / self.total_generated

    @property
    def validation_success_rate(self) -> float:
        """Calculate validation success rate."""
        if self.total_generated == 0:
            return 0.0
        return self.validation_success_count / self.total_generated

    def record_generation(
        self,
        original: str,
        surrogate: str,
        format_preserved: bool,
        is_unique: bool,
        validation_passed: bool,
    ) -> None:
        """Record a surrogate generation attempt."""
        self.total_generated += 1

        if format_preserved:
            self.format_preserved_count += 1

        if is_unique:
            self.unique_count += 1

        if validation_passed:
            self.validation_success_count += 1

    def record_collision(self, original: str, collision_value: str) -> None:
        """Record a collision occurrence."""
        self.collision_count += 1
        if len(self.collision_examples) < 10:  # Keep sample of collisions
            self.collision_examples.append(collision_value)

    def record_failure(self, original: str, error_message: str) -> None:
        """Record a generation failure."""
        if len(self.generation_failures) < 10:  # Keep sample of failures
            self.generation_failures.append(f"{original}: {error_message}")


class SurrogateGenerator:
    """
    Generate format-preserving surrogates for detected PII entities.

    Creates realistic-looking replacement values that maintain the original
    data characteristics while being completely artificial.

    Features:
    - Format-preserving generation (maintains character classes, lengths, separators)
    - Deterministic generation with seeded random values
    - Entity-specific generation rules
    - Collision detection and resolution
    - Quality metrics and validation
    """

    def __init__(self, seed: Optional[str] = None):
        """
        Initialize the surrogate generator.

        Args:
            seed: Optional seed for deterministic generation
        """
        self.seed = seed
        self._document_scope_cache: dict[str, str] = {}
        self._generated_values: set[str] = set()
        self._quality_metrics = SurrogateQualityMetrics()
        self._entity_generators = self._initialize_entity_generators()

        logger.debug(f"SurrogateGenerator initialized with seed: {seed}")

    def _initialize_entity_generators(self) -> dict[str, callable]:
        """Initialize entity-specific generators."""
        return {
            "PHONE_NUMBER": self._generate_phone_surrogate,
            "US_SSN": self._generate_ssn_surrogate,
            "EMAIL_ADDRESS": self._generate_email_surrogate,
            "CREDIT_CARD": self._generate_credit_card_surrogate,
            "PERSON": self._generate_name_surrogate,
            "LOCATION": self._generate_address_surrogate,
        }

    def generate_surrogate(self, original_text: str, entity_type: str) -> str:
        """
        Generate a format-preserving surrogate for the given text.

        Args:
            original_text: The original PII text to replace
            entity_type: Type of entity (e.g., 'PHONE_NUMBER', 'EMAIL_ADDRESS')

        Returns:
            str: Generated surrogate that preserves format characteristics
        """
        if not original_text:
            return original_text

        # Create deterministic seed for this specific input
        input_seed = self._create_deterministic_seed(original_text, entity_type)

        # Check if we've already generated a surrogate for this input
        cache_key = f"{original_text}:{entity_type}:{input_seed}"
        if cache_key in self._document_scope_cache:
            return self._document_scope_cache[cache_key]

        try:
            # Generate surrogate with collision detection
            surrogate = self._generate_with_collision_detection(
                original_text, entity_type, input_seed
            )

            # Validate and record metrics (check uniqueness before caching)
            format_preserved = self._validate_format_preservation(
                original_text, surrogate
            )
            is_unique = (
                surrogate not in self._generated_values
            )  # Check before adding to set
            validation_passed = self._validate_surrogate_quality(
                original_text, surrogate, entity_type
            )

            # Cache the result after uniqueness check
            self._document_scope_cache[cache_key] = surrogate
            self._generated_values.add(surrogate)

            self._quality_metrics.record_generation(
                original_text, surrogate, format_preserved, is_unique, validation_passed
            )

            return surrogate

        except Exception as e:
            logger.error(f"Failed to generate surrogate for {entity_type}: {e}")
            self._quality_metrics.record_failure(original_text, str(e))

            # Fallback to simple format preservation
            return self._generate_fallback_surrogate(original_text, entity_type)

    def _create_deterministic_seed(self, original_text: str, entity_type: str) -> str:
        """Create a deterministic seed for consistent generation."""
        # Combine base seed, original text, and entity type
        seed_components = [
            self.seed or "default_seed",
            original_text,
            entity_type,
            str(len(original_text)),  # Add length for additional entropy
        ]

        combined = "|".join(seed_components)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _generate_with_collision_detection(
        self, original_text: str, entity_type: str, seed: str
    ) -> str:
        """Generate surrogate with collision detection and resolution."""
        max_attempts = 10

        for attempt in range(max_attempts):
            # Create attempt-specific seed
            attempt_seed = hashlib.sha256(f"{seed}:{attempt}".encode()).hexdigest()[:16]
            rng = random.Random(attempt_seed)

            # Generate surrogate
            surrogate = self._generate_surrogate_internal(
                original_text, entity_type, rng
            )

            # Check for collision
            if surrogate not in self._generated_values:
                return surrogate

            # Record collision
            self._quality_metrics.record_collision(original_text, surrogate)
            logger.debug(f"Collision detected for {entity_type}, attempt {attempt + 1}")

        # If we still have collisions after max attempts, add suffix
        base_surrogate = surrogate
        for suffix in range(1, 100):
            candidate = f"{base_surrogate}_{suffix}"
            if candidate not in self._generated_values:
                return candidate

        # Ultimate fallback - should be very rare
        return f"{surrogate}_{hash(original_text) % 10000}"

    def _generate_surrogate_internal(
        self, original_text: str, entity_type: str, rng: random.Random
    ) -> str:
        """Internal surrogate generation logic."""
        # Try entity-specific generator first
        if entity_type in self._entity_generators:
            try:
                return self._entity_generators[entity_type](original_text, rng)
            except Exception as e:
                logger.warning(
                    f"Entity-specific generator failed for {entity_type}: {e}"
                )

        # Fall back to pattern-based generation
        return self._generate_pattern_based_surrogate(original_text, rng)

    def _generate_pattern_based_surrogate(
        self, original_text: str, rng: random.Random
    ) -> str:
        """Generate surrogate based on analyzed pattern."""
        pattern = FormatPattern.analyze(original_text)
        result = [""] * pattern.original_length

        # First, place separators and special characters
        for pos, char in pattern.separator_positions.items():
            result[pos] = char

        # Place special characters that aren't separators
        for i, char in enumerate(original_text):
            if char in pattern.special_chars and i not in pattern.separator_positions:
                result[i] = char

        # Generate digits
        for pos in pattern.digit_positions:
            result[pos] = str(rng.randint(0, 9))

        # Generate letters
        for pos in pattern.letter_positions:
            original_char = original_text[pos]
            if original_char.isupper():
                result[pos] = rng.choice(string.ascii_uppercase)
            else:
                result[pos] = rng.choice(string.ascii_lowercase)

        return "".join(result)

    def _generate_phone_surrogate(self, original_text: str, rng: random.Random) -> str:
        """Generate format-preserving phone number surrogate."""
        pattern = FormatPattern.analyze(original_text)

        # Generate realistic phone number components
        area_codes = [
            200,
            201,
            202,
            203,
            205,
            206,
            207,
            208,
            209,
            210,
        ]  # Sample valid area codes
        area_code = rng.choice(area_codes)
        exchange = rng.randint(200, 999)  # Exchange codes start from 200
        number = rng.randint(1000, 9999)  # Last 4 digits

        # Format according to original pattern
        if pattern.character_classes == "DDDDDDDDDD":  # 10 digits no separators
            return f"{area_code:03d}{exchange:03d}{number:04d}"
        elif pattern.character_classes == "DDDSDDDSDDD":  # XXX-XXX-XXXX
            sep = list(pattern.separator_positions.values())[0]
            return f"{area_code:03d}{sep}{exchange:03d}{sep}{number:04d}"
        elif "(" in original_text and ")" in original_text:  # (XXX) XXX-XXXX
            return f"({area_code:03d}) {exchange:03d}-{number:04d}"
        elif original_text.startswith("+"):  # International format
            return f"+1 {area_code:03d}-{exchange:03d}-{number:04d}"
        else:
            # Use pattern-based fallback
            return self._generate_pattern_based_surrogate(original_text, rng)

    def _generate_ssn_surrogate(self, original_text: str, rng: random.Random) -> str:
        """Generate format-preserving SSN surrogate."""
        # Generate realistic SSN components (avoiding invalid ranges)
        area = rng.randint(100, 899)  # Avoid 000, 666, 900+
        while area == 666:
            area = rng.randint(100, 899)

        group = rng.randint(10, 99)  # Avoid 00
        serial = rng.randint(1000, 9999)  # Avoid 0000

        # Format according to original
        if "-" in original_text:
            return f"{area:03d}-{group:02d}-{serial:04d}"
        else:
            return f"{area:03d}{group:02d}{serial:04d}"

    def _generate_email_surrogate(self, original_text: str, rng: random.Random) -> str:
        """Generate format-preserving email surrogate."""
        if "@" not in original_text:
            return self._generate_pattern_based_surrogate(original_text, rng)

        username_part, domain_part = original_text.split("@", 1)

        # Generate surrogate username preserving length and character types
        surrogate_username = self._generate_pattern_based_surrogate(username_part, rng)

        # Generate surrogate domain
        if "." in domain_part:
            domain_parts = domain_part.split(".")
            surrogate_parts = []

            # Common safe domains for surrogate generation
            safe_domains = ["example", "test", "sample", "demo"]
            safe_tlds = ["com", "org", "net", "edu"]

            for i, part in enumerate(domain_parts):
                if i == len(domain_parts) - 1:  # TLD
                    if len(part) <= 3:
                        surrogate_parts.append(rng.choice(safe_tlds))
                    else:
                        # Preserve pattern for longer TLDs
                        surrogate_parts.append(
                            self._generate_pattern_based_surrogate(part, rng)
                        )
                else:  # Domain parts
                    if len(part) <= 8:
                        surrogate_parts.append(rng.choice(safe_domains))
                    else:
                        # Preserve pattern for longer domains
                        surrogate_parts.append(
                            self._generate_pattern_based_surrogate(part, rng)
                        )

            surrogate_domain = ".".join(surrogate_parts)
        else:
            # Simple domain without dots
            surrogate_domain = self._generate_pattern_based_surrogate(domain_part, rng)

        return f"{surrogate_username}@{surrogate_domain}"

    def _generate_credit_card_surrogate(
        self, original_text: str, rng: random.Random
    ) -> str:
        """Generate format-preserving credit card surrogate."""
        # Generate test credit card numbers (not real)
        test_prefixes = ["4000", "4111", "4444", "5000", "5555"]  # Test card prefixes
        prefix = rng.choice(test_prefixes)

        # Generate remaining digits
        remaining_digits = 16 - len(prefix)
        remaining = "".join(str(rng.randint(0, 9)) for _ in range(remaining_digits))

        card_number = prefix + remaining

        # Apply original formatting
        if "-" in original_text:
            # Format as XXXX-XXXX-XXXX-XXXX
            return f"{card_number[:4]}-{card_number[4:8]}-{card_number[8:12]}-{card_number[12:]}"
        elif " " in original_text:
            # Format as XXXX XXXX XXXX XXXX
            return f"{card_number[:4]} {card_number[4:8]} {card_number[8:12]} {card_number[12:]}"
        else:
            # No formatting
            return card_number

    def _generate_name_surrogate(self, original_text: str, rng: random.Random) -> str:
        """Generate format-preserving name surrogate."""
        # Common safe names for surrogate generation
        first_names = ["Alex", "Jordan", "Casey", "Riley", "Morgan", "Cameron"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"]

        parts = original_text.split()
        if len(parts) == 1:
            # Single name
            return rng.choice(first_names + last_names)
        elif len(parts) == 2:
            # First + Last
            return f"{rng.choice(first_names)} {rng.choice(last_names)}"
        else:
            # Multiple parts - preserve structure
            result_parts = []
            for i, part in enumerate(parts):
                if i == 0:  # First name
                    result_parts.append(rng.choice(first_names))
                elif i == len(parts) - 1:  # Last name
                    result_parts.append(rng.choice(last_names))
                else:  # Middle names/initials
                    if len(part) == 1:  # Initial
                        result_parts.append(rng.choice(string.ascii_uppercase))
                    else:  # Full middle name
                        result_parts.append(rng.choice(first_names))

            return " ".join(result_parts)

    def _generate_address_surrogate(
        self, original_text: str, rng: random.Random
    ) -> str:
        """Generate format-preserving address surrogate."""
        # For now, use pattern-based generation
        # Could be enhanced with realistic address components
        return self._generate_pattern_based_surrogate(original_text, rng)

    def generate_from_pattern(self, pattern: str) -> str:
        """
        Generate surrogate from explicit pattern.

        Pattern characters:
        - X: uppercase letter
        - x: lowercase letter
        - 9: digit
        - ?: alphanumeric
        - Other chars: literal

        Args:
            pattern: Pattern string like "XXX-XX-9999"

        Returns:
            str: Generated text following the pattern
        """
        seed = self._create_deterministic_seed(pattern, "PATTERN")
        rng = random.Random(seed)

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

    def _generate_fallback_surrogate(self, original_text: str, entity_type: str) -> str:
        """Generate fallback surrogate when primary generation fails."""
        # Simple pattern preservation as fallback
        seed = self._create_deterministic_seed(original_text, entity_type)
        rng = random.Random(seed)

        result = ""
        for char in original_text:
            if char.isdigit():
                result += str(rng.randint(0, 9))
            elif char.isalpha():
                if char.isupper():
                    result += rng.choice(string.ascii_uppercase)
                else:
                    result += rng.choice(string.ascii_lowercase)
            else:
                result += char

        return result

    def _validate_format_preservation(self, original: str, surrogate: str) -> bool:
        """Validate that surrogate preserves the format of original."""
        if len(original) != len(surrogate):
            return False

        for _i, (orig_char, surr_char) in enumerate(zip(original, surrogate)):
            # Check character class preservation
            if orig_char.isdigit() and not surr_char.isdigit():
                return False
            elif orig_char.isalpha() and not surr_char.isalpha():
                return False
            elif not orig_char.isalnum() and orig_char != surr_char:
                # Non-alphanumeric characters should be preserved exactly
                return False

        return True

    def _validate_surrogate_quality(
        self, original: str, surrogate: str, entity_type: str
    ) -> bool:
        """Validate overall quality of generated surrogate."""
        # Check that it's different from original
        if surrogate == original:
            return False

        # Check format preservation
        if not self._validate_format_preservation(original, surrogate):
            return False

        # Check for obvious real data patterns (basic check)
        if self._contains_suspicious_patterns(surrogate):
            return False

        return True

    def _contains_suspicious_patterns(self, text: str) -> bool:
        """Check if text contains patterns that might be real data."""
        # Very basic check for obviously fake vs real patterns
        # Could be expanded with more sophisticated validation

        # Check for test/sample indicators
        test_indicators = ["test", "sample", "demo", "example", "fake"]
        text_lower = text.lower()

        for indicator in test_indicators:
            if indicator in text_lower:
                return False  # Good - contains test indicator

        # Additional checks could include:
        # - Known test credit card prefixes
        # - Reserved phone number ranges
        # - Sample domain names
        # etc.

        return False  # For now, assume all generated surrogates are safe

    def get_quality_metrics(self) -> SurrogateQualityMetrics:
        """Get current quality metrics."""
        return self._quality_metrics

    def reset_document_scope(self) -> None:
        """Reset document scope cache for new document processing."""
        self._document_scope_cache.clear()
        self._generated_values.clear()
        logger.debug("Document scope reset for new document")
