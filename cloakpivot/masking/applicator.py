"""StrategyApplicator for generating masked replacement tokens."""

import logging
import hashlib
import random
import string
from typing import Any, Dict, Optional

from ..core.strategies import Strategy, StrategyKind

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
        
        logger.debug(f"StrategyApplicator initialized with seed: {seed}")
    
    def apply_strategy(
        self,
        original_text: str,
        entity_type: str,
        strategy: Strategy,
        confidence: float
    ) -> str:
        """
        Apply a masking strategy to generate a replacement token.
        
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
        logger.debug(f"Applying {strategy.kind.value} strategy to {entity_type}")
        
        if strategy.kind == StrategyKind.REDACT:
            return self._apply_redact_strategy(original_text, strategy)
        elif strategy.kind == StrategyKind.TEMPLATE:
            return self._apply_template_strategy(original_text, entity_type, strategy)
        elif strategy.kind == StrategyKind.HASH:
            return self._apply_hash_strategy(original_text, strategy)
        elif strategy.kind == StrategyKind.PARTIAL:
            return self._apply_partial_strategy(original_text, strategy)
        elif strategy.kind == StrategyKind.SURROGATE:
            return self._apply_surrogate_strategy(original_text, entity_type, strategy)
        elif strategy.kind == StrategyKind.CUSTOM:
            return self._apply_custom_strategy(original_text, entity_type, confidence, strategy)
        else:
            raise NotImplementedError(f"Strategy {strategy.kind.value} not implemented")
    
    def _apply_redact_strategy(self, original_text: str, strategy: Strategy) -> str:
        """Apply redaction strategy - replace with redaction characters."""
        redact_char = strategy.get_parameter("redact_char", "*")
        preserve_length = strategy.get_parameter("preserve_length", True)
        
        if preserve_length:
            return redact_char * len(original_text)
        else:
            # Fixed length redaction
            redaction_length = strategy.get_parameter("redaction_length", 8)
            return redact_char * redaction_length
    
    def _apply_template_strategy(self, original_text: str, entity_type: str, strategy: Strategy) -> str:
        """Apply template strategy - replace with fixed tokens."""
        template = strategy.get_parameter("template")
        if not template:
            raise ValueError("Template strategy requires 'template' parameter")
        
        # Support placeholder substitution
        placeholders = {
            "entity_type": entity_type,
            "length": len(original_text),
        }
        
        result = template
        for key, value in placeholders.items():
            result = result.replace(f"{{{key}}}", str(value))
        
        return result
    
    def _apply_hash_strategy(self, original_text: str, strategy: Strategy) -> str:
        """Apply hash strategy - replace with hashed values."""
        algorithm = strategy.get_parameter("algorithm", "sha256")
        salt = strategy.get_parameter("salt", "")
        truncate = strategy.get_parameter("truncate", None)
        prefix = strategy.get_parameter("prefix", "")
        
        # Combine text with salt for hashing
        content_to_hash = original_text + salt
        
        # Compute hash
        if algorithm == "md5":
            hash_obj = hashlib.md5()
        elif algorithm == "sha1":
            hash_obj = hashlib.sha1()
        elif algorithm == "sha256":
            hash_obj = hashlib.sha256()
        elif algorithm == "sha384":
            hash_obj = hashlib.sha384()
        elif algorithm == "sha512":
            hash_obj = hashlib.sha512()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        hash_obj.update(content_to_hash.encode('utf-8'))
        hash_result = hash_obj.hexdigest()
        
        # Truncate if requested
        if truncate and isinstance(truncate, int) and truncate > 0:
            hash_result = hash_result[:truncate]
        
        return prefix + hash_result
    
    def _apply_partial_strategy(self, original_text: str, strategy: Strategy) -> str:
        """Apply partial strategy - show some chars, mask others."""
        visible_chars = strategy.get_parameter("visible_chars", 4)
        position = strategy.get_parameter("position", "end")
        mask_char = strategy.get_parameter("mask_char", "*")
        min_length = strategy.get_parameter("min_length", 1)
        
        if len(original_text) < min_length:
            # Text too short, mask completely
            return mask_char * len(original_text)
        
        if visible_chars >= len(original_text):
            # Would show everything, apply minimal masking
            if len(original_text) <= 2:
                return mask_char * len(original_text)
            else:
                return original_text[0] + mask_char * (len(original_text) - 2) + original_text[-1]
        
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
                    original_text,
                    strategy.with_parameters(position="end")
                )
            
            start_part = original_text[:start_chars]
            end_part = original_text[-end_chars:] if end_chars > 0 else ""
            middle_length = len(original_text) - start_chars - end_chars
            masked_part = mask_char * middle_length
            
            return start_part + masked_part + end_part
        else:
            raise ValueError(f"Invalid position for partial strategy: {position}")
    
    def _apply_surrogate_strategy(self, original_text: str, entity_type: str, strategy: Strategy) -> str:
        """Apply surrogate strategy - generate fake data in same format."""
        format_type = strategy.get_parameter("format_type", entity_type.lower())
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
        return ''.join(local_random.choice(chars) for _ in range(len(original_text)))
    
    def _apply_custom_strategy(
        self, 
        original_text: str, 
        entity_type: str, 
        confidence: float, 
        strategy: Strategy
    ) -> str:
        """Apply custom strategy - use callback function."""
        callback = strategy.get_parameter("callback")
        if not callback or not callable(callback):
            raise ValueError("Custom strategy requires a callable 'callback' parameter")
        
        try:
            result = callback(
                original_text=original_text,
                entity_type=entity_type,
                confidence=confidence
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
        exchange = rng.randint(200, 999)   # Exchange codes start from 200
        number = rng.randint(0, 9999)      # Last 4 digits
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
        group = rng.randint(10, 99)   # Avoid 00
        serial = rng.randint(1000, 9999)  # Avoid 0000
        return f"{area:03d}-{group:02d}-{serial:04d}"
    
    def _generate_surrogate_credit_card(self, rng: random.Random) -> str:
        """Generate a surrogate credit card number."""
        # Simple 16-digit format
        digits = [str(rng.randint(0, 9)) for _ in range(16)]
        return ''.join(digits)
    
    def _generate_surrogate_name(self, rng: random.Random) -> str:
        """Generate a surrogate person name."""
        first_names = ["John", "Jane", "Alex", "Sam", "Chris", "Jordan"]
        last_names = ["Smith", "Johnson", "Brown", "Davis", "Wilson", "Anderson"]
        
        first = rng.choice(first_names)
        last = rng.choice(last_names)
        return f"{first} {last}"
    
    def _generate_from_pattern(self, pattern: str, rng: random.Random) -> str:
        """Generate text from a pattern string (simple implementation)."""
        # Simple pattern replacement:
        # X = random letter, 9 = random digit, ? = random alphanumeric
        result = ""
        for char in pattern:
            if char == 'X':
                result += rng.choice(string.ascii_uppercase)
            elif char == 'x':
                result += rng.choice(string.ascii_lowercase)
            elif char == '9':
                result += str(rng.randint(0, 9))
            elif char == '?':
                result += rng.choice(string.ascii_letters + string.digits)
            else:
                result += char
        return result