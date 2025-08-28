"""Strategy system for defining masking behaviors."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union
import re


class StrategyKind(Enum):
    """Types of masking strategies available."""
    
    REDACT = "redact"
    TEMPLATE = "template"
    HASH = "hash"
    SURROGATE = "surrogate"
    PARTIAL = "partial"
    CUSTOM = "custom"


@dataclass(frozen=True)
class Strategy:
    """
    A masking strategy that defines how to transform detected PII entities.
    
    Attributes:
        kind: The type of masking strategy to apply
        parameters: Strategy-specific parameters for configuration
    
    Examples:
        >>> # Simple redaction
        >>> redact_strategy = Strategy(StrategyKind.REDACT)
        
        >>> # Template replacement
        >>> phone_strategy = Strategy(
        ...     StrategyKind.TEMPLATE,
        ...     parameters={"template": "[PHONE]"}
        ... )
        
        >>> # Partial masking (show last 4 digits)
        >>> ssn_strategy = Strategy(
        ...     StrategyKind.PARTIAL,
        ...     parameters={"visible_chars": 4, "position": "end", "mask_char": "*"}
        ... )
        
        >>> # Hash with salt
        >>> hash_strategy = Strategy(
        ...     StrategyKind.HASH,
        ...     parameters={"algorithm": "sha256", "salt": "my-salt", "truncate": 8}
        ... )
    """
    
    kind: StrategyKind
    parameters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate strategy parameters after initialization."""
        if self.parameters is None:
            object.__setattr__(self, 'parameters', {})
        
        # Validate parameters based on strategy kind
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate parameters for the specific strategy kind."""
        params = self.parameters or {}
        
        if self.kind == StrategyKind.REDACT:
            self._validate_redact_params(params)
        elif self.kind == StrategyKind.TEMPLATE:
            self._validate_template_params(params)
        elif self.kind == StrategyKind.HASH:
            self._validate_hash_params(params)
        elif self.kind == StrategyKind.SURROGATE:
            self._validate_surrogate_params(params)
        elif self.kind == StrategyKind.PARTIAL:
            self._validate_partial_params(params)
        elif self.kind == StrategyKind.CUSTOM:
            self._validate_custom_params(params)
    
    def _validate_redact_params(self, params: Dict[str, Any]) -> None:
        """Validate parameters for redact strategy."""
        # Default redaction character is '*'
        if "redact_char" in params:
            if not isinstance(params["redact_char"], str) or len(params["redact_char"]) != 1:
                raise ValueError("redact_char must be a single character string")
        
        # Optional: maintain original length
        if "preserve_length" in params and not isinstance(params["preserve_length"], bool):
            raise ValueError("preserve_length must be a boolean")
    
    def _validate_template_params(self, params: Dict[str, Any]) -> None:
        """Validate parameters for template strategy."""
        if "template" not in params:
            raise ValueError("Template strategy requires 'template' parameter")
        
        template = params["template"]
        if not isinstance(template, str):
            raise ValueError("Template must be a string")
        
        # Optional: template validation (check for placeholders)
        if "validate_placeholders" in params and params["validate_placeholders"]:
            # Check for common placeholder patterns like {entity_type}, {index}, etc.
            placeholder_pattern = r"\{[a-zA-Z_][a-zA-Z0-9_]*\}"
            if not re.search(placeholder_pattern, template):
                raise ValueError("Template should contain placeholders like {entity_type}")
    
    def _validate_hash_params(self, params: Dict[str, Any]) -> None:
        """Validate parameters for hash strategy."""
        valid_algorithms = {"md5", "sha1", "sha256", "sha384", "sha512"}
        
        algorithm = params.get("algorithm", "sha256")
        if algorithm not in valid_algorithms:
            raise ValueError(f"Hash algorithm must be one of: {valid_algorithms}")
        
        if "salt" in params and not isinstance(params["salt"], str):
            raise ValueError("Salt must be a string")
        
        if "truncate" in params:
            truncate = params["truncate"]
            if not isinstance(truncate, int) or truncate < 1:
                raise ValueError("Truncate must be a positive integer")
        
        if "prefix" in params and not isinstance(params["prefix"], str):
            raise ValueError("Prefix must be a string")
    
    def _validate_surrogate_params(self, params: Dict[str, Any]) -> None:
        """Validate parameters for surrogate strategy."""
        if "format_type" in params:
            valid_formats = {"phone", "ssn", "credit_card", "email", "name", "address", "custom"}
            if params["format_type"] not in valid_formats:
                raise ValueError(f"Format type must be one of: {valid_formats}")
        
        if "seed" in params and not isinstance(params["seed"], str):
            raise ValueError("Seed must be a string")
        
        if "dictionary" in params and not isinstance(params["dictionary"], (list, tuple)):
            raise ValueError("Dictionary must be a list or tuple")
        
        # Custom format pattern validation
        if "pattern" in params and not isinstance(params["pattern"], str):
            raise ValueError("Pattern must be a string")
    
    def _validate_partial_params(self, params: Dict[str, Any]) -> None:
        """Validate parameters for partial masking strategy."""
        if "visible_chars" not in params:
            raise ValueError("Partial strategy requires 'visible_chars' parameter")
        
        visible_chars = params["visible_chars"]
        if not isinstance(visible_chars, int) or visible_chars < 0:
            raise ValueError("visible_chars must be a non-negative integer")
        
        position = params.get("position", "end")
        if position not in {"start", "end", "middle"}:
            raise ValueError("Position must be 'start', 'end', or 'middle'")
        
        mask_char = params.get("mask_char", "*")
        if not isinstance(mask_char, str) or len(mask_char) != 1:
            raise ValueError("mask_char must be a single character string")
        
        # Minimum length threshold
        if "min_length" in params:
            min_length = params["min_length"]
            if not isinstance(min_length, int) or min_length < 1:
                raise ValueError("min_length must be a positive integer")
    
    def _validate_custom_params(self, params: Dict[str, Any]) -> None:
        """Validate parameters for custom strategy."""
        if "callback" not in params:
            raise ValueError("Custom strategy requires 'callback' parameter")
        
        callback = params["callback"]
        if not callable(callback):
            raise ValueError("Callback must be a callable function")
        
        # Optional: validate callback signature
        if "validate_signature" in params and params["validate_signature"]:
            import inspect
            sig = inspect.signature(callback)
            expected_params = {"original_text", "entity_type", "confidence"}
            actual_params = set(sig.parameters.keys())
            
            if not expected_params.issubset(actual_params):
                missing = expected_params - actual_params
                raise ValueError(f"Callback missing required parameters: {missing}")
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get a parameter value with optional default."""
        return (self.parameters or {}).get(key, default)
    
    def with_parameters(self, **new_params: Any) -> "Strategy":
        """Create a new Strategy with updated parameters."""
        merged_params = {**(self.parameters or {}), **new_params}
        return Strategy(kind=self.kind, parameters=merged_params)


# Predefined common strategies for convenience
DEFAULT_REDACT = Strategy(StrategyKind.REDACT)
PHONE_TEMPLATE = Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"})
EMAIL_TEMPLATE = Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL]"})
SSN_PARTIAL = Strategy(StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end"})
HASH_SHA256 = Strategy(StrategyKind.HASH, {"algorithm": "sha256", "truncate": 8})