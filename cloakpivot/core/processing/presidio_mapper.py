"""Strategy to Presidio OperatorConfig mapping functionality."""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from presidio_anonymizer.entities import OperatorConfig
else:
    import signal

    def timeout_handler(_signum, _frame):
        raise TimeoutError("Import timed out")

    # Try to import with timeout
    old_handler = (
        signal.signal(signal.SIGALRM, timeout_handler) if hasattr(signal, "SIGALRM") else None
    )
    if old_handler is not None:
        signal.alarm(2)  # 2 second timeout

    try:
        from presidio_anonymizer.entities import OperatorConfig

        if old_handler is not None:
            signal.alarm(0)  # Cancel alarm
            signal.signal(signal.SIGALRM, old_handler)
    except (ImportError, TimeoutError):
        if old_handler is not None:
            signal.alarm(0)  # Cancel alarm
            signal.signal(signal.SIGALRM, old_handler)

        # Create a mock OperatorConfig for when Presidio is not available
        class OperatorConfig:
            def __init__(self, operator_name: str, params: dict[str, Any] = None):
                self.operator_name = operator_name
                self.params = params or {}


from ..policies.policies import MaskingPolicy
from ..types.strategies import Strategy, StrategyKind

logger = logging.getLogger(__name__)


class StrategyToOperatorMapper:
    """Maps CloakPivot strategies to Presidio operators.

    This class provides the mapping layer between CloakPivot's Strategy objects
    and Presidio's OperatorConfig objects, enabling seamless integration with
    Presidio's AnonymizerEngine while preserving all CloakPivot functionality.

    The mapper ensures that all Strategy objects can be converted to valid
    OperatorConfig objects, with appropriate parameter translation and fallback
    handling for unsupported parameter combinations.

    Examples:
        >>> mapper = StrategyToOperatorMapper()
        >>> strategy = Strategy(StrategyKind.REDACT, {"redact_char": "#"})
        >>> operator = mapper.strategy_to_operator(strategy)
        >>> print(operator.operator_name)  # "redact"

        >>> policy = MaskingPolicy(per_entity={
        ...     "EMAIL_ADDRESS": Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL]"})
        ... })
        >>> operators = mapper.policy_to_operators(policy)
        >>> print(operators["EMAIL_ADDRESS"].operator_name)  # "replace"
    """

    def __init__(self) -> None:
        """Initialize the mapper with strategy mapping functions."""
        self._strategy_mapping: dict[StrategyKind, Callable[[Strategy], OperatorConfig]] = {
            StrategyKind.REDACT: self._map_redact_strategy,
            StrategyKind.TEMPLATE: self._map_template_strategy,
            StrategyKind.HASH: self._map_hash_strategy,
            StrategyKind.PARTIAL: self._map_partial_strategy,
            StrategyKind.SURROGATE: self._map_surrogate_strategy,
            StrategyKind.CUSTOM: self._map_custom_strategy,
        }
        # Cache for strategy to operator mappings (LRU cache with max 128 entries)
        self._operator_cache: dict[tuple, OperatorConfig] = {}
        self._cache_order: list[tuple] = []
        self._max_cache_size = 128

    def strategy_to_operator(self, strategy: Strategy) -> OperatorConfig:
        """Convert a single Strategy to OperatorConfig with caching.

        Args:
            strategy: The CloakPivot Strategy to convert

        Returns:
            A Presidio OperatorConfig object

        Raises:
            ValueError: If strategy kind is not supported
        """
        # Create cache key from strategy (excluding non-hashable elements)
        cache_key = self._create_cache_key(strategy)
        
        # Check cache first
        if cache_key in self._operator_cache:
            # Move to end for LRU
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return self._operator_cache[cache_key]

        if strategy.kind not in self._strategy_mapping:
            logger.error(f"Unsupported strategy kind: {strategy.kind}")
            # Fallback to redaction for unsupported strategies
            fallback = OperatorConfig("redact", {"redact_char": "*"})
            self._cache_operator(cache_key, fallback)
            return fallback

        try:
            operator = self._strategy_mapping[strategy.kind](strategy)
            self._cache_operator(cache_key, operator)
            return operator
        except Exception as e:
            logger.warning(f"Failed to map strategy {strategy.kind}: {e}. Using fallback.")
            # Fallback to redaction with error logging
            fallback = OperatorConfig("redact", {"redact_char": "*"})
            self._cache_operator(cache_key, fallback)
            return fallback

    def policy_to_operators(self, policy: MaskingPolicy) -> dict[str, OperatorConfig]:
        """Convert a MaskingPolicy to operator dictionary.

        Maps all entity-specific strategies in the policy to their corresponding
        Presidio operators, with the default strategy applied to entities not
        explicitly configured.

        Args:
            policy: The MaskingPolicy to convert

        Returns:
            Dictionary mapping entity types to OperatorConfig objects
        """
        operators: dict[str, OperatorConfig] = {}

        # Map per-entity strategies
        for entity_type, strategy in policy.per_entity.items():
            try:
                operators[entity_type] = self.strategy_to_operator(strategy)
            except Exception as e:
                logger.warning(f"Failed to map strategy for {entity_type}: {e}")
                # Use default strategy as fallback
                operators[entity_type] = self.strategy_to_operator(policy.default_strategy)

        # Note: Default strategy is not included in the operators dict as Presidio
        # will use the default_operator parameter in AnonymizerEngine.anonymize()

        return operators

    def _map_redact_strategy(self, strategy: Strategy) -> OperatorConfig:
        """Map REDACT strategy to Presidio redact operator."""
        params = strategy.parameters or {}

        # Map redact_char parameter - support both 'char' and 'redact_char'
        redact_char = params.get("char", params.get("redact_char", "*"))

        operator_params = {"redact_char": redact_char}

        # Handle preserve_length parameter if present
        if "preserve_length" in params and not params.get("preserve_length", True):
            # If preserve_length is False, we could potentially use a different approach
            # but Presidio's redact operator always preserves length, so we log this
            logger.info("preserve_length=False not supported by Presidio redact operator")

        return OperatorConfig("redact", operator_params)

    def _map_template_strategy(self, strategy: Strategy) -> OperatorConfig:
        """Map TEMPLATE strategy to Presidio replace operator."""
        params = strategy.parameters or {}

        # Handle auto_generate template creation
        if params.get("auto_generate", False):
            # For auto-generate, we create a generic template based on common patterns
            template = self._generate_auto_template(params)
        elif "template" in params:
            template = params["template"]
        else:
            # Fallback template - this shouldn't happen with validated strategies
            template = "[REDACTED]"

        operator_params = {"new_value": template}

        # Log unsupported preserve_format parameter
        if params.get("preserve_format", False):
            logger.info("preserve_format parameter requires custom implementation")

        return OperatorConfig("replace", operator_params)

    def _map_hash_strategy(self, strategy: Strategy) -> OperatorConfig:
        """Map HASH strategy to Presidio hash operator."""
        params = strategy.parameters or {}

        # Map hash algorithm
        algorithm = params.get("algorithm", "sha256")
        hash_type = algorithm  # Presidio uses hash_type parameter name

        operator_params = {"hash_type": hash_type}

        # Map salt parameter
        if "salt" in params:
            operator_params["salt"] = params["salt"]

        # Handle per_entity_salt by using the default salt value
        if "per_entity_salt" in params:
            per_entity_salt = params["per_entity_salt"]
            if isinstance(per_entity_salt, dict):
                # Use the default salt or first available salt
                salt_value = per_entity_salt.get("default") or next(
                    iter(per_entity_salt.values()), ""
                )
                if salt_value:
                    operator_params["salt"] = salt_value
            logger.info("per_entity_salt simplified to single salt value for Presidio")

        # Log unsupported parameters
        unsupported = [
            "truncate",
            "prefix",
            "format_output",
            "consistent_length",
            "preserve_format_structure",
        ]
        for param in unsupported:
            if param in params:
                logger.info(f"Hash parameter '{param}' not directly supported by Presidio")

        return OperatorConfig("hash", operator_params)

    def _map_partial_strategy(self, strategy: Strategy) -> OperatorConfig:
        """Map PARTIAL strategy to Presidio mask operator."""
        params = strategy.parameters or {}

        visible_chars = params.get("visible_chars", 4)
        position = params.get("position", "end")
        mask_char = params.get("mask_char", "*")

        # Convert CloakPivot position to Presidio parameters
        if position == "start":
            # Show first N characters, mask the rest
            chars_to_mask = -1  # Mask from position N to end
            from_end = False
        elif position == "end":
            # Show last N characters, mask from beginning
            chars_to_mask = visible_chars
            from_end = True
        elif position == "middle":
            # For middle position, default to end behavior with warning
            logger.warning("Middle position not directly supported, using end position")
            chars_to_mask = visible_chars
            from_end = True
        elif position == "random":
            # Random position not supported, default to end
            logger.warning("Random position not supported, using end position")
            chars_to_mask = visible_chars
            from_end = True
        else:
            chars_to_mask = visible_chars
            from_end = True

        operator_params = {
            "masking_char": mask_char,
            "chars_to_mask": chars_to_mask,
            "from_end": from_end,
        }

        # Log unsupported parameters
        unsupported = [
            "min_length",
            "format_aware",
            "preserve_delimiters",
            "deterministic",
        ]
        for param in unsupported:
            if param in params:
                logger.info(f"Partial parameter '{param}' not directly supported by Presidio")

        return OperatorConfig("mask", operator_params)

    def _map_surrogate_strategy(self, strategy: Strategy) -> OperatorConfig:
        """Map SURROGATE strategy to Presidio replace operator with fake values."""
        params = strategy.parameters or {}

        # Generate appropriate fake value based on format_type
        format_type = params.get("format_type", "custom")
        fake_value = self._generate_surrogate_value(format_type, params)

        operator_params = {"new_value": fake_value}

        # Log note about limited faker integration
        logger.info(
            "Surrogate strategy mapped to static replacement. "
            "For dynamic faker integration, use custom operator."
        )

        return OperatorConfig("replace", operator_params)

    def _map_custom_strategy(self, strategy: Strategy) -> OperatorConfig:
        """Map CUSTOM strategy to Presidio custom operator."""
        params = strategy.parameters or {}

        if "callback" not in params:
            logger.error("Custom strategy missing required callback parameter")
            return OperatorConfig("redact", {"redact_char": "*"})

        callback = params["callback"]
        if not callable(callback):
            logger.error("Custom strategy callback is not callable")
            return OperatorConfig("redact", {"redact_char": "*"})

        # Presidio custom operator expects a lambda function
        operator_params = {"lambda": callback}

        return OperatorConfig("custom", operator_params)

    def _generate_auto_template(self, params: dict[str, Any]) -> str:
        """Generate automatic template for template strategy."""
        # Simple template generation - could be enhanced with entity type detection
        return "[MASKED]"

    def _generate_surrogate_value(self, format_type: str, params: dict[str, Any]) -> str:
        """Generate surrogate value based on format type."""
        # Static surrogate values - real implementation would use faker
        surrogate_map = {
            "phone": "(555) 123-4567",
            "ssn": "123-45-6789",
            "credit_card": "4111-1111-1111-1111",
            "email": "user@example.com",
            "name": "John Doe",
            "address": "123 Main St, City, State 12345",
            "custom": "[SURROGATE]",
        }

        # Check for custom dictionary
        if "dictionary" in params and params["dictionary"]:
            dictionary = params["dictionary"]
            if isinstance(dictionary, list | tuple) and dictionary:
                return str(dictionary[0])  # Use first item as static replacement

        # Check for custom pattern
        if "pattern" in params:
            return f"[PATTERN:{params['pattern']}]"

        return surrogate_map.get(format_type, "[SURROGATE]")

    def _create_cache_key(self, strategy: Strategy) -> tuple:
        """Create a hashable cache key from strategy.
        
        Args:
            strategy: Strategy to create key for
            
        Returns:
            Tuple representing the strategy for caching
        """
        # Convert parameters to a hashable form
        params_key = ()
        if strategy.parameters:
            # Sort parameters and handle non-hashable values
            sorted_params = []
            for key, value in sorted(strategy.parameters.items()):
                if callable(value):
                    # For callbacks, use their string representation
                    sorted_params.append((key, str(value)))
                elif isinstance(value, dict | list):
                    # Convert containers to tuples
                    sorted_params.append((key, str(sorted(value.items()) if isinstance(value, dict) else value)))
                else:
                    sorted_params.append((key, value))
            params_key = tuple(sorted_params)
        
        return (strategy.kind, params_key)
    
    def _cache_operator(self, cache_key: tuple, operator: OperatorConfig) -> None:
        """Cache an operator config with LRU eviction.
        
        Args:
            cache_key: Cache key for the operator
            operator: OperatorConfig to cache
        """
        # Remove oldest entries if cache is full
        while len(self._operator_cache) >= self._max_cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._operator_cache[oldest_key]
        
        # Add to cache
        self._operator_cache[cache_key] = operator
        self._cache_order.append(cache_key)
