"""Policy system for defining masking rules and behaviors."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .strategies import DEFAULT_REDACT, Strategy, StrategyKind


class PrivacyLevel(Enum):
    """Privacy level enumeration for policies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class MaskingPolicy:
    """
    Configuration for how different types of PII entities should be masked.

    This policy defines default behavior and entity-specific overrides for
    masking operations, along with confidence thresholds and locale settings.

    Attributes:
        default_strategy: Strategy to use for entities not in per_entity map
        per_entity: Entity type to strategy mappings for specific overrides
        thresholds: Confidence thresholds per entity type (0.0-1.0)
        locale: Language/locale for recognition (e.g., 'en', 'es', 'fr')
        seed: Optional seed for deterministic operations (hashing, surrogates)
        custom_callbacks: Optional custom transformation functions
        allow_list: Entity values to never mask (exact matches)
        deny_list: Entity values to always mask regardless of confidence
        context_rules: Context-specific masking rules
        min_entity_length: Minimum length for entities to be considered

    Examples:
        >>> # Basic policy with template strategies
        >>> policy = MaskingPolicy(
        ...     default_strategy=DEFAULT_REDACT,
        ...     per_entity={
        ...         "PHONE_NUMBER": Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"}),
        ...         "EMAIL_ADDRESS": Strategy(StrategyKind.PARTIAL, {"visible_chars": 3, "position": "start"})
        ...     },
        ...     thresholds={"PHONE_NUMBER": 0.8, "EMAIL_ADDRESS": 0.7}
        ... )

        >>> # Policy with deterministic operations
        >>> policy = MaskingPolicy(
        ...     default_strategy=Strategy(StrategyKind.HASH, {"algorithm": "sha256"}),
        ...     seed="my-application-seed",
        ...     locale="en"
        ... )
    """

    default_strategy: Strategy = field(default=DEFAULT_REDACT)
    per_entity: dict[str, Strategy] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    locale: str = field(default="en")
    seed: Optional[str] = field(default=None)
    custom_callbacks: Optional[dict[str, Callable[[str, str, float], str]]] = field(
        default=None
    )
    allow_list: set[str] = field(default_factory=set)
    deny_list: set[str] = field(default_factory=set)
    context_rules: dict[str, dict[str, Any]] = field(default_factory=dict)
    min_entity_length: int = field(default=1)
    privacy_level: PrivacyLevel = field(default=PrivacyLevel.MEDIUM)

    def __post_init__(self) -> None:
        """Validate policy configuration after initialization."""
        self._validate_privacy_level()
        self._validate_thresholds()
        self._validate_locale()
        self._validate_seed()
        self._validate_entity_length()
        self._validate_context_rules()
        self._validate_callbacks()

    def _validate_privacy_level(self) -> None:
        """Validate and convert privacy level from string if needed."""
        if isinstance(self.privacy_level, str):
            try:
                # Convert string to enum
                object.__setattr__(self, 'privacy_level', PrivacyLevel(self.privacy_level))
            except ValueError:
                # Invalid string, use default
                object.__setattr__(self, 'privacy_level', PrivacyLevel.MEDIUM)
        elif not isinstance(self.privacy_level, PrivacyLevel):
            # Neither string nor enum, use default
            object.__setattr__(self, 'privacy_level', PrivacyLevel.MEDIUM)

    def _validate_thresholds(self) -> None:
        """Validate confidence threshold values."""
        for entity_type, threshold in self.thresholds.items():
            if not isinstance(threshold, (int, float)):
                raise ValueError(f"Threshold for {entity_type} must be a number")

            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    f"Threshold for {entity_type} must be between 0.0 and 1.0, got {threshold}"
                )

    def _validate_locale(self) -> None:
        """Validate locale format."""
        if not isinstance(self.locale, str):
            raise ValueError("Locale must be a string")

        # Basic locale format validation (language or language-country)
        locale_pattern = r"^[a-z]{2}(-[A-Z]{2})?$"
        if not re.match(locale_pattern, self.locale):
            raise ValueError(
                f"Locale must follow format 'xx' or 'xx-YY', got '{self.locale}'"
            )

    def _validate_seed(self) -> None:
        """Validate seed configuration."""
        if self.seed is not None and not isinstance(self.seed, str):
            raise ValueError("Seed must be a string or None")

    def _validate_entity_length(self) -> None:
        """Validate minimum entity length."""
        if not isinstance(self.min_entity_length, int) or self.min_entity_length < 0:
            raise ValueError("min_entity_length must be a non-negative integer")

    def _validate_context_rules(self) -> None:
        """Validate context rule configuration."""
        valid_contexts = {"heading", "table", "footer", "header", "list", "paragraph"}

        for context, rules in self.context_rules.items():
            if context not in valid_contexts:
                raise ValueError(
                    f"Unknown context type '{context}', valid contexts: {valid_contexts}"
                )

            if not isinstance(rules, dict):
                raise ValueError(f"Context rules for '{context}' must be a dictionary")

            # Validate rule keys
            valid_rule_keys = {
                "strategy",
                "threshold",
                "enabled",
                "strategy_overrides",
                "threshold_overrides",
            }
            for rule_key in rules.keys():
                if rule_key not in valid_rule_keys:
                    raise ValueError(
                        f"Unknown rule key '{rule_key}', valid keys: {valid_rule_keys}"
                    )

    def _validate_callbacks(self) -> None:
        """Validate custom callback functions."""
        if self.custom_callbacks is None:
            return

        if not isinstance(self.custom_callbacks, dict):
            raise ValueError("custom_callbacks must be a dictionary or None")

        for entity_type, callback in self.custom_callbacks.items():
            if not callable(callback):
                raise ValueError(f"Callback for '{entity_type}' must be callable")

            # Check callback signature
            import inspect

            sig = inspect.signature(callback)
            expected_params = {"original_text", "entity_type", "confidence"}
            actual_params = set(sig.parameters.keys())

            if not expected_params.issubset(actual_params):
                missing = expected_params - actual_params
                raise ValueError(
                    f"Callback for '{entity_type}' missing parameters: {missing}"
                )

    def get_strategy_for_entity(
        self, entity_type: str, context: Optional[str] = None
    ) -> Strategy:
        """
        Get the appropriate strategy for a given entity type and context.

        Args:
            entity_type: The type of entity (e.g., 'PHONE_NUMBER', 'EMAIL_ADDRESS')
            context: Optional context where the entity appears (e.g., 'heading', 'table')

        Returns:
            The strategy to use for masking this entity type
        """
        # Check context-specific rules first
        if context and context in self.context_rules:
            context_rule = self.context_rules[context]
            if not context_rule.get("enabled", True):
                # Context disabled - return a no-op strategy that preserves original text
                return Strategy(
                    StrategyKind.REDACT, {"redact_char": "*", "preserve_length": False}
                )

            if "strategy" in context_rule:
                strategy = context_rule["strategy"]
                if isinstance(strategy, Strategy):
                    return strategy

        # Check entity-specific strategy
        if entity_type in self.per_entity:
            return self.per_entity[entity_type]

        # Fall back to default strategy
        return self.default_strategy

    def get_threshold_for_entity(
        self, entity_type: str, context: Optional[str] = None
    ) -> float:
        """
        Get the confidence threshold for a given entity type and context.

        Args:
            entity_type: The type of entity
            context: Optional context where the entity appears

        Returns:
            The confidence threshold (0.0-1.0) for this entity type
        """
        # Check context-specific threshold
        if context and context in self.context_rules:
            context_rule = self.context_rules[context]
            if "threshold" in context_rule:
                threshold = context_rule["threshold"]
                if isinstance(threshold, (int, float)):
                    return float(threshold)

        # Check entity-specific threshold
        if entity_type in self.thresholds:
            return self.thresholds[entity_type]

        # Default threshold
        return 0.5

    def should_mask_entity(
        self,
        original_text: str,
        entity_type: str,
        confidence: float,
        context: Optional[str] = None,
    ) -> bool:
        """
        Determine if an entity should be masked based on policy rules.

        Args:
            original_text: The original text of the entity
            entity_type: The type of entity
            confidence: The confidence score (0.0-1.0)
            context: Optional context where the entity appears

        Returns:
            True if the entity should be masked, False otherwise
        """
        # Check deny list first (always mask)
        if original_text in self.deny_list:
            return True

        # Check allow list (never mask)
        if original_text in self.allow_list:
            return False

        # Check minimum entity length
        if len(original_text) < self.min_entity_length:
            return False

        # Check confidence threshold
        threshold = self.get_threshold_for_entity(entity_type, context)
        if confidence < threshold:
            return False

        # Check context rules
        if context and context in self.context_rules:
            context_rule = self.context_rules[context]
            if not context_rule.get("enabled", True):
                return False

        return True

    def get_custom_callback(
        self, entity_type: str
    ) -> Optional[Callable[[str, str, float], str]]:
        """Get custom callback function for an entity type if available."""
        if self.custom_callbacks is None:
            return None
        return self.custom_callbacks.get(entity_type)

    def with_entity_strategy(
        self, entity_type: str, strategy: Strategy
    ) -> "MaskingPolicy":
        """Create a new policy with an additional entity strategy."""
        new_per_entity = {**self.per_entity, entity_type: strategy}
        return MaskingPolicy(
            default_strategy=self.default_strategy,
            per_entity=new_per_entity,
            thresholds=self.thresholds,
            locale=self.locale,
            seed=self.seed,
            custom_callbacks=self.custom_callbacks,
            allow_list=self.allow_list,
            deny_list=self.deny_list,
            context_rules=self.context_rules,
            min_entity_length=self.min_entity_length,
            privacy_level=self.privacy_level,
        )

    def with_threshold(self, entity_type: str, threshold: float) -> "MaskingPolicy":
        """Create a new policy with an updated threshold for an entity type."""
        new_thresholds = {**self.thresholds, entity_type: threshold}
        return MaskingPolicy(
            default_strategy=self.default_strategy,
            per_entity=self.per_entity,
            thresholds=new_thresholds,
            locale=self.locale,
            seed=self.seed,
            custom_callbacks=self.custom_callbacks,
            allow_list=self.allow_list,
            deny_list=self.deny_list,
            context_rules=self.context_rules,
            min_entity_length=self.min_entity_length,
            privacy_level=self.privacy_level,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert policy to dictionary for serialization."""
        return {
            "default_strategy": {
                "kind": self.default_strategy.kind.value,
                "parameters": self.default_strategy.parameters,
            },
            "per_entity": {
                entity_type: {
                    "kind": strategy.kind.value,
                    "parameters": strategy.parameters,
                }
                for entity_type, strategy in self.per_entity.items()
            },
            "thresholds": dict(self.thresholds),
            "locale": self.locale,
            "seed": self.seed,
            "allow_list": list(self.allow_list),
            "deny_list": list(self.deny_list),
            "context_rules": dict(self.context_rules),
            "min_entity_length": self.min_entity_length,
            "privacy_level": self.privacy_level.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaskingPolicy":
        """Create policy from dictionary representation."""
        # Convert default strategy
        default_strategy_data = data.get(
            "default_strategy", {"kind": "redact", "parameters": {}}
        )
        default_strategy = Strategy(
            kind=StrategyKind(default_strategy_data["kind"]),
            parameters=default_strategy_data.get("parameters", {}),
        )

        # Convert per-entity strategies
        per_entity = {}
        for entity_type, strategy_data in data.get("per_entity", {}).items():
            per_entity[entity_type] = Strategy(
                kind=StrategyKind(strategy_data["kind"]),
                parameters=strategy_data.get("parameters", {}),
            )

        # Convert privacy level
        privacy_level = PrivacyLevel.MEDIUM  # default
        if "privacy_level" in data:
            try:
                privacy_level = PrivacyLevel(data["privacy_level"])
            except ValueError:
                privacy_level = PrivacyLevel.MEDIUM

        return cls(
            default_strategy=default_strategy,
            per_entity=per_entity,
            thresholds=data.get("thresholds", {}),
            locale=data.get("locale", "en"),
            seed=data.get("seed"),
            allow_list=set(data.get("allow_list", [])),
            deny_list=set(data.get("deny_list", [])),
            context_rules=data.get("context_rules", {}),
            min_entity_length=data.get("min_entity_length", 1),
            privacy_level=privacy_level,
        )


# Predefined common policies for convenience
CONSERVATIVE_POLICY = MaskingPolicy(
    default_strategy=DEFAULT_REDACT,
    thresholds={
        "PHONE_NUMBER": 0.9,
        "EMAIL_ADDRESS": 0.9,
        "CREDIT_CARD": 0.8,
        "US_SSN": 0.9,
        "PERSON": 0.8,
    },
)

TEMPLATE_POLICY = MaskingPolicy(
    per_entity={
        "PHONE_NUMBER": Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"}),
        "EMAIL_ADDRESS": Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL]"}),
        "CREDIT_CARD": Strategy(StrategyKind.TEMPLATE, {"template": "[CREDIT_CARD]"}),
        "US_SSN": Strategy(StrategyKind.TEMPLATE, {"template": "[SSN]"}),
        "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[NAME]"}),
    }
)

PARTIAL_POLICY = MaskingPolicy(
    per_entity={
        "PHONE_NUMBER": Strategy(
            StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end"}
        ),
        "EMAIL_ADDRESS": Strategy(
            StrategyKind.PARTIAL, {"visible_chars": 3, "position": "start"}
        ),
        "CREDIT_CARD": Strategy(
            StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end"}
        ),
        "US_SSN": Strategy(
            StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end"}
        ),
    }
)

# Enhanced policies for new features
FORMAT_AWARE_TEMPLATE_POLICY = MaskingPolicy(
    per_entity={
        "PHONE_NUMBER": Strategy(
            StrategyKind.TEMPLATE, {"auto_generate": True, "preserve_format": True}
        ),
        "EMAIL_ADDRESS": Strategy(
            StrategyKind.TEMPLATE, {"auto_generate": True, "preserve_format": True}
        ),
        "US_SSN": Strategy(
            StrategyKind.TEMPLATE, {"template": "XXX-XX-XXXX", "preserve_format": True}
        ),
        "CREDIT_CARD": Strategy(
            StrategyKind.TEMPLATE,
            {"template": "XXXX-XXXX-XXXX-XXXX", "preserve_format": True},
        ),
    }
)

FORMAT_AWARE_PARTIAL_POLICY = MaskingPolicy(
    per_entity={
        "PHONE_NUMBER": Strategy(
            StrategyKind.PARTIAL,
            {
                "visible_chars": 4,
                "position": "end",
                "format_aware": True,
                "preserve_delimiters": True,
                "deterministic": True,
            },
        ),
        "EMAIL_ADDRESS": Strategy(
            StrategyKind.PARTIAL,
            {
                "visible_chars": 3,
                "position": "start",
                "format_aware": True,
                "preserve_delimiters": True,
                "deterministic": True,
            },
        ),
        "CREDIT_CARD": Strategy(
            StrategyKind.PARTIAL,
            {
                "visible_chars": 4,
                "position": "end",
                "format_aware": True,
                "preserve_delimiters": True,
                "deterministic": True,
            },
        ),
        "US_SSN": Strategy(
            StrategyKind.PARTIAL,
            {
                "visible_chars": 4,
                "position": "end",
                "format_aware": True,
                "preserve_delimiters": True,
                "deterministic": True,
            },
        ),
    }
)

DETERMINISTIC_HASH_POLICY = MaskingPolicy(
    default_strategy=Strategy(
        StrategyKind.HASH,
        {
            "algorithm": "sha256",
            "truncate": 8,
            "format_output": "hex",
            "consistent_length": True,
            "per_entity_salt": {
                "PHONE_NUMBER": "phone_salt_v1",
                "EMAIL_ADDRESS": "email_salt_v1",
                "CREDIT_CARD": "cc_salt_v1",
                "US_SSN": "ssn_salt_v1",
                "PERSON": "name_salt_v1",
                "default": "default_salt_v1",
            },
        },
    ),
    seed="deterministic-hash-policy-v1",
)

MIXED_STRATEGY_POLICY = MaskingPolicy(
    per_entity={
        "PHONE_NUMBER": Strategy(
            StrategyKind.PARTIAL,
            {
                "visible_chars": 4,
                "position": "end",
                "format_aware": True,
                "preserve_delimiters": True,
            },
        ),
        "EMAIL_ADDRESS": Strategy(
            StrategyKind.TEMPLATE, {"auto_generate": True, "preserve_format": True}
        ),
        "CREDIT_CARD": Strategy(
            StrategyKind.HASH,
            {
                "algorithm": "sha256",
                "truncate": 12,
                "prefix": "CC_",
                "per_entity_salt": {"CREDIT_CARD": "cc_secure_v1"},
            },
        ),
        "US_SSN": Strategy(StrategyKind.TEMPLATE, {"template": "XXX-XX-XXXX"}),
        "PERSON": Strategy(
            StrategyKind.HASH,
            {
                "algorithm": "sha256",
                "truncate": 8,
                "prefix": "NAME_",
                "per_entity_salt": {"PERSON": "name_secure_v1"},
            },
        ),
    },
    thresholds={
        "PHONE_NUMBER": 0.8,
        "EMAIL_ADDRESS": 0.7,
        "CREDIT_CARD": 0.9,
        "US_SSN": 0.9,
        "PERSON": 0.75,
    },
)
