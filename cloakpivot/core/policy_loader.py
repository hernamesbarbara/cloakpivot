"""Policy loading system with inheritance and composition support."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from .policies import MaskingPolicy
from .strategies import Strategy, StrategyKind


class PolicyValidationError(Exception):
    """Raised when policy validation fails."""

    pass


class PolicyInheritanceError(Exception):
    """Raised when policy inheritance cannot be resolved."""

    pass


@dataclass
class PolicyLoadContext:
    """Context for loading policies, tracks inheritance chain."""

    current_file: Path
    base_path: Path
    inheritance_chain: list[Path]

    def derive_path(self, relative_path: str) -> Path:
        """Resolve relative path from current policy file location."""
        if Path(relative_path).is_absolute():
            return Path(relative_path)
        return (self.current_file.parent / relative_path).resolve()


class StrategyConfig(BaseModel):
    """Pydantic model for strategy configuration validation."""

    kind: str = Field(..., description="Strategy type")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Strategy parameters"
    )

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, v: Any) -> Any:
        """Validate strategy kind is supported."""
        try:
            StrategyKind(v)
        except ValueError as e:
            valid_kinds = [k.value for k in StrategyKind]
            raise ValueError(
                f"Invalid strategy kind '{v}'. Valid kinds: {valid_kinds}"
            ) from e
        return v


class EntityConfig(BaseModel):
    """Pydantic model for per-entity configuration."""

    kind: Optional[str] = Field(None, description="Strategy type override")
    parameters: Optional[dict[str, Any]] = Field(
        None, description="Strategy parameters override"
    )
    threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence threshold"
    )
    enabled: Optional[bool] = Field(
        None, description="Whether entity recognition is enabled"
    )

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, v: Any) -> Any:
        """Validate strategy kind if provided."""
        if v is not None:
            try:
                StrategyKind(v)
            except ValueError as e:
                valid_kinds = [k.value for k in StrategyKind]
                raise ValueError(
                    f"Invalid strategy kind '{v}'. Valid kinds: {valid_kinds}"
                ) from e
        return v


class LocaleConfig(BaseModel):
    """Pydantic model for locale-specific configuration."""

    recognizers: Optional[list[str]] = Field(
        None, description="Custom recognizers for locale"
    )
    entity_overrides: Optional[dict[str, EntityConfig]] = Field(
        None, description="Entity overrides for locale"
    )


class ContextRuleConfig(BaseModel):
    """Pydantic model for context-specific rules."""

    enabled: Optional[bool] = Field(
        None, description="Whether context is enabled for masking"
    )
    strategy_overrides: Optional[dict[str, EntityConfig]] = Field(
        None, description="Strategy overrides for context"
    )
    threshold_overrides: Optional[dict[str, float]] = Field(
        None, description="Threshold overrides for context"
    )


class AllowListItem(BaseModel):
    """Pydantic model for allow list items."""

    pattern: Optional[str] = Field(None, description="Regex pattern for matching")
    value: Optional[str] = Field(None, description="Exact value to match")

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: Any) -> Any:
        """Validate regex pattern if provided."""
        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}") from e
        return v


class PolicyCompositionConfig(BaseModel):
    """Pydantic model for policy composition settings."""

    merge_strategy: str = Field(
        "override", description="How to merge inherited policies"
    )
    validation_level: str = Field("strict", description="Validation strictness level")

    @field_validator("merge_strategy")
    @classmethod
    def validate_merge_strategy(cls, v: Any) -> Any:
        """Validate merge strategy."""
        valid_strategies = ["override", "merge", "strict"]
        if v not in valid_strategies:
            raise ValueError(
                f"Invalid merge strategy '{v}'. Valid strategies: {valid_strategies}"
            )
        return v

    @field_validator("validation_level")
    @classmethod
    def validate_validation_level(cls, v: Any) -> Any:
        """Validate validation level."""
        valid_levels = ["strict", "warn", "permissive"]
        if v not in valid_levels:
            raise ValueError(
                f"Invalid validation level '{v}'. Valid levels: {valid_levels}"
            )
        return v


class PolicyFileSchema(BaseModel):
    """Pydantic model for policy file schema validation."""

    version: Optional[str] = Field("1.0", description="Policy schema version")
    name: Optional[str] = Field(None, description="Policy name")
    description: Optional[str] = Field(None, description="Policy description")
    extends: Optional[Union[str, list[str]]] = Field(
        None, description="Base policy files to inherit from"
    )

    # Core configuration
    locale: Optional[str] = Field("en", description="Default locale")
    seed: Optional[str] = Field(None, description="Seed for deterministic operations")

    # Strategy configuration
    default_strategy: Optional[StrategyConfig] = Field(
        None, description="Default masking strategy"
    )
    per_entity: Optional[dict[str, EntityConfig]] = Field(
        None, description="Per-entity configurations"
    )

    # Thresholds and filtering
    thresholds: Optional[dict[str, float]] = Field(
        None, description="Per-entity confidence thresholds"
    )
    allow_list: Optional[list[Union[str, AllowListItem]]] = Field(
        None, description="Values to never mask"
    )
    deny_list: Optional[list[str]] = Field(None, description="Values to always mask")
    min_entity_length: Optional[int] = Field(
        1, ge=0, description="Minimum entity length to consider"
    )

    # Locale support
    locales: Optional[dict[str, LocaleConfig]] = Field(
        None, description="Locale-specific configurations"
    )

    # Context rules
    context_rules: Optional[dict[str, ContextRuleConfig]] = Field(
        None, description="Context-specific rules"
    )

    # Policy composition
    policy_composition: Optional[PolicyCompositionConfig] = Field(
        None, description="Policy composition settings"
    )

    @field_validator("locale")
    @classmethod
    def validate_locale(cls, v: Any) -> Any:
        """Validate locale format."""
        if v is not None:
            locale_pattern = r"^[a-z]{2}(-[A-Z]{2})?$"
            if not re.match(locale_pattern, v):
                raise ValueError(
                    f"Locale must follow format 'xx' or 'xx-YY', got '{v}'"
                )
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: Any) -> Any:
        """Validate version format."""
        if v is not None:
            version_pattern = r"^\d+\.\d+(\.\d+)?$"
            if not re.match(version_pattern, v):
                raise ValueError(
                    f"Version must follow format 'x.y' or 'x.y.z', got '{v}'"
                )
        return v


class PolicyLoader:
    """
    Advanced policy loader with inheritance, composition, and validation support.

    Features:
    - YAML policy file loading with schema validation
    - Policy inheritance from base templates
    - Composition and merging of multiple policies
    - Locale-specific configuration
    - Context-aware rules
    - Comprehensive validation with helpful error messages
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize policy loader.

        Args:
            base_path: Base directory for resolving relative policy paths
        """
        self.base_path = base_path or Path.cwd()
        self._policy_cache: dict[Path, PolicyFileSchema] = {}

    def load_policy(self, policy_path: Union[str, Path]) -> MaskingPolicy:
        """
        Load a masking policy from file with full inheritance support.

        Args:
            policy_path: Path to policy YAML file

        Returns:
            Fully resolved MaskingPolicy instance

        Raises:
            PolicyValidationError: If policy validation fails
            PolicyInheritanceError: If inheritance cannot be resolved
            FileNotFoundError: If policy file doesn't exist
        """
        policy_path = Path(policy_path)
        if not policy_path.is_absolute():
            policy_path = self.base_path / policy_path

        context = PolicyLoadContext(
            current_file=policy_path, base_path=self.base_path, inheritance_chain=[]
        )

        try:
            policy_schema = self._load_policy_file(policy_path, context)
            return self._schema_to_masking_policy(policy_schema)
        except ValidationError as e:
            raise PolicyValidationError(
                f"Policy validation failed for {policy_path}: {e}"
            ) from e
        except (PolicyValidationError, PolicyInheritanceError, FileNotFoundError):
            # Re-raise these exceptions without wrapping
            raise
        except Exception as e:
            raise PolicyValidationError(
                f"Failed to load policy {policy_path}: {e}"
            ) from e

    def _load_policy_file(
        self, policy_path: Path, context: PolicyLoadContext
    ) -> PolicyFileSchema:
        """
        Load and resolve a single policy file with full inheritance support.

        This method handles the complete policy loading pipeline including:
        - YAML parsing and schema validation
        - Circular dependency detection in inheritance chains
        - Recursive loading of base policies via 'extends' directive
        - Policy composition and merging using configured strategies
        - Caching of loaded policies for performance

        Args:
            policy_path: Path to the policy YAML file to load
            context: Loading context with inheritance chain and base path

        Returns:
            Fully resolved PolicyFileSchema with all inheritance applied

        Raises:
            PolicyInheritanceError: If circular inheritance is detected
            PolicyValidationError: If YAML parsing or schema validation fails
            FileNotFoundError: If policy file or inherited files don't exist
        """
        # Resolve path to compare consistently (handles symlinks, relative paths, etc.)
        resolved_policy_path = policy_path.resolve()
        resolved_chain = [p.resolve() for p in context.inheritance_chain]

        if resolved_policy_path in resolved_chain:
            chain_str = " -> ".join(
                str(p) for p in context.inheritance_chain + [policy_path]
            )
            raise PolicyInheritanceError(f"Circular inheritance detected: {chain_str}")

        if policy_path in self._policy_cache:
            return self._policy_cache[policy_path]

        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

        try:
            with open(policy_path, encoding="utf-8") as f:
                policy_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise PolicyValidationError(f"Invalid YAML in {policy_path}: {e}") from e

        # Validate schema
        try:
            policy_schema = PolicyFileSchema(**policy_data)
        except ValidationError as e:
            raise PolicyValidationError(
                f"Schema validation failed for {policy_path}: {e}"
            ) from e

        # Handle inheritance
        if policy_schema.extends:
            new_context = PolicyLoadContext(
                current_file=policy_path,
                base_path=context.base_path,
                inheritance_chain=context.inheritance_chain + [policy_path],
            )

            base_policies = []
            extends_list = (
                policy_schema.extends
                if isinstance(policy_schema.extends, list)
                else [policy_schema.extends]
            )

            for base_path_str in extends_list:
                base_path = new_context.derive_path(base_path_str)
                base_policy = self._load_policy_file(base_path, new_context)
                base_policies.append(base_policy)

            # Merge base policies with current policy
            policy_schema = self._merge_policies(base_policies + [policy_schema])

        self._policy_cache[policy_path] = policy_schema
        return policy_schema

    def _merge_policies(self, policies: list[PolicyFileSchema]) -> PolicyFileSchema:
        """
        Merge multiple policy schemas using intelligent composition strategies.

        This method implements sophisticated policy merging that preserves the semantic
        meaning of configurations while allowing for flexible inheritance patterns:

        - **Strategy Merging**: When merging per_entity configs, if only thresholds differ,
          the base strategy is preserved with the new threshold applied
        - **List Composition**: Allow/deny lists are merged by combining unique values
        - **Context Rules**: Context-specific rules are merged with child overriding parent
        - **Locale Support**: Locale configurations are merged with child locale settings
          taking precedence over parent settings

        Args:
            policies: List of policy schemas to merge, in inheritance order
                     (base policies first, child policies last)

        Returns:
            Single merged PolicyFileSchema with all inheritance applied

        Raises:
            ValueError: If policies list is empty or merge conflicts cannot be resolved
        """
        if not policies:
            raise ValueError("Cannot merge empty policy list")

        if len(policies) == 1:
            return policies[0]

        # Start with first policy as base
        result = policies[0].model_copy(deep=True)

        # Merge each subsequent policy
        for policy in policies[1:]:
            result = self._merge_two_policies(result, policy)

        return result

    def _merge_two_policies(
        self, base: PolicyFileSchema, override: PolicyFileSchema
    ) -> PolicyFileSchema:
        """Merge two policy schemas."""
        # Get composition settings from override policy
        composition_config = override.policy_composition or PolicyCompositionConfig(merge_strategy="override", validation_level="strict")

        # Create merged policy data
        merged_data = base.model_dump()
        override_data = override.model_dump()

        # Apply merge strategy
        if composition_config.merge_strategy == "override":
            # Override replaces base for all non-None values
            for key, value in override_data.items():
                if value is not None:
                    if key == "per_entity" and isinstance(value, dict):
                        # For per_entity, we need special handling to merge entity configs
                        base_entities = merged_data.get(key, {}) or {}
                        merged_entities = {**base_entities}  # Start with base entities

                        for entity_type, entity_config in value.items():
                            if entity_type in base_entities:
                                # Merge entity config with base
                                base_entity_config = base_entities[entity_type]
                                merged_entity_config = {**base_entity_config}

                                # Override specific fields from child policy
                                for field, field_value in entity_config.items():
                                    if field_value is not None:
                                        merged_entity_config[field] = field_value

                                merged_entities[entity_type] = merged_entity_config
                            else:
                                # New entity, just add it
                                merged_entities[entity_type] = entity_config

                        merged_data[key] = merged_entities
                    elif key in [
                        "thresholds",
                        "locales",
                        "context_rules",
                    ] and isinstance(value, dict):
                        # For other dict fields, merge keys
                        base_value = merged_data.get(key, {}) or {}
                        merged_data[key] = {**base_value, **value}
                    elif key in ["allow_list", "deny_list"] and isinstance(value, list):
                        # For list fields, extend
                        base_value = merged_data.get(key, []) or []
                        merged_data[key] = base_value + value
                    else:
                        merged_data[key] = value

        elif composition_config.merge_strategy == "strict":
            # Only allow overriding specific fields
            allowed_override_fields = {
                "name",
                "description",
                "per_entity",
                "thresholds",
                "allow_list",
                "deny_list",
            }
            for key, value in override_data.items():
                if value is not None and key in allowed_override_fields:
                    merged_data[key] = value

        # Note: "merge" strategy would require more complex field-by-field merging logic

        return PolicyFileSchema(**merged_data)

    def _schema_to_masking_policy(self, schema: PolicyFileSchema) -> MaskingPolicy:
        """Convert validated policy schema to MaskingPolicy instance."""
        # Convert default strategy
        default_strategy = None
        if schema.default_strategy:
            default_strategy = Strategy(
                kind=StrategyKind(schema.default_strategy.kind),
                parameters=schema.default_strategy.parameters,
            )

        # Convert per-entity strategies
        per_entity = {}
        if schema.per_entity:
            for entity_type, config in schema.per_entity.items():
                if config.kind:
                    per_entity[entity_type] = Strategy(
                        kind=StrategyKind(config.kind),
                        parameters=config.parameters or {},
                    )
                # Note: Entities with only threshold/enabled overrides will not have strategies
                # in per_entity dict, but their thresholds will be handled separately

        # Convert thresholds
        thresholds = {}
        if schema.thresholds:
            thresholds.update(schema.thresholds)

        # Add thresholds from per_entity configs
        if schema.per_entity:
            for entity_type, config in schema.per_entity.items():
                if config.threshold is not None:
                    thresholds[entity_type] = config.threshold

        # Convert allow/deny lists
        allow_list = set()
        deny_list = set()

        if schema.allow_list:
            for item in schema.allow_list:
                if isinstance(item, str):
                    allow_list.add(item)
                # Note: Pattern-based allow list would need special handling in the policy

        if schema.deny_list:
            deny_list.update(schema.deny_list)

        # Convert context rules
        context_rules: dict[str, Any] = {}
        if schema.context_rules:
            for context, rule_config in schema.context_rules.items():
                context_rule: dict[str, Any] = {"enabled": rule_config.enabled}

                if rule_config.strategy_overrides:
                    # Convert strategy overrides to Strategy objects
                    strategy_overrides = {}
                    for entity_type, config in rule_config.strategy_overrides.items():
                        if config.kind:
                            strategy_overrides[entity_type] = Strategy(
                                kind=StrategyKind(config.kind),
                                parameters=config.parameters or {},
                            )
                    if strategy_overrides:
                        context_rule["strategy_overrides"] = strategy_overrides

                if rule_config.threshold_overrides:
                    context_rule["threshold_overrides"] = (
                        rule_config.threshold_overrides
                    )

                context_rules[context] = context_rule

        return MaskingPolicy(
            default_strategy=default_strategy if default_strategy is not None else Strategy(kind=StrategyKind.REDACT),
            per_entity=per_entity,
            thresholds=thresholds,
            locale=schema.locale or "en",
            seed=schema.seed,
            allow_list=allow_list,
            deny_list=deny_list,
            context_rules=context_rules,
            min_entity_length=schema.min_entity_length or 1,
        )

    def validate_policy_file(self, policy_path: Union[str, Path]) -> list[str]:
        """
        Validate a policy file and return any validation errors.

        Args:
            policy_path: Path to policy file to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        try:
            self.load_policy(policy_path)
            return []
        except (PolicyValidationError, PolicyInheritanceError, FileNotFoundError) as e:
            return [str(e)]
        except Exception as e:
            return [f"Unexpected error: {e}"]
