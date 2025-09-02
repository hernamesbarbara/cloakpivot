"""Extensions to MaskingPolicy for plugin support."""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.policies import MaskingPolicy
from ..core.strategies import Strategy, StrategyKind

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PluginConfiguration:
    """Configuration for a plugin within a masking policy."""

    plugin_name: str
    plugin_type: str  # 'strategy' or 'recognizer'
    config: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0  # Higher priority plugins run first
    fallback_strategy: Optional[Strategy] = None

    def __post_init__(self) -> None:
        """Validate plugin configuration."""
        if not self.plugin_name:
            raise ValueError("Plugin name cannot be empty")
        if self.plugin_type not in ("strategy", "recognizer"):
            raise ValueError("Plugin type must be 'strategy' or 'recognizer'")
        if not isinstance(self.config, dict):
            raise ValueError("Plugin config must be a dictionary")


@dataclass(frozen=True)
class EnhancedMaskingPolicy(MaskingPolicy):
    """
    Enhanced MaskingPolicy with plugin support.

    This extends the base MaskingPolicy to support custom strategy and
    recognizer plugins while maintaining backward compatibility.

    Examples:
        >>> # Policy with custom strategy plugin
        >>> policy = EnhancedMaskingPolicy(
        ...     plugin_configurations={
        ...         "rot13_strategy": PluginConfiguration(
        ...             plugin_name="rot13_strategy",
        ...             plugin_type="strategy",
        ...             config={"reverse_on_unmask": True}
        ...         )
        ...     },
        ...     plugin_strategy_mapping={
        ...         "PHONE_NUMBER": "rot13_strategy"
        ...     }
        ... )

        >>> # Policy with custom recognizer plugin
        >>> policy = EnhancedMaskingPolicy(
        ...     plugin_configurations={
        ...         "custom_phone_recognizer": PluginConfiguration(
        ...             plugin_name="custom_phone_recognizer",
        ...             plugin_type="recognizer",
        ...             config={"country_codes": ["US", "CA"]}
        ...         )
        ...     },
        ...     enabled_recognizer_plugins=["custom_phone_recognizer"]
        ... )
    """

    # Plugin-specific fields
    plugin_configurations: dict[str, PluginConfiguration] = field(default_factory=dict)
    plugin_strategy_mapping: dict[str, str] = field(
        default_factory=dict
    )  # entity_type -> plugin_name
    enabled_strategy_plugins: list[str] = field(default_factory=list)
    enabled_recognizer_plugins: list[str] = field(default_factory=list)
    plugin_fallback_enabled: bool = field(default=True)
    plugin_error_handling: str = field(default="fallback")  # "fallback", "skip", "fail"

    def __post_init__(self) -> None:
        """Validate enhanced policy configuration."""
        # Run base validation first
        super().__post_init__()

        # Validate plugin configurations
        self._validate_plugin_configurations()
        self._validate_plugin_mappings()
        self._validate_error_handling()

    def _validate_plugin_configurations(self) -> None:
        """Validate plugin configuration entries."""
        for name, config in self.plugin_configurations.items():
            if not isinstance(config, PluginConfiguration):
                raise ValueError(
                    f"Plugin configuration for {name} must be PluginConfiguration instance"
                )

            # Validate plugin name matches key
            if config.plugin_name != name:
                raise ValueError(
                    f"Plugin name mismatch: key={name}, config.plugin_name={config.plugin_name}"
                )

    def _validate_plugin_mappings(self) -> None:
        """Validate plugin mapping references."""
        for _entity_type, plugin_name in self.plugin_strategy_mapping.items():
            if plugin_name not in self.plugin_configurations:
                raise ValueError(
                    f"Strategy mapping references unknown plugin: {plugin_name}"
                )

            config = self.plugin_configurations[plugin_name]
            if config.plugin_type != "strategy":
                raise ValueError(
                    f"Strategy mapping references non-strategy plugin: {plugin_name}"
                )

        # Validate enabled plugin lists
        for plugin_name in self.enabled_strategy_plugins:
            if plugin_name not in self.plugin_configurations:
                raise ValueError(
                    f"Enabled strategy plugins references unknown plugin: {plugin_name}"
                )

            config = self.plugin_configurations[plugin_name]
            if config.plugin_type != "strategy":
                raise ValueError(
                    f"Enabled strategy plugins references non-strategy plugin: {plugin_name}"
                )

        for plugin_name in self.enabled_recognizer_plugins:
            if plugin_name not in self.plugin_configurations:
                raise ValueError(
                    f"Enabled recognizer plugins references unknown plugin: {plugin_name}"
                )

            config = self.plugin_configurations[plugin_name]
            if config.plugin_type != "recognizer":
                raise ValueError(
                    f"Enabled recognizer plugins references non-recognizer plugin: {plugin_name}"
                )

    def _validate_error_handling(self) -> None:
        """Validate error handling configuration."""
        valid_modes = {"fallback", "skip", "fail"}
        if self.plugin_error_handling not in valid_modes:
            raise ValueError(f"plugin_error_handling must be one of {valid_modes}")

    def get_strategy_for_entity(
        self, entity_type: str, context: Optional[str] = None
    ) -> Strategy:
        """
        Get strategy for entity, checking plugin mappings first.

        Args:
            entity_type: The type of entity
            context: Optional context

        Returns:
            Strategy to use (plugin or traditional)
        """
        # Check for plugin strategy mapping first
        if entity_type in self.plugin_strategy_mapping:
            plugin_name = self.plugin_strategy_mapping[entity_type]
            plugin_config = self.plugin_configurations.get(plugin_name)

            if plugin_config and plugin_config.enabled:
                # Return a custom strategy that indicates plugin usage
                # Add a dummy callback to satisfy Strategy validation
                def plugin_callback(
                    original_text: str, entity_type: str, confidence: float
                ) -> str:
                    return f"PLUGIN_{plugin_name}[{original_text}]"

                return Strategy(
                    StrategyKind.CUSTOM,
                    {
                        "callback": plugin_callback,
                        "plugin_name": plugin_name,
                        "plugin_config": plugin_config.config,
                        "fallback_strategy": plugin_config.fallback_strategy,
                    },
                )

        # Fall back to base implementation
        return super().get_strategy_for_entity(entity_type, context)

    def get_plugin_configuration(
        self, plugin_name: str
    ) -> Optional[PluginConfiguration]:
        """Get configuration for a specific plugin."""
        return self.plugin_configurations.get(plugin_name)

    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled."""
        config = self.plugin_configurations.get(plugin_name)
        return config is not None and config.enabled

    def get_enabled_strategy_plugins(self) -> list[str]:
        """Get list of enabled strategy plugin names."""
        return [
            name
            for name in self.enabled_strategy_plugins
            if self.is_plugin_enabled(name)
        ]

    def get_enabled_recognizer_plugins(self) -> list[str]:
        """Get list of enabled recognizer plugin names."""
        return [
            name
            for name in self.enabled_recognizer_plugins
            if self.is_plugin_enabled(name)
        ]

    def get_strategy_plugins_by_priority(self) -> list[str]:
        """Get strategy plugins sorted by priority (highest first)."""
        enabled_plugins = self.get_enabled_strategy_plugins()

        return sorted(
            enabled_plugins,
            key=lambda name: self.plugin_configurations[name].priority,
            reverse=True,
        )

    def get_recognizer_plugins_by_priority(self) -> list[str]:
        """Get recognizer plugins sorted by priority (highest first)."""
        enabled_plugins = self.get_enabled_recognizer_plugins()

        return sorted(
            enabled_plugins,
            key=lambda name: self.plugin_configurations[name].priority,
            reverse=True,
        )

    def with_plugin_configuration(
        self, plugin_name: str, plugin_config: PluginConfiguration
    ) -> "EnhancedMaskingPolicy":
        """Create new policy with additional plugin configuration."""
        new_configs = {**self.plugin_configurations, plugin_name: plugin_config}

        return EnhancedMaskingPolicy(
            # Base MaskingPolicy fields
            default_strategy=self.default_strategy,
            per_entity=self.per_entity,
            thresholds=self.thresholds,
            locale=self.locale,
            seed=self.seed,
            custom_callbacks=self.custom_callbacks,
            allow_list=self.allow_list,
            deny_list=self.deny_list,
            context_rules=self.context_rules,
            min_entity_length=self.min_entity_length,
            # Enhanced fields
            plugin_configurations=new_configs,
            plugin_strategy_mapping=self.plugin_strategy_mapping,
            enabled_strategy_plugins=self.enabled_strategy_plugins,
            enabled_recognizer_plugins=self.enabled_recognizer_plugins,
            plugin_fallback_enabled=self.plugin_fallback_enabled,
            plugin_error_handling=self.plugin_error_handling,
        )

    def with_plugin_strategy_mapping(
        self, entity_type: str, plugin_name: str
    ) -> "EnhancedMaskingPolicy":
        """Create new policy with additional plugin strategy mapping."""
        new_mapping = {**self.plugin_strategy_mapping, entity_type: plugin_name}

        return EnhancedMaskingPolicy(
            # Base MaskingPolicy fields
            default_strategy=self.default_strategy,
            per_entity=self.per_entity,
            thresholds=self.thresholds,
            locale=self.locale,
            seed=self.seed,
            custom_callbacks=self.custom_callbacks,
            allow_list=self.allow_list,
            deny_list=self.deny_list,
            context_rules=self.context_rules,
            min_entity_length=self.min_entity_length,
            # Enhanced fields
            plugin_configurations=self.plugin_configurations,
            plugin_strategy_mapping=new_mapping,
            enabled_strategy_plugins=self.enabled_strategy_plugins,
            enabled_recognizer_plugins=self.enabled_recognizer_plugins,
            plugin_fallback_enabled=self.plugin_fallback_enabled,
            plugin_error_handling=self.plugin_error_handling,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert enhanced policy to dictionary."""
        base_dict = super().to_dict()

        # Add plugin-specific fields
        base_dict.update(
            {
                "plugin_configurations": {
                    name: {
                        "plugin_name": config.plugin_name,
                        "plugin_type": config.plugin_type,
                        "config": config.config,
                        "enabled": config.enabled,
                        "priority": config.priority,
                        "fallback_strategy": (
                            {
                                "kind": config.fallback_strategy.kind.value,
                                "parameters": config.fallback_strategy.parameters,
                            }
                            if config.fallback_strategy
                            else None
                        ),
                    }
                    for name, config in self.plugin_configurations.items()
                },
                "plugin_strategy_mapping": dict(self.plugin_strategy_mapping),
                "enabled_strategy_plugins": list(self.enabled_strategy_plugins),
                "enabled_recognizer_plugins": list(self.enabled_recognizer_plugins),
                "plugin_fallback_enabled": self.plugin_fallback_enabled,
                "plugin_error_handling": self.plugin_error_handling,
            }
        )

        return base_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnhancedMaskingPolicy":
        """Create enhanced policy from dictionary."""
        # Create base policy first
        base_policy = super().from_dict(data)

        # Extract plugin configurations
        plugin_configs = {}
        plugin_config_data = data.get("plugin_configurations", {})

        for name, config_data in plugin_config_data.items():
            fallback_strategy = None
            if config_data.get("fallback_strategy"):
                fs_data = config_data["fallback_strategy"]
                fallback_strategy = Strategy(
                    kind=StrategyKind(fs_data["kind"]),
                    parameters=fs_data.get("parameters", {}),
                )

            plugin_configs[name] = PluginConfiguration(
                plugin_name=config_data["plugin_name"],
                plugin_type=config_data["plugin_type"],
                config=config_data.get("config", {}),
                enabled=config_data.get("enabled", True),
                priority=config_data.get("priority", 0),
                fallback_strategy=fallback_strategy,
            )

        return cls(
            # Base fields
            default_strategy=base_policy.default_strategy,
            per_entity=base_policy.per_entity,
            thresholds=base_policy.thresholds,
            locale=base_policy.locale,
            seed=base_policy.seed,
            custom_callbacks=base_policy.custom_callbacks,
            allow_list=base_policy.allow_list,
            deny_list=base_policy.deny_list,
            context_rules=base_policy.context_rules,
            min_entity_length=base_policy.min_entity_length,
            # Enhanced fields
            plugin_configurations=plugin_configs,
            plugin_strategy_mapping=data.get("plugin_strategy_mapping", {}),
            enabled_strategy_plugins=data.get("enabled_strategy_plugins", []),
            enabled_recognizer_plugins=data.get("enabled_recognizer_plugins", []),
            plugin_fallback_enabled=data.get("plugin_fallback_enabled", True),
            plugin_error_handling=data.get("plugin_error_handling", "fallback"),
        )


def create_plugin_enabled_policy(
    base_policy: MaskingPolicy,
    plugin_configs: Optional[dict[str, PluginConfiguration]] = None,
) -> EnhancedMaskingPolicy:
    """
    Convert a base MaskingPolicy to an EnhancedMaskingPolicy.

    Args:
        base_policy: Base policy to enhance
        plugin_configs: Optional plugin configurations to add

    Returns:
        EnhancedMaskingPolicy with plugin support
    """
    return EnhancedMaskingPolicy(
        # Copy base fields
        default_strategy=base_policy.default_strategy,
        per_entity=base_policy.per_entity,
        thresholds=base_policy.thresholds,
        locale=base_policy.locale,
        seed=base_policy.seed,
        custom_callbacks=base_policy.custom_callbacks,
        allow_list=base_policy.allow_list,
        deny_list=base_policy.deny_list,
        context_rules=base_policy.context_rules,
        min_entity_length=base_policy.min_entity_length,
        # Add plugin configurations
        plugin_configurations=plugin_configs or {},
    )
