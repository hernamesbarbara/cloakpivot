"""Tests for the plugin system."""

from typing import Any, Optional

import pytest

from cloakpivot.core.strategies import StrategyKind
from cloakpivot.plugins.base import PluginInfo, PluginStatus
from cloakpivot.plugins.exceptions import (
    PluginError,
    PluginRegistrationError,
)
from cloakpivot.plugins.policy_extensions import (
    EnhancedMaskingPolicy,
    PluginConfiguration,
)
from cloakpivot.plugins.recognizers.base import (
    BaseRecognizerPlugin,
    RecognizerPluginResult,
)
from cloakpivot.plugins.registry import (
    get_plugin_registry,
    reset_plugin_registry,
)
from cloakpivot.plugins.strategies.base import BaseStrategyPlugin, StrategyPluginResult


class MockStrategyPlugin(BaseStrategyPlugin):
    """Mock strategy plugin for testing."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        self.apply_count = 0
        self.should_fail = False

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="mock_strategy",
            version="1.0.0",
            description="Mock strategy plugin for testing",
            author="Test Suite",
            plugin_type="strategy"
        )

    def apply_strategy(
        self,
        original_text: str,
        entity_type: str,
        confidence: float,
        context: Optional[dict[str, Any]] = None
    ) -> StrategyPluginResult:
        self.apply_count += 1

        if self.should_fail:
            raise Exception("Mock strategy failure")

        return StrategyPluginResult(
            masked_text=f"MOCK[{original_text}]",
            execution_time_ms=0.0,
            metadata={"mock": True, "call_count": self.apply_count}
        )


class MockRecognizerPlugin(BaseRecognizerPlugin):
    """Mock recognizer plugin for testing."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        self.analyze_count = 0
        self.should_fail = False

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="mock_recognizer",
            version="1.0.0",
            description="Mock recognizer plugin for testing",
            author="Test Suite",
            plugin_type="recognizer"
        )

    def analyze_text(
        self,
        text: str,
        language: str = "en",
        context: Optional[dict[str, Any]] = None
    ) -> list[RecognizerPluginResult]:
        self.analyze_count += 1

        if self.should_fail:
            raise Exception("Mock recognizer failure")

        # Simple mock: find "test" as TEST entity
        results = []
        start = text.lower().find("test")
        if start != -1:
            results.append(RecognizerPluginResult(
                entity_type="TEST",
                start=start,
                end=start + 4,
                confidence=0.9,
                text="test",
                metadata={"mock": True, "call_count": self.analyze_count}
            ))

        return results


class TestPluginBase:
    """Test base plugin functionality."""

    def test_plugin_info_validation(self):
        """Test plugin info validation."""
        with pytest.raises(ValueError, match="Plugin name cannot be empty"):
            PluginInfo(name="", version="1.0.0", description="Test", author="Test", plugin_type="strategy")

        with pytest.raises(ValueError, match="Plugin version cannot be empty"):
            PluginInfo(name="test", version="", description="Test", author="Test", plugin_type="strategy")

    def test_mock_strategy_plugin(self):
        """Test mock strategy plugin."""
        plugin = MockStrategyPlugin()
        plugin.initialize()

        result = plugin.apply_strategy_safe("test_text", "TEST", 0.9)

        assert result.success
        assert result.masked_text == "MOCK[test_text]"
        assert result.metadata["mock"] is True
        assert result.metadata["call_count"] == 1
        assert plugin.apply_count == 1

    def test_mock_recognizer_plugin(self):
        """Test mock recognizer plugin."""
        plugin = MockRecognizerPlugin()
        plugin.initialize()

        results = plugin.analyze_text_safe("This is a test message")

        assert len(results) == 1
        assert results[0].entity_type == "TEST"
        assert results[0].text == "test"
        assert results[0].confidence == 0.9
        assert results[0].metadata["mock"] is True


class TestPluginRegistry:
    """Test plugin registry functionality."""

    def test_plugin_registry_singleton(self, reset_registries):
        """Test registry singleton behavior."""
        registry1 = get_plugin_registry()
        registry2 = get_plugin_registry()

        assert registry1 is registry2

    def test_register_strategy_plugin(self, reset_registries):
        """Test registering a strategy plugin."""
        registry = get_plugin_registry()
        plugin = MockStrategyPlugin()

        registry.register_plugin(plugin)

        assert "mock_strategy" in registry._plugins
        assert "mock_strategy" in registry._strategy_plugins
        assert len(registry._recognizer_plugins) == 0

    def test_register_recognizer_plugin(self, reset_registries):
        """Test registering a recognizer plugin."""
        registry = get_plugin_registry()
        plugin = MockRecognizerPlugin()

        registry.register_plugin(plugin)

        assert "mock_recognizer" in registry._plugins
        assert "mock_recognizer" in registry._recognizer_plugins
        assert len(registry._strategy_plugins) == 0

    def test_duplicate_plugin_registration(self, reset_registries):
        """Test duplicate plugin registration fails."""
        registry = get_plugin_registry()
        plugin1 = MockStrategyPlugin()
        plugin2 = MockStrategyPlugin()

        registry.register_plugin(plugin1)

        with pytest.raises(PluginRegistrationError, match="already registered"):
            registry.register_plugin(plugin2)

    def test_initialize_plugin(self, reset_registries):
        """Test plugin initialization."""
        registry = get_plugin_registry()
        plugin = MockStrategyPlugin()
        registry.register_plugin(plugin)

        registry.initialize_plugin("mock_strategy")

        plugin_info = registry.get_plugin_info("mock_strategy")
        assert plugin_info.status == PluginStatus.ACTIVE
        assert plugin.is_initialized

    def test_plugin_not_found(self, reset_registries):
        """Test error when plugin not found."""
        registry = get_plugin_registry()

        with pytest.raises(PluginError, match="Plugin nonexistent not found"):
            registry.initialize_plugin("nonexistent")

    def test_get_registry_status(self, reset_registries):
        """Test getting registry status."""
        registry = get_plugin_registry()
        plugin1 = MockStrategyPlugin()
        plugin2 = MockRecognizerPlugin()

        registry.register_plugin(plugin1)
        registry.register_plugin(plugin2)
        registry.initialize_plugin("mock_strategy")

        status = registry.get_registry_status()

        assert status["total_plugins"] == 2
        assert status["strategy_plugins"] == 1
        assert status["recognizer_plugins"] == 1
        assert status["status_counts"]["loaded"] == 1
        assert status["status_counts"]["active"] == 1


class TestPluginConfiguration:
    """Test plugin configuration system."""

    def test_plugin_configuration_validation(self):
        """Test plugin configuration validation."""
        with pytest.raises(ValueError, match="Plugin name cannot be empty"):
            PluginConfiguration(plugin_name="", plugin_type="strategy")

        with pytest.raises(ValueError, match="Plugin type must be"):
            PluginConfiguration(plugin_name="test", plugin_type="invalid")

    def test_enhanced_masking_policy(self):
        """Test enhanced masking policy with plugins."""
        plugin_config = PluginConfiguration(
            plugin_name="test_strategy",
            plugin_type="strategy",
            config={"param": "value"}
        )

        policy = EnhancedMaskingPolicy(
            plugin_configurations={"test_strategy": plugin_config},
            plugin_strategy_mapping={"PHONE_NUMBER": "test_strategy"}
        )

        strategy = policy.get_strategy_for_entity("PHONE_NUMBER")

        assert strategy.kind == StrategyKind.CUSTOM
        assert strategy.get_parameter("plugin_name") == "test_strategy"
        assert strategy.get_parameter("plugin_config") == {"param": "value"}

    def test_policy_serialization(self):
        """Test policy serialization and deserialization."""
        plugin_config = PluginConfiguration(
            plugin_name="test_strategy",
            plugin_type="strategy",
            config={"param": "value"},
            enabled=True,
            priority=5
        )

        policy = EnhancedMaskingPolicy(
            plugin_configurations={"test_strategy": plugin_config},
            enabled_strategy_plugins=["test_strategy"]
        )

        # Serialize to dict
        policy_dict = policy.to_dict()

        assert "plugin_configurations" in policy_dict
        assert "test_strategy" in policy_dict["plugin_configurations"]

        # Deserialize from dict
        restored_policy = EnhancedMaskingPolicy.from_dict(policy_dict)

        assert "test_strategy" in restored_policy.plugin_configurations
        assert restored_policy.plugin_configurations["test_strategy"].priority == 5


class TestErrorHandling:
    """Test plugin error handling."""

    def test_strategy_plugin_failure(self):
        """Test handling of strategy plugin failures."""
        plugin = MockStrategyPlugin()
        plugin.should_fail = True
        plugin.initialize()

        result = plugin.apply_strategy_safe("test", "TEST", 0.9)

        assert not result.success
        assert "Mock strategy failure" in result.error_message
        assert result.masked_text == "test"  # Should return original text on failure

    def test_recognizer_plugin_failure(self):
        """Test handling of recognizer plugin failures."""
        plugin = MockRecognizerPlugin()
        plugin.should_fail = True
        plugin.initialize()

        results = plugin.analyze_text_safe("test message")

        assert results == []  # Should return empty list on failure

    def test_plugin_not_initialized(self):
        """Test behavior when plugin not initialized."""
        plugin = MockStrategyPlugin()
        # Don't initialize

        result = plugin.apply_strategy_safe("test", "TEST", 0.9)

        assert not result.success
        assert "not initialized" in result.error_message


if __name__ == "__main__":
    pytest.main([__file__])
