"""Simple tests for plugin-aware strategy applicator to achieve basic coverage."""

from cloakpivot.plugins.strategies.registry import StrategyPluginRegistry
from cloakpivot.plugins.strategy_applicator import PluginAwareStrategyApplicator


class TestPluginAwareStrategyApplicator:
    """Test PluginAwareStrategyApplicator basic functionality."""

    def test_initialization_default(self):
        """Test applicator can be created with defaults."""
        applicator = PluginAwareStrategyApplicator()

        assert applicator is not None
        assert applicator.strategy_registry is not None
        assert isinstance(applicator.strategy_registry, StrategyPluginRegistry)

    def test_initialization_with_seed(self):
        """Test applicator initialization with custom seed."""
        applicator = PluginAwareStrategyApplicator(seed="test_seed")

        assert applicator is not None
        assert applicator.strategy_registry is not None

    def test_initialization_with_registry(self):
        """Test applicator initialization with custom registry."""
        custom_registry = StrategyPluginRegistry()
        applicator = PluginAwareStrategyApplicator(strategy_registry=custom_registry)

        assert applicator.strategy_registry == custom_registry

    def test_has_strategy_registry_attribute(self):
        """Test that applicator has the expected strategy registry attribute."""
        applicator = PluginAwareStrategyApplicator()

        assert hasattr(applicator, "strategy_registry")
        assert applicator.strategy_registry is not None

    def test_extends_strategy_applicator(self):
        """Test that PluginAwareStrategyApplicator extends base StrategyApplicator."""
        applicator = PluginAwareStrategyApplicator()

        # Should have the apply_strategy method from base class
        assert hasattr(applicator, "apply_strategy")
        assert callable(applicator.apply_strategy)

    def test_registry_integration(self):
        """Test basic registry integration."""
        registry = StrategyPluginRegistry()
        applicator = PluginAwareStrategyApplicator(strategy_registry=registry)

        # Should be able to access registry methods
        assert hasattr(applicator.strategy_registry, "register_strategy_plugin")
        assert callable(applicator.strategy_registry.register_strategy_plugin)
