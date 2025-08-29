"""Exception classes for the plugin system."""

from typing import Optional


class PluginError(Exception):
    """Base exception for plugin-related errors."""

    def __init__(self, message: str, plugin_name: Optional[str] = None) -> None:
        super().__init__(message)
        self.plugin_name = plugin_name


class PluginRegistrationError(PluginError):
    """Raised when plugin registration fails."""

    pass


class PluginValidationError(PluginError):
    """Raised when plugin validation fails."""

    pass


class PluginExecutionError(PluginError):
    """Raised when plugin execution fails."""

    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, plugin_name)
        self.original_exception = original_exception


class PluginConfigurationError(PluginError):
    """Raised when plugin configuration is invalid."""

    pass


class PluginDependencyError(PluginError):
    """Raised when plugin dependencies are not satisfied."""

    pass