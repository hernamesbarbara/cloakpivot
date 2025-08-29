"""Plugin system for CloakPivot custom masking strategies and recognizers."""

from .base import BasePlugin, PluginInfo, PluginStatus
from .exceptions import (
    PluginError,
    PluginExecutionError,
    PluginRegistrationError,
    PluginValidationError,
)
from .recognizers import BaseRecognizerPlugin
from .registry import PluginRegistry, get_plugin_registry
from .strategies import BaseStrategyPlugin

__all__ = [
    "PluginRegistry",
    "get_plugin_registry",
    "BasePlugin",
    "BaseStrategyPlugin",
    "BaseRecognizerPlugin",
    "PluginInfo",
    "PluginStatus",
    "PluginError",
    "PluginRegistrationError",
    "PluginValidationError",
    "PluginExecutionError",
]
