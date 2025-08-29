"""Plugin system for CloakPivot custom masking strategies and recognizers."""

from .registry import PluginRegistry, get_plugin_registry
from .base import BasePlugin, PluginInfo, PluginStatus
from .strategies import BaseStrategyPlugin
from .recognizers import BaseRecognizerPlugin
from .exceptions import (
    PluginError,
    PluginRegistrationError,
    PluginValidationError,
    PluginExecutionError,
)

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