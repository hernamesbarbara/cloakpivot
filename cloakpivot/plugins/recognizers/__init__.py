"""Recognizer plugin framework for custom Presidio recognizers."""

from .base import BaseRecognizerPlugin, RecognizerPluginResult
from .registry import RecognizerPluginRegistry

__all__ = [
    "BaseRecognizerPlugin",
    "RecognizerPluginResult",
    "RecognizerPluginRegistry",
]