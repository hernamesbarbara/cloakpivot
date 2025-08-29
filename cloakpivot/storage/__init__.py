"""
Storage backend system for CloakMaps.

This module provides pluggable storage backends for CloakMaps, supporting
local file systems, cloud storage services, and databases. The storage
system is designed to be extensible while maintaining backward compatibility
with existing file-based operations.

Key Components:
- StorageBackend: Abstract base class for all storage implementations
- StorageConfig: Configuration management for storage backends
- StorageRegistry: Backend discovery and registration system
- Built-in backends: LocalStorage, S3Storage, GCSStorage, DatabaseStorage

Example:
    >>> from cloakpivot.storage import StorageRegistry, LocalStorage
    >>> from cloakpivot.core.cloakmap import CloakMap
    >>>
    >>> # Create storage backend
    >>> storage = LocalStorage("/path/to/storage")
    >>>
    >>> # Save CloakMap
    >>> cloakmap = CloakMap.create(...)
    >>> storage.save("my_document.cmap", cloakmap)
    >>>
    >>> # Load CloakMap
    >>> loaded_map = storage.load("my_document.cmap")
"""

from .backends import (
    DatabaseStorage,
    GCSStorage,
    LocalStorage,
    S3Storage,
    StorageBackend,
)
from .config import StorageConfig
from .registry import StorageRegistry

__all__ = [
    "StorageBackend",
    "LocalStorage",
    "S3Storage",
    "GCSStorage",
    "DatabaseStorage",
    "StorageConfig",
    "StorageRegistry",
]
