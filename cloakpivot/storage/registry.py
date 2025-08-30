"""
Storage backend registry for CloakMaps.

Provides backend discovery, registration, and factory functionality
for creating storage backend instances with proper configuration.
"""

from typing import Any, Optional

from .backends.base import StorageBackend


class StorageRegistry:
    """
    Registry for storage backend types and factory for creating instances.

    Manages registration and discovery of storage backends, allowing
    for plugin-style architecture where new backends can be added
    without modifying core code.

    Features:
    - Backend type registration and lookup
    - Configuration-based backend creation
    - Plugin discovery support
    - Backend validation and health checking

    Examples:
        >>> registry = StorageRegistry()
        >>> backend = registry.create_backend("s3", {"bucket_name": "my-bucket"})
        >>>
        >>> # Register custom backend
        >>> registry.register_backend("custom", CustomStorageBackend)
    """

    def __init__(self):
        """Initialize storage registry with built-in backends."""
        self._backends: dict[str, type[StorageBackend]] = {}
        self._aliases: dict[str, str] = {}

        # Register built-in backends
        self._register_builtin_backends()

    def _register_builtin_backends(self) -> None:
        """Register all built-in storage backends."""
        # Import backends locally to avoid circular imports
        from .backends.database import DatabaseStorage
        from .backends.gcs import GCSStorage
        from .backends.local import LocalStorage
        from .backends.s3 import S3Storage

        # Register backends with primary names
        self._backends["local_filesystem"] = LocalStorage
        self._backends["aws_s3"] = S3Storage
        self._backends["google_cloud_storage"] = GCSStorage
        self._backends["database"] = DatabaseStorage

        # Register common aliases
        self._aliases.update({
            "local": "local_filesystem",
            "file": "local_filesystem",
            "filesystem": "local_filesystem",
            "s3": "aws_s3",
            "gcs": "google_cloud_storage",
            "google": "google_cloud_storage",
            "db": "database",
            "sql": "database",
            "sqlite": "database",
            "postgresql": "database",
            "postgres": "database",
        })

    def register_backend(
        self,
        backend_type: str,
        backend_class: type[StorageBackend],
        aliases: Optional[list[str]] = None
    ) -> None:
        """
        Register a storage backend type.

        Args:
            backend_type: Unique identifier for the backend type
            backend_class: Storage backend class (must inherit from StorageBackend)
            aliases: Optional list of alias names for the backend

        Raises:
            ValueError: If backend_type is already registered or backend_class is invalid
        """
        if not backend_type or not isinstance(backend_type, str):
            raise ValueError("backend_type must be a non-empty string")

        if backend_type in self._backends:
            raise ValueError(f"Backend type '{backend_type}' is already registered")

        if not issubclass(backend_class, StorageBackend):
            raise ValueError("backend_class must inherit from StorageBackend")

        # Register the backend
        self._backends[backend_type] = backend_class

        # Register aliases
        if aliases:
            for alias in aliases:
                if alias in self._aliases:
                    raise ValueError(f"Alias '{alias}' is already registered")
                self._aliases[alias] = backend_type

    def unregister_backend(self, backend_type: str) -> None:
        """
        Unregister a storage backend type.

        Args:
            backend_type: Backend type to unregister

        Raises:
            KeyError: If backend_type is not registered
        """
        if backend_type not in self._backends:
            raise KeyError(f"Backend type '{backend_type}' is not registered")

        # Remove from backends
        del self._backends[backend_type]

        # Remove aliases that point to this backend
        aliases_to_remove = [
            alias for alias, target in self._aliases.items()
            if target == backend_type
        ]
        for alias in aliases_to_remove:
            del self._aliases[alias]

    def resolve_backend_type(self, backend_type: str) -> str:
        """
        Resolve backend type, handling aliases.

        Args:
            backend_type: Backend type or alias to resolve

        Returns:
            Resolved backend type

        Raises:
            KeyError: If backend_type is not found
        """
        # Check if it's an alias first
        if backend_type in self._aliases:
            return self._aliases[backend_type]

        # Check if it's a direct backend type
        if backend_type in self._backends:
            return backend_type

        raise KeyError(f"Unknown backend type: '{backend_type}'")

    def get_backend_class(self, backend_type: str) -> type[StorageBackend]:
        """
        Get the storage backend class for a given type.

        Args:
            backend_type: Backend type or alias

        Returns:
            Storage backend class

        Raises:
            KeyError: If backend_type is not found
        """
        resolved_type = self.resolve_backend_type(backend_type)
        return self._backends[resolved_type]

    def create_backend(
        self,
        backend_type: str,
        config: Optional[dict[str, Any]] = None
    ) -> StorageBackend:
        """
        Create a storage backend instance.

        Args:
            backend_type: Backend type or alias
            config: Backend-specific configuration

        Returns:
            Configured storage backend instance

        Raises:
            KeyError: If backend_type is not found
            ValueError: If configuration is invalid
        """
        backend_class = self.get_backend_class(backend_type)

        try:
            # Handle different backend constructor signatures
            if backend_type in ["local", "local_filesystem"]:
                # LocalStorage expects base_path as first argument
                if config and "base_path" in config:
                    base_path = config.pop("base_path")
                    return backend_class(base_path, config)
                else:
                    raise ValueError("base_path is required for local storage backend")
            else:
                # Other backends expect config dict
                return backend_class(config=config)

        except Exception as e:
            raise ValueError(f"Failed to create {backend_type} backend: {e}") from e

    def list_backend_types(self) -> list[str]:
        """
        List all registered backend types.

        Returns:
            List of backend type identifiers
        """
        return list(self._backends.keys())

    def list_aliases(self) -> dict[str, str]:
        """
        List all registered aliases and their targets.

        Returns:
            Dictionary mapping aliases to backend types
        """
        return dict(self._aliases)

    def get_backend_info(self, backend_type: str) -> dict[str, Any]:
        """
        Get information about a backend type.

        Args:
            backend_type: Backend type or alias to get info for

        Returns:
            Dictionary with backend information

        Raises:
            KeyError: If backend_type is not found
        """
        resolved_type = self.resolve_backend_type(backend_type)
        backend_class = self._backends[resolved_type]

        # Find aliases for this backend
        aliases = [
            alias for alias, target in self._aliases.items()
            if target == resolved_type
        ]

        return {
            "backend_type": resolved_type,
            "class_name": backend_class.__name__,
            "module": backend_class.__module__,
            "aliases": aliases,
            "docstring": backend_class.__doc__,
        }

    def validate_backend(
        self,
        backend_type: str,
        config: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Validate a backend configuration without creating an instance.

        Args:
            backend_type: Backend type to validate
            config: Configuration to validate

        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "backend_type": backend_type,
        }

        try:
            resolved_type = self.resolve_backend_type(backend_type)
            results["resolved_type"] = resolved_type

            # Try to create backend
            backend = self.create_backend(backend_type, config)
            results["creation_successful"] = True

            # Try health check
            health = backend.health_check()
            results["health_check"] = health

            if health.get("status") != "healthy":
                results["warnings"].append(f"Health check failed: {health.get('error')}")

        except KeyError as e:
            results["valid"] = False
            results["errors"].append(f"Unknown backend type: {e}")
        except ValueError as e:
            results["valid"] = False
            results["errors"].append(f"Configuration error: {e}")
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Validation failed: {e}")

        return results

    def discover_plugins(self) -> list[str]:
        """
        Discover storage backend plugins using entry points.

        Looks for plugins registered under the 'cloakpivot.storage.backends'
        entry point group and registers them automatically.

        Returns:
            List of discovered and registered plugin names
        """
        discovered = []

        try:
            # Try to use importlib.metadata (Python 3.8+)
            try:
                from importlib.metadata import entry_points
            except ImportError:
                # Fallback for Python < 3.8
                from importlib_metadata import entry_points

            # Look for storage backend plugins
            eps = entry_points()
            if hasattr(eps, 'select'):
                # Python 3.10+ style
                storage_eps = eps.select(group='cloakpivot.storage.backends')
            else:
                # Older style
                storage_eps = eps.get('cloakpivot.storage.backends', [])

            for entry_point in storage_eps:
                try:
                    # Load the plugin class
                    plugin_class = entry_point.load()

                    # Register the plugin
                    plugin_name = entry_point.name
                    self.register_backend(plugin_name, plugin_class)
                    discovered.append(plugin_name)

                except Exception as e:
                    # Log plugin loading errors but continue
                    import warnings
                    warnings.warn(f"Failed to load storage backend plugin '{entry_point.name}': {e}", stacklevel=2)

        except ImportError:
            # Entry points not available
            pass

        return discovered

    def health_check_all(self) -> dict[str, Any]:
        """
        Perform health check on all registered backends.

        Creates instances with minimal configuration and tests
        their health check functionality.

        Returns:
            Dictionary with health check results for all backends
        """
        results = {
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
            "backends": {},
            "summary": {
                "total": 0,
                "healthy": 0,
                "unhealthy": 0,
                "failed": 0,
            }
        }

        for backend_type in self.list_backend_types():
            results["summary"]["total"] += 1

            try:
                # Create minimal config for testing
                test_config = self._get_test_config(backend_type)
                backend = self.create_backend(backend_type, test_config)

                # Perform health check
                health = backend.health_check()
                results["backends"][backend_type] = health

                if health.get("status") == "healthy":
                    results["summary"]["healthy"] += 1
                else:
                    results["summary"]["unhealthy"] += 1

            except Exception as e:
                results["backends"][backend_type] = {
                    "status": "failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                results["summary"]["failed"] += 1

        return results

    def _get_test_config(self, backend_type: str) -> dict[str, Any]:
        """Get minimal test configuration for a backend type."""
        resolved_type = self.resolve_backend_type(backend_type)

        if resolved_type == "local_filesystem":
            import tempfile
            return {"base_path": tempfile.gettempdir()}
        elif resolved_type == "aws_s3":
            return {"bucket_name": "test-bucket-nonexistent"}
        elif resolved_type == "google_cloud_storage":
            return {"bucket_name": "test-bucket-nonexistent"}
        elif resolved_type == "database":
            return {"database_url": "sqlite:///:memory:"}
        else:
            return {}

    def __str__(self) -> str:
        """String representation of the registry."""
        backend_count = len(self._backends)
        alias_count = len(self._aliases)
        return f"StorageRegistry({backend_count} backends, {alias_count} aliases)"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"StorageRegistry("
                f"backends={list(self._backends.keys())}, "
                f"aliases={list(self._aliases.keys())})")


# Global registry instance
_global_registry: Optional[StorageRegistry] = None


def get_storage_registry() -> StorageRegistry:
    """
    Get the global storage registry instance.

    Creates a new registry on first access and discovers plugins.

    Returns:
        Global StorageRegistry instance
    """
    global _global_registry

    if _global_registry is None:
        _global_registry = StorageRegistry()

        # Auto-discover plugins
        try:
            _global_registry.discover_plugins()
        except Exception:
            # Don't fail if plugin discovery fails
            pass

    return _global_registry


def register_backend(
    backend_type: str,
    backend_class: type[StorageBackend],
    aliases: Optional[list[str]] = None
) -> None:
    """
    Register a storage backend with the global registry.

    Convenience function for plugin authors to register backends.

    Args:
        backend_type: Unique identifier for the backend type
        backend_class: Storage backend class
        aliases: Optional list of alias names
    """
    registry = get_storage_registry()
    registry.register_backend(backend_type, backend_class, aliases)


def reset_storage_registry() -> None:
    """
    Reset the global storage registry.

    This is primarily for testing purposes to ensure test isolation.
    """
    global _global_registry
    _global_registry = None
