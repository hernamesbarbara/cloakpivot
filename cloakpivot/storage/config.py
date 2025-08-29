"""
Storage configuration system for CloakMaps.

Provides configuration management, backend selection, and validation
for storage backends with support for environment variables and
configuration files.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from .backends.base import StorageBackend


@dataclass
class StorageConfig:
    """
    Configuration for storage backends.

    Manages backend selection, configuration validation, and provides
    a unified interface for configuring different storage systems.

    Attributes:
        backend_type: Type of storage backend to use
        config: Backend-specific configuration parameters
        fallback_backends: List of fallback backend configurations
        default_backend: Default backend type if none specified
        environment_prefix: Prefix for environment variable lookup

    Examples:
        >>> # Local storage
        >>> config = StorageConfig(
        ...     backend_type="local",
        ...     config={"base_path": "/path/to/storage"}
        ... )
        >>>
        >>> # S3 storage with fallback
        >>> config = StorageConfig(
        ...     backend_type="s3",
        ...     config={
        ...         "bucket_name": "my-bucket",
        ...         "region_name": "us-west-2"
        ...     },
        ...     fallback_backends=[
        ...         {"backend_type": "local", "config": {"base_path": "/tmp/backup"}}
        ...     ]
        ... )
    """

    backend_type: str = "local"
    config: dict[str, Any] = field(default_factory=dict)
    fallback_backends: list[dict[str, Any]] = field(default_factory=list)
    default_backend: str = "local"
    environment_prefix: str = "CLOAKPIVOT_STORAGE"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_backend_type()
        self._apply_environment_overrides()

    def _validate_backend_type(self) -> None:
        """Validate that backend_type is supported."""
        supported_backends = {
            "local", "local_filesystem",
            "s3", "aws_s3",
            "gcs", "google_cloud_storage",
            "database", "db"
        }

        if self.backend_type not in supported_backends:
            raise ValueError(
                f"Unsupported backend_type '{self.backend_type}'. "
                f"Supported types: {sorted(supported_backends)}"
            )

    def _apply_environment_overrides(self) -> None:
        """Apply configuration overrides from environment variables."""
        # Backend type override
        backend_env = f"{self.environment_prefix}_BACKEND"
        if os.getenv(backend_env):
            self.backend_type = os.getenv(backend_env)

        # Configuration overrides
        config_overrides = {}

        # Common configuration variables
        env_mappings = {
            f"{self.environment_prefix}_BASE_PATH": "base_path",
            f"{self.environment_prefix}_BUCKET_NAME": "bucket_name",
            f"{self.environment_prefix}_DATABASE_URL": "database_url",
            f"{self.environment_prefix}_REGION": "region_name",
            f"{self.environment_prefix}_PROJECT_ID": "project_id",
            f"{self.environment_prefix}_CREDENTIALS_PATH": "credentials_path",
        }

        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                config_overrides[config_key] = value

        # AWS specific variables
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_DEFAULT_REGION")

        if aws_access_key:
            config_overrides["aws_access_key_id"] = aws_access_key
        if aws_secret_key:
            config_overrides["aws_secret_access_key"] = aws_secret_key
        if aws_region and "region_name" not in config_overrides:
            config_overrides["region_name"] = aws_region

        # Google Cloud specific variables
        gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT")
        gcp_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if gcp_project:
            config_overrides["project_id"] = gcp_project
        if gcp_credentials:
            config_overrides["credentials_path"] = gcp_credentials

        # Apply overrides
        self.config.update(config_overrides)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StorageConfig":
        """Create StorageConfig from dictionary."""
        return cls(
            backend_type=data.get("backend_type", "local"),
            config=data.get("config", {}),
            fallback_backends=data.get("fallback_backends", []),
            default_backend=data.get("default_backend", "local"),
            environment_prefix=data.get("environment_prefix", "CLOAKPIVOT_STORAGE"),
        )

    @classmethod
    def from_yaml_file(cls, file_path: Union[str, Path]) -> "StorageConfig":
        """Load StorageConfig from YAML file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Storage config file not found: {file_path}")

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # Handle nested storage config
            if "storage" in data:
                data = data["storage"]

            return cls.from_dict(data)

        except Exception as e:
            raise ValueError(f"Failed to load storage config from {file_path}: {e}") from e

    @classmethod
    def from_environment(
        cls,
        environment_prefix: str = "CLOAKPIVOT_STORAGE"
    ) -> "StorageConfig":
        """Create StorageConfig from environment variables only."""
        config = cls(environment_prefix=environment_prefix)
        # Environment overrides are applied in __post_init__
        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert StorageConfig to dictionary."""
        return {
            "backend_type": self.backend_type,
            "config": self.config,
            "fallback_backends": self.fallback_backends,
            "default_backend": self.default_backend,
            "environment_prefix": self.environment_prefix,
        }

    def to_yaml(self) -> str:
        """Convert StorageConfig to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save StorageConfig to YAML file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())

    def create_backend(self) -> StorageBackend:
        """Create a storage backend instance from this configuration."""
        from .registry import StorageRegistry

        registry = StorageRegistry()
        return registry.create_backend(self.backend_type, self.config)

    def create_backend_with_fallbacks(self) -> StorageBackend:
        """
        Create storage backend with fallback support.

        Returns a backend that will automatically retry operations
        on fallback backends if the primary backend fails.
        """
        primary_backend = self.create_backend()

        if not self.fallback_backends:
            return primary_backend

        # Create fallback backends
        fallback_instances = []
        for fallback_config in self.fallback_backends:
            fallback_type = fallback_config.get("backend_type", self.default_backend)
            fallback_params = fallback_config.get("config", {})

            from .registry import StorageRegistry
            registry = StorageRegistry()
            fallback_backend = registry.create_backend(fallback_type, fallback_params)
            fallback_instances.append(fallback_backend)

        # Return fallback-enabled backend
        return FallbackStorageBackend(primary_backend, fallback_instances)

    def validate(self) -> dict[str, Any]:
        """
        Validate the storage configuration.

        Returns:
            Dictionary with validation results and any errors
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "backend_type": self.backend_type,
        }

        try:
            # Test backend creation
            backend = self.create_backend()

            # Test health check
            health = backend.health_check()
            results["health_check"] = health

            if health.get("status") != "healthy":
                results["warnings"].append(f"Backend health check failed: {health.get('error')}")

        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Failed to create backend: {e}")

        # Validate fallback backends
        for i, fallback_config in enumerate(self.fallback_backends):
            try:
                fallback_type = fallback_config.get("backend_type", self.default_backend)
                fallback_params = fallback_config.get("config", {})

                from .registry import StorageRegistry
                registry = StorageRegistry()
                fallback_backend = registry.create_backend(fallback_type, fallback_params)

                fallback_health = fallback_backend.health_check()
                results[f"fallback_{i}_health"] = fallback_health

                if fallback_health.get("status") != "healthy":
                    results["warnings"].append(
                        f"Fallback backend {i} health check failed: {fallback_health.get('error')}"
                    )

            except Exception as e:
                results["warnings"].append(f"Fallback backend {i} validation failed: {e}")

        return results


class FallbackStorageBackend(StorageBackend):
    """
    Storage backend wrapper that provides fallback support.

    Automatically retries operations on fallback backends if the
    primary backend fails, providing resilience and high availability.
    """

    def __init__(
        self,
        primary_backend: StorageBackend,
        fallback_backends: list[StorageBackend]
    ):
        """
        Initialize fallback storage backend.

        Args:
            primary_backend: Primary storage backend to use
            fallback_backends: List of fallback backends to try on failure
        """
        self.primary_backend = primary_backend
        self.fallback_backends = fallback_backends
        super().__init__()

    @property
    def backend_type(self) -> str:
        """Return composite backend type."""
        fallback_types = [b.backend_type for b in self.fallback_backends]
        return f"fallback({self.primary_backend.backend_type}→{','.join(fallback_types)})"

    def _validate_config(self) -> None:
        """Validate configuration (delegated to underlying backends)."""
        pass

    def _try_operation(self, operation_name: str, *args, **kwargs):
        """Try an operation on primary backend, fall back to alternatives on failure."""
        backends_to_try = [self.primary_backend] + self.fallback_backends
        last_exception = None

        for i, backend in enumerate(backends_to_try):
            try:
                operation = getattr(backend, operation_name)
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e

                # For certain operations, don't retry on all backends
                if operation_name in ["save", "delete"] and i > 0:
                    # Only try save/delete on primary backend
                    break

                continue

        # All backends failed
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"All backends failed for operation: {operation_name}")

    def save(
        self,
        key: str,
        cloakmap: Any,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any
    ) -> Any:
        """Save with fallback (only tries primary backend for consistency)."""
        return self.primary_backend.save(key, cloakmap, metadata, **kwargs)

    def load(self, key: str, **kwargs: Any) -> Any:
        """Load with fallback support."""
        return self._try_operation("load", key, **kwargs)

    def exists(self, key: str, **kwargs: Any) -> bool:
        """Check existence with fallback support."""
        return self._try_operation("exists", key, **kwargs)

    def delete(self, key: str, **kwargs: Any) -> bool:
        """Delete with fallback (only tries primary backend for consistency)."""
        return self.primary_backend.delete(key, **kwargs)

    def list_keys(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs: Any
    ) -> list[str]:
        """List keys with fallback support."""
        return self._try_operation("list_keys", prefix, limit, **kwargs)

    def get_metadata(self, key: str, **kwargs: Any) -> Any:
        """Get metadata with fallback support."""
        return self._try_operation("get_metadata", key, **kwargs)

    def health_check(self) -> dict[str, Any]:
        """Perform health check on all backends."""
        primary_health = self.primary_backend.health_check()

        fallback_healths = []
        for i, backend in enumerate(self.fallback_backends):
            try:
                health = backend.health_check()
                health["fallback_index"] = i
                fallback_healths.append(health)
            except Exception as e:
                fallback_healths.append({
                    "fallback_index": i,
                    "status": "error",
                    "error": str(e),
                })

        return {
            "status": primary_health.get("status", "unknown"),
            "backend_type": self.backend_type,
            "primary": primary_health,
            "fallbacks": fallback_healths,
            "fallback_count": len(self.fallback_backends),
        }


def create_storage_config_from_policy(policy_data: dict[str, Any]) -> Optional[StorageConfig]:
    """
    Create StorageConfig from policy data.

    Extracts storage configuration from a policy file, allowing
    storage backend selection to be driven by masking policies.

    Args:
        policy_data: Policy dictionary that may contain storage config

    Returns:
        StorageConfig if found in policy, None otherwise
    """
    storage_data = policy_data.get("storage")
    if not storage_data:
        return None

    return StorageConfig.from_dict(storage_data)


def get_default_storage_config() -> StorageConfig:
    """
    Get default storage configuration.

    Creates a sensible default configuration that works out of the box,
    with environment variable overrides applied.

    Returns:
        Default StorageConfig instance
    """
    # Check for user config file
    config_paths = [
        Path.home() / ".cloakpivot" / "storage.yaml",
        Path.cwd() / "cloakpivot-storage.yaml",
        Path.cwd() / "storage.yaml",
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                return StorageConfig.from_yaml_file(config_path)
            except Exception:
                continue

    # Create default with environment overrides
    default_config = StorageConfig(
        backend_type="local",
        config={
            "base_path": str(Path.home() / ".cloakpivot" / "storage"),
            "create_dirs": True,
            "file_extension": ".cmap",
            "metadata_extension": ".meta",
        }
    )

    return default_config
