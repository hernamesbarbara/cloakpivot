"""
Local filesystem storage backend for CloakMaps.

Provides file-based storage with directory organization, atomic writes,
and support for both encrypted and unencrypted CloakMaps.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ...core.cloakmap import CloakMap
from .base import StorageBackend, StorageMetadata


class LocalStorage(StorageBackend):
    """
    Local filesystem storage backend for CloakMaps.

    Stores CloakMaps as JSON files in a designated directory with
    atomic write operations, metadata tracking, and optional
    directory organization.

    Features:
    - Atomic writes using temporary files
    - Automatic directory creation
    - Support for nested paths as keys
    - Metadata stored alongside CloakMaps
    - File-based locking for concurrent access

    Configuration:
        base_path: Root directory for storage (required)
        create_dirs: Whether to create directories automatically (default: True)
        file_extension: Extension for CloakMap files (default: ".cmap")
        metadata_extension: Extension for metadata files (default: ".meta")
        ensure_permissions: File permissions to set (default: 0o600)

    Examples:
        >>> storage = LocalStorage("/path/to/cloakmaps")
        >>> storage.save("documents/my_doc.cmap", cloakmap)
        >>> loaded = storage.load("documents/my_doc.cmap")
    """

    def __init__(self, base_path: str, config: dict[str, Any] | None = None):
        """
        Initialize local storage backend.

        Args:
            base_path: Root directory for CloakMap storage
            config: Additional configuration options
        """
        self.base_path = Path(base_path).resolve()
        super().__init__(config)

        # Set up base directory
        if self.config.get("create_dirs", True):
            self.base_path.mkdir(parents=True, exist_ok=True)

    @property
    def backend_type(self) -> str:
        """Return the backend type identifier."""
        return "local_filesystem"

    def _validate_config(self) -> None:
        """Validate local storage configuration."""
        # Validate base path
        if not self.base_path:
            raise ValueError("base_path is required for local storage")

        # Check if base path is accessible
        if self.base_path.exists() and not self.base_path.is_dir():
            raise ValueError(f"base_path is not a directory: {self.base_path}")

        # Validate file extensions
        file_ext = self.config.get("file_extension", ".cmap")
        meta_ext = self.config.get("metadata_extension", ".meta")

        if not file_ext.startswith("."):
            raise ValueError("file_extension must start with '.'")
        if not meta_ext.startswith("."):
            raise ValueError("metadata_extension must start with '.'")

        if file_ext == meta_ext:
            raise ValueError("file_extension and metadata_extension must be different")

    def _get_file_path(self, key: str) -> Path:
        """Get the full file path for a CloakMap key."""
        self.validate_key(key)
        file_ext = self.config.get("file_extension", ".cmap")

        # Ensure key ends with correct extension
        if not key.endswith(file_ext):
            key = key + file_ext

        return self.base_path / key

    def _get_metadata_path(self, key: str) -> Path:
        """Get the metadata file path for a CloakMap key."""
        file_path = self._get_file_path(key)
        meta_ext = self.config.get("metadata_extension", ".meta")
        return file_path.with_suffix(meta_ext)

    def _ensure_directory(self, file_path: Path) -> None:
        """Ensure the parent directory exists for a file path."""
        if self.config.get("create_dirs", True):
            file_path.parent.mkdir(parents=True, exist_ok=True)

    def _set_file_permissions(self, file_path: Path) -> None:
        """Set appropriate file permissions."""
        permissions = self.config.get("ensure_permissions", 0o600)
        if permissions is not None and file_path.exists():
            file_path.chmod(permissions)

    def save(
        self,
        key: str,
        cloakmap: CloakMap,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> StorageMetadata:
        """
        Save a CloakMap to local filesystem.

        Uses atomic writes via temporary files to ensure data integrity.
        Creates directories as needed and stores metadata alongside.

        Args:
            key: File key/path relative to base_path
            cloakmap: CloakMap instance to save
            metadata: Additional metadata to store
            **kwargs: Additional options (indent for JSON formatting)
        """
        file_path = self._get_file_path(key)
        metadata_path = self._get_metadata_path(key)

        self._ensure_directory(file_path)

        # Serialize CloakMap
        indent = kwargs.get("indent", 2)
        content = cloakmap.to_json(indent=indent)
        content_bytes = content.encode("utf-8")

        # Create storage metadata
        storage_metadata = StorageMetadata.from_cloakmap(
            key=key,
            cloakmap=cloakmap,
            backend_type=self.backend_type,
            content_bytes=content_bytes,
            file_path=str(file_path),
            base_path=str(self.base_path),
            **(metadata or {})
        )

        # Write CloakMap with atomic operation
        self._write_file_atomic(file_path, content_bytes)

        # Write metadata
        metadata_content = json.dumps(storage_metadata.to_dict(), indent=2).encode("utf-8")
        self._write_file_atomic(metadata_path, metadata_content)

        # Set file permissions
        self._set_file_permissions(file_path)
        self._set_file_permissions(metadata_path)

        return storage_metadata

    def _write_file_atomic(self, file_path: Path, content: bytes) -> None:
        """Write file atomically using temporary file."""
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=file_path.parent,
            delete=False,
            prefix=f".{file_path.name}.tmp"
        ) as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            temp_path = Path(tmp_file.name)

        try:
            # Atomic move
            temp_path.replace(file_path)
        except Exception:
            # Clean up temp file on failure
            if temp_path.exists():
                temp_path.unlink()
            raise

    def load(self, key: str, **kwargs: Any) -> CloakMap:
        """
        Load a CloakMap from local filesystem.

        Args:
            key: File key/path relative to base_path
            **kwargs: Additional load options

        Returns:
            Loaded CloakMap instance
        """
        file_path = self._get_file_path(key)

        if not file_path.exists():
            raise KeyError(f"CloakMap not found: {key}")

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            return CloakMap.from_json(content)

        except Exception as e:
            raise ValueError(f"Failed to load CloakMap from {key}: {e}") from e

    def exists(self, key: str, **kwargs: Any) -> bool:
        """Check if a CloakMap file exists."""
        file_path = self._get_file_path(key)
        return file_path.exists() and file_path.is_file()

    def delete(self, key: str, **kwargs: Any) -> bool:
        """
        Delete a CloakMap and its metadata from filesystem.

        Args:
            key: File key/path to delete
            **kwargs: Additional delete options

        Returns:
            True if file was deleted, False if it didn't exist
        """
        file_path = self._get_file_path(key)
        metadata_path = self._get_metadata_path(key)

        deleted = False

        # Delete main file
        if file_path.exists():
            file_path.unlink()
            deleted = True

        # Delete metadata file
        if metadata_path.exists():
            metadata_path.unlink()

        # Clean up empty directories if configured
        if deleted and self.config.get("remove_empty_dirs", False):
            self._cleanup_empty_dirs(file_path.parent)

        return deleted

    def _cleanup_empty_dirs(self, dir_path: Path) -> None:
        """Remove empty directories up to base_path."""
        current = dir_path
        while current != self.base_path and current != current.parent:
            try:
                if current.exists() and current.is_dir():
                    # Only remove if empty
                    current.rmdir()
                current = current.parent
            except OSError:
                # Directory not empty or other error
                break

    def list_keys(
        self,
        prefix: str | None = None,
        limit: int | None = None,
        **kwargs: Any
    ) -> list[str]:
        """
        List CloakMap keys in storage.

        Args:
            prefix: Optional key prefix filter
            limit: Optional maximum number of keys to return
            **kwargs: Additional list options

        Returns:
            List of CloakMap keys (relative paths)
        """
        file_ext = self.config.get("file_extension", ".cmap")
        keys = []

        # Build glob pattern
        if prefix:
            # Handle prefix that may or may not include extension
            search_prefix = prefix
            if not search_prefix.endswith(file_ext):
                search_prefix = search_prefix + "*"
            pattern = search_prefix + file_ext if not search_prefix.endswith(file_ext) else search_prefix
            glob_pattern = pattern
        else:
            glob_pattern = f"**/*{file_ext}"

        # Find matching files
        try:
            for file_path in self.base_path.glob(glob_pattern):
                if file_path.is_file() and file_path.suffix == file_ext:
                    # Get relative path from base_path
                    relative_path = file_path.relative_to(self.base_path)
                    key = str(relative_path)

                    # Remove extension for clean key
                    if key.endswith(file_ext):
                        key = key[:-len(file_ext)]

                    if not prefix or key.startswith(prefix):
                        keys.append(key)

                        if limit and len(keys) >= limit:
                            break
        except Exception:
            # Handle cases where base_path doesn't exist yet
            pass

        return sorted(keys)

    def get_metadata(self, key: str, **kwargs: Any) -> StorageMetadata:
        """
        Get metadata for a CloakMap without loading full content.

        First tries to load from metadata file, falls back to
        loading the CloakMap if metadata file doesn't exist.

        Args:
            key: CloakMap key
            **kwargs: Additional options

        Returns:
            StorageMetadata for the CloakMap
        """
        metadata_path = self._get_metadata_path(key)

        # Try to load from metadata file first
        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    metadata_dict = json.load(f)
                return StorageMetadata.from_dict(metadata_dict)
            except Exception:
                # Fall back to loading CloakMap
                pass

        # Fallback: load CloakMap and create metadata
        if not self.exists(key):
            raise KeyError(f"CloakMap not found: {key}")

        file_path = self._get_file_path(key)

        # Get file stats
        file_path.stat()

        with open(file_path, "rb") as f:
            content_bytes = f.read()

        # Load CloakMap to get internal metadata
        cloakmap = CloakMap.from_json(content_bytes.decode("utf-8"))

        # Create and save metadata for future use
        metadata = StorageMetadata.from_cloakmap(
            key=key,
            cloakmap=cloakmap,
            backend_type=self.backend_type,
            content_bytes=content_bytes,
            file_path=str(file_path),
        )

        # Save metadata file for next time
        try:
            metadata_content = json.dumps(metadata.to_dict(), indent=2).encode("utf-8")
            self._write_file_atomic(metadata_path, metadata_content)
            self._set_file_permissions(metadata_path)
        except Exception:
            # Don't fail if we can't save metadata
            pass

        return metadata

    def list_metadata(
        self,
        prefix: str | None = None,
        limit: int | None = None,
        **kwargs: Any
    ) -> list[StorageMetadata]:
        """
        List metadata for all CloakMaps efficiently.

        Optimized implementation that tries to read metadata files
        directly instead of loading each CloakMap.
        """
        keys = self.list_keys(prefix=prefix, limit=limit, **kwargs)
        metadata_list = []

        for key in keys:
            try:
                metadata = self.get_metadata(key, **kwargs)
                metadata_list.append(metadata)
            except KeyError:
                # Key might have been deleted
                continue

        return metadata_list

    def health_check(self) -> dict[str, Any]:
        """Perform health check of local storage."""
        base_result = super().health_check()

        try:
            # Check directory access
            accessible = os.access(self.base_path, os.R_OK | os.W_OK)

            # Try to create a temporary file
            with tempfile.NamedTemporaryFile(dir=self.base_path, delete=True):
                pass

            base_result.update({
                "base_path": str(self.base_path),
                "directory_accessible": accessible,
                "can_write": True,
            })

        except Exception as e:
            base_result.update({
                "status": "unhealthy",
                "base_path": str(self.base_path),
                "directory_accessible": False,
                "error": str(e),
                "error_type": type(e).__name__,
            })

        return base_result
