"""
Base storage backend interface for CloakMaps.

Defines the abstract interface that all storage backends must implement,
providing a consistent API for storing and retrieving CloakMaps across
different storage systems.
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ...core.cloakmap import CloakMap


@dataclass
class StorageMetadata:
    """
    Metadata associated with stored CloakMaps.

    This includes information about the storage operation, integrity checks,
    and backend-specific metadata for efficient retrieval and validation.
    """

    # Core metadata
    key: str
    size_bytes: int
    content_hash: str
    created_at: datetime
    modified_at: datetime

    # CloakMap specific metadata
    doc_id: str
    version: str
    anchor_count: int
    is_encrypted: bool

    # Backend specific metadata
    backend_type: str
    backend_metadata: dict[str, Any] = field(default_factory=dict)

    # Storage integrity
    checksum: str | None = None
    storage_version: str | None = None

    @classmethod
    def from_cloakmap(
        cls,
        key: str,
        cloakmap: CloakMap,
        backend_type: str,
        content_bytes: bytes,
        **backend_metadata: Any,
    ) -> "StorageMetadata":
        """Create metadata from a CloakMap and its serialized content."""
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        now = datetime.utcnow()

        return cls(
            key=key,
            size_bytes=len(content_bytes),
            content_hash=content_hash,
            created_at=now,
            modified_at=now,
            doc_id=cloakmap.doc_id,
            version=cloakmap.version,
            anchor_count=cloakmap.anchor_count,
            is_encrypted=cloakmap.is_encrypted,
            backend_type=backend_type,
            backend_metadata=backend_metadata,
            checksum=content_hash,  # Use content hash as checksum
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "key": self.key,
            "size_bytes": self.size_bytes,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "doc_id": self.doc_id,
            "version": self.version,
            "anchor_count": self.anchor_count,
            "is_encrypted": self.is_encrypted,
            "backend_type": self.backend_type,
            "backend_metadata": self.backend_metadata,
            "checksum": self.checksum,
            "storage_version": self.storage_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StorageMetadata":
        """Create metadata from dictionary representation."""
        return cls(
            key=data["key"],
            size_bytes=data["size_bytes"],
            content_hash=data["content_hash"],
            created_at=datetime.fromisoformat(data["created_at"]),
            modified_at=datetime.fromisoformat(data["modified_at"]),
            doc_id=data["doc_id"],
            version=data["version"],
            anchor_count=data["anchor_count"],
            is_encrypted=data["is_encrypted"],
            backend_type=data["backend_type"],
            backend_metadata=data.get("backend_metadata", {}),
            checksum=data.get("checksum"),
            storage_version=data.get("storage_version"),
        )


class StorageBackend(ABC):
    """
    Abstract base class for CloakMap storage backends.

    This defines the interface that all storage implementations must provide,
    ensuring consistent behavior across different storage systems like local
    files, cloud storage, and databases.

    Key Operations:
    - save: Store a CloakMap with a given key
    - load: Retrieve a CloakMap by key
    - exists: Check if a CloakMap exists
    - delete: Remove a CloakMap
    - list: List stored CloakMaps with optional filtering
    - get_metadata: Get metadata without loading full content

    Implementations should handle:
    - Authentication and authorization
    - Network failures and retries (for cloud backends)
    - Data integrity and checksums
    - Encryption at rest where supported
    - Connection pooling and resource management
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize storage backend with configuration.

        Args:
            config: Backend-specific configuration parameters
        """
        self.config = config or {}
        self._validate_config()

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Return the backend type identifier."""
        pass

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate backend-specific configuration."""
        pass

    @abstractmethod
    def save(
        self,
        key: str,
        cloakmap: CloakMap,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> StorageMetadata:
        """
        Save a CloakMap to storage.

        Args:
            key: Unique identifier for the CloakMap
            cloakmap: CloakMap instance to save
            metadata: Optional additional metadata
            **kwargs: Backend-specific save options

        Returns:
            StorageMetadata for the saved CloakMap

        Raises:
            ValueError: If key is invalid or CloakMap validation fails
            ConnectionError: If storage system is unreachable
            PermissionError: If write access is denied
        """
        pass

    @abstractmethod
    def load(self, key: str, **kwargs: Any) -> CloakMap:
        """
        Load a CloakMap from storage.

        Args:
            key: Unique identifier for the CloakMap
            **kwargs: Backend-specific load options

        Returns:
            Loaded CloakMap instance

        Raises:
            KeyError: If CloakMap with key doesn't exist
            ValueError: If stored data is invalid or corrupted
            ConnectionError: If storage system is unreachable
            PermissionError: If read access is denied
        """
        pass

    @abstractmethod
    def exists(self, key: str, **kwargs: Any) -> bool:
        """
        Check if a CloakMap exists in storage.

        Args:
            key: Unique identifier to check
            **kwargs: Backend-specific options

        Returns:
            True if CloakMap exists, False otherwise

        Raises:
            ConnectionError: If storage system is unreachable
        """
        pass

    @abstractmethod
    def delete(self, key: str, **kwargs: Any) -> bool:
        """
        Delete a CloakMap from storage.

        Args:
            key: Unique identifier for the CloakMap to delete
            **kwargs: Backend-specific delete options

        Returns:
            True if CloakMap was deleted, False if it didn't exist

        Raises:
            ConnectionError: If storage system is unreachable
            PermissionError: If delete access is denied
        """
        pass

    @abstractmethod
    def list_keys(
        self, prefix: str | None = None, limit: int | None = None, **kwargs: Any
    ) -> list[str]:
        """
        List CloakMap keys in storage.

        Args:
            prefix: Optional key prefix filter
            limit: Optional maximum number of keys to return
            **kwargs: Backend-specific list options

        Returns:
            List of CloakMap keys

        Raises:
            ConnectionError: If storage system is unreachable
            PermissionError: If list access is denied
        """
        pass

    @abstractmethod
    def get_metadata(self, key: str, **kwargs: Any) -> StorageMetadata:
        """
        Get metadata for a CloakMap without loading full content.

        Args:
            key: Unique identifier for the CloakMap
            **kwargs: Backend-specific options

        Returns:
            StorageMetadata for the CloakMap

        Raises:
            KeyError: If CloakMap with key doesn't exist
            ConnectionError: If storage system is unreachable
        """
        pass

    def list_metadata(
        self, prefix: str | None = None, limit: int | None = None, **kwargs: Any
    ) -> list[StorageMetadata]:
        """
        List metadata for all CloakMaps matching criteria.

        Default implementation calls get_metadata for each key from list_keys.
        Backends may override for more efficient bulk operations.

        Args:
            prefix: Optional key prefix filter
            limit: Optional maximum number of items to return
            **kwargs: Backend-specific options

        Returns:
            List of StorageMetadata objects
        """
        keys = self.list_keys(prefix=prefix, limit=limit, **kwargs)
        metadata = []

        for key in keys:
            try:
                meta = self.get_metadata(key, **kwargs)
                metadata.append(meta)
            except KeyError:
                # Key might have been deleted between list and get_metadata
                continue

        return metadata

    def copy(self, source_key: str, dest_key: str, **kwargs: Any) -> StorageMetadata:
        """
        Copy a CloakMap to a new key.

        Default implementation loads and saves. Backends may override
        for more efficient server-side copying.

        Args:
            source_key: Source CloakMap key
            dest_key: Destination key
            **kwargs: Backend-specific options

        Returns:
            StorageMetadata for the copied CloakMap
        """
        cloakmap = self.load(source_key, **kwargs)
        return self.save(dest_key, cloakmap, **kwargs)

    def move(self, source_key: str, dest_key: str, **kwargs: Any) -> StorageMetadata:
        """
        Move a CloakMap to a new key.

        Default implementation copies then deletes. Backends may override
        for atomic move operations.

        Args:
            source_key: Source CloakMap key
            dest_key: Destination key
            **kwargs: Backend-specific options

        Returns:
            StorageMetadata for the moved CloakMap
        """
        metadata = self.copy(source_key, dest_key, **kwargs)
        self.delete(source_key, **kwargs)
        return metadata

    def validate_key(self, key: str) -> None:
        """
        Validate that a key is acceptable for this backend.

        Args:
            key: Key to validate

        Raises:
            ValueError: If key is invalid for this backend
        """
        if not key or not key.strip():
            raise ValueError("Key cannot be empty")

        if len(key) > 1024:
            raise ValueError("Key too long (max 1024 characters)")

        # Check for potentially problematic characters
        invalid_chars = set("\x00\r\n")
        if any(c in invalid_chars for c in key):
            raise ValueError("Key contains invalid characters")

    def health_check(self) -> dict[str, Any]:
        """
        Perform a health check of the storage backend.

        Returns:
            Dictionary with health status and diagnostic information
        """
        try:
            # Basic connectivity test - try to list keys with limit 1
            self.list_keys(limit=1)
            return {
                "status": "healthy",
                "backend_type": self.backend_type,
                "timestamp": datetime.utcnow().isoformat(),
                "config": {
                    k: "***" if "key" in k.lower() or "secret" in k.lower() else v
                    for k, v in self.config.items()
                },
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend_type": self.backend_type,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def __str__(self) -> str:
        """String representation of the storage backend."""
        return f"{self.__class__.__name__}({self.backend_type})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(backend_type='{self.backend_type}', config={self.config})"
