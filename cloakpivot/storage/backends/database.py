"""
Database storage backend for CloakMaps.

Provides database-based storage with support for SQLite, PostgreSQL,
and other SQL databases, with connection pooling and transaction support.
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ...core.cloakmap import CloakMap
from .base import StorageBackend, StorageMetadata


class DatabaseStorage(StorageBackend):
    """
    Database storage backend for CloakMaps.

    Stores CloakMaps and metadata in a relational database with support
    for SQLite, PostgreSQL, and other SQL databases. Provides ACID
    transactions, concurrent access, and efficient querying.

    Features:
    - ACID transaction support
    - Connection pooling for PostgreSQL
    - Automatic schema creation and migration
    - Efficient metadata queries without loading content
    - Support for concurrent access with proper locking
    - Indexing for fast key lookups and prefix searches

    Configuration:
        database_url: Database connection URL (required)
            - SQLite: "sqlite:///path/to/database.db"
            - PostgreSQL: "postgresql://user:pass@host:port/database"
        pool_size: Connection pool size for PostgreSQL (default: 5)
        pool_max_overflow: Maximum overflow connections (default: 10)
        timeout: Connection timeout in seconds (default: 30)
        create_schema: Whether to create schema automatically (default: True)

    Examples:
        >>> # SQLite
        >>> config = {"database_url": "sqlite:///cloakmaps.db"}
        >>> storage = DatabaseStorage(config=config)
        >>>
        >>> # PostgreSQL
        >>> config = {
        ...     "database_url": "postgresql://user:pass@localhost/cloakmaps",
        ...     "pool_size": 10
        ... }
        >>> storage = DatabaseStorage(config=config)
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        Initialize database storage backend.

        Args:
            config: Database-specific configuration including connection URL
        """
        super().__init__(config)
        self._connection = None
        self._engine = None
        self._lock = threading.RLock()

        # Initialize schema if requested
        if self.config.get("create_schema", True):
            self._create_schema()

    @property
    def backend_type(self) -> str:
        """Return the backend type identifier."""
        return "database"

    def _validate_config(self) -> None:
        """Validate database configuration."""
        if not self.config.get("database_url"):
            raise ValueError("database_url is required for database storage")

        database_url = self.config["database_url"]
        if not database_url.startswith(("sqlite://", "postgresql://", "mysql://")):
            raise ValueError("database_url must start with sqlite://, postgresql://, or mysql://")

    @property
    def connection(self):
        """Get database connection with lazy initialization."""
        if self._connection is None:
            with self._lock:
                if self._connection is None:
                    self._connection = self._create_connection()
        return self._connection

    def _create_connection(self):
        """Create database connection based on URL scheme."""
        database_url = self.config["database_url"]

        if database_url.startswith("sqlite://"):
            # SQLite connection
            db_path = database_url[9:]  # Remove "sqlite://"

            # Handle memory databases (remove leading slash if present)
            if db_path == ":memory:" or db_path == "/:memory:":
                conn = sqlite3.connect(":memory:", check_same_thread=False)
            else:
                # Ensure directory exists
                db_path = Path(db_path).resolve()
                db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(db_path), check_same_thread=False)

            # Configure SQLite for better concurrent access
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")

            return conn

        elif database_url.startswith("postgresql://"):
            # PostgreSQL connection (requires psycopg2)
            try:
                import psycopg2

                # Create connection pool
                pool_size = self.config.get("pool_size", 5)
                max_overflow = self.config.get("pool_max_overflow", 10)

                self._pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=pool_size + max_overflow,
                    dsn=database_url
                )

                # Get initial connection from pool
                return self._pool.getconn()

            except ImportError as e:
                raise ValueError("psycopg2 is required for PostgreSQL storage backend") from e

        else:
            raise ValueError(f"Unsupported database URL: {database_url}")

    def _create_schema(self) -> None:
        """Create database schema for CloakMaps."""
        database_url = self.config["database_url"]

        if database_url.startswith("sqlite://"):
            self._create_sqlite_schema()
        elif database_url.startswith("postgresql://"):
            self._create_postgresql_schema()

    def _create_sqlite_schema(self) -> None:
        """Create SQLite schema."""
        with self._lock:
            conn = self.connection
            cursor = conn.cursor()

            # Create cloakmaps table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cloakmaps (
                    key TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    anchor_count INTEGER NOT NULL,
                    is_encrypted BOOLEAN NOT NULL,
                    content_hash TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    modified_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cloakmap_metadata (
                    key TEXT PRIMARY KEY,
                    metadata_json TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    modified_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (key) REFERENCES cloakmaps (key) ON DELETE CASCADE
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cloakmaps_doc_id ON cloakmaps (doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cloakmaps_created_at ON cloakmaps (created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cloakmaps_version ON cloakmaps (version)")

            conn.commit()

    def _create_postgresql_schema(self) -> None:
        """Create PostgreSQL schema."""
        with self._lock:
            conn = self.connection
            cursor = conn.cursor()

            # Create cloakmaps table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cloakmaps (
                    key TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    anchor_count INTEGER NOT NULL,
                    is_encrypted BOOLEAN NOT NULL,
                    content_hash TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    modified_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cloakmap_metadata (
                    key TEXT PRIMARY KEY,
                    metadata_json TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    modified_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (key) REFERENCES cloakmaps (key) ON DELETE CASCADE
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cloakmaps_doc_id ON cloakmaps (doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cloakmaps_created_at ON cloakmaps (created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cloakmaps_version ON cloakmaps (version)")

            conn.commit()

    def save(
        self,
        key: str,
        cloakmap: CloakMap,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any
    ) -> StorageMetadata:
        """
        Save a CloakMap to database.

        Args:
            key: Database key for the CloakMap
            cloakmap: CloakMap instance to save
            metadata: Additional metadata to store
            **kwargs: Additional database options
        """
        try:
            self.validate_key(key)

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
                database_url=self.config["database_url"],
                **(metadata or {})
            )

            with self._lock:
                conn = self.connection
                cursor = conn.cursor()

                # Insert/update CloakMap
                if self.config["database_url"].startswith("sqlite://"):
                    cursor.execute("""
                        INSERT OR REPLACE INTO cloakmaps
                        (key, content, doc_id, version, anchor_count, is_encrypted,
                         content_hash, size_bytes, created_at, modified_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        key, content, cloakmap.doc_id, cloakmap.version,
                        cloakmap.anchor_count, cloakmap.is_encrypted,
                        storage_metadata.content_hash, storage_metadata.size_bytes,
                        storage_metadata.created_at, storage_metadata.modified_at
                    ))
                else:
                    cursor.execute("""
                        INSERT INTO cloakmaps
                        (key, content, doc_id, version, anchor_count, is_encrypted,
                         content_hash, size_bytes, created_at, modified_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (key) DO UPDATE SET
                        content = EXCLUDED.content,
                        doc_id = EXCLUDED.doc_id,
                        version = EXCLUDED.version,
                        anchor_count = EXCLUDED.anchor_count,
                        is_encrypted = EXCLUDED.is_encrypted,
                        content_hash = EXCLUDED.content_hash,
                        size_bytes = EXCLUDED.size_bytes,
                        modified_at = EXCLUDED.modified_at
                    """, (
                        key, content, cloakmap.doc_id, cloakmap.version,
                        cloakmap.anchor_count, cloakmap.is_encrypted,
                        storage_metadata.content_hash, storage_metadata.size_bytes,
                        storage_metadata.created_at, storage_metadata.modified_at
                    ))

                # Insert/update metadata
                metadata_json = json.dumps(storage_metadata.to_dict())

                if self.config["database_url"].startswith("sqlite://"):
                    cursor.execute("""
                        INSERT OR REPLACE INTO cloakmap_metadata
                        (key, metadata_json, created_at, modified_at)
                        VALUES (?, ?, ?, ?)
                    """, (
                        key, metadata_json, storage_metadata.created_at,
                        storage_metadata.modified_at
                    ))
                else:
                    cursor.execute("""
                        INSERT INTO cloakmap_metadata
                        (key, metadata_json, created_at, modified_at)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (key) DO UPDATE SET
                        metadata_json = EXCLUDED.metadata_json,
                        modified_at = EXCLUDED.modified_at
                    """, (
                        key, metadata_json, storage_metadata.created_at,
                        storage_metadata.modified_at
                    ))

                conn.commit()

            return storage_metadata

        except Exception as e:
            if hasattr(conn, 'rollback'):
                conn.rollback()
            raise ValueError(f"Failed to save CloakMap to database: {e}") from e

    def load(self, key: str, **kwargs: Any) -> CloakMap:
        """
        Load a CloakMap from database.

        Args:
            key: Database key for the CloakMap
            **kwargs: Additional database options

        Returns:
            Loaded CloakMap instance
        """
        try:
            with self._lock:
                conn = self.connection
                cursor = conn.cursor()

                if self.config["database_url"].startswith("sqlite://"):
                    cursor.execute("SELECT content FROM cloakmaps WHERE key = ?", (key,))
                else:
                    cursor.execute("SELECT content FROM cloakmaps WHERE key = %s", (key,))

                row = cursor.fetchone()
                if not row:
                    raise KeyError(f"CloakMap not found: {key}")

                content = row[0]
                return CloakMap.from_json(content)

        except Exception as e:
            if isinstance(e, KeyError):
                raise
            raise ValueError(f"Failed to load CloakMap from database: {e}") from e

    def exists(self, key: str, **kwargs: Any) -> bool:
        """Check if a CloakMap exists in database."""
        try:
            with self._lock:
                conn = self.connection
                cursor = conn.cursor()

                if self.config["database_url"].startswith("sqlite://"):
                    cursor.execute("SELECT 1 FROM cloakmaps WHERE key = ?", (key,))
                else:
                    cursor.execute("SELECT 1 FROM cloakmaps WHERE key = %s", (key,))

                return cursor.fetchone() is not None

        except Exception:
            return False

    def delete(self, key: str, **kwargs: Any) -> bool:
        """
        Delete a CloakMap from database.

        Args:
            key: Database key to delete
            **kwargs: Additional database options

        Returns:
            True if CloakMap was deleted, False if it didn't exist
        """
        try:
            with self._lock:
                conn = self.connection
                cursor = conn.cursor()

                # Check if exists first
                if not self.exists(key):
                    return False

                # Delete (metadata will be cascade deleted)
                if self.config["database_url"].startswith("sqlite://"):
                    cursor.execute("DELETE FROM cloakmaps WHERE key = ?", (key,))
                else:
                    cursor.execute("DELETE FROM cloakmaps WHERE key = %s", (key,))

                conn.commit()
                return True

        except Exception as e:
            if hasattr(conn, 'rollback'):
                conn.rollback()
            raise ConnectionError(f"Failed to delete from database: {e}") from e

    def list_keys(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs: Any
    ) -> list[str]:
        """
        List CloakMap keys in database.

        Args:
            prefix: Optional key prefix filter
            limit: Optional maximum number of keys to return
            **kwargs: Additional database options

        Returns:
            List of CloakMap keys
        """
        try:
            with self._lock:
                conn = self.connection
                cursor = conn.cursor()

                # Build query
                query = "SELECT key FROM cloakmaps"
                params = []

                if prefix:
                    if self.config["database_url"].startswith("sqlite://"):
                        query += " WHERE key LIKE ?"
                        params.append(f"{prefix}%")
                    else:
                        query += " WHERE key LIKE %s"
                        params.append(f"{prefix}%")

                query += " ORDER BY key"

                if limit:
                    if self.config["database_url"].startswith("sqlite://"):
                        query += " LIMIT ?"
                        params.append(limit)
                    else:
                        query += " LIMIT %s"
                        params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                return [row[0] for row in rows]

        except Exception as e:
            raise ConnectionError(f"Failed to list database keys: {e}") from e

    def get_metadata(self, key: str, **kwargs: Any) -> StorageMetadata:
        """
        Get metadata for a CloakMap from database.

        Args:
            key: Database key
            **kwargs: Additional database options

        Returns:
            StorageMetadata for the CloakMap
        """
        try:
            with self._lock:
                conn = self.connection
                cursor = conn.cursor()

                # Try to get from metadata table first
                if self.config["database_url"].startswith("sqlite://"):
                    cursor.execute("SELECT metadata_json FROM cloakmap_metadata WHERE key = ?", (key,))
                else:
                    cursor.execute("SELECT metadata_json FROM cloakmap_metadata WHERE key = %s", (key,))

                row = cursor.fetchone()
                if row:
                    metadata_dict = json.loads(row[0])
                    return StorageMetadata.from_dict(metadata_dict)

                # Fallback to constructing from main table
                if self.config["database_url"].startswith("sqlite://"):
                    cursor.execute("""
                        SELECT doc_id, version, anchor_count, is_encrypted,
                               content_hash, size_bytes, created_at, modified_at
                        FROM cloakmaps WHERE key = ?
                    """, (key,))
                else:
                    cursor.execute("""
                        SELECT doc_id, version, anchor_count, is_encrypted,
                               content_hash, size_bytes, created_at, modified_at
                        FROM cloakmaps WHERE key = %s
                    """, (key,))

                row = cursor.fetchone()
                if not row:
                    raise KeyError(f"CloakMap not found: {key}")

                doc_id, version, anchor_count, is_encrypted, content_hash, size_bytes, created_at, modified_at = row

                # Parse timestamps
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at)
                if isinstance(modified_at, str):
                    modified_at = datetime.fromisoformat(modified_at)

                return StorageMetadata(
                    key=key,
                    size_bytes=size_bytes,
                    content_hash=content_hash,
                    created_at=created_at,
                    modified_at=modified_at,
                    doc_id=doc_id,
                    version=version,
                    anchor_count=anchor_count,
                    is_encrypted=bool(is_encrypted),
                    backend_type=self.backend_type,
                )

        except Exception as e:
            if isinstance(e, KeyError):
                raise
            raise ConnectionError(f"Failed to get database metadata: {e}") from e

    def health_check(self) -> dict[str, Any]:
        """Perform health check of database storage."""
        base_result = super().health_check()

        try:
            with self._lock:
                conn = self.connection
                cursor = conn.cursor()

                # Test basic query
                cursor.execute("SELECT COUNT(*) FROM cloakmaps")
                count = cursor.fetchone()[0]

            base_result.update({
                "database_url": self.config["database_url"].split("@")[-1],  # Hide credentials
                "connection_active": True,
                "cloakmap_count": count,
            })

        except Exception as e:
            base_result.update({
                "status": "unhealthy",
                "database_url": self.config.get("database_url", "").split("@")[-1],
                "connection_active": False,
                "error": str(e),
                "error_type": type(e).__name__,
            })

        return base_result
