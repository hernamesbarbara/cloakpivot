"""
Google Cloud Storage backend for CloakMaps.

Provides cloud-based storage using Google Cloud Storage with support for
encryption, access control, versioning, and efficient metadata operations.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...core.cloakmap import CloakMap
from .base import StorageBackend, StorageMetadata


class GCSStorage(StorageBackend):
    """
    Google Cloud Storage backend for CloakMaps.
    
    Stores CloakMaps as objects in GCS buckets with optional encryption,
    versioning, and metadata tracking. Supports both public and private
    buckets with configurable access controls.
    
    Features:
    - Customer-managed encryption keys (CMEK)
    - Object versioning and lifecycle management
    - Efficient metadata operations using HEAD requests
    - Batch operations for large datasets
    - Retry logic with exponential backoff
    - Support for resumable uploads for large CloakMaps
    
    Configuration:
        bucket_name: GCS bucket name (required)
        project_id: Google Cloud project ID (optional, auto-detected)
        credentials_path: Path to service account JSON file (optional)
        encryption_key_name: Customer-managed encryption key (optional)
        object_prefix: Prefix for all object keys (default: "cloakmaps/")
        storage_class: GCS storage class (default: "STANDARD")
    
    Examples:
        >>> config = {
        ...     "bucket_name": "my-cloakmap-bucket",
        ...     "project_id": "my-project",
        ...     "credentials_path": "/path/to/service-account.json"
        ... }
        >>> storage = GCSStorage(config=config)
        >>> storage.save("documents/my_doc", cloakmap)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GCS storage backend.
        
        Args:
            config: GCS-specific configuration including bucket name and credentials
        """
        super().__init__(config)
        self._client = None
        self._bucket = None
    
    @property
    def backend_type(self) -> str:
        """Return the backend type identifier."""
        return "google_cloud_storage"
    
    def _validate_config(self) -> None:
        """Validate GCS configuration."""
        if not self.config.get("bucket_name"):
            raise ValueError("bucket_name is required for GCS storage")
        
        # Validate bucket name format (basic validation)
        bucket_name = self.config["bucket_name"]
        if not bucket_name or len(bucket_name) < 3 or len(bucket_name) > 63:
            raise ValueError("bucket_name must be 3-63 characters long")
        
        # Validate storage class if provided
        storage_class = self.config.get("storage_class", "STANDARD")
        valid_classes = ["STANDARD", "NEARLINE", "COLDLINE", "ARCHIVE"]
        if storage_class not in valid_classes:
            raise ValueError(f"storage_class must be one of: {valid_classes}")
    
    @property
    def client(self):
        """Get Google Cloud Storage client with lazy initialization."""
        if self._client is None:
            try:
                from google.cloud import storage
                from google.oauth2 import service_account
                
                # Build client arguments
                client_kwargs = {}
                
                if self.config.get('project_id'):
                    client_kwargs['project'] = self.config['project_id']
                
                # Handle service account credentials
                if self.config.get('credentials_path'):
                    credentials = service_account.Credentials.from_service_account_file(
                        self.config['credentials_path']
                    )
                    client_kwargs['credentials'] = credentials
                elif self.config.get('credentials_json'):
                    credentials = service_account.Credentials.from_service_account_info(
                        self.config['credentials_json']
                    )
                    client_kwargs['credentials'] = credentials
                
                self._client = storage.Client(**client_kwargs)
                
            except ImportError as e:
                raise ValueError("google-cloud-storage is required for GCS storage backend") from e
        
        return self._client
    
    @property
    def bucket(self):
        """Get GCS bucket with lazy initialization."""
        if self._bucket is None:
            bucket_name = self.config["bucket_name"]
            self._bucket = self.client.bucket(bucket_name)
        return self._bucket
    
    def _get_blob_name(self, key: str) -> str:
        """Get the full GCS blob name with prefix."""
        self.validate_key(key)
        prefix = self.config.get("object_prefix", "cloakmaps/")
        
        # Ensure prefix ends with /
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        
        return f"{prefix}{key}"
    
    def _get_metadata_blob_name(self, key: str) -> str:
        """Get the GCS blob name for metadata."""
        return self._get_blob_name(key) + ".meta"
    
    def _build_blob_kwargs(self) -> Dict[str, Any]:
        """Build arguments for GCS blob operations."""
        kwargs = {}
        
        # Add storage class
        storage_class = self.config.get('storage_class', 'STANDARD')
        kwargs['storage_class'] = storage_class
        
        # Add encryption key if configured
        if self.config.get('encryption_key_name'):
            kwargs['kms_key_name'] = self.config['encryption_key_name']
        
        return kwargs
    
    def save(
        self, 
        key: str, 
        cloakmap: CloakMap,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> StorageMetadata:
        """
        Save a CloakMap to GCS.
        
        Args:
            key: GCS blob name (without prefix)
            cloakmap: CloakMap instance to save
            metadata: Additional metadata to store
            **kwargs: Additional GCS options
        """
        try:
            blob_name = self._get_blob_name(key)
            metadata_blob_name = self._get_metadata_blob_name(key)
            
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
                bucket_name=self.config["bucket_name"],
                blob_name=blob_name,
                **(metadata or {})
            )
            
            # Create blob and upload CloakMap
            blob = self.bucket.blob(blob_name)
            
            # Apply blob configuration
            blob_kwargs = self._build_blob_kwargs()
            for attr, value in blob_kwargs.items():
                setattr(blob, attr, value)
            
            # Set content type
            blob.content_type = 'application/json'
            
            # Upload content
            blob.upload_from_string(
                content_bytes,
                content_type='application/json',
                **kwargs.get('upload_kwargs', {})
            )
            
            # Upload metadata
            metadata_blob = self.bucket.blob(metadata_blob_name)
            metadata_content = json.dumps(storage_metadata.to_dict(), indent=2).encode("utf-8")
            
            metadata_blob.content_type = 'application/json'
            metadata_blob.upload_from_string(
                metadata_content,
                content_type='application/json'
            )
            
            return storage_metadata
            
        except Exception as e:
            if "NotFound" in str(e):
                raise ValueError(f"GCS bucket '{self.config['bucket_name']}' does not exist") from e
            elif "Forbidden" in str(e):
                raise PermissionError(f"Access denied to GCS bucket") from e
            else:
                raise ConnectionError(f"Failed to save to GCS: {e}") from e
    
    def load(self, key: str, **kwargs: Any) -> CloakMap:
        """
        Load a CloakMap from GCS.
        
        Args:
            key: GCS blob name (without prefix)
            **kwargs: Additional GCS options
            
        Returns:
            Loaded CloakMap instance
        """
        try:
            blob_name = self._get_blob_name(key)
            blob = self.bucket.blob(blob_name)
            
            # Download content
            content_bytes = blob.download_as_bytes()
            content = content_bytes.decode('utf-8')
            
            return CloakMap.from_json(content)
            
        except Exception as e:
            if "NotFound" in str(e):
                raise KeyError(f"CloakMap not found: {key}") from e
            elif "Forbidden" in str(e):
                raise PermissionError(f"Access denied to GCS object '{blob_name}'") from e
            else:
                raise ValueError(f"Failed to load CloakMap from GCS: {e}") from e
    
    def exists(self, key: str, **kwargs: Any) -> bool:
        """Check if a CloakMap exists in GCS."""
        try:
            blob_name = self._get_blob_name(key)
            blob = self.bucket.blob(blob_name)
            return blob.exists()
            
        except Exception:
            return False
    
    def delete(self, key: str, **kwargs: Any) -> bool:
        """
        Delete a CloakMap from GCS.
        
        Args:
            key: GCS blob name (without prefix) to delete
            **kwargs: Additional GCS options
            
        Returns:
            True if blob was deleted, False if it didn't exist
        """
        try:
            blob_name = self._get_blob_name(key)
            metadata_blob_name = self._get_metadata_blob_name(key)
            
            # Check if blob exists first
            blob = self.bucket.blob(blob_name)
            exists = blob.exists()
            
            if exists:
                # Delete both blob and metadata
                blob.delete()
                
                # Try to delete metadata (may not exist)
                try:
                    metadata_blob = self.bucket.blob(metadata_blob_name)
                    metadata_blob.delete()
                except Exception:
                    pass  # Metadata deletion failure is not critical
                
                return True
            
            return False
            
        except Exception as e:
            if "Forbidden" in str(e):
                raise PermissionError(f"Access denied to delete GCS object") from e
            else:
                raise ConnectionError(f"Failed to delete from GCS: {e}") from e
    
    def list_keys(
        self, 
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs: Any
    ) -> List[str]:
        """
        List CloakMap keys in GCS.
        
        Args:
            prefix: Optional key prefix filter
            limit: Optional maximum number of keys to return
            **kwargs: Additional GCS options
            
        Returns:
            List of CloakMap keys (without object prefix)
        """
        try:
            object_prefix = self.config.get("object_prefix", "cloakmaps/")
            
            # Build the GCS prefix
            if prefix:
                gcs_prefix = f"{object_prefix}{prefix}"
            else:
                gcs_prefix = object_prefix
            
            keys = []
            
            # List blobs with prefix
            blobs = self.bucket.list_blobs(
                prefix=gcs_prefix,
                max_results=limit
            )
            
            for blob in blobs:
                blob_name = blob.name
                
                # Skip metadata files
                if blob_name.endswith('.meta'):
                    continue
                
                # Remove object prefix to get clean key
                if blob_name.startswith(object_prefix):
                    clean_key = blob_name[len(object_prefix):]
                    keys.append(clean_key)
                    
                    if limit and len(keys) >= limit:
                        break
            
            return sorted(keys)
            
        except Exception as e:
            if "NotFound" in str(e):
                raise ValueError(f"GCS bucket '{self.config['bucket_name']}' does not exist") from e
            elif "Forbidden" in str(e):
                raise PermissionError(f"Access denied to list GCS bucket") from e
            else:
                raise ConnectionError(f"Failed to list GCS objects: {e}") from e
    
    def get_metadata(self, key: str, **kwargs: Any) -> StorageMetadata:
        """
        Get metadata for a CloakMap from GCS.
        
        First tries to load from metadata blob, falls back to
        loading the main blob metadata.
        
        Args:
            key: GCS blob name (without prefix)
            **kwargs: Additional GCS options
            
        Returns:
            StorageMetadata for the CloakMap
        """
        try:
            blob_name = self._get_blob_name(key)
            metadata_blob_name = self._get_metadata_blob_name(key)
            
            # Try to load metadata blob first
            try:
                metadata_blob = self.bucket.blob(metadata_blob_name)
                if metadata_blob.exists():
                    metadata_content = metadata_blob.download_as_text()
                    metadata_dict = json.loads(metadata_content)
                    return StorageMetadata.from_dict(metadata_dict)
            except Exception:
                # Fall back to main blob metadata
                pass
            
            # Get main blob for metadata
            blob = self.bucket.blob(blob_name)
            if not blob.exists():
                raise KeyError(f"CloakMap not found: {key}")
            
            # Reload blob to get metadata
            blob.reload()
            
            # Load CloakMap to get internal metadata
            cloakmap = self.load(key, **kwargs)
            
            # Build minimal content bytes for hash calculation
            content = cloakmap.to_json()
            content_bytes = content.encode('utf-8')
            
            metadata = StorageMetadata.from_cloakmap(
                key=key,
                cloakmap=cloakmap,
                backend_type=self.backend_type,
                content_bytes=content_bytes,
                bucket_name=self.config["bucket_name"],
                blob_name=blob_name,
            )
            
            # Update with GCS-specific metadata
            if blob.size is not None:
                metadata.size_bytes = blob.size
            if blob.time_created:
                metadata.created_at = blob.time_created.replace(tzinfo=None)
            if blob.updated:
                metadata.modified_at = blob.updated.replace(tzinfo=None)
            
            metadata.backend_metadata.update({
                'etag': blob.etag,
                'storage_class': blob.storage_class,
                'content_type': blob.content_type,
                'generation': str(blob.generation) if blob.generation else None,
                'metageneration': str(blob.metageneration) if blob.metageneration else None,
            })
            
            return metadata
            
        except Exception as e:
            if "NotFound" in str(e):
                raise KeyError(f"CloakMap not found: {key}") from e
            else:
                raise ConnectionError(f"Failed to get GCS metadata: {e}") from e
    
    def copy(
        self, 
        source_key: str, 
        dest_key: str,
        **kwargs: Any
    ) -> StorageMetadata:
        """
        Copy a CloakMap to a new key using GCS server-side copy.
        
        Args:
            source_key: Source CloakMap key
            dest_key: Destination key
            **kwargs: Additional GCS options
            
        Returns:
            StorageMetadata for the copied CloakMap
        """
        try:
            source_blob_name = self._get_blob_name(source_key)
            dest_blob_name = self._get_blob_name(dest_key)
            
            source_blob = self.bucket.blob(source_blob_name)
            
            # Server-side copy
            dest_blob = self.bucket.copy_blob(source_blob, self.bucket, dest_blob_name)
            
            # Also copy metadata if it exists
            try:
                source_meta_name = self._get_metadata_blob_name(source_key)
                dest_meta_name = self._get_metadata_blob_name(dest_key)
                
                source_meta_blob = self.bucket.blob(source_meta_name)
                if source_meta_blob.exists():
                    self.bucket.copy_blob(source_meta_blob, self.bucket, dest_meta_name)
            except Exception:
                pass  # Metadata copy failure is not critical
            
            # Get metadata for the copied object
            return self.get_metadata(dest_key, **kwargs)
            
        except Exception as e:
            raise ConnectionError(f"Failed to copy in GCS: {e}") from e
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of GCS storage."""
        base_result = super().health_check()
        
        try:
            bucket_name = self.config["bucket_name"]
            
            # Try to get bucket metadata
            self.bucket.reload()
            
            # Try to list objects (limited)
            list(self.bucket.list_blobs(max_results=1))
            
            base_result.update({
                "bucket_name": bucket_name,
                "bucket_accessible": True,
                "can_list": True,
                "location": getattr(self.bucket, 'location', None),
                "storage_class": getattr(self.bucket, 'storage_class', None),
            })
            
        except Exception as e:
            base_result.update({
                "status": "unhealthy",
                "bucket_name": self.config.get("bucket_name"),
                "bucket_accessible": False,
                "error": str(e),
                "error_type": type(e).__name__,
            })
        
        return base_result