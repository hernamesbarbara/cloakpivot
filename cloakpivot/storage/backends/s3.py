"""
Amazon S3 storage backend for CloakMaps.

Provides cloud-based storage using AWS S3 with support for encryption,
access control, versioning, and efficient metadata operations.
"""

import json
from typing import Any, Dict, List, Optional

from ...core.cloakmap import CloakMap
from .base import StorageBackend, StorageMetadata


class S3Storage(StorageBackend):
    """
    Amazon S3 storage backend for CloakMaps.
    
    Stores CloakMaps as objects in S3 buckets with optional encryption,
    versioning, and metadata tracking. Supports both public and private
    buckets with configurable access controls.
    
    Features:
    - Server-side encryption (SSE-S3, SSE-KMS)
    - Object versioning and lifecycle management
    - Efficient metadata operations using HEAD requests
    - Batch operations for large datasets
    - Retry logic with exponential backoff
    - Support for multi-part uploads for large CloakMaps
    
    Configuration:
        bucket_name: S3 bucket name (required)
        aws_access_key_id: AWS access key (optional, uses boto3 defaults)
        aws_secret_access_key: AWS secret key (optional, uses boto3 defaults)
        region_name: AWS region (optional, uses boto3 defaults)
        encryption: Server-side encryption configuration
        object_prefix: Prefix for all object keys (default: "cloakmaps/")
        storage_class: S3 storage class (default: "STANDARD")
    
    Examples:
        >>> config = {
        ...     "bucket_name": "my-cloakmap-bucket",
        ...     "region_name": "us-west-2",
        ...     "encryption": {"type": "SSE-S3"}
        ... }
        >>> storage = S3Storage(config=config)
        >>> storage.save("documents/my_doc", cloakmap)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize S3 storage backend.
        
        Args:
            config: S3-specific configuration including bucket name and credentials
        """
        super().__init__(config)
        self._s3_client = None
        self._s3_resource = None

    @property
    def backend_type(self) -> str:
        """Return the backend type identifier."""
        return "aws_s3"

    def _validate_config(self) -> None:
        """Validate S3 configuration."""
        if not self.config.get("bucket_name"):
            raise ValueError("bucket_name is required for S3 storage")

        # Validate bucket name format (basic validation)
        bucket_name = self.config["bucket_name"]
        if not bucket_name or len(bucket_name) < 3 or len(bucket_name) > 63:
            raise ValueError("bucket_name must be 3-63 characters long")

        # Validate encryption configuration if provided
        encryption = self.config.get("encryption", {})
        if encryption:
            enc_type = encryption.get("type")
            if enc_type not in ["SSE-S3", "SSE-KMS", "SSE-C"]:
                raise ValueError("encryption type must be SSE-S3, SSE-KMS, or SSE-C")

    @property
    def s3_client(self):
        """Get boto3 S3 client with lazy initialization."""
        if self._s3_client is None:
            try:
                import boto3
                from botocore.config import Config

                # Configure retry strategy
                config = Config(
                    retries={
                        'max_attempts': self.config.get('max_retries', 3),
                        'mode': 'adaptive'
                    }
                )

                # Build client arguments
                client_kwargs = {'config': config}

                if self.config.get('aws_access_key_id'):
                    client_kwargs['aws_access_key_id'] = self.config['aws_access_key_id']

                if self.config.get('aws_secret_access_key'):
                    client_kwargs['aws_secret_access_key'] = self.config['aws_secret_access_key']

                if self.config.get('region_name'):
                    client_kwargs['region_name'] = self.config['region_name']

                self._s3_client = boto3.client('s3', **client_kwargs)

            except ImportError as e:
                raise ValueError("boto3 is required for S3 storage backend") from e

        return self._s3_client

    @property
    def s3_resource(self):
        """Get boto3 S3 resource with lazy initialization."""
        if self._s3_resource is None:
            try:
                import boto3

                # Build resource arguments
                resource_kwargs = {}

                if self.config.get('aws_access_key_id'):
                    resource_kwargs['aws_access_key_id'] = self.config['aws_access_key_id']

                if self.config.get('aws_secret_access_key'):
                    resource_kwargs['aws_secret_access_key'] = self.config['aws_secret_access_key']

                if self.config.get('region_name'):
                    resource_kwargs['region_name'] = self.config['region_name']

                self._s3_resource = boto3.resource('s3', **resource_kwargs)

            except ImportError as e:
                raise ValueError("boto3 is required for S3 storage backend") from e

        return self._s3_resource

    def _get_object_key(self, key: str) -> str:
        """Get the full S3 object key with prefix."""
        self.validate_key(key)
        prefix = self.config.get("object_prefix", "cloakmaps/")

        # Ensure prefix ends with /
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        return f"{prefix}{key}"

    def _get_metadata_key(self, key: str) -> str:
        """Get the S3 object key for metadata."""
        return self._get_object_key(key) + ".meta"

    def _build_put_args(self, content_bytes: bytes) -> dict[str, Any]:
        """Build arguments for S3 put operations including encryption."""
        args = {
            'Body': content_bytes,
            'ContentType': 'application/json',
        }

        # Add storage class
        storage_class = self.config.get('storage_class', 'STANDARD')
        args['StorageClass'] = storage_class

        # Add encryption
        encryption = self.config.get('encryption', {})
        if encryption:
            enc_type = encryption.get('type')

            if enc_type == 'SSE-S3':
                args['ServerSideEncryption'] = 'AES256'

            elif enc_type == 'SSE-KMS':
                args['ServerSideEncryption'] = 'aws:kms'
                if encryption.get('kms_key_id'):
                    args['SSEKMSKeyId'] = encryption['kms_key_id']

            elif enc_type == 'SSE-C':
                if not encryption.get('customer_key'):
                    raise ValueError("customer_key required for SSE-C encryption")
                args['SSECustomerAlgorithm'] = 'AES256'
                args['SSECustomerKey'] = encryption['customer_key']

        return args

    def save(
        self,
        key: str,
        cloakmap: CloakMap,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> StorageMetadata:
        """
        Save a CloakMap to S3.
        
        Args:
            key: S3 object key (without prefix)
            cloakmap: CloakMap instance to save
            metadata: Additional metadata to store
            **kwargs: Additional S3 options
        """
        try:
            bucket_name = self.config["bucket_name"]
            object_key = self._get_object_key(key)
            metadata_key = self._get_metadata_key(key)

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
                bucket_name=bucket_name,
                object_key=object_key,
                **(metadata or {})
            )

            # Build put arguments
            put_args = self._build_put_args(content_bytes)

            # Upload CloakMap
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=object_key,
                **put_args
            )

            # Upload metadata
            metadata_content = json.dumps(storage_metadata.to_dict(), indent=2).encode("utf-8")
            metadata_put_args = self._build_put_args(metadata_content)

            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=metadata_key,
                **metadata_put_args
            )

            return storage_metadata

        except Exception as e:
            if "NoSuchBucket" in str(e):
                raise ValueError(f"S3 bucket '{bucket_name}' does not exist") from e
            elif "AccessDenied" in str(e):
                raise PermissionError(f"Access denied to S3 bucket '{bucket_name}'") from e
            else:
                raise ConnectionError(f"Failed to save to S3: {e}") from e

    def load(self, key: str, **kwargs: Any) -> CloakMap:
        """
        Load a CloakMap from S3.
        
        Args:
            key: S3 object key (without prefix)
            **kwargs: Additional S3 options
            
        Returns:
            Loaded CloakMap instance
        """
        try:
            bucket_name = self.config["bucket_name"]
            object_key = self._get_object_key(key)

            # Get object
            response = self.s3_client.get_object(
                Bucket=bucket_name,
                Key=object_key
            )

            content = response['Body'].read().decode('utf-8')
            return CloakMap.from_json(content)

        except Exception as e:
            if "NoSuchKey" in str(e):
                raise KeyError(f"CloakMap not found: {key}") from e
            elif "NoSuchBucket" in str(e):
                raise ValueError(f"S3 bucket '{bucket_name}' does not exist") from e
            elif "AccessDenied" in str(e):
                raise PermissionError(f"Access denied to S3 object '{object_key}'") from e
            else:
                raise ValueError(f"Failed to load CloakMap from S3: {e}") from e

    def exists(self, key: str, **kwargs: Any) -> bool:
        """Check if a CloakMap exists in S3."""
        try:
            bucket_name = self.config["bucket_name"]
            object_key = self._get_object_key(key)

            self.s3_client.head_object(
                Bucket=bucket_name,
                Key=object_key
            )
            return True

        except Exception:
            return False

    def delete(self, key: str, **kwargs: Any) -> bool:
        """
        Delete a CloakMap from S3.
        
        Args:
            key: S3 object key (without prefix) to delete
            **kwargs: Additional S3 options
            
        Returns:
            True if object was deleted, False if it didn't exist
        """
        try:
            bucket_name = self.config["bucket_name"]
            object_key = self._get_object_key(key)
            metadata_key = self._get_metadata_key(key)

            # Check if object exists first
            exists = self.exists(key, **kwargs)

            if exists:
                # Delete both object and metadata
                delete_objects = [
                    {'Key': object_key},
                    {'Key': metadata_key}
                ]

                self.s3_client.delete_objects(
                    Bucket=bucket_name,
                    Delete={'Objects': delete_objects}
                )

                return True

            return False

        except Exception as e:
            if "AccessDenied" in str(e):
                raise PermissionError("Access denied to delete S3 object") from e
            else:
                raise ConnectionError(f"Failed to delete from S3: {e}") from e

    def list_keys(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs: Any
    ) -> List[str]:
        """
        List CloakMap keys in S3.
        
        Args:
            prefix: Optional key prefix filter
            limit: Optional maximum number of keys to return
            **kwargs: Additional S3 options
            
        Returns:
            List of CloakMap keys (without object prefix)
        """
        try:
            bucket_name = self.config["bucket_name"]
            object_prefix = self.config.get("object_prefix", "cloakmaps/")

            # Build the S3 prefix
            if prefix:
                s3_prefix = f"{object_prefix}{prefix}"
            else:
                s3_prefix = object_prefix

            keys = []
            continuation_token = None

            while True:
                # Build list arguments
                list_args = {
                    'Bucket': bucket_name,
                    'Prefix': s3_prefix,
                    'MaxKeys': min(1000, limit or 1000)
                }

                if continuation_token:
                    list_args['ContinuationToken'] = continuation_token

                # List objects
                response = self.s3_client.list_objects_v2(**list_args)

                # Process objects
                for obj in response.get('Contents', []):
                    obj_key = obj['Key']

                    # Skip metadata files
                    if obj_key.endswith('.meta'):
                        continue

                    # Remove object prefix to get clean key
                    if obj_key.startswith(object_prefix):
                        clean_key = obj_key[len(object_prefix):]
                        keys.append(clean_key)

                        if limit and len(keys) >= limit:
                            break

                # Check if we need to continue
                if not response.get('IsTruncated', False) or (limit and len(keys) >= limit):
                    break

                continuation_token = response.get('NextContinuationToken')

            return sorted(keys)

        except Exception as e:
            if "NoSuchBucket" in str(e):
                raise ValueError(f"S3 bucket '{bucket_name}' does not exist") from e
            elif "AccessDenied" in str(e):
                raise PermissionError("Access denied to list S3 bucket") from e
            else:
                raise ConnectionError(f"Failed to list S3 objects: {e}") from e

    def get_metadata(self, key: str, **kwargs: Any) -> StorageMetadata:
        """
        Get metadata for a CloakMap from S3.
        
        First tries to load from metadata object, falls back to
        HEAD request on the main object.
        
        Args:
            key: S3 object key (without prefix)
            **kwargs: Additional S3 options
            
        Returns:
            StorageMetadata for the CloakMap
        """
        try:
            bucket_name = self.config["bucket_name"]
            object_key = self._get_object_key(key)
            metadata_key = self._get_metadata_key(key)

            # Try to load metadata object first
            try:
                response = self.s3_client.get_object(
                    Bucket=bucket_name,
                    Key=metadata_key
                )
                metadata_content = response['Body'].read().decode('utf-8')
                metadata_dict = json.loads(metadata_content)
                return StorageMetadata.from_dict(metadata_dict)

            except Exception:
                # Fall back to HEAD request and loading CloakMap
                pass

            # Get object metadata via HEAD request
            head_response = self.s3_client.head_object(
                Bucket=bucket_name,
                Key=object_key
            )

            # Load CloakMap to get internal metadata
            cloakmap = self.load(key, **kwargs)

            # Create metadata from HEAD response and CloakMap
            content_length = head_response['ContentLength']
            last_modified = head_response['LastModified']

            # Build minimal content bytes for hash calculation
            content = cloakmap.to_json()
            content_bytes = content.encode('utf-8')

            metadata = StorageMetadata.from_cloakmap(
                key=key,
                cloakmap=cloakmap,
                backend_type=self.backend_type,
                content_bytes=content_bytes,
                bucket_name=bucket_name,
                object_key=object_key,
            )

            # Update with S3-specific metadata
            metadata.size_bytes = content_length
            metadata.modified_at = last_modified.replace(tzinfo=None)
            metadata.backend_metadata.update({
                'etag': head_response.get('ETag', '').strip('"'),
                'storage_class': head_response.get('StorageClass', 'STANDARD'),
                'server_side_encryption': head_response.get('ServerSideEncryption'),
            })

            return metadata

        except Exception as e:
            if "NoSuchKey" in str(e):
                raise KeyError(f"CloakMap not found: {key}") from e
            else:
                raise ConnectionError(f"Failed to get S3 metadata: {e}") from e

    def health_check(self) -> Dict[str, Any]:
        """Perform health check of S3 storage."""
        base_result = super().health_check()

        try:
            bucket_name = self.config["bucket_name"]

            # Try to head the bucket
            self.s3_client.head_bucket(Bucket=bucket_name)

            # Try to list objects (limited)
            self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                MaxKeys=1
            )

            base_result.update({
                "bucket_name": bucket_name,
                "bucket_accessible": True,
                "can_list": True,
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
