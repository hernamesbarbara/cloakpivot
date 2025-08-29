"""CloakMap system for secure, reversible masking operations."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from .anchors import AnchorEntry, AnchorIndex
from .security import (
    CryptoUtils,
    EncryptedCloakMap,
    KeyManager,
    SecurityConfig,
    create_default_key_manager,
)


@dataclass(frozen=True)
class CloakMap:
    """
    A versioned, secure mapping artifact that stores all information needed
    for deterministic unmasking of a masked document.

    The CloakMap contains anchor entries that track original-to-masked mappings,
    policy state, cryptographic metadata, and document integrity information.
    It supports encryption and digital signatures for secure storage.

    Attributes:
        version: Schema version for compatibility management
        doc_id: Unique identifier for the source document
        doc_hash: SHA-256 hash of the original document content
        anchors: List of anchor entries mapping original to masked content
        policy_snapshot: Serialized policy state applied during masking
        crypto: Optional cryptographic metadata (keys, algorithms, etc.)
        signature: Optional HMAC signature for integrity verification
        created_at: Timestamp when the CloakMap was created
        metadata: Additional document and processing metadata

    Examples:
        >>> from .policies import MaskingPolicy
        >>> from .strategies import Strategy, StrategyKind
        >>>
        >>> # Create anchors for detected entities
        >>> anchors = [
        ...     AnchorEntry.create_from_detection(
        ...         node_id="p1",
        ...         start=10,
        ...         end=22,
        ...         entity_type="PHONE_NUMBER",
        ...         confidence=0.95,
        ...         original_text="555-123-4567",
        ...         masked_value="[PHONE]",
        ...         strategy_used="template"
        ...     )
        ... ]
        >>>
        >>> # Create CloakMap
        >>> cloakmap = CloakMap.create(
        ...     doc_id="my_document",
        ...     doc_hash="a1b2c3d4...",
        ...     anchors=anchors,
        ...     policy=MaskingPolicy()
        ... )
    """

    version: str = field(default="1.0")
    doc_id: str = ""
    doc_hash: str = ""
    anchors: list[AnchorEntry] = field(default_factory=list)
    policy_snapshot: dict[str, Any] = field(default_factory=dict)
    crypto: Optional[dict[str, Any]] = field(default=None)
    signature: Optional[str] = field(default=None)
    created_at: Optional[datetime] = field(default=None)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate CloakMap data after initialization."""
        self._validate_version()
        self._validate_doc_fields()
        self._validate_anchors()
        self._validate_crypto()

        # Set default timestamp if not provided
        if self.created_at is None:
            object.__setattr__(self, 'created_at', datetime.utcnow())

    def _validate_version(self) -> None:
        """Validate version string format."""
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("Version cannot be empty")

        # Basic semantic version validation (major.minor format)
        parts = self.version.split('.')
        if len(parts) < 2:
            raise ValueError("version must follow 'major.minor' format at minimum")

        try:
            for part in parts[:2]:  # At least major.minor must be numeric
                int(part)
        except ValueError as e:
            raise ValueError("version major and minor components must be numeric") from e

    def _validate_doc_fields(self) -> None:
        """Validate document identification fields."""
        if not isinstance(self.doc_id, str) or not self.doc_id.strip():
            raise ValueError("Document ID cannot be empty")

        if not isinstance(self.doc_hash, str) or not self.doc_hash.strip():
            raise ValueError("Document hash cannot be empty")

        # Basic SHA-256 hash validation if provided
        if len(self.doc_hash) == 64:
            try:
                int(self.doc_hash, 16)
            except ValueError as e:
                raise ValueError("doc_hash should be a valid SHA-256 hex string") from e

    def _validate_anchors(self) -> None:
        """Validate anchor entries."""
        if not isinstance(self.anchors, list):
            raise ValueError("anchors must be a list")

        # Check for duplicate replacement IDs
        replacement_ids = set()
        for anchor in self.anchors:
            if not isinstance(anchor, AnchorEntry):
                raise ValueError("all anchors must be AnchorEntry instances")

            if anchor.replacement_id in replacement_ids:
                raise ValueError(f"duplicate replacement_id: {anchor.replacement_id}")

            replacement_ids.add(anchor.replacement_id)

    def _validate_crypto(self) -> None:
        """Validate cryptographic metadata."""
        if self.crypto is not None and not isinstance(self.crypto, dict):
            raise ValueError("crypto must be a dictionary or None")

        if self.signature is not None and not isinstance(self.signature, str):
            raise ValueError("signature must be a string or None")

    @property
    def anchor_count(self) -> int:
        """Get the total number of anchors."""
        return len(self.anchors)

    @property
    def entity_count_by_type(self) -> dict[str, int]:
        """Get count of entities by type."""
        counts: dict[str, int] = {}
        for anchor in self.anchors:
            counts[anchor.entity_type] = counts.get(anchor.entity_type, 0) + 1
        return counts

    @property
    def is_encrypted(self) -> bool:
        """Check if the CloakMap has encryption metadata."""
        return self.crypto is not None and self.crypto.get("algorithm") is not None

    @property
    def is_signed(self) -> bool:
        """Check if the CloakMap has a signature."""
        return self.signature is not None

    def get_anchor_index(self) -> AnchorIndex:
        """Get an indexed view of the anchors for efficient lookups."""
        return AnchorIndex(self.anchors)

    def get_anchor_by_replacement_id(self, replacement_id: str) -> Optional[AnchorEntry]:
        """Get a specific anchor by its replacement ID."""
        for anchor in self.anchors:
            if anchor.replacement_id == replacement_id:
                return anchor
        return None

    def get_anchors_for_node(self, node_id: str) -> list[AnchorEntry]:
        """Get all anchors for a specific document node, sorted by position."""
        node_anchors = [a for a in self.anchors if a.node_id == node_id]
        return sorted(node_anchors, key=lambda a: a.start)

    def get_anchors_by_entity_type(self, entity_type: str) -> list[AnchorEntry]:
        """Get all anchors for a specific entity type."""
        return [a for a in self.anchors if a.entity_type == entity_type]

    def verify_document_hash(self, document_content: Union[str, bytes]) -> bool:
        """
        Verify that the provided document content matches the stored hash.

        Args:
            document_content: The document content to verify

        Returns:
            True if the content matches the hash, False otherwise
        """
        if not self.doc_hash:
            return False

        if isinstance(document_content, str):
            content_bytes = document_content.encode('utf-8')
        else:
            content_bytes = document_content

        computed_hash = hashlib.sha256(content_bytes).hexdigest()
        return computed_hash == self.doc_hash

    def verify_signature(self, key_manager: Optional[KeyManager] = None,
                        secret_key: Optional[str] = None,
                        config: Optional[SecurityConfig] = None) -> bool:
        """
        Verify the HMAC signature of the CloakMap.

        Args:
            key_manager: Key manager for retrieving signing keys
            secret_key: Direct secret key (deprecated, use key_manager)
            config: Security configuration

        Returns:
            True if the signature is valid, False otherwise
        """
        if not self.signature:
            return False

        if config is None:
            config = SecurityConfig()

        # Get signing key using the key_id from crypto metadata
        key_id = self.crypto.get('key_id', 'default') if self.crypto else 'default'
        signing_key = self._get_signing_key(key_manager, secret_key, key_id)
        if not signing_key:
            return False

        # Create a copy without signature for verification
        # NOTE: crypto must be None during verification to match signing content
        unsigned_map = CloakMap(
            version=self.version,
            doc_id=self.doc_id,
            doc_hash=self.doc_hash,
            anchors=self.anchors,
            policy_snapshot=self.policy_snapshot,
            crypto=None,  # Must be None to match signing content
            signature=None,
            created_at=self.created_at,
            metadata=self.metadata
        )

        # Compute expected signature
        content = json.dumps(unsigned_map.to_dict(), sort_keys=True).encode('utf-8')

        # Use enhanced crypto utilities
        algorithm = self.crypto.get('signature_algorithm', config.hmac_algorithm) if self.crypto else config.hmac_algorithm
        CryptoUtils.compute_hmac(content, signing_key, algorithm)

        return CryptoUtils.verify_hmac(
            content, signing_key, self.signature, algorithm,
            config.constant_time_verification
        )

    def with_signature(self, key_manager: Optional[KeyManager] = None,
                      secret_key: Optional[str] = None,
                      key_id: str = "default",
                      config: Optional[SecurityConfig] = None) -> "CloakMap":
        """
        Create a new CloakMap with an HMAC signature.

        Args:
            key_manager: Key manager for retrieving signing keys
            secret_key: Direct secret key (deprecated, use key_manager)
            key_id: Key identifier for key manager
            config: Security configuration

        Returns:
            New CloakMap with signature
        """
        if config is None:
            config = SecurityConfig()

        # Get signing key
        signing_key = self._get_signing_key(key_manager, secret_key, key_id)
        if not signing_key:
            raise ValueError("No signing key available")

        # Create unsigned version for signing
        unsigned_map = CloakMap(
            version=self.version,
            doc_id=self.doc_id,
            doc_hash=self.doc_hash,
            anchors=self.anchors,
            policy_snapshot=self.policy_snapshot,
            crypto=self.crypto,
            signature=None,
            created_at=self.created_at,
            metadata=self.metadata
        )

        # Generate signature with enhanced crypto
        content = json.dumps(unsigned_map.to_dict(), sort_keys=True).encode('utf-8')
        signature = CryptoUtils.compute_hmac(content, signing_key, config.hmac_algorithm)

        # Update crypto metadata with signing information
        crypto_data = self.crypto.copy() if self.crypto else {}
        crypto_data.update({
            'signature_algorithm': config.hmac_algorithm,
            'key_id': key_id,
            'signed_at': datetime.utcnow().isoformat()
        })

        return CloakMap(
            version=self.version,
            doc_id=self.doc_id,
            doc_hash=self.doc_hash,
            anchors=self.anchors,
            policy_snapshot=self.policy_snapshot,
            crypto=crypto_data,
            signature=signature,
            created_at=self.created_at,
            metadata=self.metadata
        )

    def sign(self, key_manager: Optional[KeyManager] = None,
            secret_key: Optional[str] = None,
            key_id: str = "default",
            config: Optional[SecurityConfig] = None) -> "CloakMap":
        """
        Sign the CloakMap with a secret key (alias for with_signature).

        Args:
            key_manager: Key manager for retrieving signing keys
            secret_key: Direct secret key (deprecated, use key_manager)
            key_id: Key identifier for key manager
            config: Security configuration

        Returns:
            New CloakMap with signature
        """
        return self.with_signature(key_manager, secret_key, key_id, config)

    def _get_signing_key(self, key_manager: Optional[KeyManager],
                        secret_key: Optional[str],
                        key_id: str = "default") -> Optional[bytes]:
        """
        Get signing key from manager or direct string.

        Args:
            key_manager: Key manager instance
            secret_key: Direct secret key string
            key_id: Key identifier

        Returns:
            Key bytes or None if not found
        """
        if key_manager:
            try:
                return key_manager.get_key(key_id)
            except (KeyError, ValueError):
                pass

        if secret_key:
            return secret_key.encode('utf-8')

        # Try default key manager as fallback
        try:
            default_manager = create_default_key_manager()
            return default_manager.get_key(key_id)
        except (KeyError, ValueError):
            pass

        return None

    def with_encryption_metadata(self, algorithm: str, key_id: str,
                               additional_params: Optional[dict[str, Any]] = None) -> "CloakMap":
        """
        Create a new CloakMap with encryption metadata.

        Args:
            algorithm: Encryption algorithm used (e.g., 'AES-GCM-256')
            key_id: Identifier for the encryption key
            additional_params: Additional encryption parameters

        Returns:
            New CloakMap with encryption metadata
        """
        crypto_data = {
            "algorithm": algorithm,
            "key_id": key_id,
            **(additional_params or {})
        }

        return CloakMap(
            version=self.version,
            doc_id=self.doc_id,
            doc_hash=self.doc_hash,
            anchors=self.anchors,
            policy_snapshot=self.policy_snapshot,
            crypto=crypto_data,
            signature=self.signature,
            created_at=self.created_at,
            metadata=self.metadata
        )

    def encrypt(self, key_manager: Optional[KeyManager] = None,
                key_id: str = "default", key_version: Optional[str] = None,
                config: Optional[SecurityConfig] = None) -> "EncryptedCloakMap":
        """
        Encrypt this CloakMap using AES-GCM encryption.

        Args:
            key_manager: Key manager for retrieving encryption keys
            key_id: Encryption key identifier
            key_version: Optional key version
            config: Security configuration

        Returns:
            EncryptedCloakMap with encrypted sensitive data

        Raises:
            KeyError: If encryption key is not found
            ValueError: If encryption fails
        """
        from .security import CloakMapEncryption, create_default_key_manager

        if key_manager is None:
            key_manager = create_default_key_manager()

        if config is None:
            config = SecurityConfig()

        encryption = CloakMapEncryption(key_manager, config)
        return encryption.encrypt_cloakmap(self, key_id, key_version)

    def save_encrypted(self, file_path: Union[str, Path], 
                      key_manager: Optional[KeyManager] = None,
                      key_id: str = "default", key_version: Optional[str] = None,
                      config: Optional[SecurityConfig] = None,
                      indent: int = 2) -> None:
        """
        Encrypt and save CloakMap to JSON file.

        Args:
            file_path: Path to save encrypted file
            key_manager: Key manager for encryption keys
            key_id: Encryption key identifier
            key_version: Optional key version
            config: Security configuration
            indent: JSON indentation

        Raises:
            KeyError: If encryption key is not found
            ValueError: If encryption or save fails
        """
        encrypted_map = self.encrypt(key_manager, key_id, key_version, config)
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(encrypted_map.to_json(indent=indent))
        except Exception as e:
            raise ValueError(f"Failed to save encrypted CloakMap to {file_path}: {e}") from e

    @classmethod
    def load_encrypted(cls, file_path: Union[str, Path],
                      key_manager: Optional[KeyManager] = None,
                      config: Optional[SecurityConfig] = None) -> "CloakMap":
        """
        Load and decrypt an EncryptedCloakMap from JSON file.

        Args:
            file_path: Path to encrypted CloakMap file
            key_manager: Key manager for decryption keys
            config: Security configuration

        Returns:
            Decrypted CloakMap

        Raises:
            FileNotFoundError: If file doesn't exist
            KeyError: If decryption key is not found
            ValueError: If decryption fails
        """
        from .security import CloakMapEncryption, EncryptedCloakMap, create_default_key_manager

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Encrypted CloakMap file not found: {file_path}")

        if key_manager is None:
            key_manager = create_default_key_manager()

        if config is None:
            config = SecurityConfig()

        try:
            with open(path, encoding='utf-8') as f:
                encrypted_map = EncryptedCloakMap.from_json(f.read())

            encryption = CloakMapEncryption(key_manager, config)
            return encryption.decrypt_cloakmap(encrypted_map)

        except Exception as e:
            raise ValueError(f"Failed to load encrypted CloakMap from {file_path}: {e}") from e

    @classmethod  
    def load_from_file(cls, file_path: Union[str, Path], 
                      key_manager: Optional[KeyManager] = None,
                      config: Optional[SecurityConfig] = None) -> "CloakMap":
        """
        Load CloakMap from JSON file, auto-detecting encrypted vs unencrypted format.

        Args:
            file_path: Path to CloakMap file
            key_manager: Optional key manager for encrypted files
            config: Security configuration

        Returns:
            Loaded CloakMap

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If loading fails
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CloakMap file not found: {file_path}")

        try:
            with open(path, encoding='utf-8') as f:
                content = f.read()
                data = json.loads(content)

            # Check if this is an encrypted format
            if "encrypted_content" in data:
                # This is an encrypted CloakMap
                return cls.load_encrypted(file_path, key_manager, config)
            else:
                # This is a standard unencrypted CloakMap
                return cls.from_json(content)

        except Exception as e:
            raise ValueError(f"Failed to load CloakMap from {file_path}: {e}") from e

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics about the CloakMap."""
        total_confidence = sum(a.confidence for a in self.anchors)
        avg_confidence = round(total_confidence / len(self.anchors), 10) if self.anchors else 0.0

        strategy_counts: dict[str, int] = {}
        for anchor in self.anchors:
            strategy = anchor.strategy_used
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # Calculate text change statistics
        total_original_length = sum(a.span_length for a in self.anchors)
        total_masked_length = sum(a.replacement_length for a in self.anchors)

        return {
            "version": self.version,
            "doc_id": self.doc_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "anchor_count": len(self.anchors),
            "total_anchors": len(self.anchors),
            "total_original_length": total_original_length,
            "total_masked_length": total_masked_length,
            "entity_counts": self.entity_count_by_type,
            "strategy_counts": strategy_counts,
            "average_confidence": avg_confidence,
            "text_stats": {
                "total_original_chars": total_original_length,
                "total_masked_chars": total_masked_length,
                "length_delta": total_masked_length - total_original_length
            },
            "security": {
                "is_encrypted": self.is_encrypted,
                "is_signed": self.is_signed,
                "encryption_algorithm": self.crypto.get("algorithm") if self.crypto else None
            }
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert CloakMap to dictionary for serialization."""
        return {
            "version": self.version,
            "doc_id": self.doc_id,
            "doc_hash": self.doc_hash,
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "policy_snapshot": self.policy_snapshot,
            "crypto": self.crypto,
            "signature": self.signature,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert CloakMap to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CloakMap":
        """Create CloakMap from dictionary representation."""
        # Convert anchors
        anchors = []
        for anchor_data in data.get("anchors", []):
            anchors.append(AnchorEntry.from_dict(anchor_data))

        # Convert timestamp
        created_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])

        return cls(
            version=data.get("version", "1.0"),
            doc_id=data.get("doc_id", ""),
            doc_hash=data.get("doc_hash", ""),
            anchors=anchors,
            policy_snapshot=data.get("policy_snapshot", {}),
            crypto=data.get("crypto"),
            signature=data.get("signature"),
            created_at=created_at,
            metadata=data.get("metadata", {})
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CloakMap":
        """Create CloakMap from JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}") from e



    def save_to_file(self, file_path: Union[str, Path], indent: int = 2) -> None:
        """Save CloakMap to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.to_json(indent=indent))
        except Exception as e:
            raise ValueError(f"Failed to save CloakMap to {file_path}: {e}") from e

    @classmethod
    def create(
        cls,
        doc_id: str,
        doc_hash: str,
        anchors: list[AnchorEntry],
        policy: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> "CloakMap":
        """
        Create a new CloakMap with the provided data.

        Args:
            doc_id: Document identifier
            doc_hash: Hash of the original document
            anchors: List of anchor entries
            policy: Optional masking policy (will be serialized to policy_snapshot)
            metadata: Optional additional metadata

        Returns:
            New CloakMap instance
        """
        # Serialize policy if provided
        policy_snapshot = {}
        if policy is not None and hasattr(policy, 'to_dict'):
            policy_snapshot = policy.to_dict()

        return cls(
            doc_id=doc_id,
            doc_hash=doc_hash,
            anchors=anchors,
            policy_snapshot=policy_snapshot,
            metadata=metadata or {}
        )


# Utility functions for CloakMap operations

def merge_cloakmaps(cloakmaps: list[CloakMap], target_doc_id: Optional[str] = None) -> CloakMap:
    """
    Merge multiple CloakMaps into a single consolidated map.

    This is useful when processing documents in chunks or combining
    results from different processing stages.

    Args:
        cloakmaps: List of CloakMaps to merge
        target_doc_id: Document ID for the merged result

    Returns:
        Merged CloakMap

    Raises:
        ValueError: If CloakMaps have incompatible versions or conflicting data
    """
    if not cloakmaps:
        raise ValueError("Cannot merge empty list of CloakMaps")

    # Check version compatibility
    base_version = cloakmaps[0].version
    for cm in cloakmaps[1:]:
        if cm.version != base_version:
            raise ValueError("Cannot merge CloakMaps with different versions")

    # Infer target_doc_id if not provided
    if target_doc_id is None:
        target_doc_id = cloakmaps[0].doc_id
        # Check that all have the same doc_id
        for cm in cloakmaps[1:]:
            if cm.doc_id != target_doc_id:
                raise ValueError("Cannot merge CloakMaps from different documents without explicit target_doc_id")

    # Collect all anchors and check for conflicts
    all_anchors = []
    replacement_ids = set()

    for cm in cloakmaps:
        for anchor in cm.anchors:
            if anchor.replacement_id in replacement_ids:
                raise ValueError(f"Conflicting replacement_id: {anchor.replacement_id}")

            all_anchors.append(anchor)
            replacement_ids.add(anchor.replacement_id)

    # Check for anchor overlaps within the same node
    for i, anchor1 in enumerate(all_anchors):
        for anchor2 in all_anchors[i+1:]:
            if anchor1.overlaps_with(anchor2):
                raise ValueError("Anchor overlap detected")

    # Merge metadata
    merged_metadata = {}
    for cm in cloakmaps:
        merged_metadata.update(cm.metadata)

    # Use the first non-empty doc_hash
    doc_hash = ""
    for cm in cloakmaps:
        if cm.doc_hash:
            doc_hash = cm.doc_hash
            break

    # Merge policy snapshots (use the most recent one)
    policy_snapshot = {}
    latest_created_at = None

    for cm in cloakmaps:
        if cm.created_at and (latest_created_at is None or cm.created_at > latest_created_at):
            latest_created_at = cm.created_at
            policy_snapshot = cm.policy_snapshot

    return CloakMap(
        version=base_version,
        doc_id=target_doc_id,
        doc_hash=doc_hash,
        anchors=all_anchors,
        policy_snapshot=policy_snapshot,
        metadata=merged_metadata
    )


def validate_cloakmap_integrity(cloakmap: CloakMap,
                               key_manager: Optional[KeyManager] = None,
                               secret_key: Optional[str] = None,
                               config: Optional[SecurityConfig] = None) -> dict[str, Any]:
    """
    Perform comprehensive integrity validation of a CloakMap using enhanced security.

    Args:
        cloakmap: CloakMap to validate
        key_manager: Key manager for signature verification
        secret_key: Optional direct secret key (deprecated, use key_manager)
        config: Security configuration

    Returns:
        Dictionary with detailed validation results
    """
    from .security import SecurityValidator

    # Use enhanced security validator
    if key_manager is None and secret_key:
        # Create temporary key manager for backward compatibility
        import os

        from .security import EnvironmentKeyManager
        os.environ['CLOAKPIVOT_KEY_DEFAULT'] = secret_key
        key_manager = EnvironmentKeyManager()

    validator = SecurityValidator(config=config, key_manager=key_manager)
    return validator.validate_cloakmap(cloakmap)
