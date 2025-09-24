"""CloakMap system for secure, reversible masking operations."""

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .anchors import AnchorEntry, AnchorIndex
from ..utilities.cloakmap_serializer import CloakMapSerializer
from ..utilities.cloakmap_validator import CloakMapValidator, merge_cloakmaps, validate_cloakmap_integrity

# Security features removed - simplified implementation


@dataclass(frozen=True)
class CloakMap:
    """
    A versioned, secure mapping artifact that stores all information needed
    for deterministic unmasking of a masked document.

    The CloakMap contains anchor entries that track original-to-masked mappings,
    policy state, cryptographic metadata, and document integrity information.
    It supports encryption and digital signatures for secure storage.

    Version 2.0 adds optional Presidio metadata for enhanced reversibility
    when using Presidio's AnonymizerEngine and DeanonymizerEngine.

    Attributes:
        version: Schema version for compatibility management (1.0 or 2.0)
        doc_id: Unique identifier for the source document
        doc_hash: SHA-256 hash of the original document content
        anchors: List of anchor entries mapping original to masked content
        policy_snapshot: Serialized policy state applied during masking
        crypto: Optional cryptographic metadata (keys, algorithms, etc.)
        signature: Optional HMAC signature for integrity verification
        created_at: Timestamp when the CloakMap was created
        metadata: Additional document and processing metadata
        presidio_metadata: Optional Presidio operator results for v2.0 (backward compatible)

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
        >>> # Create CloakMap v1.0 (backward compatible)
        >>> cloakmap = CloakMap.create(
        ...     doc_id="my_document",
        ...     doc_hash="a1b2c3d4...",
        ...     anchors=anchors,
        ...     policy=MaskingPolicy()
        ... )
        >>>
        >>> # Create CloakMap v2.0 with Presidio metadata
        >>> presidio_data = {
        ...     "engine_version": "2.2.x",
        ...     "operator_results": [...],
        ...     "reversible_operators": ["encrypt"]
        ... }
        >>> cloakmap_v2 = CloakMap.create_with_presidio(
        ...     doc_id="my_document",
        ...     doc_hash="a1b2c3d4...",
        ...     anchors=anchors,
        ...     policy=MaskingPolicy(),
        ...     presidio_metadata=presidio_data
        ... )
    """

    version: str = field(default="1.0")
    doc_id: str = ""
    doc_hash: str = ""
    anchors: list[AnchorEntry] = field(default_factory=list)
    policy_snapshot: dict[str, Any] = field(default_factory=dict)
    crypto: dict[str, Any] | None = field(default=None)
    signature: str | None = field(default=None)
    created_at: datetime | None = field(default=None)
    metadata: dict[str, Any] = field(default_factory=dict)
    presidio_metadata: dict[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        """Validate CloakMap data after initialization."""
        # Use validator for all validation
        CloakMapValidator.validate_version(self.version)
        CloakMapValidator.validate_doc_fields(self.doc_id, self.doc_hash)
        CloakMapValidator.validate_anchors(self.anchors)
        CloakMapValidator.validate_crypto(self.crypto, self.signature)
        CloakMapValidator.validate_presidio_metadata(self.presidio_metadata)

        # Set default timestamp if not provided
        if self.created_at is None:
            object.__setattr__(self, "created_at", datetime.now(UTC))

        # Auto-set version to 2.0 if presidio_metadata is present
        if self.presidio_metadata is not None and self.version == "1.0":
            object.__setattr__(self, "version", "2.0")

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
    def entity_mappings(self) -> dict[str, str]:
        """Get mappings from replacement IDs to masked values."""
        mappings: dict[str, str] = {}
        for anchor in self.anchors:
            mappings[anchor.replacement_id] = anchor.masked_value
        return mappings

    @property
    def is_encrypted(self) -> bool:
        """Check if the CloakMap has encryption metadata."""
        return self.crypto is not None and self.crypto.get("algorithm") is not None

    @property
    def is_signed(self) -> bool:
        """Check if the CloakMap has a signature."""
        return self.signature is not None

    @property
    def is_presidio_enabled(self) -> bool:
        """Check if the CloakMap has Presidio metadata."""
        return self.presidio_metadata is not None

    @property
    def has_reversible_operators(self) -> bool:
        """Check if the CloakMap contains reversible Presidio operators."""
        if not self.is_presidio_enabled or self.presidio_metadata is None:
            return False

        reversible = self.presidio_metadata.get("reversible_operators", [])
        return len(reversible) > 0

    @property
    def presidio_engine_version(self) -> str | None:
        """Get the Presidio engine version used to create this CloakMap."""
        if not self.is_presidio_enabled or self.presidio_metadata is None:
            return None

        return self.presidio_metadata.get("engine_version")

    def get_anchor_index(self) -> AnchorIndex:
        """Get an indexed view of the anchors for efficient lookups."""
        return AnchorIndex(self.anchors)

    def get_anchor_by_replacement_id(self, replacement_id: str) -> AnchorEntry | None:
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

    def verify_document_hash(self, document_content: str | bytes) -> bool:
        """
        Verify that the provided document content matches the stored hash.

        Args:
            document_content: The document content to verify

        Returns:
            True if the content matches the hash, False otherwise
        """
        return CloakMapValidator.verify_document_hash(self.doc_hash, document_content)

    def verify_signature(
        self,
        key_manager: Any | None = None,
        secret_key: str | None = None,
        config: Any | None = None,
    ) -> bool:
        """
        Verify the HMAC signature of the CloakMap.

        Args:
            key_manager: Key manager for retrieving signing keys
            secret_key: Direct secret key (deprecated, use key_manager)
            config: Security configuration

        Returns:
            True if the signature is valid, False otherwise
        """
        return CloakMapValidator.verify_signature(self, key_manager, secret_key, config)

    def with_signature(
        self,
        key_manager: Any | None = None,
        secret_key: str | None = None,
        key_id: str = "default",
        config: Any | None = None,
    ) -> "CloakMap":
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
        return CloakMapSerializer.sign_cloakmap(self, key_manager, secret_key, key_id, config)

    def sign(
        self,
        key_manager: Any | None = None,
        secret_key: str | None = None,
        key_id: str = "default",
        config: Any | None = None,
    ) -> "CloakMap":
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

    def with_encryption_metadata(
        self,
        algorithm: str,
        key_id: str,
        additional_params: dict[str, Any] | None = None,
    ) -> "CloakMap":
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
            **(additional_params or {}),
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
            metadata=self.metadata,
        )

    def encrypt(
        self,
        key_manager: Any | None = None,
        key_id: str = "default",
        key_version: str | None = None,
        config: Any | None = None,
    ) -> Any:
        """
        Encrypt this CloakMap using AES-GCM encryption.

        Args:
            key_manager: Key manager for retrieving encryption keys
            key_id: Encryption key identifier
            key_version: Optional key version
            config: Security configuration

        Returns:
            Any - would be EncryptedCloakMap if encryption was enabled

        Raises:
            KeyError: If encryption key is not found
            ValueError: If encryption fails
        """
        # Security features removed - simplified implementation
        raise NotImplementedError("Encryption feature has been removed in the simplified version")

    def save_encrypted(
        self,
        file_path: str | Path,
        key_manager: Any | None = None,
        key_id: str = "default",
        key_version: str | None = None,
        config: Any | None = None,
        indent: int = 2,
    ) -> None:
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
        CloakMapSerializer.save_encrypted(
            self, file_path, key_manager, key_id, key_version, config, indent
        )

    @classmethod
    def load_encrypted(
        cls,
        file_path: str | Path,
        key_manager: Any | None = None,
        config: Any | None = None,
    ) -> "CloakMap":
        """
        Load and decrypt a CloakMap from JSON file (encryption removed).

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
        return CloakMapSerializer.load_encrypted(file_path, key_manager, config)

    @classmethod
    def load_from_file(
        cls,
        file_path: str | Path,
        key_manager: Any | None = None,
        config: Any | None = None,
    ) -> "CloakMap":
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
        return CloakMapSerializer.load_from_file(file_path, key_manager, config)

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
                "length_delta": total_masked_length - total_original_length,
            },
            "security": {
                "is_encrypted": self.is_encrypted,
                "is_signed": self.is_signed,
                "encryption_algorithm": (self.crypto.get("algorithm") if self.crypto else None),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert CloakMap to dictionary for serialization."""
        return CloakMapSerializer.to_dict(self)

    def to_json(self, indent: int | None = None) -> str:
        """Convert CloakMap to JSON string."""
        return CloakMapSerializer.to_json(self, indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CloakMap":
        """Create CloakMap from dictionary representation."""
        return CloakMapSerializer.from_dict(data)

    @classmethod
    def from_json(cls, json_str: str) -> "CloakMap":
        """Create CloakMap from JSON string."""
        return CloakMapSerializer.from_json(json_str)

    def save_to_file(self, file_path: str | Path, indent: int = 2) -> None:
        """Save CloakMap to JSON file."""
        CloakMapSerializer.save_to_file(self, file_path, indent)

    @classmethod
    def create(
        cls,
        doc_id: str,
        doc_hash: str,
        anchors: list[AnchorEntry],
        policy: Any | None = None,
        metadata: dict[str, Any] | None = None,
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
            New CloakMap instance (v1.0 format)
        """
        return CloakMapSerializer.create_cloakmap(doc_id, doc_hash, anchors, policy, metadata)

    @classmethod
    def create_with_presidio(
        cls,
        doc_id: str,
        doc_hash: str,
        anchors: list[AnchorEntry],
        presidio_metadata: dict[str, Any],
        policy: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "CloakMap":
        """
        Create a new CloakMap v2.0 with Presidio metadata.

        Args:
            doc_id: Document identifier
            doc_hash: Hash of the original document
            anchors: List of anchor entries
            presidio_metadata: Presidio operator results and metadata
            policy: Optional masking policy (will be serialized to policy_snapshot)
            metadata: Optional additional metadata

        Returns:
            New CloakMap instance (v2.0 format with Presidio metadata)
        """
        return CloakMapSerializer.create_cloakmap_with_presidio(
            doc_id, doc_hash, anchors, presidio_metadata, policy, metadata
        )


# Re-export utility functions
__all__ = ["CloakMap", "AnchorEntry", "AnchorIndex", "merge_cloakmaps", "validate_cloakmap_integrity"]