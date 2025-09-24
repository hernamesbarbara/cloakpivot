"""Serialization logic for CloakMap data structures."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .anchors import AnchorEntry

if TYPE_CHECKING:
    from .cloakmap import CloakMap


class CloakMapSerializer:
    """Serializer for CloakMap data structures."""

    @staticmethod
    def to_dict(cloakmap: "CloakMap") -> dict[str, Any]:
        """Convert CloakMap to dictionary for serialization."""
        result = {
            "version": cloakmap.version,
            "doc_id": cloakmap.doc_id,
            "doc_hash": cloakmap.doc_hash,
            "anchors": [anchor.to_dict() for anchor in cloakmap.anchors],
            "policy_snapshot": cloakmap.policy_snapshot,
            "crypto": cloakmap.crypto,
            "signature": cloakmap.signature,
            "created_at": cloakmap.created_at.isoformat() if cloakmap.created_at else None,
            "metadata": cloakmap.metadata,
        }

        # Only include presidio_metadata if present (backward compatibility)
        if cloakmap.presidio_metadata is not None:
            result["presidio_metadata"] = cloakmap.presidio_metadata
            # If engine_used is in presidio_metadata, also expose it at top level for compatibility
            if "engine_used" in cloakmap.presidio_metadata:
                result["engine_used"] = cloakmap.presidio_metadata["engine_used"]

        return result

    @staticmethod
    def to_json(cloakmap: "CloakMap", indent: int | None = None) -> str:
        """Convert CloakMap to JSON string."""
        return json.dumps(
            CloakMapSerializer.to_dict(cloakmap), indent=indent, ensure_ascii=False
        )

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CloakMap":
        """Create CloakMap from dictionary representation."""
        # Import here to avoid circular dependency
        from .cloakmap import CloakMap

        # Convert anchors
        anchors = []
        for anchor_data in data.get("anchors", []):
            anchors.append(AnchorEntry.from_dict(anchor_data))

        # Convert timestamp
        created_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])

        # Handle presidio_metadata (optional for backward compatibility)
        presidio_metadata = data.get("presidio_metadata")

        # If engine_used is at top level, move it to presidio_metadata for storage
        if "engine_used" in data:
            if presidio_metadata is None:
                presidio_metadata = {}
            presidio_metadata["engine_used"] = data["engine_used"]

        return CloakMap(
            version=data.get("version", "1.0"),
            doc_id=data.get("doc_id", ""),
            doc_hash=data.get("doc_hash", ""),
            anchors=anchors,
            policy_snapshot=data.get("policy_snapshot", {}),
            crypto=data.get("crypto"),
            signature=data.get("signature"),
            created_at=created_at,
            metadata=data.get("metadata", {}),
            presidio_metadata=presidio_metadata,
        )

    @staticmethod
    def from_json(json_str: str) -> "CloakMap":
        """Create CloakMap from JSON string."""
        try:
            data = json.loads(json_str)
            return CloakMapSerializer.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}") from e

    @staticmethod
    def save_to_file(cloakmap: "CloakMap", file_path: str | Path, indent: int = 2) -> None:
        """Save CloakMap to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with path.open("w", encoding="utf-8") as f:
                f.write(CloakMapSerializer.to_json(cloakmap, indent=indent))
        except Exception as e:
            raise ValueError(f"Failed to save CloakMap to {file_path}: {e}") from e

    @staticmethod
    def load_from_file(
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
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CloakMap file not found: {file_path}")

        try:
            with path.open(encoding="utf-8") as f:
                content = f.read()
                data = json.loads(content)

            # Check if this is an encrypted format
            if "encrypted_content" in data:
                # This is an encrypted CloakMap
                return CloakMapSerializer.load_encrypted(file_path, key_manager, config)
            # This is a standard unencrypted CloakMap
            return CloakMapSerializer.from_json(content)

        except Exception as e:
            raise ValueError(f"Failed to load CloakMap from {file_path}: {e}") from e

    @staticmethod
    def save_encrypted(
        cloakmap: "CloakMap",
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
            cloakmap: CloakMap to save
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
        # Encryption removed in v2.0
        raise NotImplementedError("Encryption has been removed in v2.0")

    @staticmethod
    def load_encrypted(
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
        # Security features removed
        raise NotImplementedError("Encrypted loading has been removed in the simplified version")

    @staticmethod
    def sign_cloakmap(
        cloakmap: "CloakMap",
        key_manager: Any | None = None,
        secret_key: str | None = None,
        key_id: str = "default",
        config: Any | None = None,
    ) -> "CloakMap":
        """
        Create a new CloakMap with an HMAC signature.

        Args:
            cloakmap: CloakMap to sign
            key_manager: Key manager for retrieving signing keys
            secret_key: Direct secret key (deprecated, use key_manager)
            key_id: Key identifier for key manager
            config: Security configuration

        Returns:
            New CloakMap with signature
        """
        # Import here to avoid circular dependency
        from .cloakmap import CloakMap

        if config is None:
            config = None  # Security config removed

        # Get signing key
        signing_key = CloakMapSerializer._get_signing_key(key_manager, secret_key, key_id)
        if not signing_key:
            raise ValueError("No signing key available")

        # Create unsigned version for signing
        unsigned_map = CloakMap(
            version=cloakmap.version,
            doc_id=cloakmap.doc_id,
            doc_hash=cloakmap.doc_hash,
            anchors=cloakmap.anchors,
            policy_snapshot=cloakmap.policy_snapshot,
            crypto=cloakmap.crypto,
            signature=None,
            created_at=cloakmap.created_at,
            metadata=cloakmap.metadata,
        )

        # Generate signature with enhanced crypto
        json.dumps(CloakMapSerializer.to_dict(unsigned_map), sort_keys=True).encode("utf-8")
        # CryptoUtils removed in v2.0 - signature generation disabled
        signature = None

        # Update crypto metadata with signing information
        crypto_data = cloakmap.crypto.copy() if cloakmap.crypto else {}
        if config is not None:
            crypto_data.update(
                {
                    "signature_algorithm": config.hmac_algorithm,
                    "key_id": key_id,
                    "signed_at": datetime.now(UTC).isoformat(),
                }
            )

        return CloakMap(
            version=cloakmap.version,
            doc_id=cloakmap.doc_id,
            doc_hash=cloakmap.doc_hash,
            anchors=cloakmap.anchors,
            policy_snapshot=cloakmap.policy_snapshot,
            crypto=crypto_data,
            signature=signature,
            created_at=cloakmap.created_at,
            metadata=cloakmap.metadata,
        )

    @staticmethod
    def _get_signing_key(
        key_manager: Any | None,
        secret_key: str | None,
        key_id: str = "default",
    ) -> bytes | None:
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
                key = key_manager.get_key(key_id)
                if isinstance(key, bytes):
                    return key
                # If key is not bytes, convert it
                if isinstance(key, str):
                    return key.encode("utf-8")
                # Otherwise, skip this key_manager
            except (KeyError, ValueError):
                pass

        if secret_key:
            return secret_key.encode("utf-8")

        # Try default key manager as fallback
        try:
            # Default key manager removed in v2.0
            return None
        except (KeyError, ValueError):
            pass

        return None

    @staticmethod
    def create_cloakmap(
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
        # Import here to avoid circular dependency
        from .cloakmap import CloakMap

        # Serialize policy if provided
        policy_snapshot = {}
        if policy is not None and hasattr(policy, "to_dict"):
            policy_snapshot = policy.to_dict()

        return CloakMap(
            doc_id=doc_id,
            doc_hash=doc_hash,
            anchors=anchors,
            policy_snapshot=policy_snapshot,
            metadata=metadata or {},
        )

    @staticmethod
    def create_cloakmap_with_presidio(
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
        # Import here to avoid circular dependency
        from .cloakmap import CloakMap

        # Serialize policy if provided
        policy_snapshot = {}
        if policy is not None and hasattr(policy, "to_dict"):
            policy_snapshot = policy.to_dict()

        return CloakMap(
            version="2.0",
            doc_id=doc_id,
            doc_hash=doc_hash,
            anchors=anchors,
            policy_snapshot=policy_snapshot,
            metadata=metadata or {},
            presidio_metadata=presidio_metadata,
        )


__all__ = ["CloakMapSerializer"]