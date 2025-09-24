"""Validation logic for CloakMap data structures."""

import hashlib
import json
from typing import TYPE_CHECKING, Any

from ..types.anchors import AnchorEntry

if TYPE_CHECKING:
    from ..types.cloakmap import CloakMap


class CloakMapValidator:
    """Validator for CloakMap data integrity and structure."""

    @staticmethod
    def validate_version(version: str) -> None:
        """Validate version string format."""
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Version cannot be empty")

        # Basic semantic version validation (major.minor format)
        parts = version.split(".")
        if len(parts) < 2:
            raise ValueError("version must follow 'major.minor' format at minimum")

        try:
            for part in parts[:2]:  # At least major.minor must be numeric
                int(part)
        except ValueError as e:
            raise ValueError("version major and minor components must be numeric") from e

    @staticmethod
    def validate_doc_fields(doc_id: str, doc_hash: str) -> None:
        """Validate document identification fields."""
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("Document ID cannot be empty")

        if not isinstance(doc_hash, str) or not doc_hash.strip():
            raise ValueError("Document hash cannot be empty")

        # Basic SHA-256 hash validation if provided
        if len(doc_hash) == 64:
            try:
                int(doc_hash, 16)
            except ValueError as e:
                raise ValueError("doc_hash should be a valid SHA-256 hex string") from e

    @staticmethod
    def validate_anchors(anchors: list[AnchorEntry]) -> None:
        """Validate anchor entries."""
        if not isinstance(anchors, list):
            raise ValueError("anchors must be a list")

        # Check for duplicate replacement IDs
        replacement_ids = set()
        for anchor in anchors:
            if not isinstance(anchor, AnchorEntry):
                raise ValueError("all anchors must be AnchorEntry instances")

            if anchor.replacement_id in replacement_ids:
                raise ValueError(f"duplicate replacement_id: {anchor.replacement_id}")

            replacement_ids.add(anchor.replacement_id)

    @staticmethod
    def validate_crypto(crypto: dict[str, Any] | None, signature: str | None) -> None:
        """Validate cryptographic metadata."""
        if crypto is not None and not isinstance(crypto, dict):
            raise ValueError("crypto must be a dictionary or None")

        if signature is not None and not isinstance(signature, str):
            raise ValueError("signature must be a string or None")

    @staticmethod
    def validate_presidio_metadata(presidio_metadata: dict[str, Any] | None) -> None:
        """Validate Presidio metadata structure."""
        if presidio_metadata is None:
            return

        if not isinstance(presidio_metadata, dict):
            raise ValueError("presidio_metadata must be a dictionary or None")

        # Validate required fields if present
        if "operator_results" in presidio_metadata:
            results = presidio_metadata["operator_results"]
            if not isinstance(results, list):
                raise ValueError("operator_results must be a list")

            # Validate each operator result structure
            for i, result in enumerate(results):
                if not isinstance(result, dict):
                    raise ValueError(f"operator_result[{i}] must be a dictionary")

                # Check for required fields in operator result
                required_fields = ["entity_type", "start", "end", "operator"]
                for field in required_fields:
                    if field not in result:
                        raise ValueError(f"operator_result[{i}] missing required field: {field}")

        # Validate reversible_operators if present
        if "reversible_operators" in presidio_metadata:
            reversible = presidio_metadata["reversible_operators"]
            if not isinstance(reversible, list):
                raise ValueError("reversible_operators must be a list")

            for op in reversible:
                if not isinstance(op, str):
                    raise ValueError("all reversible_operators must be strings")

        # Validate engine_version if present
        if "engine_version" in presidio_metadata:
            engine_version = presidio_metadata["engine_version"]
            if not isinstance(engine_version, str) or not engine_version.strip():
                raise ValueError("engine_version must be a non-empty string")

    @staticmethod
    def verify_document_hash(doc_hash: str, document_content: str | bytes) -> bool:
        """
        Verify that the provided document content matches the stored hash.

        Args:
            doc_hash: The stored document hash
            document_content: The document content to verify

        Returns:
            True if the content matches the hash, False otherwise
        """
        if not doc_hash:
            return False

        if isinstance(document_content, str):
            content_bytes = document_content.encode("utf-8")
        else:
            content_bytes = document_content

        computed_hash = hashlib.sha256(content_bytes).hexdigest()
        return computed_hash == doc_hash

    @staticmethod
    def verify_signature(
        cloakmap: "CloakMap",
        key_manager: Any | None = None,
        secret_key: str | None = None,
        config: Any | None = None,
    ) -> bool:
        """
        Verify the HMAC signature of the CloakMap.

        Args:
            cloakmap: CloakMap to verify
            key_manager: Key manager for retrieving signing keys
            secret_key: Direct secret key (deprecated, use key_manager)
            config: Security configuration

        Returns:
            True if the signature is valid, False otherwise
        """
        if not cloakmap.signature:
            return False

        if config is None:
            config = None  # Security config removed

        # Get signing key using the key_id from crypto metadata
        key_id = cloakmap.crypto.get("key_id", "default") if cloakmap.crypto else "default"
        signing_key = CloakMapValidator._get_signing_key(key_manager, secret_key, key_id)
        if not signing_key:
            return False

        # Create a copy without signature for verification
        # Import here to avoid circular dependency
        from ..types.cloakmap import CloakMap

        # NOTE: crypto must be None during verification to match signing content
        unsigned_map = CloakMap(
            version=cloakmap.version,
            doc_id=cloakmap.doc_id,
            doc_hash=cloakmap.doc_hash,
            anchors=cloakmap.anchors,
            policy_snapshot=cloakmap.policy_snapshot,
            crypto=None,  # Must be None to match signing content
            signature=None,
            created_at=cloakmap.created_at,
            metadata=cloakmap.metadata,
        )

        # Compute expected signature
        json.dumps(unsigned_map.to_dict(), sort_keys=True).encode("utf-8")

        # Use enhanced crypto utilities
        if config is not None:
            (
                cloakmap.crypto.get("signature_algorithm", config.hmac_algorithm)
                if cloakmap.crypto
                else config.hmac_algorithm
            )
        # CryptoUtils removed in v2.0 - signature verification disabled
        return False

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


def validate_cloakmap_integrity(
    cloakmap: "CloakMap",
    key_manager: Any | None = None,
    secret_key: str | None = None,
    config: Any | None = None,
) -> dict[str, Any]:
    """
    Perform basic integrity validation of a CloakMap.

    Args:
        cloakmap: CloakMap to validate
        key_manager: Not used (kept for compatibility)
        secret_key: Not used (kept for compatibility)
        config: Not used (kept for compatibility)

    Returns:
        Dictionary with validation results
    """
    errors = []
    warnings = []

    # Basic validation checks
    if not cloakmap.version:
        errors.append("CloakMap missing version")

    if not cloakmap.doc_id:
        errors.append("CloakMap missing document ID")

    if not cloakmap.doc_hash:
        warnings.append("CloakMap missing document hash")

    # Validate anchors
    for i, anchor in enumerate(cloakmap.anchors):
        if not anchor.node_id:
            errors.append(f"Anchor {i} missing node_id")
        if anchor.start < 0:
            errors.append(f"Anchor {i} has invalid start position")
        if anchor.end <= anchor.start:
            errors.append(f"Anchor {i} has invalid end position")
        if not anchor.entity_type:
            errors.append(f"Anchor {i} missing entity_type")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "anchor_count": len(cloakmap.anchors),
        "version": cloakmap.version,
    }


def merge_cloakmaps(cloakmaps: list["CloakMap"], target_doc_id: str | None = None) -> "CloakMap":
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

    # Import here to avoid circular dependency
    from ..types.cloakmap import CloakMap

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
                raise ValueError(
                    "Cannot merge CloakMaps from different documents without explicit target_doc_id"
                )

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
        for anchor2 in all_anchors[i + 1 :]:
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
        metadata=merged_metadata,
    )


__all__ = ["CloakMapValidator", "validate_cloakmap_integrity", "merge_cloakmaps"]
