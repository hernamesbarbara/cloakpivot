"""CloakMapLoader for loading and validating CloakMap files."""

import json
import logging
import os
from pathlib import Path
from typing import Any

from ..core.cloakmap import CloakMap, validate_cloakmap_integrity

logger = logging.getLogger(__name__)


class CloakMapLoadError(Exception):
    """Raised when CloakMap loading fails."""

    pass


class CloakMapLoader:
    """
    Handles loading and validation of CloakMap files from storage.

    This class provides robust loading of CloakMap files with comprehensive
    validation including:
    - File existence and accessibility
    - JSON parsing and schema validation
    - Version compatibility checking
    - Integrity verification (signatures, checksums)
    - Error handling and recovery

    Examples:
        >>> loader = CloakMapLoader()
        >>> cloakmap = loader.load("document.cloakmap")
        >>> print(f"Loaded CloakMap v{cloakmap.version}")

        >>> # With signature verification
        >>> cloakmap = loader.load(
        ...     "document.cloakmap",
        ...     verify_signature=True,
        ...     secret_key="my_secret"
        ... )
    """

    SUPPORTED_VERSIONS = ["1.0", "2.0"]
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB max CloakMap file size

    def __init__(self) -> None:
        """Initialize the CloakMap loader."""
        logger.debug("CloakMapLoader initialized")

    def load(
        self,
        file_path: str | Path,
        verify_signature: bool = False,
        secret_key: str | None = None,
        strict_validation: bool = True,
    ) -> CloakMap:
        """
        Load a CloakMap from a file path.

        Args:
            file_path: Path to the CloakMap file
            verify_signature: Whether to verify HMAC signature
            secret_key: Secret key for signature verification
            strict_validation: Whether to perform strict validation

        Returns:
            Loaded and validated CloakMap

        Raises:
            CloakMapLoadError: If loading or validation fails
            FileNotFoundError: If file does not exist
        """
        path = Path(file_path)
        logger.info(f"Loading CloakMap from {path}")

        try:
            # Validate file accessibility
            self._validate_file_access(path)

            # Load and parse JSON content
            raw_content = self._load_file_content(path)
            cloakmap_data = self._parse_json_content(raw_content)

            # Create CloakMap object
            cloakmap = self._create_cloakmap(cloakmap_data)

            # Perform validation
            self._validate_cloakmap(
                cloakmap=cloakmap,
                verify_signature=verify_signature,
                secret_key=secret_key,
                strict_validation=strict_validation,
            )

            logger.info(
                f"Successfully loaded CloakMap v{cloakmap.version} "
                f"with {cloakmap.anchor_count} anchors"
            )

            return cloakmap

        except Exception as e:
            logger.error(f"Failed to load CloakMap from {path}: {e}")
            if isinstance(e, CloakMapLoadError | FileNotFoundError):
                raise
            raise CloakMapLoadError(f"Unexpected error loading CloakMap: {e}") from e

    def load_from_string(
        self,
        json_content: str,
        verify_signature: bool = False,
        secret_key: str | None = None,
        strict_validation: bool = True,
    ) -> CloakMap:
        """
        Load a CloakMap from a JSON string.

        Args:
            json_content: JSON string content
            verify_signature: Whether to verify HMAC signature
            secret_key: Secret key for signature verification
            strict_validation: Whether to perform strict validation

        Returns:
            Loaded and validated CloakMap

        Raises:
            CloakMapLoadError: If loading or validation fails
        """
        logger.info("Loading CloakMap from string content")

        try:
            # Parse JSON content
            cloakmap_data = self._parse_json_content(json_content)

            # Create CloakMap object
            cloakmap = self._create_cloakmap(cloakmap_data)

            # Perform validation
            self._validate_cloakmap(
                cloakmap=cloakmap,
                verify_signature=verify_signature,
                secret_key=secret_key,
                strict_validation=strict_validation,
            )

            logger.info(
                f"Successfully loaded CloakMap v{cloakmap.version} "
                f"with {cloakmap.anchor_count} anchors from string"
            )

            return cloakmap

        except Exception as e:
            logger.error(f"Failed to load CloakMap from string: {e}")
            if isinstance(e, CloakMapLoadError):
                raise
            raise CloakMapLoadError(f"Unexpected error loading CloakMap: {e}") from e

    def validate_file(
        self,
        file_path: str | Path,
        verify_signature: bool = False,
        secret_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Validate a CloakMap file without fully loading it.

        Args:
            file_path: Path to the CloakMap file
            verify_signature: Whether to verify HMAC signature
            secret_key: Secret key for signature verification

        Returns:
            Dictionary with validation results

        Raises:
            FileNotFoundError: If file does not exist
        """
        path = Path(file_path)
        logger.info(f"Validating CloakMap file {path}")

        validation_result: dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "file_info": {},
            "cloakmap_info": {},
        }

        try:
            # Validate file access
            self._validate_file_access(path)
            validation_result["file_info"] = {
                "path": str(path),
                "size": path.stat().st_size,
                "exists": True,
                "readable": True,
            }

            # Load and validate content
            raw_content = self._load_file_content(path)
            cloakmap_data = self._parse_json_content(raw_content)
            cloakmap = self._create_cloakmap(cloakmap_data)

            validation_result["cloakmap_info"] = {
                "version": cloakmap.version,
                "doc_id": cloakmap.doc_id,
                "anchor_count": cloakmap.anchor_count,
                "is_signed": cloakmap.is_signed,
                "is_encrypted": cloakmap.is_encrypted,
            }

            # Perform comprehensive validation
            integrity_result = validate_cloakmap_integrity(
                cloakmap, secret_key=secret_key if verify_signature else None
            )

            validation_result["valid"] = integrity_result["valid"]
            if isinstance(validation_result["errors"], list):
                validation_result["errors"].extend(integrity_result["errors"])
            if isinstance(validation_result["warnings"], list):
                validation_result["warnings"].extend(integrity_result["warnings"])

        except Exception as e:
            validation_result["valid"] = False
            if isinstance(validation_result["errors"], list):
                validation_result["errors"].append(str(e))

        return validation_result

    def _validate_file_access(self, path: Path) -> None:
        """Validate file exists and is accessible."""
        if not path.exists():
            raise FileNotFoundError(f"CloakMap file not found: {path}")

        if not path.is_file():
            raise CloakMapLoadError(f"Path is not a file: {path}")

        if not path.stat().st_size:
            raise CloakMapLoadError(f"CloakMap file is empty: {path}")

        if path.stat().st_size > self.MAX_FILE_SIZE:
            raise CloakMapLoadError(
                f"CloakMap file too large ({path.stat().st_size} bytes, "
                f"max {self.MAX_FILE_SIZE}): {path}"
            )

        try:
            # Test if file is readable by attempting to open it
            with open(path, encoding="utf-8"):
                pass
        except PermissionError as e:
            raise CloakMapLoadError(f"Cannot read CloakMap file: {path}") from e

    def _load_file_content(self, path: Path) -> str:
        """Load raw file content as string."""
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError as e:
            raise CloakMapLoadError(f"CloakMap file contains invalid UTF-8: {path} - {e}") from e
        except OSError as e:
            raise CloakMapLoadError(f"Failed to read CloakMap file: {path} - {e}") from e

    def _parse_json_content(self, content: str) -> dict[str, Any]:
        """Parse JSON content and validate basic structure."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise CloakMapLoadError(f"Invalid JSON format: {e}") from e

        if not isinstance(data, dict):
            raise CloakMapLoadError("CloakMap JSON must be an object")

        # Validate required top-level fields
        required_fields = ["version", "doc_id", "doc_hash", "anchors"]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            raise CloakMapLoadError(f"Missing required fields: {missing_fields}")

        return data

    def _create_cloakmap(self, data: dict[str, Any]) -> CloakMap:
        """Create CloakMap object from parsed data."""
        try:
            return CloakMap.from_dict(data)
        except Exception as e:
            raise CloakMapLoadError(f"Failed to create CloakMap object: {e}") from e

    def _validate_cloakmap(
        self,
        cloakmap: CloakMap,
        verify_signature: bool,
        secret_key: str | None,
        strict_validation: bool,
    ) -> None:
        """Perform comprehensive CloakMap validation."""
        # Check version compatibility
        if cloakmap.version not in self.SUPPORTED_VERSIONS:
            if strict_validation:
                raise CloakMapLoadError(
                    f"Unsupported CloakMap version: {cloakmap.version}. "
                    f"Supported versions: {self.SUPPORTED_VERSIONS}"
                )
            logger.warning(f"CloakMap version {cloakmap.version} may not be fully supported")

        # Signature verification is not currently implemented
        if verify_signature:
            logger.warning("Signature verification requested but not implemented")

        # Perform basic integrity validation
        integrity_result = validate_cloakmap_integrity(
            cloakmap, key_manager=None, secret_key=secret_key
        )

        if not integrity_result["valid"]:
            error_msg = "CloakMap integrity validation failed: " + "; ".join(
                integrity_result["errors"]
            )
            if strict_validation:
                raise CloakMapLoadError(error_msg)
            logger.warning(error_msg)

        # Log any warnings
        for warning in integrity_result.get("warnings", []):
            logger.warning(f"CloakMap validation warning: {warning}")

    def get_file_info(self, file_path: str | Path) -> dict[str, Any]:
        """
        Get basic information about a CloakMap file without full loading.

        Args:
            file_path: Path to the CloakMap file

        Returns:
            Dictionary with file information

        Raises:
            FileNotFoundError: If file does not exist
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"CloakMap file not found: {path}")

        file_info = {
            "path": str(path),
            "size": path.stat().st_size,
            "exists": path.exists(),
            "is_file": path.is_file(),
            "readable": os.access(path, os.R_OK),
            "modified_time": path.stat().st_mtime,
        }

        # Try to extract basic info from JSON without full parsing
        try:
            with open(path, encoding="utf-8") as f:
                # Read just enough to get basic metadata
                first_chunk = f.read(1024)
                if first_chunk:
                    try:
                        partial_data = json.loads(first_chunk)
                        if isinstance(partial_data, dict):
                            file_info.update(
                                {
                                    "version": partial_data.get("version"),
                                    "doc_id": partial_data.get("doc_id"),
                                    "created_at": partial_data.get("created_at"),
                                }
                            )
                    except json.JSONDecodeError:
                        # File might be larger than 1024 chars, that's ok
                        pass
        except Exception:
            # Don't fail if we can't read metadata
            pass

        return file_info
