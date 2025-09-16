"""Comprehensive validation system for early error detection.

This module provides thorough validation of inputs, configurations, system
requirements, and dependencies before processing begins, enabling early
error detection with clear, actionable error messages.
"""

import os
import sys
from pathlib import Path
from typing import Any

from .exceptions import (
    ConfigurationError,
    DependencyError,
    ValidationError,
    create_dependency_error,
    create_validation_error,
)


class SystemValidator:
    """Validates system requirements and environment setup."""

    @staticmethod
    def validate_python_version(min_version: tuple[int, int] = (3, 8)) -> None:
        """Validate Python version meets requirements."""
        current_version = sys.version_info[:2]
        if current_version < min_version:
            raise ValidationError(
                f"Python {'.'.join(map(str, min_version))} or higher required, "
                f"found {'.'.join(map(str, current_version))}",
                error_code="PYTHON_VERSION_INCOMPATIBLE",
            )

    @staticmethod
    def validate_dependencies() -> None:
        """Validate required dependencies are installed with compatible versions."""
        required_packages = {
            "presidio_analyzer": "2.0.0",
            "presidio_anonymizer": "2.0.0",
            "docpivot": "0.1.0",
            "pydantic": "2.0.0",
            "click": "8.0.0",
        }

        missing_packages = []
        version_conflicts = []

        for package, min_version in required_packages.items():
            try:
                import importlib.metadata

                installed_version = importlib.metadata.version(package)

                # Simple version comparison (major.minor.patch)
                if not SystemValidator._version_compatible(installed_version, min_version):
                    version_conflicts.append(
                        {
                            "package": package,
                            "required": min_version,
                            "installed": installed_version,
                        }
                    )

            except importlib.metadata.PackageNotFoundError:
                missing_packages.append(package)

        if missing_packages:
            raise create_dependency_error(
                ", ".join(missing_packages),
                required_version="See requirements.txt",
            )

        if version_conflicts:
            conflict_details = []
            for conflict in version_conflicts:
                conflict_details.append(
                    f"{conflict['package']}: required >={conflict['required']}, "
                    f"found {conflict['installed']}"
                )
            raise DependencyError(
                f"Version conflicts detected: {'; '.join(conflict_details)}",
                error_code="DEPENDENCY_VERSION_CONFLICT",
            )

    @staticmethod
    def _version_compatible(installed: str, required: str) -> bool:
        """Check if installed version meets minimum requirement."""

        def parse_version(v: str) -> tuple[int, ...]:
            return tuple(map(int, v.split(".")[:3]))

        try:
            return parse_version(installed) >= parse_version(required)
        except ValueError:
            # If we can't parse versions, assume compatible
            return True

    @staticmethod
    def validate_file_permissions(
        file_path: str | Path,
        require_read: bool = True,
        require_write: bool = False,
    ) -> None:
        """Validate file exists and has required permissions."""
        path = Path(file_path)

        if not path.exists():
            error = ValidationError(
                f"File does not exist: {path}",
                error_code="FILE_NOT_FOUND",
            )
            error.add_recovery_suggestion("Check the file path is correct")
            error.add_recovery_suggestion("Ensure the file exists at the specified location")
            raise error

        if require_read and not os.access(path, os.R_OK):
            raise ValidationError(
                f"File is not readable: {path}",
                error_code="FILE_NOT_READABLE",
            )

        if require_write and not os.access(path, os.W_OK):
            raise ValidationError(
                f"File is not writable: {path}",
                error_code="FILE_NOT_WRITABLE",
            )

    @staticmethod
    def validate_directory_access(
        directory_path: str | Path,
        require_read: bool = True,
        require_write: bool = False,
        create_if_missing: bool = False,
    ) -> None:
        """Validate directory exists and has required permissions."""
        path = Path(directory_path)

        if not path.exists():
            if create_if_missing:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    raise ValidationError(
                        f"Cannot create directory {path}: {e}",
                        error_code="DIRECTORY_CREATION_FAILED",
                    ) from e
            else:
                raise ValidationError(
                    f"Directory does not exist: {path}",
                    error_code="DIRECTORY_NOT_FOUND",
                )

        if not path.is_dir():
            raise ValidationError(
                f"Path is not a directory: {path}",
                error_code="NOT_A_DIRECTORY",
            )

        if require_read and not os.access(path, os.R_OK):
            raise ValidationError(
                f"Directory is not readable: {path}",
                error_code="DIRECTORY_NOT_READABLE",
            )

        if require_write and not os.access(path, os.W_OK):
            raise ValidationError(
                f"Directory is not writable: {path}",
                error_code="DIRECTORY_NOT_WRITABLE",
            )


class DocumentValidator:
    """Validates document structure and format."""

    @staticmethod
    def validate_document_format(document_path: str | Path) -> str:
        """Validate document format and return detected format."""
        path = Path(document_path)
        SystemValidator.validate_file_permissions(path, require_read=True)

        suffix = path.suffix.lower()
        supported_formats = {".pdf", ".docx", ".txt", ".md", ".html", ".json"}

        if suffix not in supported_formats:
            raise ValidationError(
                f"Unsupported document format: {suffix}. "
                f"Supported formats: {', '.join(sorted(supported_formats))}",
                error_code="UNSUPPORTED_DOCUMENT_FORMAT",
            )

        # Additional format-specific validation
        if suffix == ".json":
            DocumentValidator._validate_json_document(path)

        return suffix[1:]  # Remove the dot

    @staticmethod
    def _validate_json_document(path: Path) -> None:
        """Validate JSON document structure."""
        import json

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            # Check if it's a valid docling document structure
            if isinstance(data, dict):
                required_keys = {"name", "texts"}  # Minimal docling structure
                if not required_keys.issubset(data.keys()):
                    raise ValidationError(
                        f"Invalid document structure. Required keys: {required_keys}",
                        error_code="INVALID_DOCUMENT_STRUCTURE",
                    )
            else:
                raise ValidationError(
                    "Document must be a JSON object",
                    error_code="INVALID_JSON_STRUCTURE",
                )

        except json.JSONDecodeError as e:
            raise ValidationError(
                f"Invalid JSON format: {e}",
                error_code="INVALID_JSON_FORMAT",
            ) from e

    @staticmethod
    def validate_document_size(
        document_path: str | Path,
        max_size_mb: float = 100.0,
    ) -> None:
        """Validate document size is within acceptable limits."""
        path = Path(document_path)
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        if size_mb > max_size_mb:
            raise ValidationError(
                f"Document too large: {size_mb:.1f}MB exceeds limit of {max_size_mb:.1f}MB",
                error_code="DOCUMENT_TOO_LARGE",
            )


class PolicyValidator:
    """Validates masking policy configuration."""

    @staticmethod
    def validate_policy_structure(policy_data: dict[str, Any]) -> None:
        """Validate basic policy structure and required fields."""
        required_fields = ["default_strategy"]
        missing_fields = [field for field in required_fields if field not in policy_data]

        if missing_fields:
            raise ConfigurationError(
                f"Missing required policy fields: {', '.join(missing_fields)}",
                error_code="MISSING_POLICY_FIELDS",
            )

        # Validate default strategy
        PolicyValidator._validate_strategy(policy_data["default_strategy"])

        # Validate per-entity strategies if present
        if "per_entity" in policy_data:
            if not isinstance(policy_data["per_entity"], dict):
                raise create_validation_error(
                    "per_entity must be a dictionary",
                    "per_entity",
                    dict,
                    type(policy_data["per_entity"]),
                )

            for _entity_type, strategy in policy_data["per_entity"].items():
                PolicyValidator._validate_strategy(strategy)

    @staticmethod
    def _validate_strategy(strategy: Any) -> None:
        """Validate a masking strategy configuration."""
        if not isinstance(strategy, dict):
            raise create_validation_error(
                "Strategy must be a dictionary", "strategy", dict, type(strategy)
            )

        if "kind" not in strategy:
            raise ValidationError(
                "Strategy must specify 'kind'",
                field_name="strategy.kind",
            )

        valid_kinds = {"redact", "hash", "template", "partial"}
        if strategy["kind"] not in valid_kinds:
            raise ValidationError(
                f"Invalid strategy kind '{strategy['kind']}'. "
                f"Valid kinds: {', '.join(sorted(valid_kinds))}",
                field_name="strategy.kind",
                expected_type=f"one of {valid_kinds}",
                actual_value=strategy["kind"],
            )

    @staticmethod
    def validate_thresholds(thresholds: dict[str, float]) -> None:
        """Validate confidence thresholds are within valid range."""
        for entity_type, threshold in thresholds.items():
            if not isinstance(threshold, int | float):
                raise create_validation_error(
                    f"Threshold for {entity_type} must be numeric",
                    f"thresholds.{entity_type}",
                    float,
                    threshold,
                )

            if not 0.0 <= threshold <= 1.0:
                raise ValidationError(
                    f"Threshold for {entity_type} must be between 0.0 and 1.0, got {threshold}",
                    field_name=f"thresholds.{entity_type}",
                    expected_type="float in range [0.0, 1.0]",
                    actual_value=threshold,
                )


class CloakMapValidator:
    """Validates CloakMap structure and integrity."""

    @staticmethod
    def validate_cloakmap_structure(cloakmap_data: dict[str, Any]) -> None:
        """Validate CloakMap has required structure."""
        required_fields = ["doc_id", "version", "anchors", "created_at"]
        missing_fields = [field for field in required_fields if field not in cloakmap_data]

        if missing_fields:
            raise ValidationError(
                f"Missing required CloakMap fields: {', '.join(missing_fields)}",
                error_code="INVALID_CLOAKMAP_STRUCTURE",
            )

        # Validate anchors is a list
        if not isinstance(cloakmap_data["anchors"], list):
            raise create_validation_error(
                "CloakMap anchors must be a list",
                "anchors",
                list,
                type(cloakmap_data["anchors"]),
            )

        # Validate each anchor
        for i, anchor in enumerate(cloakmap_data["anchors"]):
            CloakMapValidator._validate_anchor(anchor, i)

    @staticmethod
    def _validate_anchor(anchor: Any, index: int) -> None:
        """Validate a single anchor entry."""
        if not isinstance(anchor, dict):
            raise create_validation_error(
                f"Anchor at index {index} must be a dictionary",
                f"anchors[{index}]",
                dict,
                type(anchor),
            )

        required_fields = ["anchor_id", "node_id", "start", "end", "entity_type"]
        missing_fields = [field for field in required_fields if field not in anchor]

        if missing_fields:
            raise ValidationError(
                f"Anchor at index {index} missing required fields: {', '.join(missing_fields)}",
                error_code="INVALID_ANCHOR_STRUCTURE",
            )

        # Validate position values
        if not isinstance(anchor["start"], int) or anchor["start"] < 0:
            raise ValidationError(
                f"Anchor at index {index} has invalid start position: {anchor['start']}",
                field_name=f"anchors[{index}].start",
                expected_type="non-negative integer",
                actual_value=anchor["start"],
            )

        if not isinstance(anchor["end"], int) or anchor["end"] < anchor["start"]:
            raise ValidationError(
                f"Anchor at index {index} has invalid end position: {anchor['end']}",
                field_name=f"anchors[{index}].end",
                expected_type="integer >= start position",
                actual_value=anchor["end"],
            )


class InputValidator:
    """Main validator class that orchestrates all validation checks."""

    def __init__(self) -> None:
        self.system_validator = SystemValidator()
        self.document_validator = DocumentValidator()
        self.policy_validator = PolicyValidator()
        self.cloakmap_validator = CloakMapValidator()

    def validate_masking_inputs(
        self,
        document_path: str | Path,
        policy_data: dict[str, Any] | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        """Validate all inputs required for masking operation."""
        # System validation
        self.system_validator.validate_python_version()
        self.system_validator.validate_dependencies()

        # Document validation
        self.document_validator.validate_document_format(document_path)
        self.document_validator.validate_document_size(document_path)

        # Policy validation
        if policy_data:
            self.policy_validator.validate_policy_structure(policy_data)
            if "thresholds" in policy_data:
                self.policy_validator.validate_thresholds(policy_data["thresholds"])

        # Output validation
        if output_path:
            output_dir = Path(output_path).parent
            self.system_validator.validate_directory_access(
                output_dir,
                require_write=True,
                create_if_missing=True,
            )

    def validate_unmasking_inputs(
        self,
        masked_document_path: str | Path,
        cloakmap_path: str | Path,
        output_path: str | Path | None = None,
    ) -> None:
        """Validate all inputs required for unmasking operation."""
        # System validation
        self.system_validator.validate_python_version()
        self.system_validator.validate_dependencies()

        # Input file validation
        self.system_validator.validate_file_permissions(masked_document_path, require_read=True)
        self.system_validator.validate_file_permissions(cloakmap_path, require_read=True)

        # CloakMap structure validation
        import json

        try:
            with open(cloakmap_path, encoding="utf-8") as f:
                cloakmap_data = json.load(f)
            self.cloakmap_validator.validate_cloakmap_structure(cloakmap_data)
        except json.JSONDecodeError as e:
            raise ValidationError(
                f"Invalid CloakMap JSON format: {e}",
                error_code="INVALID_CLOAKMAP_JSON",
            ) from e

        # Output validation
        if output_path:
            output_dir = Path(output_path).parent
            self.system_validator.validate_directory_access(
                output_dir,
                require_write=True,
                create_if_missing=True,
            )

    def validate_configuration(self, config: dict[str, Any]) -> list[str]:
        """Validate configuration and return list of warnings (non-fatal issues)."""
        warnings = []

        # Check for deprecated settings
        deprecated_keys = {"legacy_mode", "old_strategy_format"}
        for key in deprecated_keys:
            if key in config:
                warnings.append(f"Configuration key '{key}' is deprecated and will be ignored")

        # Check for potentially problematic settings
        if config.get("max_file_size_mb", 100) > 500:
            warnings.append(
                f"Large file size limit ({config['max_file_size_mb']}MB) may cause memory issues"
            )

        # Check performance settings
        if config.get("parallel_processing", True) and config.get("max_workers", 4) > 8:
            warnings.append(f"High worker count ({config['max_workers']}) may impact performance")

        return warnings


# Convenience functions for common validation scenarios


def validate_for_masking(
    document_path: str | Path,
    policy_data: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> list[str]:
    """Validate inputs for masking operation and return any warnings."""
    validator = InputValidator()
    validator.validate_masking_inputs(document_path, policy_data, output_path)

    # Generate configuration warnings if policy provided
    warnings = []
    if policy_data:
        warnings = validator.validate_configuration(policy_data)

    return warnings


def validate_for_unmasking(
    masked_document_path: str | Path,
    cloakmap_path: str | Path,
    output_path: str | Path | None = None,
) -> list[str]:
    """Validate inputs for unmasking operation and return any warnings."""
    validator = InputValidator()
    validator.validate_unmasking_inputs(masked_document_path, cloakmap_path, output_path)
    return []
