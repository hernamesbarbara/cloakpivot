"""CloakPivot exception hierarchy for comprehensive error handling.

This module provides a structured exception hierarchy that enables precise error
categorization, partial failure isolation, and enhanced error recovery throughout
the CloakPivot system.
"""

from typing import Any, Dict, List, Optional, Union


class CloakPivotError(Exception):
    """Base exception for all CloakPivot-related errors.
    
    Provides common functionality for error context, recovery guidance,
    and structured error information.
    
    Attributes:
        message: Human-readable error description
        error_code: Machine-readable error identifier
        context: Additional error context and metadata
        recovery_suggestions: List of suggested recovery actions
        component: Component where the error originated
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
        component: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self._default_error_code()
        self.context = context or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.component = component or self._infer_component()
    
    def _default_error_code(self) -> str:
        """Generate default error code based on exception class name."""
        return self.__class__.__name__.upper().replace("ERROR", "_ERROR")
    
    def _infer_component(self) -> str:
        """Infer component name from exception class."""
        name = self.__class__.__name__.lower()
        if "validation" in name:
            return "validation"
        elif "detection" in name:
            return "detection"
        elif "masking" in name:
            return "masking"
        elif "unmasking" in name:
            return "unmasking"
        elif "policy" in name:
            return "policy"
        else:
            return "core"
    
    def add_context(self, key: str, value: Any) -> None:
        """Add additional context to the error."""
        self.context[key] = value
    
    def add_recovery_suggestion(self, suggestion: str) -> None:
        """Add a recovery suggestion to help users resolve the error."""
        if suggestion not in self.recovery_suggestions:
            self.recovery_suggestions.append(suggestion)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to structured dictionary for logging/reporting."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "component": self.component,
            "context": self.context,
            "recovery_suggestions": self.recovery_suggestions,
        }


class ValidationError(CloakPivotError):
    """Raised when input validation fails.
    
    Used for configuration validation, document format validation,
    and other input validation scenarios.
    """
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if field_name:
            self.add_context("field_name", field_name)
        if expected_type:
            self.add_context("expected_type", expected_type)
        if actual_value is not None:
            self.add_context("actual_value", str(actual_value))


class ProcessingError(CloakPivotError):
    """Raised when document processing fails.
    
    Used for DocPivot integration errors, document parsing failures,
    and other processing-related issues.
    """
    
    def __init__(
        self,
        message: str,
        document_path: Optional[str] = None,
        processing_stage: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if document_path:
            self.add_context("document_path", document_path)
        if processing_stage:
            self.add_context("processing_stage", processing_stage)


class DetectionError(CloakPivotError):
    """Raised when PII detection fails.
    
    Used for Presidio analyzer errors, entity detection failures,
    and recognition pipeline issues.
    """
    
    def __init__(
        self,
        message: str,
        entity_type: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if entity_type:
            self.add_context("entity_type", entity_type)
        if confidence_threshold is not None:
            self.add_context("confidence_threshold", confidence_threshold)


class MaskingError(CloakPivotError):
    """Raised when masking operations fail.
    
    Used for strategy application failures, anchor generation issues,
    and masking pipeline errors.
    """
    
    def __init__(
        self,
        message: str,
        strategy_type: Optional[str] = None,
        entity_count: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if strategy_type:
            self.add_context("strategy_type", strategy_type)
        if entity_count is not None:
            self.add_context("entity_count", entity_count)


class UnmaskingError(CloakPivotError):
    """Raised when unmasking operations fail.
    
    Used for CloakMap compatibility issues, anchor resolution failures,
    and unmasking pipeline errors.
    """
    
    def __init__(
        self,
        message: str,
        cloakmap_version: Optional[str] = None,
        anchor_count: Optional[int] = None,
        failed_anchors: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if cloakmap_version:
            self.add_context("cloakmap_version", cloakmap_version)
        if anchor_count is not None:
            self.add_context("anchor_count", anchor_count)
        if failed_anchors:
            self.add_context("failed_anchors", failed_anchors)


class PolicyError(CloakPivotError):
    """Raised when policy-related operations fail.
    
    Used for policy loading errors, inheritance failures,
    and policy validation issues.
    """
    
    def __init__(
        self,
        message: str,
        policy_file: Optional[str] = None,
        policy_version: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if policy_file:
            self.add_context("policy_file", policy_file)
        if policy_version:
            self.add_context("policy_version", policy_version)


class IntegrityError(CloakPivotError):
    """Raised when data integrity violations are detected.
    
    Used for CloakMap corruption, document hash mismatches,
    and other integrity-related issues.
    """
    
    def __init__(
        self,
        message: str,
        expected_hash: Optional[str] = None,
        actual_hash: Optional[str] = None,
        corruption_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if expected_hash:
            self.add_context("expected_hash", expected_hash)
        if actual_hash:
            self.add_context("actual_hash", actual_hash)
        if corruption_type:
            self.add_context("corruption_type", corruption_type)


class PartialProcessingError(CloakPivotError):
    """Raised when partial processing completes with some failures.
    
    This is not a fatal error but indicates that some operations
    failed while others succeeded. Contains detailed information
    about both successful and failed operations.
    """
    
    def __init__(
        self,
        message: str,
        total_operations: int,
        successful_operations: int,
        failed_operations: int,
        failures: List[Dict[str, Any]],
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.add_context("total_operations", total_operations)
        self.add_context("successful_operations", successful_operations)
        self.add_context("failed_operations", failed_operations)
        self.add_context("failures", failures)
        self.add_context("success_rate", successful_operations / total_operations)


class ConfigurationError(ValidationError):
    """Raised when configuration is invalid or incomplete.
    
    Specialized validation error for configuration-specific issues.
    """
    
    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        config_section: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if config_file:
            self.add_context("config_file", config_file)
        if config_section:
            self.add_context("config_section", config_section)


class DependencyError(CloakPivotError):
    """Raised when required dependencies are missing or incompatible.
    
    Used for missing packages, version incompatibilities,
    and system requirement failures.
    """
    
    def __init__(
        self,
        message: str,
        dependency_name: Optional[str] = None,
        required_version: Optional[str] = None,
        installed_version: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if dependency_name:
            self.add_context("dependency_name", dependency_name)
        if required_version:
            self.add_context("required_version", required_version)
        if installed_version:
            self.add_context("installed_version", installed_version)


# Convenience functions for creating common exception scenarios

def create_validation_error(
    message: str,
    field_name: str,
    expected: Union[str, type],
    actual: Any,
) -> ValidationError:
    """Create a validation error with standard context."""
    expected_str = expected.__name__ if isinstance(expected, type) else str(expected)
    
    error = ValidationError(
        message=message,
        field_name=field_name,
        expected_type=expected_str,
        actual_value=actual,
    )
    
    error.add_recovery_suggestion(f"Ensure {field_name} is of type {expected_str}")
    return error


def create_processing_error(
    message: str,
    document_path: str,
    stage: str,
    original_error: Optional[Exception] = None,
) -> ProcessingError:
    """Create a processing error with standard context."""
    error = ProcessingError(
        message=message,
        document_path=document_path,
        processing_stage=stage,
    )
    
    if original_error:
        error.add_context("original_error", str(original_error))
        error.add_context("original_error_type", type(original_error).__name__)
    
    error.add_recovery_suggestion("Check document format and accessibility")
    error.add_recovery_suggestion("Verify document is not corrupted")
    
    return error


def create_dependency_error(
    dependency: str,
    required_version: Optional[str] = None,
    installed_version: Optional[str] = None,
) -> DependencyError:
    """Create a dependency error with installation guidance."""
    if required_version and installed_version:
        message = f"Incompatible {dependency} version: required {required_version}, found {installed_version}"
    elif required_version:
        message = f"Missing required dependency: {dependency} >= {required_version}"
    else:
        message = f"Missing required dependency: {dependency}"
    
    error = DependencyError(
        message=message,
        dependency_name=dependency,
        required_version=required_version,
        installed_version=installed_version,
    )
    
    if required_version:
        error.add_recovery_suggestion(f"Install with: pip install '{dependency}>={required_version}'")
    else:
        error.add_recovery_suggestion(f"Install with: pip install {dependency}")
    
    return error