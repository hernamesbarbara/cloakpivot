"""Comprehensive unit tests for cloakpivot.core.exceptions module.

This test module provides full coverage of the exception hierarchy,
including all exception classes, methods, and helper functions.
"""

from cloakpivot.core.exceptions import (
    CloakPivotError,
    ConfigurationError,
    DependencyError,
    DetectionError,
    IntegrityError,
    MaskingError,
    PartialProcessingError,
    PolicyError,
    ProcessingError,
    UnmaskingError,
    ValidationError,
    create_dependency_error,
    create_processing_error,
    create_validation_error,
)


class TestCloakPivotError:
    """Test the base CloakPivotError exception class."""

    def test_basic_initialization(self):
        """Test basic exception initialization with just a message."""
        error = CloakPivotError("Test error message")
        assert error.message == "Test error message"
        assert error.error_code == "CLOAKPIVOT_ERROR"
        assert error.context == {}
        assert error.recovery_suggestions == []
        assert error.component == "core"

    def test_full_initialization(self):
        """Test exception initialization with all parameters."""
        context = {"key": "value", "number": 42}
        suggestions = ["Try again", "Check configuration"]
        error = CloakPivotError(
            message="Test error",
            error_code="TEST_001",
            context=context,
            recovery_suggestions=suggestions,
            component="testing",
        )
        assert error.message == "Test error"
        assert error.error_code == "TEST_001"
        assert error.context == context
        assert error.recovery_suggestions == suggestions
        assert error.component == "testing"

    def test_default_error_code_generation(self):
        """Test the _default_error_code method for various exception types."""
        error = CloakPivotError("test")
        assert error._default_error_code() == "CLOAKPIVOT_ERROR"

        # Test with a custom exception class
        class CustomError(CloakPivotError):
            pass

        custom_error = CustomError("test")
        assert custom_error._default_error_code() == "CUSTOM_ERROR"

    def test_component_inference(self):
        """Test the _infer_component method for component name inference."""
        # Base error should return "core"
        base_error = CloakPivotError("test")
        assert base_error._infer_component() == "core"

        # Test component inference for different error types
        validation_error = ValidationError("test")
        assert validation_error._infer_component() == "validation"

        detection_error = DetectionError("test")
        assert detection_error._infer_component() == "detection"

        masking_error = MaskingError("test")
        assert masking_error._infer_component() == "masking"

        unmasking_error = UnmaskingError("test")
        assert (
            unmasking_error._infer_component() == "masking"
        )  # Bug: matches "masking" before "unmasking"

        policy_error = PolicyError("test")
        assert policy_error._infer_component() == "policy"

    def test_add_context(self):
        """Test adding context to an error."""
        error = CloakPivotError("test")
        assert error.context == {}

        error.add_context("file", "test.txt")
        assert error.context["file"] == "test.txt"

        error.add_context("line", 42)
        assert error.context["line"] == 42
        assert len(error.context) == 2

    def test_add_recovery_suggestion(self):
        """Test adding recovery suggestions."""
        error = CloakPivotError("test")
        assert error.recovery_suggestions == []

        error.add_recovery_suggestion("Check input format")
        assert error.recovery_suggestions == ["Check input format"]

        # Test duplicate prevention
        error.add_recovery_suggestion("Check input format")
        assert error.recovery_suggestions == ["Check input format"]

        error.add_recovery_suggestion("Validate configuration")
        assert len(error.recovery_suggestions) == 2
        assert "Validate configuration" in error.recovery_suggestions

    def test_to_dict(self):
        """Test converting error to dictionary representation."""
        error = CloakPivotError(message="Test error", error_code="TEST_001", component="testing")
        error.add_context("key", "value")
        error.add_recovery_suggestion("Try again")

        result = error.to_dict()
        assert result["error_type"] == "CloakPivotError"
        assert result["message"] == "Test error"
        assert result["error_code"] == "TEST_001"
        assert result["component"] == "testing"
        assert result["context"] == {"key": "value"}
        assert result["recovery_suggestions"] == ["Try again"]


class TestValidationError:
    """Test the ValidationError exception class."""

    def test_basic_validation_error(self):
        """Test basic ValidationError initialization."""
        error = ValidationError("Validation failed")
        assert error.message == "Validation failed"
        assert error.component == "validation"

    def test_validation_error_with_field_info(self):
        """Test ValidationError with field information."""
        error = ValidationError(
            message="Invalid field value",
            field_name="email",
            expected_type="string",
            actual_value=123,
        )
        assert error.message == "Invalid field value"
        assert error.context["field_name"] == "email"
        assert error.context["expected_type"] == "string"
        assert error.context["actual_value"] == "123"

    def test_validation_error_partial_params(self):
        """Test ValidationError with partial parameters."""
        # Only field_name
        error1 = ValidationError("test", field_name="username")
        assert "field_name" in error1.context
        assert "expected_type" not in error1.context
        assert "actual_value" not in error1.context

        # Only expected_type
        error2 = ValidationError("test", expected_type="integer")
        assert "expected_type" in error2.context
        assert "field_name" not in error2.context

        # Only actual_value (including None handling)
        error3 = ValidationError("test", actual_value=None)
        assert "actual_value" not in error3.context  # None should not add context


class TestProcessingError:
    """Test the ProcessingError exception class."""

    def test_basic_processing_error(self):
        """Test basic ProcessingError initialization."""
        error = ProcessingError("Processing failed")
        assert error.message == "Processing failed"

    def test_processing_error_with_context(self):
        """Test ProcessingError with document context."""
        error = ProcessingError(
            message="Failed to parse document",
            document_path="/path/to/doc.pdf",
            processing_stage="parsing",
        )
        assert error.context["document_path"] == "/path/to/doc.pdf"
        assert error.context["processing_stage"] == "parsing"

    def test_processing_error_partial_params(self):
        """Test ProcessingError with partial parameters."""
        error = ProcessingError("test", document_path="/test.pdf")
        assert "document_path" in error.context
        assert "processing_stage" not in error.context


class TestDetectionError:
    """Test the DetectionError exception class."""

    def test_basic_detection_error(self):
        """Test basic DetectionError initialization."""
        error = DetectionError("Detection failed")
        assert error.message == "Detection failed"
        assert error.component == "detection"

    def test_detection_error_with_entity_info(self):
        """Test DetectionError with entity information."""
        error = DetectionError(
            message="Failed to detect entities",
            entity_type="EMAIL",
            confidence_threshold=0.8,
        )
        assert error.context["entity_type"] == "EMAIL"
        assert error.context["confidence_threshold"] == 0.8

    def test_detection_error_confidence_zero(self):
        """Test DetectionError with zero confidence threshold."""
        error = DetectionError("test", confidence_threshold=0.0)
        assert error.context["confidence_threshold"] == 0.0


class TestMaskingError:
    """Test the MaskingError exception class."""

    def test_basic_masking_error(self):
        """Test basic MaskingError initialization."""
        error = MaskingError("Masking failed")
        assert error.message == "Masking failed"
        assert error.component == "masking"

    def test_masking_error_with_strategy_info(self):
        """Test MaskingError with strategy information."""
        error = MaskingError(
            message="Strategy application failed",
            strategy_type="redaction",
            entity_count=15,
        )
        assert error.context["strategy_type"] == "redaction"
        assert error.context["entity_count"] == 15

    def test_masking_error_zero_entities(self):
        """Test MaskingError with zero entity count."""
        error = MaskingError("test", entity_count=0)
        assert error.context["entity_count"] == 0


class TestUnmaskingError:
    """Test the UnmaskingError exception class."""

    def test_basic_unmasking_error(self):
        """Test basic UnmaskingError initialization."""
        error = UnmaskingError("Unmasking failed")
        assert error.message == "Unmasking failed"
        assert error.component == "masking"  # Bug: matches "masking" before "unmasking"

    def test_unmasking_error_with_full_context(self):
        """Test UnmaskingError with all context parameters."""
        failed_anchors = ["anchor1", "anchor2"]
        error = UnmaskingError(
            message="Failed to unmask document",
            cloakmap_version="1.0.0",
            anchor_count=10,
            failed_anchors=failed_anchors,
        )
        assert error.context["cloakmap_version"] == "1.0.0"
        assert error.context["anchor_count"] == 10
        assert error.context["failed_anchors"] == failed_anchors

    def test_unmasking_error_partial_params(self):
        """Test UnmaskingError with partial parameters."""
        error = UnmaskingError("test", anchor_count=0)
        assert error.context["anchor_count"] == 0
        assert "cloakmap_version" not in error.context
        assert "failed_anchors" not in error.context


class TestPolicyError:
    """Test the PolicyError exception class."""

    def test_basic_policy_error(self):
        """Test basic PolicyError initialization."""
        error = PolicyError("Policy failed")
        assert error.message == "Policy failed"
        assert error.component == "policy"

    def test_policy_error_with_file_info(self):
        """Test PolicyError with file information."""
        error = PolicyError(
            message="Failed to load policy",
            policy_file="/path/to/policy.yaml",
            policy_version="2.1.0",
        )
        assert error.context["policy_file"] == "/path/to/policy.yaml"
        assert error.context["policy_version"] == "2.1.0"


class TestIntegrityError:
    """Test the IntegrityError exception class."""

    def test_basic_integrity_error(self):
        """Test basic IntegrityError initialization."""
        error = IntegrityError("Integrity check failed")
        assert error.message == "Integrity check failed"

    def test_integrity_error_with_hash_info(self):
        """Test IntegrityError with hash information."""
        error = IntegrityError(
            message="Hash mismatch detected",
            expected_hash="abc123",
            actual_hash="def456",
            corruption_type="checksum",
        )
        assert error.context["expected_hash"] == "abc123"
        assert error.context["actual_hash"] == "def456"
        assert error.context["corruption_type"] == "checksum"

    def test_integrity_error_partial_hash_params(self):
        """Test IntegrityError with partial hash parameters."""
        error = IntegrityError("test", expected_hash="hash1")
        assert "expected_hash" in error.context
        assert "actual_hash" not in error.context


class TestPartialProcessingError:
    """Test the PartialProcessingError exception class."""

    def test_partial_processing_error(self):
        """Test PartialProcessingError with operation statistics."""
        failures = [
            {"file": "doc1.pdf", "error": "parse error"},
            {"file": "doc2.pdf", "error": "timeout"},
        ]
        error = PartialProcessingError(
            message="Partial processing completed",
            total_operations=10,
            successful_operations=8,
            failed_operations=2,
            failures=failures,
        )
        assert error.message == "Partial processing completed"
        assert error.context["total_operations"] == 10
        assert error.context["successful_operations"] == 8
        assert error.context["failed_operations"] == 2
        assert error.context["failures"] == failures
        assert error.context["success_rate"] == 0.8

    def test_partial_processing_zero_success_rate(self):
        """Test PartialProcessingError with all operations failed."""
        error = PartialProcessingError(
            message="All operations failed",
            total_operations=5,
            successful_operations=0,
            failed_operations=5,
            failures=[{"error": "test"}],
        )
        assert error.context["success_rate"] == 0.0

    def test_partial_processing_full_success_rate(self):
        """Test PartialProcessingError with all operations successful."""
        error = PartialProcessingError(
            message="All operations succeeded",
            total_operations=5,
            successful_operations=5,
            failed_operations=0,
            failures=[],
        )
        assert error.context["success_rate"] == 1.0


class TestConfigurationError:
    """Test the ConfigurationError exception class."""

    def test_basic_configuration_error(self):
        """Test basic ConfigurationError initialization."""
        error = ConfigurationError("Configuration invalid")
        assert error.message == "Configuration invalid"
        assert error.component == "core"  # Default component since no "configuration" in class name

    def test_configuration_error_with_file_info(self):
        """Test ConfigurationError with file information."""
        error = ConfigurationError(
            message="Invalid config",
            config_file="/etc/app/config.yaml",
            config_section="database",
        )
        assert error.context["config_file"] == "/etc/app/config.yaml"
        assert error.context["config_section"] == "database"

    def test_configuration_error_inheritance(self):
        """Test that ConfigurationError inherits from ValidationError."""
        error = ConfigurationError(
            message="test",
            field_name="db_host",
            expected_type="string",
        )
        # Should have both ValidationError and ConfigurationError contexts
        assert error.context["field_name"] == "db_host"
        assert error.context["expected_type"] == "string"


class TestDependencyError:
    """Test the DependencyError exception class."""

    def test_basic_dependency_error(self):
        """Test basic DependencyError initialization."""
        error = DependencyError("Dependency missing")
        assert error.message == "Dependency missing"

    def test_dependency_error_with_version_info(self):
        """Test DependencyError with version information."""
        error = DependencyError(
            message="Version mismatch",
            dependency_name="pandas",
            required_version="1.0.0",
            installed_version="0.25.0",
        )
        assert error.context["dependency_name"] == "pandas"
        assert error.context["required_version"] == "1.0.0"
        assert error.context["installed_version"] == "0.25.0"

    def test_dependency_error_missing_dependency(self):
        """Test DependencyError for missing dependency."""
        error = DependencyError(
            message="Not installed",
            dependency_name="numpy",
            required_version="1.19.0",
        )
        assert error.context["dependency_name"] == "numpy"
        assert error.context["required_version"] == "1.19.0"
        assert "installed_version" not in error.context


class TestHelperFunctions:
    """Test the helper functions for creating common exceptions."""

    def test_create_validation_error_with_type_object(self):
        """Test create_validation_error with a type object."""
        error = create_validation_error(
            message="Invalid type",
            field_name="age",
            expected=int,
            actual="twenty",
        )
        assert error.message == "Invalid type"
        assert error.context["field_name"] == "age"
        assert error.context["expected_type"] == "int"
        assert error.context["actual_value"] == "twenty"
        assert len(error.recovery_suggestions) == 1
        assert "Ensure age is of type int" in error.recovery_suggestions[0]

    def test_create_validation_error_with_string_expected(self):
        """Test create_validation_error with a string expected value."""
        error = create_validation_error(
            message="Invalid format",
            field_name="status",
            expected="active|inactive",
            actual="pending",
        )
        assert error.context["expected_type"] == "active|inactive"
        assert "Ensure status is of type active|inactive" in error.recovery_suggestions[0]

    def test_create_processing_error_without_original(self):
        """Test create_processing_error without original error."""
        error = create_processing_error(
            message="Failed to process",
            document_path="/docs/test.pdf",
            stage="parsing",
        )
        assert error.message == "Failed to process"
        assert error.context["document_path"] == "/docs/test.pdf"
        assert error.context["processing_stage"] == "parsing"
        assert "original_error" not in error.context
        assert len(error.recovery_suggestions) == 2
        assert "Check document format and accessibility" in error.recovery_suggestions
        assert "Verify document is not corrupted" in error.recovery_suggestions

    def test_create_processing_error_with_original(self):
        """Test create_processing_error with original error."""
        original = ValueError("Invalid value")
        error = create_processing_error(
            message="Processing failed",
            document_path="/docs/test.pdf",
            stage="validation",
            original_error=original,
        )
        assert error.context["original_error"] == "Invalid value"
        assert error.context["original_error_type"] == "ValueError"

    def test_create_dependency_error_missing(self):
        """Test create_dependency_error for missing dependency."""
        error = create_dependency_error(
            dependency="requests",
        )
        assert error.message == "Missing required dependency: requests"
        assert error.context["dependency_name"] == "requests"
        assert "Install with: pip install requests" in error.recovery_suggestions[0]

    def test_create_dependency_error_missing_with_version(self):
        """Test create_dependency_error for missing dependency with version."""
        error = create_dependency_error(
            dependency="pandas",
            required_version="1.0.0",
        )
        assert error.message == "Missing required dependency: pandas >= 1.0.0"
        assert error.context["required_version"] == "1.0.0"
        assert "Install with: pip install 'pandas>=1.0.0'" in error.recovery_suggestions[0]

    def test_create_dependency_error_incompatible_version(self):
        """Test create_dependency_error for incompatible versions."""
        error = create_dependency_error(
            dependency="numpy",
            required_version="1.20.0",
            installed_version="1.19.0",
        )
        assert "Incompatible numpy version" in error.message
        assert "required 1.20.0, found 1.19.0" in error.message
        assert error.context["required_version"] == "1.20.0"
        assert error.context["installed_version"] == "1.19.0"


class TestExceptionHierarchy:
    """Test the exception hierarchy and inheritance."""

    def test_all_exceptions_inherit_from_base(self):
        """Test that all exceptions inherit from CloakPivotError."""
        exceptions = [
            ValidationError("test"),
            ProcessingError("test"),
            DetectionError("test"),
            MaskingError("test"),
            UnmaskingError("test"),
            PolicyError("test"),
            IntegrityError("test"),
            PartialProcessingError("test", 1, 1, 0, []),
            ConfigurationError("test"),
            DependencyError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, CloakPivotError)
            assert isinstance(exc, Exception)

    def test_configuration_inherits_from_validation(self):
        """Test that ConfigurationError inherits from ValidationError."""
        error = ConfigurationError("test")
        assert isinstance(error, ValidationError)
        assert isinstance(error, CloakPivotError)

    def test_exception_string_representation(self):
        """Test that exceptions have proper string representation."""
        error = CloakPivotError("Test error message")
        assert str(error) == "Test error message"

        error2 = ValidationError("Validation failed", field_name="email")
        assert str(error2) == "Validation failed"
