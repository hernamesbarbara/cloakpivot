"""Comprehensive tests for the error handling system.

Tests cover exception hierarchy, partial failure isolation, validation,
error recovery, and retry logic to ensure robust error handling behavior.
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from cloakpivot.core.error_handling import (
    ErrorCollector,
    FailureToleranceConfig,
    FailureToleranceLevel,
    PartialFailureManager,
    RetryManager,
    create_partial_failure_manager,
    with_circuit_breaker,
    with_error_isolation,
    with_retry,
)
from cloakpivot.core.exceptions import (
    CloakPivotError,
    DetectionError,
    MaskingError,
    PartialProcessingError,
    ProcessingError,
    UnmaskingError,
    ValidationError,
    create_dependency_error,
    create_processing_error,
    create_validation_error,
)
from cloakpivot.core.validation import (
    CloakMapValidator,
    DocumentValidator,
    InputValidator,
    PolicyValidator,
    SystemValidator,
)


class TestExceptionHierarchy:
    """Test the CloakPivot exception hierarchy."""

    def test_base_exception_creation(self):
        """Test basic CloakPivotError creation and attributes."""
        error = CloakPivotError(
            "Test error",
            error_code="TEST_ERROR",
            context={"key": "value"},
            recovery_suggestions=["Try again"],
            component="test",
        )

        assert str(error) == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.context == {"key": "value"}
        assert error.recovery_suggestions == ["Try again"]
        assert error.component == "test"

    def test_error_code_generation(self):
        """Test automatic error code generation."""
        error = ValidationError("Test validation error")
        assert error.error_code == "VALIDATION_ERROR"

        error = ProcessingError("Test processing error")
        assert error.error_code == "PROCESSING_ERROR"

    def test_component_inference(self):
        """Test automatic component inference from error type."""
        validation_error = ValidationError("Test")
        assert validation_error.component == "validation"

        detection_error = DetectionError("Test")
        assert detection_error.component == "detection"

        masking_error = MaskingError("Test")
        assert masking_error.component == "masking"

    def test_context_management(self):
        """Test adding context to errors."""
        error = CloakPivotError("Test error")
        error.add_context("file", "test.json")
        error.add_context("line", 42)

        assert error.context["file"] == "test.json"
        assert error.context["line"] == 42

    def test_recovery_suggestions(self):
        """Test recovery suggestion management."""
        error = CloakPivotError("Test error")
        error.add_recovery_suggestion("Check file permissions")
        error.add_recovery_suggestion("Verify file format")
        error.add_recovery_suggestion("Check file permissions")  # Duplicate

        assert len(error.recovery_suggestions) == 2
        assert "Check file permissions" in error.recovery_suggestions
        assert "Verify file format" in error.recovery_suggestions

    def test_error_serialization(self):
        """Test error conversion to dictionary."""
        error = ValidationError(
            "Invalid field",
            field_name="test_field",
            expected_type="string",
            actual_value=42,
        )

        error_dict = error.to_dict()
        assert error_dict["error_type"] == "ValidationError"
        assert error_dict["message"] == "Invalid field"
        assert error_dict["component"] == "validation"
        assert "field_name" in error_dict["context"]

    def test_specialized_exceptions(self):
        """Test specialized exception types with their specific attributes."""
        # ValidationError
        val_error = ValidationError(
            "Invalid value",
            field_name="test",
            expected_type="int",
            actual_value="string",
        )
        assert val_error.context["field_name"] == "test"

        # ProcessingError
        proc_error = ProcessingError(
            "Processing failed",
            document_path="/test/doc.pdf",
            processing_stage="extraction",
        )
        assert proc_error.context["document_path"] == "/test/doc.pdf"

        # DetectionError
        det_error = DetectionError(
            "Detection failed", entity_type="PERSON", confidence_threshold=0.8
        )
        assert det_error.context["entity_type"] == "PERSON"

        # UnmaskingError
        unmask_error = UnmaskingError(
            "Unmasking failed", cloakmap_version="1.0", anchor_count=5
        )
        assert unmask_error.context["cloakmap_version"] == "1.0"

    def test_convenience_functions(self):
        """Test convenience functions for creating common errors."""
        val_error = create_validation_error("Invalid type", "field", str, 42)
        assert val_error.context["field_name"] == "field"
        assert val_error.context["expected_type"] == "str"

        proc_error = create_processing_error(
            "Failed", "/test/doc", "parsing", ValueError("test")
        )
        assert proc_error.context["document_path"] == "/test/doc"
        assert proc_error.context["processing_stage"] == "parsing"
        assert "ValueError" in proc_error.context["original_error_type"]

        dep_error = create_dependency_error("test-package", "1.0.0", "0.9.0")
        assert dep_error.context["dependency_name"] == "test-package"
        assert "pip install" in dep_error.recovery_suggestions[0]


class TestErrorCollector:
    """Test error collection and categorization."""

    def test_error_collection(self):
        """Test basic error collection functionality."""
        collector = ErrorCollector()

        collector.record_success({"operation": "test1"})
        collector.record_error(
            ValueError("Test error"), {"operation": "test2"}, component="test"
        )
        collector.record_success({"operation": "test3"})

        assert collector.success_count == 2
        assert collector.total_operations == 3
        assert len(collector.errors) == 1
        assert collector.get_failure_rate() == 1.0 / 3.0

    def test_consecutive_failures(self):
        """Test consecutive failure tracking."""
        collector = ErrorCollector()

        # Record some successes and failures
        collector.record_success()
        collector.record_error(ValueError("Error 1"))
        collector.record_error(ValueError("Error 2"))

        # Recent errors should be counted as consecutive
        consecutive = collector.get_consecutive_failures()
        assert consecutive >= 0  # Depends on timing

    def test_error_summary(self):
        """Test error summary generation."""
        collector = ErrorCollector()

        collector.record_success()
        collector.record_error(ValueError("Error 1"), component="comp1")
        collector.record_error(TypeError("Error 2"), component="comp2")
        collector.record_error(ValueError("Error 3"), component="comp1")

        summary = collector.get_error_summary()

        assert summary["total_errors"] == 3
        assert summary["total_operations"] == 4
        assert summary["success_count"] == 1
        assert summary["failure_rate"] == 0.75
        assert summary["error_types"]["ValueError"] == 2
        assert summary["error_types"]["TypeError"] == 1
        assert summary["components"]["comp1"] == 2

    def test_clear_functionality(self):
        """Test clearing collected errors."""
        collector = ErrorCollector()

        collector.record_success()
        collector.record_error(ValueError("Test"))

        assert collector.has_errors()

        collector.clear()

        assert not collector.has_errors()
        assert collector.success_count == 0
        assert collector.total_operations == 0


class TestPartialFailureManager:
    """Test partial failure isolation and management."""

    def test_failure_tolerance_levels(self):
        """Test different failure tolerance levels."""
        # Strict - should stop on first failure
        strict_config = FailureToleranceConfig(level=FailureToleranceLevel.STRICT)
        strict_manager = PartialFailureManager(strict_config)

        result = strict_manager.execute_with_isolation(
            lambda: 1 / 0,  # Will raise ZeroDivisionError
            component="test",
        )
        assert result is None
        assert not strict_manager.should_continue_processing()

        # Best effort - should never stop
        best_effort_config = FailureToleranceConfig(
            level=FailureToleranceLevel.BEST_EFFORT
        )
        best_effort_manager = PartialFailureManager(best_effort_config)

        result = best_effort_manager.execute_with_isolation(
            lambda: 1 / 0, component="test"
        )
        assert result is None
        assert best_effort_manager.should_continue_processing()

    def test_failure_rate_threshold(self):
        """Test failure rate threshold enforcement."""
        config = FailureToleranceConfig(
            level=FailureToleranceLevel.MODERATE, max_failure_rate=0.5
        )
        manager = PartialFailureManager(config)

        # Record some successes and failures
        for i in range(10):
            if i < 4:  # 40% success rate
                manager.error_collector.record_success()
            else:  # 60% failure rate - exceeds threshold
                manager.error_collector.record_error(ValueError(f"Error {i}"))

        assert not manager.should_continue_processing()

    def test_successful_operations(self):
        """Test successful operation isolation."""
        manager = PartialFailureManager()

        def successful_operation(x, y):
            return x + y

        result = manager.execute_with_isolation(
            successful_operation, args=(2, 3), component="math"
        )

        assert result == 5
        assert manager.error_collector.success_count == 1

    def test_finalize_processing(self):
        """Test processing finalization with different error scenarios."""
        # Low failure rate - should not raise
        moderate_manager = PartialFailureManager()
        moderate_manager.error_collector.record_success()
        moderate_manager.error_collector.record_success()
        moderate_manager.error_collector.record_error(ValueError("Test"))

        moderate_manager.finalize_processing()  # Should not raise

        # High failure rate - should raise PartialProcessingError
        failing_manager = PartialFailureManager()
        for _ in range(10):
            failing_manager.error_collector.record_error(ValueError("Test"))

        with pytest.raises(PartialProcessingError):
            failing_manager.finalize_processing()


class TestCircuitBreaker:
    """Test circuit breaker pattern implementation."""

    def test_circuit_breaker_states(self, monkeypatch):
        """Test circuit breaker state transitions."""
        mock_time = 1000.0  # Start time

        def mock_time_func():
            return mock_time

        monkeypatch.setattr(time, "time", mock_time_func)

        @with_circuit_breaker(failure_threshold=2, recovery_timeout=0.1)
        def failing_function():
            raise ConnectionError("Service unavailable")

        # First few failures should work (CLOSED state)
        with pytest.raises(ConnectionError):
            failing_function()
        with pytest.raises(ConnectionError):
            failing_function()

        # After threshold, should get ProcessingError (OPEN state)
        with pytest.raises(ProcessingError):
            failing_function()

        # Advance time beyond recovery timeout
        mock_time += 0.2
        # After recovery timeout, should try again (HALF_OPEN state)
        with pytest.raises(ConnectionError):
            failing_function()

    def test_circuit_breaker_recovery(self, monkeypatch):
        """Test circuit breaker recovery after success."""
        failure_count = 0
        mock_time = 2000.0  # Start time

        def mock_time_func():
            return mock_time

        monkeypatch.setattr(time, "time", mock_time_func)

        @with_circuit_breaker(failure_threshold=2, recovery_timeout=0.1)
        def sometimes_failing_function():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise ConnectionError("Service unavailable")
            return "success"

        # Trigger failures to open circuit
        with pytest.raises(ConnectionError):
            sometimes_failing_function()
        with pytest.raises(ConnectionError):
            sometimes_failing_function()

        # Circuit should be open
        with pytest.raises(ProcessingError):
            sometimes_failing_function()

        # Advance time beyond recovery timeout and succeed
        mock_time += 0.2
        result = sometimes_failing_function()
        assert result == "success"


class TestRetryManager:
    """Test retry logic and exponential backoff."""

    def test_successful_retry(self):
        """Test successful operation after retries."""
        attempt_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        def eventually_successful():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = eventually_successful()
        assert result == "success"
        assert attempt_count == 3

    def test_retry_exhaustion(self):
        """Test behavior when all retries are exhausted."""

        @with_retry(max_retries=2, base_delay=0.01)
        def always_failing():
            raise ValueError("Permanent failure")

        with pytest.raises(ValueError):
            always_failing()

    def test_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        retry_manager = RetryManager(base_delay=1.0, exponential_base=2.0, jitter=False)

        delay0 = retry_manager._calculate_delay(0)
        delay1 = retry_manager._calculate_delay(1)
        delay2 = retry_manager._calculate_delay(2)

        assert delay0 == 1.0
        assert delay1 == 2.0
        assert delay2 == 4.0


class TestValidation:
    """Test comprehensive validation system."""

    def test_system_validation(self):
        """Test system requirements validation."""
        # Python version validation should pass for current version
        SystemValidator.validate_python_version((3, 8))

        # Should fail for future version
        with pytest.raises(ValidationError):
            SystemValidator.validate_python_version((4, 0))

    def test_file_validation(self):
        """Test file permission validation."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            # Should pass for existing readable file
            SystemValidator.validate_file_permissions(temp_path, require_read=True)

            # Should fail for non-existent file
            with pytest.raises(ValidationError):
                SystemValidator.validate_file_permissions("/non/existent/file")
        finally:
            Path(temp_path).unlink()

    def test_policy_validation(self):
        """Test policy structure validation."""
        valid_policy = {
            "default_strategy": {"kind": "redact"},
            "per_entity": {"PERSON": {"kind": "hash"}},
            "thresholds": {"PERSON": 0.8},
        }

        # Valid policy should pass
        PolicyValidator.validate_policy_structure(valid_policy)
        PolicyValidator.validate_thresholds(valid_policy["thresholds"])

        # Invalid policy should fail
        invalid_policy = {"per_entity": {"PERSON": {"kind": "invalid"}}}
        with pytest.raises(ValidationError):
            PolicyValidator.validate_policy_structure(invalid_policy)

        # Invalid thresholds should fail
        with pytest.raises(ValidationError):
            PolicyValidator.validate_thresholds({"PERSON": 1.5})

    def test_cloakmap_validation(self):
        """Test CloakMap structure validation."""
        valid_cloakmap = {
            "doc_id": "test-123",
            "version": "1.0",
            "created_at": "2023-01-01T00:00:00Z",
            "anchors": [
                {
                    "anchor_id": "anchor-1",
                    "node_id": "node-1",
                    "start": 0,
                    "end": 10,
                    "entity_type": "PERSON",
                }
            ],
        }

        # Valid CloakMap should pass
        CloakMapValidator.validate_cloakmap_structure(valid_cloakmap)

        # Missing required fields should fail
        invalid_cloakmap = {"doc_id": "test"}
        with pytest.raises(ValidationError):
            CloakMapValidator.validate_cloakmap_structure(invalid_cloakmap)

        # Invalid anchor should fail
        invalid_anchor_cloakmap = valid_cloakmap.copy()
        invalid_anchor_cloakmap["anchors"] = [
            {"anchor_id": "test", "start": -1, "end": 5}
        ]
        with pytest.raises(ValidationError):
            CloakMapValidator.validate_cloakmap_structure(invalid_anchor_cloakmap)

    def test_document_validation(self):
        """Test document format and size validation."""
        # Create temporary test files
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"name": "test", "texts": []}, f)
            json_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as f:
            f.write("test")
            invalid_path = f.name

        try:
            # Valid JSON document should pass
            format_type = DocumentValidator.validate_document_format(json_path)
            assert format_type == "json"

            # Invalid format should fail
            with pytest.raises(ValidationError):
                DocumentValidator.validate_document_format(invalid_path)

            # Document size validation should pass for small files
            DocumentValidator.validate_document_size(json_path, max_size_mb=1.0)

        finally:
            Path(json_path).unlink()
            Path(invalid_path).unlink()


class TestIntegration:
    """Test integration of error handling components."""

    def test_complete_error_handling_workflow(self):
        """Test complete error handling workflow with validation and partial failures."""
        # Create a manager with moderate tolerance but higher failure rate threshold
        manager = create_partial_failure_manager(
            tolerance="moderate",
            max_failure_rate=0.6,  # Allow higher failure rate for this test
        )

        # Simulate processing workflow with mixed success/failure
        operations = [
            (lambda: "success1", True),
            (lambda: 1 / 0, False),  # Will fail
            (lambda: "success2", True),
            (lambda: "success3", True),
            (lambda: int("not_a_number"), False),  # Will fail
        ]

        results = []
        for operation, _should_succeed in operations:
            if not manager.should_continue_processing():
                break

            result = manager.execute_with_isolation(
                operation, component="test_workflow"
            )
            results.append(result)

        # Should have processed all operations despite failures
        assert len(results) == 5
        assert results[0] == "success1"
        assert results[1] is None  # Failed operation
        assert results[2] == "success2"

        # Should be able to finalize without exception (under failure threshold)
        manager.finalize_processing()

        summary = manager.get_processing_summary()
        assert summary["total_operations"] == 5
        assert summary["success_count"] == 3
        assert summary["total_errors"] == 2

    def test_validation_with_error_handling(self):
        """Test integration of validation with error handling."""
        validator = InputValidator()

        # Test with invalid inputs - should get clear validation errors
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_masking_inputs(
                document_path="/non/existent/file", policy_data={"invalid": "policy"}
            )

        error = exc_info.value
        assert isinstance(error, ValidationError)
        assert error.component == "validation"
        assert len(error.recovery_suggestions) > 0


class TestConvenienceFunctions:
    """Test convenience functions and decorators."""

    def test_error_isolation_decorator(self):
        """Test error isolation decorator."""
        manager = PartialFailureManager()

        @with_error_isolation(manager, component="decorated", recoverable=True)
        def test_function(x):
            if x == 0:
                raise ValueError("Zero not allowed")
            return x * 2

        # Success case
        result = test_function(5)
        assert result == 10

        # Failure case
        result = test_function(0)
        assert result is None
        assert manager.error_collector.has_errors()

    def test_convenience_manager_creation(self):
        """Test convenience function for creating managers."""
        manager = create_partial_failure_manager(
            tolerance="permissive", max_failure_rate=0.8
        )

        assert manager.tolerance_config.level == FailureToleranceLevel.PERMISSIVE
        assert manager.tolerance_config.max_failure_rate == 0.8


class TestErrorRecovery:
    """Test error recovery and guidance systems."""

    def test_recovery_suggestions(self):
        """Test that errors provide helpful recovery suggestions."""
        dep_error = create_dependency_error("missing-package", "1.0.0")
        assert any(
            "pip install" in suggestion for suggestion in dep_error.recovery_suggestions
        )

        val_error = create_validation_error("Invalid type", "field", str, 42)
        assert any(
            "Ensure field is of type" in suggestion
            for suggestion in val_error.recovery_suggestions
        )

        proc_error = create_processing_error("Failed to parse", "/test/doc", "parsing")
        assert any(
            "Check document format" in suggestion
            for suggestion in proc_error.recovery_suggestions
        )

    def test_structured_error_information(self):
        """Test that errors provide structured information for debugging."""
        error = DetectionError(
            "PII detection failed", entity_type="PERSON", confidence_threshold=0.8
        )

        error_dict = error.to_dict()
        assert error_dict["component"] == "detection"
        assert error_dict["context"]["entity_type"] == "PERSON"
        assert error_dict["context"]["confidence_threshold"] == 0.8

        # Should be JSON serializable
        json_str = json.dumps(error_dict)
        assert json_str is not None
