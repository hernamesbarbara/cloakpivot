"""Comprehensive error handling and partial failure isolation system.

This module provides robust error handling capabilities including partial failure
isolation, error collection and categorization, and recovery strategies for
different types of failures throughout the CloakPivot pipeline.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from .exceptions import (
    CloakPivotError,
    PartialProcessingError,
    ProcessingError,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


class FailureToleranceLevel(Enum):
    """Defines how tolerant the system should be to failures."""

    STRICT = "strict"  # Fail on first error
    MODERATE = "moderate"  # Allow some failures but fail if too many
    PERMISSIVE = "permissive"  # Continue processing despite failures
    BEST_EFFORT = "best_effort"  # Never fail, always return partial results


@dataclass
class FailureToleranceConfig:
    """Configuration for failure tolerance behavior."""

    level: FailureToleranceLevel = FailureToleranceLevel.MODERATE
    max_failure_rate: float = 0.3  # Maximum allowed failure rate (0.0 - 1.0)
    max_consecutive_failures: int = 5  # Max consecutive failures before stopping
    min_success_count: int = 1  # Minimum successful operations required


@dataclass
class ErrorRecord:
    """Record of a single error that occurred during processing."""

    error: Exception
    timestamp: float = field(default_factory=time.time)
    context: dict[str, Any] = field(default_factory=dict)
    recoverable: bool = False
    retry_count: int = 0
    component: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Convert error record to dictionary for serialization."""
        return {
            "error_type": type(self.error).__name__,
            "message": str(self.error),
            "timestamp": self.timestamp,
            "context": self.context,
            "recoverable": self.recoverable,
            "retry_count": self.retry_count,
            "component": self.component,
        }


class ErrorCollector:
    """Collects and categorizes errors during processing operations."""

    def __init__(self) -> None:
        self.errors: list[ErrorRecord] = []
        self.success_count = 0
        self.total_operations = 0

    def record_success(self, context: dict[str, Any] | None = None) -> None:
        """Record a successful operation."""
        self.success_count += 1
        self.total_operations += 1
        logger.debug(
            f"Operation succeeded "
            f"(total: {self.total_operations}, successes: {self.success_count})"
        )

    def record_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        recoverable: bool = False,
        component: str = "unknown",
    ) -> None:
        """Record an error that occurred during processing."""
        record = ErrorRecord(
            error=error,
            context=context or {},
            recoverable=recoverable,
            component=component,
        )
        self.errors.append(record)
        self.total_operations += 1

        logger.warning(
            f"Error recorded in {component}: {error} "
            f"(total: {self.total_operations}, failures: {len(self.errors)})"
        )

    def get_failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_operations == 0:
            return 0.0
        return len(self.errors) / self.total_operations

    def get_consecutive_failures(self) -> int:
        """Get count of consecutive failures from the end."""
        consecutive = 0
        # Work backwards through recent operations to find consecutive failures
        recent_errors = self.errors[-10:] if self.errors else []
        for error in reversed(recent_errors):
            if error.timestamp > (time.time() - 60):  # Within last minute
                consecutive += 1
            else:
                break
        return consecutive

    def has_errors(self) -> bool:
        """Check if any errors have been recorded."""
        return len(self.errors) > 0

    def get_error_summary(self) -> dict[str, Any]:
        """Get summary of all recorded errors."""
        error_types: dict[str, int] = {}
        components: dict[str, int] = {}

        for error_record in self.errors:
            error_type = type(error_record.error).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1
            components[error_record.component] = components.get(error_record.component, 0) + 1

        return {
            "total_errors": len(self.errors),
            "total_operations": self.total_operations,
            "success_count": self.success_count,
            "failure_rate": self.get_failure_rate(),
            "error_types": error_types,
            "components": components,
            "consecutive_failures": self.get_consecutive_failures(),
        }

    def clear(self) -> None:
        """Clear all recorded errors and reset counters."""
        self.errors.clear()
        self.success_count = 0
        self.total_operations = 0


class PartialFailureManager:
    """Manages partial failure isolation and recovery strategies."""

    def __init__(
        self,
        tolerance_config: FailureToleranceConfig | None = None,
    ):
        self.tolerance_config = tolerance_config or FailureToleranceConfig()
        self.error_collector = ErrorCollector()

    def should_continue_processing(self) -> bool:
        """Determine if processing should continue based on current error state."""
        if self.tolerance_config.level == FailureToleranceLevel.STRICT:
            return not self.error_collector.has_errors()

        if self.tolerance_config.level == FailureToleranceLevel.BEST_EFFORT:
            return True

        # Check failure rate threshold
        failure_rate = self.error_collector.get_failure_rate()
        if failure_rate > self.tolerance_config.max_failure_rate:
            logger.warning(
                f"Failure rate {failure_rate:.2%} exceeds threshold "
                f"{self.tolerance_config.max_failure_rate:.2%}"
            )
            return False

        # Check consecutive failures
        consecutive = self.error_collector.get_consecutive_failures()
        if consecutive >= self.tolerance_config.max_consecutive_failures:
            logger.warning(
                f"Consecutive failures {consecutive} exceeds threshold "
                f"{self.tolerance_config.max_consecutive_failures}"
            )
            return False

        # Check minimum success requirement
        if (
            self.error_collector.total_operations >= 10
            and self.error_collector.success_count < self.tolerance_config.min_success_count
        ):
            logger.warning(
                f"Success count {self.error_collector.success_count} below minimum "
                f"{self.tolerance_config.min_success_count}"
            )
            return False

        return True

    def execute_with_isolation(
        self,
        operation: Callable[..., T],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        component: str = "unknown",
        recoverable: bool = True,
    ) -> T | None:
        """Execute an operation with error isolation and collection."""
        kwargs = kwargs or {}

        try:
            result = operation(*args, **kwargs)
            self.error_collector.record_success({"component": component})
            return result
        except Exception as e:
            # Determine if this is a recoverable error
            is_recoverable = recoverable and self._is_error_recoverable(e)

            self.error_collector.record_error(
                error=e,
                context={"args": str(args), "kwargs": str(kwargs)},
                recoverable=is_recoverable,
                component=component,
            )

            logger.error(f"Operation failed in {component}: {e}")
            return None

    def _is_error_recoverable(self, error: Exception) -> bool:
        """Determine if an error is potentially recoverable."""
        # Network-related errors are often recoverable
        if any(keyword in str(error).lower() for keyword in ["timeout", "connection", "network"]):
            return True

        # Some CloakPivot errors are recoverable
        if isinstance(error, CloakPivotError):
            return error.component not in ["validation", "integrity"]

        # File not found might be recoverable (temporary issue)
        if isinstance(error, FileNotFoundError):
            return True

        # Memory errors are not recoverable
        if isinstance(error, MemoryError):
            return False

        return False

    def finalize_processing(self) -> None:
        """Finalize processing and handle any accumulated errors."""
        if not self.error_collector.has_errors():
            return

        summary = self.error_collector.get_error_summary()

        # If we have a high failure rate or very few successes, raise an error
        if self.tolerance_config.level != FailureToleranceLevel.BEST_EFFORT and (
            summary["failure_rate"] > 0.8 or summary["success_count"] == 0
        ):
            failures = [record.to_dict() for record in self.error_collector.errors]
            raise PartialProcessingError(
                message=(
                    f"Processing completed with high failure rate: "
                    f"{summary['failure_rate']:.2%}"
                ),
                total_operations=summary["total_operations"],
                successful_operations=summary["success_count"],
                failed_operations=summary["total_errors"],
                failures=failures,
            )

    def get_processing_summary(self) -> dict[str, Any]:
        """Get comprehensive summary of processing results."""
        summary = self.error_collector.get_error_summary()
        summary["should_continue"] = self.should_continue_processing()
        summary["tolerance_level"] = self.tolerance_config.level.value
        return summary


class CircuitBreaker:
    """Circuit breaker pattern implementation for external service calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to apply circuit breaker to a function."""

        def wrapper(*args: Any, **kwargs: Any) -> T:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise ProcessingError(
                        "Circuit breaker is OPEN - service unavailable",
                        component="circuit_breaker",
                    )

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e

        return wrapper

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time is not None
            and time.time() - self.last_failure_time >= self.recovery_timeout
        )

    def _on_success(self) -> None:
        """Handle successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class RetryManager:
    """Manages retry logic with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def execute_with_retry(
        self,
        operation: Callable[..., T],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> T:
        """Execute operation with retry logic."""
        kwargs = kwargs or {}
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except retryable_exceptions as e:
                last_exception = e

                if attempt == self.max_retries:
                    logger.error(f"Operation failed after {self.max_retries} retries: {e}")
                    raise e

                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Operation failed on attempt {attempt + 1}, " f"retrying in {delay:.2f}s: {e}"
                )
                time.sleep(delay)

        # Should never reach here, but just in case
        raise last_exception or Exception("Unexpected retry loop exit")

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        import random

        delay = self.base_delay * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add ±25% jitter to prevent thundering herd
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)

        return delay


# Convenience functions for common error handling patterns


def create_partial_failure_manager(
    tolerance: str = "moderate",
    max_failure_rate: float = 0.3,
) -> PartialFailureManager:
    """Create a partial failure manager with common settings."""
    level = FailureToleranceLevel(tolerance.lower())
    config = FailureToleranceConfig(
        level=level,
        max_failure_rate=max_failure_rate,
    )
    return PartialFailureManager(config)


def with_error_isolation(
    manager: PartialFailureManager,
    component: str = "unknown",
    recoverable: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T | None]]:
    """Decorator to add error isolation to a function."""

    def decorator(func: Callable[..., T]) -> Callable[..., T | None]:
        def wrapper(*args: Any, **kwargs: Any) -> T | None:
            return manager.execute_with_isolation(func, args, kwargs, component, recoverable)

        return wrapper

    return decorator


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add retry logic to a function."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        retry_manager = RetryManager(max_retries=max_retries, base_delay=base_delay)

        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry_manager.execute_with_retry(func, args, kwargs, retryable_exceptions)

        return wrapper

    return decorator


def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type[Exception] = Exception,
) -> CircuitBreaker:
    """Decorator to add circuit breaker pattern to a function."""
    return CircuitBreaker(failure_threshold, recovery_timeout, expected_exception)
