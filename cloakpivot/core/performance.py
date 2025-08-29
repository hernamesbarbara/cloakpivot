"""Performance monitoring and metrics collection for CloakPivot operations."""

import functools
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generator, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class PerformanceMetric:
    """Individual performance measurement."""
    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_delta_mb: Optional[float] = None
    
    @property
    def timestamp(self) -> str:
        """Get ISO timestamp for this metric."""
        return datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat()


@dataclass  
class OperationStats:
    """Aggregated statistics for an operation type."""
    operation: str
    total_calls: int
    total_duration_ms: float
    average_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    last_call_time: float
    success_rate: float = 1.0
    failure_count: int = 0


class PerformanceProfiler:
    """
    Performance profiler for tracking timing and resource usage across operations.
    
    This profiler provides:
    - Method-level timing instrumentation via decorators
    - Context managers for ad-hoc performance measurement
    - Aggregated statistics and reporting
    - Integration with memory monitoring
    """
    
    def __init__(
        self,
        enable_memory_tracking: bool = True,
        enable_detailed_logging: bool = False,
        auto_report_threshold_ms: float = 1000.0,
    ) -> None:
        """
        Initialize performance profiler.
        
        Args:
            enable_memory_tracking: Track memory usage deltas during operations
            enable_detailed_logging: Log detailed performance information
            auto_report_threshold_ms: Auto-log operations exceeding this duration
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_detailed_logging = enable_detailed_logging
        self.auto_report_threshold_ms = auto_report_threshold_ms
        
        self._metrics: list[PerformanceMetric] = []
        self._operation_stats: Dict[str, OperationStats] = {}
        
        if self.enable_memory_tracking:
            try:
                from .memory_optimization import MemoryMonitor
                self._memory_monitor = MemoryMonitor()
            except ImportError:
                logger.warning("Memory monitoring unavailable - disabling memory tracking")
                self.enable_memory_tracking = False
                self._memory_monitor = None
        else:
            self._memory_monitor = None
        
        logger.debug(
            f"PerformanceProfiler initialized: memory_tracking={enable_memory_tracking}, "
            f"detailed_logging={enable_detailed_logging}, "
            f"auto_report_threshold={auto_report_threshold_ms}ms"
        )
    
    @contextmanager
    def measure_operation(
        self, 
        operation: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Generator[PerformanceMetric, None, None]:
        """
        Context manager for measuring operation performance.
        
        Args:
            operation: Name of the operation being measured
            metadata: Additional metadata to include with the measurement
            
        Yields:
            PerformanceMetric: Metric object that will be populated with timing data
        """
        start_time = time.perf_counter()
        start_memory = None
        
        if self._memory_monitor and self.enable_memory_tracking:
            self._memory_monitor.set_baseline()
            start_memory = self._memory_monitor.get_memory_stats().process_memory_mb
        
        # Create metric object
        metric = PerformanceMetric(
            operation=operation,
            start_time=start_time,
            end_time=0.0,
            duration_ms=0.0,
            metadata=metadata or {},
        )
        
        try:
            yield metric
            success = True
        except Exception:
            success = False
            raise
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Update metric
            metric.end_time = end_time
            metric.duration_ms = duration_ms
            
            # Add memory delta if tracking enabled
            if self._memory_monitor and self.enable_memory_tracking and start_memory:
                current_memory = self._memory_monitor.get_memory_stats().process_memory_mb
                metric.memory_delta_mb = current_memory - start_memory
            
            # Record metric
            self._record_metric(metric, success)
            
            # Auto-report slow operations
            if duration_ms >= self.auto_report_threshold_ms:
                logger.warning(
                    f"Slow operation detected: {operation} took {duration_ms:.1f}ms"
                )
    
    def timing_decorator(
        self, 
        operation: Optional[str] = None, 
        include_args: bool = False
    ) -> Callable[[F], F]:
        """
        Decorator for automatic method timing.
        
        Args:
            operation: Name for the operation (defaults to function name)
            include_args: Whether to include function arguments in metadata
            
        Returns:
            Decorated function with timing instrumentation
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation or f"{func.__module__}.{func.__name__}"
                
                metadata = {}
                if include_args:
                    metadata['args_count'] = len(args)
                    metadata['kwargs_keys'] = list(kwargs.keys())
                
                with self.measure_operation(op_name, metadata):
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def _record_metric(self, metric: PerformanceMetric, success: bool) -> None:
        """Record a performance metric and update statistics."""
        self._metrics.append(metric)
        
        # Update operation statistics
        op_name = metric.operation
        
        if op_name not in self._operation_stats:
            self._operation_stats[op_name] = OperationStats(
                operation=op_name,
                total_calls=0,
                total_duration_ms=0.0,
                average_duration_ms=0.0,
                min_duration_ms=float('inf'),
                max_duration_ms=0.0,
                last_call_time=metric.start_time,
                success_rate=0.0,
                failure_count=0,
            )
        
        stats = self._operation_stats[op_name]
        stats.total_calls += 1
        stats.total_duration_ms += metric.duration_ms
        stats.average_duration_ms = stats.total_duration_ms / stats.total_calls
        stats.min_duration_ms = min(stats.min_duration_ms, metric.duration_ms)
        stats.max_duration_ms = max(stats.max_duration_ms, metric.duration_ms)
        stats.last_call_time = metric.start_time
        
        if not success:
            stats.failure_count += 1
        
        stats.success_rate = (stats.total_calls - stats.failure_count) / stats.total_calls
        
        # Log detailed information if enabled
        if self.enable_detailed_logging:
            memory_info = ""
            if metric.memory_delta_mb is not None:
                memory_info = f" (Δ{metric.memory_delta_mb:+.1f} MB)"
            
            logger.debug(
                f"Performance metric: {op_name} completed in {metric.duration_ms:.1f}ms{memory_info}"
            )
    
    def get_operation_stats(self, operation: Optional[str] = None) -> Union[OperationStats, Dict[str, OperationStats]]:
        """
        Get performance statistics for operations.
        
        Args:
            operation: Specific operation to get stats for (None for all)
            
        Returns:
            OperationStats for specific operation or dict of all stats
        """
        if operation:
            return self._operation_stats.get(operation, OperationStats(
                operation=operation,
                total_calls=0,
                total_duration_ms=0.0,
                average_duration_ms=0.0,
                min_duration_ms=0.0,
                max_duration_ms=0.0,
                last_call_time=0.0,
            ))
        
        return self._operation_stats.copy()
    
    def get_recent_metrics(
        self, 
        operation: Optional[str] = None, 
        limit: int = 100
    ) -> list[PerformanceMetric]:
        """
        Get recent performance metrics.
        
        Args:
            operation: Filter by operation name (None for all)
            limit: Maximum number of metrics to return
            
        Returns:
            List of recent PerformanceMetric objects
        """
        metrics = self._metrics
        
        if operation:
            metrics = [m for m in metrics if m.operation == operation]
        
        # Return most recent metrics first
        return sorted(metrics, key=lambda m: m.start_time, reverse=True)[:limit]
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dictionary containing performance analysis and recommendations
        """
        total_operations = sum(stats.total_calls for stats in self._operation_stats.values())
        total_time = sum(stats.total_duration_ms for stats in self._operation_stats.values())
        
        # Find slowest operations
        slowest_ops = sorted(
            self._operation_stats.values(),
            key=lambda s: s.average_duration_ms,
            reverse=True
        )[:5]
        
        # Find most frequent operations
        frequent_ops = sorted(
            self._operation_stats.values(),
            key=lambda s: s.total_calls,
            reverse=True
        )[:5]
        
        # Calculate success rates
        failed_ops = [s for s in self._operation_stats.values() if s.failure_count > 0]
        
        report = {
            "summary": {
                "total_operations": total_operations,
                "total_time_ms": total_time,
                "unique_operation_types": len(self._operation_stats),
                "overall_success_rate": (
                    sum(s.total_calls - s.failure_count for s in self._operation_stats.values()) /
                    max(total_operations, 1)
                ),
            },
            "slowest_operations": [
                {
                    "operation": op.operation,
                    "average_duration_ms": op.average_duration_ms,
                    "total_calls": op.total_calls,
                    "max_duration_ms": op.max_duration_ms,
                }
                for op in slowest_ops
            ],
            "most_frequent_operations": [
                {
                    "operation": op.operation,
                    "total_calls": op.total_calls,
                    "average_duration_ms": op.average_duration_ms,
                    "total_time_ms": op.total_duration_ms,
                }
                for op in frequent_ops
            ],
            "operations_with_failures": [
                {
                    "operation": op.operation,
                    "failure_count": op.failure_count,
                    "success_rate": op.success_rate,
                    "total_calls": op.total_calls,
                }
                for op in failed_ops
            ],
            "recommendations": self._generate_recommendations(),
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        return report
    
    def _generate_recommendations(self) -> list[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Check for slow operations
        slow_ops = [
            s for s in self._operation_stats.values() 
            if s.average_duration_ms > self.auto_report_threshold_ms
        ]
        
        if slow_ops:
            recommendations.append(
                f"Consider optimizing {len(slow_ops)} operations with average duration "
                f"> {self.auto_report_threshold_ms}ms"
            )
        
        # Check for failed operations
        failed_ops = [s for s in self._operation_stats.values() if s.failure_count > 0]
        
        if failed_ops:
            recommendations.append(
                f"Investigate {len(failed_ops)} operations with failures to improve reliability"
            )
        
        # Check for high-frequency operations
        high_freq_ops = [
            s for s in self._operation_stats.values()
            if s.total_calls > 100 and s.average_duration_ms > 100
        ]
        
        if high_freq_ops:
            recommendations.append(
                f"Focus optimization on {len(high_freq_ops)} high-frequency operations"
            )
        
        return recommendations
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics and statistics."""
        self._metrics.clear()
        self._operation_stats.clear()
        logger.info("Performance metrics reset")
    
    def export_metrics_to_log(self) -> None:
        """Export performance metrics to structured logs for analysis."""
        report = self.generate_performance_report()
        
        logger.info(
            "Performance Report",
            extra={
                "performance_summary": report["summary"],
                "slowest_operations": report["slowest_operations"],
                "most_frequent_operations": report["most_frequent_operations"],
                "operations_with_failures": report["operations_with_failures"],
                "recommendations": report["recommendations"],
            }
        )


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """Get or create the global performance profiler instance."""
    global _global_profiler
    
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    
    return _global_profiler


def profile_method(operation: Optional[str] = None, include_args: bool = False) -> Callable[[F], F]:
    """
    Convenience decorator for profiling methods using global profiler.
    
    Args:
        operation: Name for the operation (defaults to function name)
        include_args: Whether to include function arguments in metadata
        
    Returns:
        Decorated function with timing instrumentation
    """
    return get_profiler().timing_decorator(operation, include_args)