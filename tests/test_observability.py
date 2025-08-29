"""Tests for observability and monitoring features."""

import json
import threading
import time
from datetime import datetime
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

import pytest

from cloakpivot.observability.config import (
    ObservabilityConfig,
    LoggingConfig,
    MetricsConfig,
    PrometheusConfig,
    StatsDConfig,
    WebhookConfig,
    get_config,
    set_config,
)
from cloakpivot.observability.logging import (
    configure_logging,
    get_logger,
    correlation_context,
    trace_operation,
    TraceContext,
)
from cloakpivot.observability.metrics import (
    MetricEvent,
    MetricType,
    MetricStore,
    MetricsCollector,
    collect_metrics,
    get_metrics_collector,
    set_metrics_collector,
)
from cloakpivot.observability.health import (
    HealthStatus,
    HealthCheckResult,
    SystemStatus,
    SystemResourcesCheck,
    DependenciesCheck,
    ConfigurationCheck,
    HealthMonitor,
)


class TestObservabilityConfig:
    """Test observability configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ObservabilityConfig()
        
        assert config.logging.level == "INFO"
        assert config.logging.format == "json"
        assert config.logging.enable_tracing is True
        assert config.metrics.enabled is True
        assert config.exporters.prometheus.enabled is True
        assert config.exporters.prometheus.port == 9090
        assert config.exporters.statsd.enabled is False

    def test_from_env(self):
        """Test loading configuration from environment variables."""
        with patch.dict('os.environ', {
            'CLOAKPIVOT_LOG_LEVEL': 'DEBUG',
            'CLOAKPIVOT_LOG_FORMAT': 'text',
            'CLOAKPIVOT_METRICS_ENABLED': 'false',
            'CLOAKPIVOT_PROMETHEUS_PORT': '9091',
        }):
            config = ObservabilityConfig.from_env()
            
            assert config.logging.level == "DEBUG"
            assert config.logging.format == "text"
            assert config.metrics.enabled is False
            assert config.exporters.prometheus.port == 9091

    def test_from_file(self, tmp_path):
        """Test loading configuration from file."""
        config_content = """
observability:
  logging:
    level: WARNING
    format: text
  metrics:
    enabled: false
  exporters:
    prometheus:
      port: 8080
    statsd:
      enabled: true
      host: statsd.example.com
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        config = ObservabilityConfig.from_file(config_file)
        
        assert config.logging.level == "WARNING"
        assert config.logging.format == "text"
        assert config.metrics.enabled is False
        assert config.exporters.prometheus.port == 8080
        assert config.exporters.statsd.enabled is True
        assert config.exporters.statsd.host == "statsd.example.com"

    def test_to_dict(self):
        """Test configuration serialization."""
        config = ObservabilityConfig()
        data = config.to_dict()
        
        assert isinstance(data, dict)
        assert "logging" in data
        assert "metrics" in data
        assert "exporters" in data

    def test_global_config(self):
        """Test global configuration management."""
        original_config = get_config()
        
        new_config = ObservabilityConfig()
        new_config.logging.level = "DEBUG"
        set_config(new_config)
        
        assert get_config().logging.level == "DEBUG"
        
        # Restore original
        set_config(original_config)


class TestStructuredLogging:
    """Test structured logging functionality."""

    def test_configure_logging(self):
        """Test logging configuration."""
        config = LoggingConfig(level="DEBUG", format="json")
        
        # This should not raise an exception
        configure_logging(config)

    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger(__name__)
        assert logger is not None

    def test_correlation_context(self):
        """Test correlation ID context."""
        with correlation_context() as corr_id:
            assert corr_id is not None
            assert len(corr_id) > 0

    def test_correlation_context_with_id(self):
        """Test correlation ID context with provided ID."""
        test_id = "test-correlation-123"
        with correlation_context(test_id) as corr_id:
            assert corr_id == test_id

    def test_trace_operation(self):
        """Test operation tracing."""
        with trace_operation("test_operation", param1="value1") as trace:
            assert isinstance(trace, TraceContext)
            assert trace.operation == "test_operation"
            assert "param1" in trace.attributes
            
            trace.set_attribute("result", "success")
            assert trace.attributes["result"] == "success"

    def test_trace_operation_with_exception(self):
        """Test operation tracing with exception."""
        with pytest.raises(ValueError, match="test error"):
            with trace_operation("failing_operation") as trace:
                trace.log_info("About to fail")
                raise ValueError("test error")


class TestMetrics:
    """Test metrics collection and management."""

    def test_metric_event_creation(self):
        """Test metric event creation."""
        event = MetricEvent(
            name="test_counter",
            value=1.0,
            labels={"type": "test"},
            metric_type=MetricType.COUNTER,
        )
        
        assert event.name == "test_counter"
        assert event.value == 1.0
        assert event.labels["type"] == "test"
        assert event.metric_type == MetricType.COUNTER

    def test_metric_event_validation(self):
        """Test metric event validation."""
        with pytest.raises(ValueError, match="Metric name cannot be empty"):
            MetricEvent(name="", value=1.0)
        
        with pytest.raises(ValueError, match="Metric value must be numeric"):
            MetricEvent(name="test", value="invalid")  # type: ignore

    def test_metric_store(self):
        """Test metric store functionality."""
        store = MetricStore(buffer_size=10)
        
        # Test counter
        store.record_counter("requests", 1.0, {"method": "GET"})
        store.record_counter("requests", 2.0, {"method": "GET"})
        
        counters = store.get_counters()
        assert len(counters) == 1
        key = list(counters.keys())[0]
        assert counters[key][0] == 3.0  # 1.0 + 2.0
        assert counters[key][1]["method"] == "GET"
        
        # Test gauge
        store.record_gauge("temperature", 25.5, {"sensor": "A1"})
        
        gauges = store.get_gauges()
        assert len(gauges) == 1
        key = list(gauges.keys())[0]
        assert gauges[key][0] == 25.5
        
        # Test histogram
        store.record_histogram("response_time", 0.1)
        store.record_histogram("response_time", 0.2)
        
        histograms = store.get_histograms()
        assert len(histograms) == 1
        key = list(histograms.keys())[0]
        assert len(histograms[key][0]) == 2
        
        # Test recent events
        events = store.get_recent_events(5)
        assert len(events) <= 5

    def test_metrics_collector(self):
        """Test metrics collector."""
        config = MetricsConfig(enabled=True, collection_interval=1)
        collector = MetricsCollector(config)
        
        # Test metric recording
        collector.counter("test_counter", 1.0, {"label": "value"})
        collector.gauge("test_gauge", 100.0)
        collector.histogram("test_histogram", 0.5)
        collector.timing("test_timing", 0.123)
        
        store = collector.get_store()
        counters = store.get_counters()
        gauges = store.get_gauges()
        histograms = store.get_histograms()
        
        assert len(counters) >= 1
        assert len(gauges) >= 1
        assert len(histograms) >= 2  # histogram + timing

    def test_collect_metrics_decorator(self):
        """Test metrics collection decorator."""
        collector = MetricsCollector()
        set_metrics_collector(collector)
        
        @collect_metrics("test_function")
        def test_func(x: int) -> int:
            return x * 2
        
        # Call the decorated function
        result = test_func(5)
        assert result == 10
        
        # Check metrics were collected
        store = collector.get_store()
        events = store.get_recent_events()
        assert len(events) >= 2  # calls_total and duration

    def test_collect_metrics_decorator_with_exception(self):
        """Test metrics collection decorator with exception."""
        collector = MetricsCollector()
        set_metrics_collector(collector)
        
        @collect_metrics("failing_function")
        def failing_func() -> None:
            raise ValueError("test error")
        
        with pytest.raises(ValueError):
            failing_func()
        
        # Check error metrics were collected
        store = collector.get_store()
        counters = store.get_counters()
        
        # Look for error counter
        error_counters = [
            key for key in counters.keys() 
            if "error" in key and "failing_function" in key
        ]
        assert len(error_counters) > 0


class TestHealthChecks:
    """Test health monitoring functionality."""

    def test_health_check_result(self):
        """Test health check result creation."""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="All good",
            details={"key": "value"},
        )
        
        assert result.name == "test_check"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
        assert result.details["key"] == "value"
        
        # Test serialization
        data = result.to_dict()
        assert data["name"] == "test_check"
        assert data["status"] == "healthy"

    def test_system_status(self):
        """Test system status creation."""
        check1 = HealthCheckResult("check1", HealthStatus.HEALTHY)
        check2 = HealthCheckResult("check2", HealthStatus.DEGRADED)
        
        status = SystemStatus(
            status=HealthStatus.DEGRADED,
            checks=[check1, check2],
            uptime_seconds=3600.0,
        )
        
        assert status.status == HealthStatus.DEGRADED
        assert len(status.checks) == 2
        assert status.uptime_seconds == 3600.0
        
        # Test serialization
        data = status.to_dict()
        assert data["status"] == "degraded"
        assert len(data["checks"]) == 2

    def test_system_resources_check(self):
        """Test system resources health check."""
        check = SystemResourcesCheck(
            memory_threshold=0.99,  # Very high to ensure healthy status
            disk_threshold=0.99,
            cpu_threshold=0.99,
        )
        
        result = check.check()
        
        assert result.name == "system_resources"
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert "memory" in result.details
        assert "disk" in result.details
        assert "cpu" in result.details

    def test_dependencies_check(self):
        """Test dependencies health check."""
        check = DependenciesCheck()
        result = check.check()
        
        assert result.name == "dependencies"
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert "dependencies" in result.details

    def test_configuration_check(self):
        """Test configuration health check."""
        check = ConfigurationCheck()
        result = check.check()
        
        assert result.name == "configuration"
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY]
        assert "config_summary" in result.details

    def test_health_monitor(self):
        """Test health monitor."""
        monitor = HealthMonitor()
        
        # Add a custom check
        class TestCheck:
            def __init__(self):
                self.name = "test_check"
                self.timeout = 30.0
            
            def check(self):
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Test check passed",
                )
        
        monitor.add_check(TestCheck())
        
        status = monitor.get_status()
        
        assert isinstance(status, SystemStatus)
        assert status.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert len(status.checks) >= 1  # At least our test check
        assert status.uptime_seconds >= 0
        assert "platform" in status.system_info
        
        # Test removing check
        monitor.remove_check("test_check")
        new_status = monitor.get_status()
        
        test_checks = [check for check in new_status.checks if check.name == "test_check"]
        assert len(test_checks) == 0


class TestIntegration:
    """Test integration between components."""

    def test_full_observability_stack(self):
        """Test the full observability stack together."""
        # Configure observability
        config = ObservabilityConfig()
        config.logging.level = "DEBUG"
        config.metrics.enabled = True
        set_config(config)
        
        # Configure logging
        configure_logging(config.logging)
        
        # Set up metrics collector
        collector = MetricsCollector(config.metrics)
        set_metrics_collector(collector)
        
        # Set up health monitor
        monitor = HealthMonitor(config.health)
        
        # Use the stack
        logger = get_logger(__name__)
        
        with correlation_context("integration-test") as corr_id:
            with trace_operation("integration_test") as trace:
                # Log something
                logger.info("Integration test starting")
                
                # Record metrics
                collector.counter("integration_test_counter")
                collector.gauge("integration_test_gauge", 42.0)
                collector.timing("integration_test_timing", 0.123)
                
                # Check health
                health_status = monitor.get_status()
                
                trace.log_info("Integration test completed", 
                             health_status=health_status.status.value)
        
        # Verify everything worked
        assert corr_id is not None
        
        store = collector.get_store()
        events = store.get_recent_events()
        assert len(events) >= 3  # counter, gauge, timing
        
        assert health_status.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert len(health_status.checks) >= 3  # Default checks

    def test_concurrent_operations(self):
        """Test observability under concurrent load."""
        collector = MetricsCollector()
        set_metrics_collector(collector)
        
        def worker(worker_id: int) -> None:
            """Worker function for concurrent testing."""
            with correlation_context(f"worker-{worker_id}"):
                with trace_operation(f"worker_{worker_id}_operation"):
                    for i in range(10):
                        collector.counter("concurrent_counter", 1.0, {"worker": str(worker_id)})
                        collector.gauge("concurrent_gauge", float(i), {"worker": str(worker_id)})
                        time.sleep(0.01)  # Small delay to simulate work
        
        # Start multiple workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify metrics were collected
        store = collector.get_store()
        counters = store.get_counters()
        gauges = store.get_gauges()
        
        # Should have metrics from all workers
        assert len(counters) >= 5  # One per worker
        assert len(gauges) >= 5   # One per worker
        
        # Total counter value should be 50 (5 workers * 10 increments)
        total_counter_value = sum(value for value, labels in counters.values() 
                                if "concurrent_counter" in list(counters.keys())[0])
        assert total_counter_value >= 50