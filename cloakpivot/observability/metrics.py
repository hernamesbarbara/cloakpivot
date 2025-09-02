"""Metrics collection and management."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar

from .config import get_config
from .logging import get_logger

F = TypeVar("F", bound=Callable[..., Any])


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricEvent:
    """A single metric event."""

    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metric_type: MetricType = MetricType.COUNTER

    def __post_init__(self) -> None:
        """Validate metric event."""
        if not self.name:
            raise ValueError("Metric name cannot be empty")
        if not isinstance(self.value, (int, float)):
            raise ValueError("Metric value must be numeric")


@dataclass
class HistogramBucket:
    """A histogram bucket."""

    le: float  # less than or equal to
    count: int = 0


class MetricStore:
    """Thread-safe metric storage."""

    def __init__(self, buffer_size: int = 1000):
        self._buffer_size = buffer_size
        self._lock = threading.RLock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._events: deque[MetricEvent] = deque(maxlen=buffer_size)
        self._labels: dict[str, dict[str, str]] = {}

    def record_counter(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Record a counter metric."""
        labels = labels or {}
        key = self._make_key(name, labels)

        with self._lock:
            self._counters[key] += value
            self._labels[key] = labels
            self._events.append(
                MetricEvent(
                    name=name,
                    value=value,
                    labels=labels,
                    metric_type=MetricType.COUNTER,
                )
            )

    def record_gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a gauge metric."""
        labels = labels or {}
        key = self._make_key(name, labels)

        with self._lock:
            self._gauges[key] = value
            self._labels[key] = labels
            self._events.append(
                MetricEvent(
                    name=name, value=value, labels=labels, metric_type=MetricType.GAUGE
                )
            )

    def record_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a histogram metric."""
        labels = labels or {}
        key = self._make_key(name, labels)

        with self._lock:
            self._histograms[key].append(value)
            # Keep only recent values to prevent memory growth
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-500:]

            self._labels[key] = labels
            self._events.append(
                MetricEvent(
                    name=name,
                    value=value,
                    labels=labels,
                    metric_type=MetricType.HISTOGRAM,
                )
            )

    def get_counters(self) -> dict[str, tuple[float, dict[str, str]]]:
        """Get all counter values with labels."""
        with self._lock:
            return {
                key: (value, self._labels.get(key, {}))
                for key, value in self._counters.items()
            }

    def get_gauges(self) -> dict[str, tuple[float, dict[str, str]]]:
        """Get all gauge values with labels."""
        with self._lock:
            return {
                key: (value, self._labels.get(key, {}))
                for key, value in self._gauges.items()
            }

    def get_histograms(self) -> dict[str, tuple[list[float], dict[str, str]]]:
        """Get all histogram values with labels."""
        with self._lock:
            return {
                key: (values[:], self._labels.get(key, {}))
                for key, values in self._histograms.items()
            }

    def get_recent_events(self, count: int | None = None) -> list[MetricEvent]:
        """Get recent metric events."""
        with self._lock:
            events = list(self._events)
            if count is not None:
                events = events[-count:]
            return events

    def clear(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._events.clear()
            self._labels.clear()

    @staticmethod
    def _make_key(name: str, labels: dict[str, str]) -> str:
        """Create a unique key for metric with labels."""
        if not labels:
            return name
        sorted_labels = sorted(labels.items())
        label_str = ",".join(f"{k}={v}" for k, v in sorted_labels)
        return f"{name}{{{label_str}}}"


class MetricExporter(ABC):
    """Abstract base class for metric exporters."""

    @abstractmethod
    def export_metrics(self, store: MetricStore) -> None:
        """Export metrics from the store."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the exporter and clean up resources."""
        pass


class MetricsCollector:
    """Main metrics collector."""

    def __init__(self, config: Any | None = None):
        self._config = config or get_config().metrics
        self._store = MetricStore(buffer_size=self._config.buffer_size)
        self._exporters: list[MetricExporter] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._logger = get_logger(__name__)

    def add_exporter(self, exporter: MetricExporter) -> None:
        """Add a metric exporter."""
        self._exporters.append(exporter)

    def start(self) -> None:
        """Start the metrics collector."""
        if self._running or not self._config.enabled:
            return

        self._running = True
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()
        self._logger.info("Metrics collector started")

    def stop(self) -> None:
        """Stop the metrics collector."""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

        for exporter in self._exporters:
            try:
                exporter.close()
            except Exception as e:
                self._logger.error(f"Error closing exporter: {e}")

        self._logger.info("Metrics collector stopped")

    def counter(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Record a counter metric."""
        if self._config.enabled:
            self._store.record_counter(name, value, labels)

    def gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a gauge metric."""
        if self._config.enabled:
            self._store.record_gauge(name, value, labels)

    def histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a histogram metric."""
        if self._config.enabled:
            self._store.record_histogram(name, value, labels)

    def timing(
        self, name: str, duration: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a timing metric (histogram)."""
        self.histogram(f"{name}_duration_seconds", duration, labels)

    def get_store(self) -> MetricStore:
        """Get the metric store."""
        return self._store

    def _collection_loop(self) -> None:
        """Main collection loop."""
        while self._running:
            try:
                for exporter in self._exporters:
                    try:
                        exporter.export_metrics(self._store)
                    except Exception as e:
                        self._logger.error(f"Error in metric exporter: {e}")

                time.sleep(self._config.collection_interval)
            except Exception as e:
                self._logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(1)


# Global metrics collector
_collector: MetricsCollector | None = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _collector
    with _collector_lock:
        if _collector is None:
            _collector = MetricsCollector()
        return _collector


def set_metrics_collector(collector: MetricsCollector) -> None:
    """Set the global metrics collector."""
    global _collector
    with _collector_lock:
        if _collector and _collector._running:
            _collector.stop()
        _collector = collector


def collect_metrics(
    metric_name: str,
    labels: dict[str, str] | None = None,
    count_calls: bool = True,
    measure_time: bool = True,
) -> Callable[[F], F]:
    """Decorator to collect metrics for function calls."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            collector = get_metrics_collector()
            start_time = time.time()

            # Count function calls
            if count_calls:
                call_labels = {**(labels or {}), "function": func.__name__}
                collector.counter(f"{metric_name}_calls_total", 1.0, call_labels)

            try:
                result = func(*args, **kwargs)

                # Record success
                if count_calls:
                    success_labels = {
                        **(labels or {}),
                        "function": func.__name__,
                        "status": "success",
                    }
                    collector.counter(
                        f"{metric_name}_calls_success", 1.0, success_labels
                    )

                return result

            except Exception as e:
                # Record error
                if count_calls:
                    error_labels = {
                        **(labels or {}),
                        "function": func.__name__,
                        "status": "error",
                        "error_type": type(e).__name__,
                    }
                    collector.counter(f"{metric_name}_calls_error", 1.0, error_labels)
                raise

            finally:
                # Record timing
                if measure_time:
                    duration = time.time() - start_time
                    timing_labels = {**(labels or {}), "function": func.__name__}
                    collector.timing(f"{metric_name}_duration", duration, timing_labels)

        return wrapper  # type: ignore

    return decorator
