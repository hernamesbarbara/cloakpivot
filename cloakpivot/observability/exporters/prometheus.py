"""Prometheus metrics exporter."""

from __future__ import annotations

import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from ..config import PrometheusConfig
from ..logging import get_logger
from ..metrics import MetricExporter, MetricStore


def sanitize_metric_name(name: str) -> str:
    """Sanitize metric name for Prometheus."""
    # Replace invalid characters with underscores
    name = re.sub(r"[^a-zA-Z0-9_:]", "_", name)
    # Ensure it starts with a letter or underscore
    if name and not re.match(r"^[a-zA-Z_]", name):
        name = f"_{name}"
    return name


def format_labels(labels: dict[str, str]) -> str:
    """Format labels for Prometheus format."""
    if not labels:
        return ""

    formatted = []
    for key, value in sorted(labels.items()):
        # Escape quotes and backslashes in label values
        escaped_value = value.replace("\\", "\\\\").replace('"', '\\"')
        formatted.append(f'{key}="{escaped_value}"')

    return "{" + ",".join(formatted) + "}"


class PrometheusMetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""

    def __init__(self, metric_store: MetricStore, namespace: str, *args: Any, **kwargs: Any):
        self.metric_store = metric_store
        self.namespace = namespace
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        """Handle GET request for metrics."""
        if self.path != "/metrics":
            self.send_error(404)
            return

        try:
            metrics_output = self._generate_prometheus_metrics()

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.end_headers()
            self.wfile.write(metrics_output.encode("utf-8"))

        except Exception as e:
            self.send_error(500, str(e))

    def _generate_prometheus_metrics(self) -> str:
        """Generate Prometheus format metrics."""
        lines = []

        # Add counters
        counters = self.metric_store.get_counters()
        counter_names = set()
        for key, (_value, _labels) in counters.items():
            base_name = key.split("{")[0] if "{" in key else key
            counter_names.add(base_name)

        for name in sorted(counter_names):
            full_name = f"{self.namespace}_{sanitize_metric_name(name)}"
            lines.append(f"# HELP {full_name} Counter metric")
            lines.append(f"# TYPE {full_name} counter")

            for key, (value, labels) in counters.items():
                base_name = key.split("{")[0] if "{" in key else key
                if base_name == name:
                    label_str = format_labels(labels)
                    lines.append(f"{full_name}{label_str} {value}")

        # Add gauges
        gauges = self.metric_store.get_gauges()
        gauge_names = set()
        for key, (_value, _labels) in gauges.items():
            base_name = key.split("{")[0] if "{" in key else key
            gauge_names.add(base_name)

        for name in sorted(gauge_names):
            full_name = f"{self.namespace}_{sanitize_metric_name(name)}"
            lines.append(f"# HELP {full_name} Gauge metric")
            lines.append(f"# TYPE {full_name} gauge")

            for key, (value, labels) in gauges.items():
                base_name = key.split("{")[0] if "{" in key else key
                if base_name == name:
                    label_str = format_labels(labels)
                    lines.append(f"{full_name}{label_str} {value}")

        # Add histograms (simplified as summaries)
        histograms = self.metric_store.get_histograms()
        histogram_names = set()
        for key, (_values, _labels) in histograms.items():
            base_name = key.split("{")[0] if "{" in key else key
            histogram_names.add(base_name)

        for name in sorted(histogram_names):
            full_name = f"{self.namespace}_{sanitize_metric_name(name)}"
            lines.append(f"# HELP {full_name} Histogram metric")
            lines.append(f"# TYPE {full_name} histogram")

            for key, (values, labels) in histograms.items():
                base_name = key.split("{")[0] if "{" in key else key
                if base_name == name and values:
                    label_str = format_labels(labels)

                    # Create histogram buckets
                    buckets = [0.001, 0.01, 0.1, 1, 5, 10, 30, 60, 300]
                    bucket_counts = dict.fromkeys(buckets, 0)

                    for value in values:
                        for bucket in buckets:
                            if value <= bucket:
                                bucket_counts[bucket] += 1

                    total_count = len(values)
                    for bucket in buckets:
                        bucket_labels = {**labels, "le": str(bucket)}
                        bucket_label_str = format_labels(bucket_labels)
                        lines.append(f"{full_name}_bucket{bucket_label_str} {bucket_counts[bucket]}")

                    # Add +Inf bucket
                    inf_labels = {**labels, "le": "+Inf"}
                    inf_label_str = format_labels(inf_labels)
                    lines.append(f"{full_name}_bucket{inf_label_str} {total_count}")

                    # Add count and sum
                    lines.append(f"{full_name}_count{label_str} {total_count}")
                    if values:
                        total_sum = sum(values)
                        lines.append(f"{full_name}_sum{label_str} {total_sum}")

        return "\n".join(lines) + "\n"

    def log_message(self, format: str, *args: Any) -> None:
        """Override to suppress default logging."""
        pass


class PrometheusExporter(MetricExporter):
    """Prometheus metrics exporter."""

    def __init__(self, config: PrometheusConfig, metric_store: MetricStore):
        self.config = config
        self.metric_store = metric_store
        self.server: HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.logger = get_logger(__name__)

        if self.config.enabled:
            self._start_server()

    def _start_server(self) -> None:
        """Start the HTTP server for metrics."""
        try:
            def handler_factory(*args: Any, **kwargs: Any) -> PrometheusMetricsHandler:
                return PrometheusMetricsHandler(
                    self.metric_store, self.config.namespace, *args, **kwargs
                )

            self.server = HTTPServer(("", self.config.port), handler_factory)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()

            self.logger.info(
                f"Prometheus metrics server started on port {self.config.port}{self.config.path}"
            )

        except Exception as e:
            self.logger.error(f"Failed to start Prometheus server: {e}")
            raise

    def export_metrics(self, store: MetricStore) -> None:
        """Export metrics (no-op for Prometheus as it's pull-based)."""
        # Prometheus is pull-based, so we don't need to push metrics
        pass

    def close(self) -> None:
        """Close the exporter and stop the server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()

        if self.thread:
            self.thread.join(timeout=5.0)

        self.logger.info("Prometheus exporter closed")
