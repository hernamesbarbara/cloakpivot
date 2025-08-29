"""StatsD metrics exporter."""

from __future__ import annotations

import socket
from typing import Optional

from ..config import StatsDConfig
from ..logging import get_logger
from ..metrics import MetricExporter, MetricStore


class StatsDExporter(MetricExporter):
    """StatsD metrics exporter."""

    def __init__(self, config: StatsDConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.socket: Optional[socket.socket] = None
        
        if self.config.enabled:
            self._setup_socket()

    def _setup_socket(self) -> None:
        """Setup socket connection to StatsD."""
        try:
            if self.config.protocol.lower() == "tcp":
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.config.host, self.config.port))
            else:  # UDP
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            self.logger.info(
                f"StatsD exporter connected to {self.config.host}:{self.config.port} "
                f"via {self.config.protocol.upper()}"
            )
        except Exception as e:
            self.logger.error(f"Failed to setup StatsD socket: {e}")
            self.socket = None

    def export_metrics(self, store: MetricStore) -> None:
        """Export metrics to StatsD."""
        if not self.config.enabled or not self.socket:
            return

        try:
            self._export_counters(store)
            self._export_gauges(store)
            self._export_histograms(store)
        except Exception as e:
            self.logger.error(f"Failed to export metrics to StatsD: {e}")
            # Try to reconnect
            self._setup_socket()

    def _export_counters(self, store: MetricStore) -> None:
        """Export counter metrics."""
        counters = store.get_counters()
        for key, (value, labels) in counters.items():
            metric_name = self._format_metric_name(key, labels)
            message = f"{metric_name}:{value}|c"
            self._send_metric(message)

    def _export_gauges(self, store: MetricStore) -> None:
        """Export gauge metrics."""
        gauges = store.get_gauges()
        for key, (value, labels) in gauges.items():
            metric_name = self._format_metric_name(key, labels)
            message = f"{metric_name}:{value}|g"
            self._send_metric(message)

    def _export_histograms(self, store: MetricStore) -> None:
        """Export histogram metrics."""
        histograms = store.get_histograms()
        for key, (values, labels) in histograms.items():
            if not values:
                continue
                
            metric_name = self._format_metric_name(key, labels)
            
            # Send individual timing values
            for value in values:
                message = f"{metric_name}:{value}|ms"
                self._send_metric(message)

    def _format_metric_name(self, key: str, labels: dict[str, str]) -> str:
        """Format metric name with prefix and labels."""
        base_name = key.split("{")[0] if "{" in key else key
        metric_name = f"{self.config.prefix}.{base_name}"
        
        # Add labels as tags (if supported by StatsD implementation)
        if labels:
            tags = ",".join(f"{k}:{v}" for k, v in sorted(labels.items()))
            metric_name = f"{metric_name}|#{tags}"
        
        return metric_name

    def _send_metric(self, message: str) -> None:
        """Send metric message to StatsD."""
        if not self.socket:
            return

        try:
            data = message.encode("utf-8")
            if self.config.protocol.lower() == "tcp":
                self.socket.send(data + b"\n")
            else:  # UDP
                self.socket.sendto(data, (self.config.host, self.config.port))
        except Exception as e:
            self.logger.error(f"Failed to send metric to StatsD: {e}")

    def close(self) -> None:
        """Close the exporter and socket connection."""
        if self.socket:
            try:
                self.socket.close()
            except Exception as e:
                self.logger.error(f"Error closing StatsD socket: {e}")
            finally:
                self.socket = None
        
        self.logger.info("StatsD exporter closed")