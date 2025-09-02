"""Configuration for observability and monitoring."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PrometheusConfig(BaseModel):
    """Configuration for Prometheus metrics export."""

    enabled: bool = Field(default=True, description="Enable Prometheus export")
    port: int = Field(default=9090, description="Port for metrics endpoint")
    path: str = Field(default="/metrics", description="Path for metrics endpoint")
    namespace: str = Field(default="cloakpivot", description="Metric namespace")


class StatsDConfig(BaseModel):
    """Configuration for StatsD metrics export."""

    enabled: bool = Field(default=False, description="Enable StatsD export")
    host: str = Field(default="localhost", description="StatsD host")
    port: int = Field(default=8125, description="StatsD port")
    protocol: str = Field(default="udp", description="Protocol (udp or tcp)")
    prefix: str = Field(default="cloakpivot", description="Metric prefix")


class WebhookConfig(BaseModel):
    """Configuration for webhook notifications."""

    enabled: bool = Field(default=False, description="Enable webhook notifications")
    url: str = Field(description="Webhook URL")
    severity_levels: list[str] = Field(
        default=["error", "critical"], description="Severity levels to send"
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")
    retry_count: int = Field(default=3, description="Number of retries")


class LoggingConfig(BaseModel):
    """Configuration for structured logging."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format (json or text)")
    enable_tracing: bool = Field(default=True, description="Enable distributed tracing")
    enable_correlation: bool = Field(default=True, description="Enable correlation IDs")
    output: str = Field(default="stdout", description="Log output (stdout or file)")
    file_path: str | None = Field(default=None, description="Log file path")


class HealthConfig(BaseModel):
    """Configuration for health monitoring."""

    enabled: bool = Field(default=True, description="Enable health checks")
    endpoint: str = Field(default="/health", description="Health check endpoint")
    check_interval: int = Field(
        default=60, description="Health check interval in seconds"
    )


class MetricsConfig(BaseModel):
    """Configuration for metrics collection."""

    enabled: bool = Field(default=True, description="Enable metrics collection")
    collection_interval: int = Field(
        default=10, description="Collection interval in seconds"
    )
    buffer_size: int = Field(default=1000, description="Metrics buffer size")


class ExportersConfig(BaseModel):
    """Configuration for metric exporters."""

    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    statsd: StatsDConfig = Field(default_factory=StatsDConfig)
    webhooks: list[WebhookConfig] = Field(default_factory=list)


class ObservabilityConfig(BaseModel):
    """Main configuration for observability and monitoring."""

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    exporters: ExportersConfig = Field(default_factory=ExportersConfig)

    @classmethod
    def from_file(cls, config_path: Path | str) -> ObservabilityConfig:
        """Load configuration from a YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        observability_data = config_data.get("observability", {})
        return cls(**observability_data)

    @classmethod
    def from_env(cls) -> ObservabilityConfig:
        """Load configuration from environment variables."""
        config = cls()

        # Logging configuration
        config.logging.level = os.getenv("CLOAKPIVOT_LOG_LEVEL", config.logging.level)
        config.logging.format = os.getenv(
            "CLOAKPIVOT_LOG_FORMAT", config.logging.format
        )
        config.logging.enable_tracing = (
            os.getenv("CLOAKPIVOT_LOG_TRACING", "true").lower() == "true"
        )

        # Metrics configuration
        config.metrics.enabled = (
            os.getenv("CLOAKPIVOT_METRICS_ENABLED", "true").lower() == "true"
        )

        # Prometheus configuration
        config.exporters.prometheus.enabled = (
            os.getenv("CLOAKPIVOT_PROMETHEUS_ENABLED", "true").lower() == "true"
        )
        if port := os.getenv("CLOAKPIVOT_PROMETHEUS_PORT"):
            config.exporters.prometheus.port = int(port)

        # StatsD configuration
        config.exporters.statsd.enabled = (
            os.getenv("CLOAKPIVOT_STATSD_ENABLED", "false").lower() == "true"
        )
        config.exporters.statsd.host = os.getenv(
            "CLOAKPIVOT_STATSD_HOST", config.exporters.statsd.host
        )
        if port := os.getenv("CLOAKPIVOT_STATSD_PORT"):
            config.exporters.statsd.port = int(port)

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()


# Global configuration instance
_config: ObservabilityConfig | None = None


def get_config() -> ObservabilityConfig:
    """Get the global observability configuration."""
    global _config
    if _config is None:
        _config = ObservabilityConfig.from_env()
    return _config


def set_config(config: ObservabilityConfig) -> None:
    """Set the global observability configuration."""
    global _config
    _config = config


def load_config(config_path: Path | str | None = None) -> ObservabilityConfig:
    """Load and set the global configuration."""
    if config_path:
        config = ObservabilityConfig.from_file(config_path)
    else:
        config = ObservabilityConfig.from_env()
    set_config(config)
    return config
