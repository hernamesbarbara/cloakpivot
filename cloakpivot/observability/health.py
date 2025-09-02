"""Health monitoring and system status."""

from __future__ import annotations

import json
import os
import platform
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from http.server import BaseHTTPRequestHandler
from typing import Any

import psutil

from .config import HealthConfig, get_config
from .logging import get_logger


class HealthStatus(Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
        }


@dataclass
class SystemStatus:
    """Overall system status."""

    status: HealthStatus
    checks: list[HealthCheckResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    uptime_seconds: float = 0.0
    system_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "system_info": self.system_info,
            "checks": [check.to_dict() for check in self.checks],
        }


class HealthCheck(ABC):
    """Abstract base class for health checks."""

    def __init__(self, name: str, timeout: float = 30.0) -> None:
        self.name = name
        self.timeout = timeout

    @abstractmethod
    def check(self) -> HealthCheckResult:
        """Perform the health check."""
        pass


class SystemResourcesCheck(HealthCheck):
    """Check system resource usage."""

    def __init__(
        self,
        memory_threshold: float = 0.9,
        disk_threshold: float = 0.9,
        cpu_threshold: float = 0.95,
    ):
        super().__init__("system_resources")
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
        self.cpu_threshold = cpu_threshold

    def check(self) -> HealthCheckResult:
        """Check system resources."""
        start_time = time.time()

        try:
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0

            # Disk usage for current directory
            disk = psutil.disk_usage(os.getcwd())
            disk_usage = disk.used / disk.total

            # CPU usage (average over 1 second)
            cpu_usage = psutil.cpu_percent(interval=1.0) / 100.0

            details = {
                "memory": {
                    "usage_percent": memory_usage * 100,
                    "available_bytes": memory.available,
                    "total_bytes": memory.total,
                },
                "disk": {
                    "usage_percent": disk_usage * 100,
                    "free_bytes": disk.free,
                    "total_bytes": disk.total,
                },
                "cpu": {
                    "usage_percent": cpu_usage * 100,
                },
            }

            # Determine status
            if (
                memory_usage > self.memory_threshold
                or disk_usage > self.disk_threshold
                or cpu_usage > self.cpu_threshold
            ):
                status = HealthStatus.UNHEALTHY
                message = "System resources exceeded thresholds"
            elif (
                memory_usage > self.memory_threshold * 0.8
                or disk_usage > self.disk_threshold * 0.8
                or cpu_usage > self.cpu_threshold * 0.8
            ):
                status = HealthStatus.DEGRADED
                message = "System resources approaching limits"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources within normal limits"

            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details=details,
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check system resources: {e}",
                duration_ms=(time.time() - start_time) * 1000,
            )


class DependenciesCheck(HealthCheck):
    """Check external dependencies."""

    def __init__(self) -> None:
        super().__init__("dependencies")

    def check(self) -> HealthCheckResult:
        """Check dependencies."""
        start_time = time.time()

        try:
            dependencies = {}

            # Check Presidio
            try:
                import presidio_analyzer

                dependencies["presidio_analyzer"] = {
                    "available": True,
                    "version": getattr(presidio_analyzer, "__version__", "unknown"),
                }
            except ImportError:
                dependencies["presidio_analyzer"] = {
                    "available": False,
                    "error": "Not installed",
                }

            # Check DocPivot
            try:
                import docpivot

                dependencies["docpivot"] = {
                    "available": True,
                    "version": getattr(docpivot, "__version__", "unknown"),
                }
            except ImportError:
                dependencies["docpivot"] = {
                    "available": False,
                    "error": "Not installed",
                }

            # Check structlog
            try:
                import structlog

                dependencies["structlog"] = {
                    "available": True,
                    "version": getattr(structlog, "__version__", "unknown"),
                }
            except ImportError:
                dependencies["structlog"] = {
                    "available": False,
                    "error": "Not installed",
                }

            # Determine status
            unavailable = [
                name for name, info in dependencies.items() if not info["available"]
            ]

            if unavailable:
                if "presidio_analyzer" in unavailable or "docpivot" in unavailable:
                    status = HealthStatus.UNHEALTHY
                    message = (
                        f"Critical dependencies unavailable: {', '.join(unavailable)}"
                    )
                else:
                    status = HealthStatus.DEGRADED
                    message = (
                        f"Optional dependencies unavailable: {', '.join(unavailable)}"
                    )
            else:
                status = HealthStatus.HEALTHY
                message = "All dependencies available"

            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details={"dependencies": dependencies},
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check dependencies: {e}",
                duration_ms=(time.time() - start_time) * 1000,
            )


class ConfigurationCheck(HealthCheck):
    """Check configuration validity."""

    def __init__(self) -> None:
        super().__init__("configuration")

    def check(self) -> HealthCheckResult:
        """Check configuration."""
        start_time = time.time()

        try:
            config = get_config()

            issues = []

            # Check logging configuration
            if config.logging.level not in [
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
                "CRITICAL",
            ]:
                issues.append(f"Invalid log level: {config.logging.level}")

            # Check Prometheus configuration
            if config.exporters.prometheus.enabled:
                if not (1 <= config.exporters.prometheus.port <= 65535):
                    issues.append(
                        f"Invalid Prometheus port: {config.exporters.prometheus.port}"
                    )

            # Check StatsD configuration
            if config.exporters.statsd.enabled:
                if not (1 <= config.exporters.statsd.port <= 65535):
                    issues.append(
                        f"Invalid StatsD port: {config.exporters.statsd.port}"
                    )
                if config.exporters.statsd.protocol not in ["udp", "tcp"]:
                    issues.append(
                        f"Invalid StatsD protocol: {config.exporters.statsd.protocol}"
                    )

            # Determine status
            if issues:
                status = HealthStatus.UNHEALTHY
                message = f"Configuration issues: {'; '.join(issues)}"
            else:
                status = HealthStatus.HEALTHY
                message = "Configuration is valid"

            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details={"config_summary": config.to_dict(), "issues": issues},
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check configuration: {e}",
                duration_ms=(time.time() - start_time) * 1000,
            )


class HealthMonitor:
    """Health monitoring service."""

    def __init__(self, config: HealthConfig | None = None) -> None:
        self.config = config or get_config().health
        self.logger = get_logger(__name__)
        self.checks: list[HealthCheck] = []
        self.start_time = datetime.utcnow()
        self.last_status: SystemStatus | None = None
        self._lock = threading.RLock()

        # Add default checks
        self.add_check(SystemResourcesCheck())
        self.add_check(DependenciesCheck())
        self.add_check(ConfigurationCheck())

    def add_check(self, check: HealthCheck) -> None:
        """Add a health check."""
        with self._lock:
            self.checks.append(check)

    def remove_check(self, name: str) -> None:
        """Remove a health check by name."""
        with self._lock:
            self.checks = [check for check in self.checks if check.name != name]

    def get_status(self) -> SystemStatus:
        """Get current system status."""
        with self._lock:
            check_results = []

            for check in self.checks:
                try:
                    result = check.check()
                    check_results.append(result)
                except Exception as e:
                    self.logger.error(f"Health check {check.name} failed: {e}")
                    check_results.append(
                        HealthCheckResult(
                            name=check.name,
                            status=HealthStatus.UNHEALTHY,
                            message=f"Check failed: {e}",
                        )
                    )

            # Determine overall status
            if any(result.status == HealthStatus.UNHEALTHY for result in check_results):
                overall_status = HealthStatus.UNHEALTHY
            elif any(
                result.status == HealthStatus.DEGRADED for result in check_results
            ):
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.HEALTHY

            # Calculate uptime
            uptime = (datetime.utcnow() - self.start_time).total_seconds()

            # Get system info
            system_info = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "process_id": os.getpid(),
                "working_directory": os.getcwd(),
            }

            status = SystemStatus(
                status=overall_status,
                checks=check_results,
                uptime_seconds=uptime,
                system_info=system_info,
            )

            self.last_status = status
            return status


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health endpoints."""

    def __init__(
        self, health_monitor: HealthMonitor, *args: Any, **kwargs: Any
    ) -> None:
        self.health_monitor = health_monitor
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        """Handle GET request."""
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/health/ready":
            self._handle_readiness()
        elif self.path == "/health/live":
            self._handle_liveness()
        else:
            self.send_error(404)

    def _handle_health(self) -> None:
        """Handle full health check."""
        try:
            status = self.health_monitor.get_status()
            response_data = status.to_dict()

            # Set HTTP status based on health
            if status.status == HealthStatus.HEALTHY:
                http_status = 200
            elif status.status == HealthStatus.DEGRADED:
                http_status = 200  # Still operational
            else:
                http_status = 503  # Service unavailable

            self._send_json_response(http_status, response_data)

        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_readiness(self) -> None:
        """Handle readiness probe."""
        # Simple check - are we able to respond?
        try:
            status = self.health_monitor.get_status()
            if status.status != HealthStatus.UNHEALTHY:
                self._send_json_response(200, {"status": "ready"})
            else:
                self._send_json_response(503, {"status": "not ready"})
        except Exception:
            self._send_json_response(503, {"status": "not ready"})

    def _handle_liveness(self) -> None:
        """Handle liveness probe."""
        # Very basic liveness - just return 200 if we can respond
        self._send_json_response(200, {"status": "alive"})

    def _send_json_response(self, status_code: int, data: dict[str, Any]) -> None:
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:
        """Override to suppress default logging."""
        pass


# Global health monitor
_health_monitor: HealthMonitor | None = None
_monitor_lock = threading.Lock()


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor."""
    global _health_monitor
    with _monitor_lock:
        if _health_monitor is None:
            _health_monitor = HealthMonitor()
        return _health_monitor


def set_health_monitor(monitor: HealthMonitor) -> None:
    """Set the global health monitor."""
    global _health_monitor
    with _monitor_lock:
        _health_monitor = monitor
