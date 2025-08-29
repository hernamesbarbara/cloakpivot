"""Structured logging with tracing support."""

from __future__ import annotations

import contextvars
import logging
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Optional

try:
    import structlog
    from structlog import processors
    from structlog.typing import FilteringBoundLogger

    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False
    structlog = None  # type: ignore
    FilteringBoundLogger = None  # type: ignore

from .config import get_config

# Context variable for correlation IDs
correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


def add_correlation_id(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add correlation ID to log events."""
    corr_id = correlation_id.get()
    if corr_id:
        event_dict["correlation_id"] = corr_id
    return event_dict


def add_timestamp(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add timestamp to log events."""
    event_dict["timestamp"] = datetime.utcnow().isoformat()
    return event_dict


def add_level(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add log level to event dict."""
    event_dict["level"] = method_name.upper()
    return event_dict


def configure_logging(config: Optional[Any] = None) -> None:
    """Configure structured logging."""
    if config is None:
        config = get_config().logging

    if not HAS_STRUCTLOG:
        # Fallback to standard logging with enhanced formatting
        _configure_standard_logging(config)
        return

    # Configure structlog
    processors_list = [
        structlog.stdlib.filter_by_level,
        add_timestamp,
        add_level,
        structlog.stdlib.add_logger_name,
        add_correlation_id,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if config.format == "json":
        processors_list.append(structlog.processors.JSONRenderer())
    else:
        processors_list.extend(
            [
                structlog.processors.CallsiteParameterAdder(
                    parameters={structlog.processors.CallsiteParameter.FUNC_NAME}
                ),
                structlog.dev.ConsoleRenderer(colors=True),
            ]
        )

    structlog.configure(
        processors=processors_list,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout if config.output == "stdout" else None,
        filename=config.file_path if config.output == "file" else None,
        level=getattr(logging, config.level.upper(), logging.INFO),
    )


def _configure_standard_logging(config: Any) -> None:
    """Configure standard logging when structlog is not available."""
    if config.format == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))


class JsonFormatter(logging.Formatter):
    """JSON formatter for standard logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        import json

        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }

        # Add correlation ID if available
        corr_id = correlation_id.get()
        if corr_id:
            log_data["correlation_id"] = corr_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
            }:
                log_data[key] = value

        return json.dumps(log_data)


def get_logger(name: str) -> Any:
    """Get a logger instance."""
    if HAS_STRUCTLOG:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


@contextmanager
def correlation_context(corr_id: Optional[str] = None) -> Generator[str, None, None]:
    """Context manager for correlation ID."""
    if corr_id is None:
        corr_id = str(uuid.uuid4())

    token = correlation_id.set(corr_id)
    try:
        yield corr_id
    finally:
        correlation_id.reset(token)


@contextmanager
def trace_operation(
    operation: str, **kwargs: Any
) -> Generator[TraceContext, None, None]:
    """Context manager for tracing operations."""
    config = get_config()
    if not config.logging.enable_tracing:
        yield TraceContext(operation, **kwargs)
        return

    logger = get_logger(__name__)
    start_time = datetime.utcnow()
    trace_id = str(uuid.uuid4())

    # Start trace
    logger.info(
        "Operation started",
        operation=operation,
        trace_id=trace_id,
        start_time=start_time.isoformat(),
        **kwargs,
    )

    context = TraceContext(operation, trace_id=trace_id, logger=logger, **kwargs)

    try:
        yield context
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            "Operation completed",
            operation=operation,
            trace_id=trace_id,
            duration_seconds=duration,
            status="success",
        )
    except Exception as e:
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.error(
            "Operation failed",
            operation=operation,
            trace_id=trace_id,
            duration_seconds=duration,
            status="error",
            error=str(e),
            error_type=type(e).__name__,
        )
        raise


class TraceContext:
    """Context for traced operations."""

    def __init__(
        self,
        operation: str,
        trace_id: Optional[str] = None,
        logger: Optional[Any] = None,
        **kwargs: Any,
    ):
        self.operation = operation
        self.trace_id = trace_id or str(uuid.uuid4())
        self.logger = logger or get_logger(__name__)
        self.attributes: dict[str, Any] = kwargs

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a trace attribute."""
        self.attributes[key] = value

    def log_info(self, message: str, **kwargs: Any) -> None:
        """Log info message with trace context."""
        self.logger.info(
            message,
            operation=self.operation,
            trace_id=self.trace_id,
            **self.attributes,
            **kwargs,
        )

    def log_error(self, message: str, error: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log error message with trace context."""
        log_data = {
            "operation": self.operation,
            "trace_id": self.trace_id,
            **self.attributes,
            **kwargs,
        }
        
        if error:
            log_data["error"] = str(error)
            log_data["error_type"] = type(error).__name__
            
        self.logger.error(message, **log_data)

    def log_warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with trace context."""
        self.logger.warning(
            message,
            operation=self.operation,
            trace_id=self.trace_id,
            **self.attributes,
            **kwargs,
        )