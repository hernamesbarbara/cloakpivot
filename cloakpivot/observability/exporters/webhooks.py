"""Webhook notifications exporter."""

from __future__ import annotations

import json
import time
from typing import Any, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from ..config import WebhookConfig
from ..logging import get_logger
from ..metrics import MetricExporter, MetricStore, MetricEvent


class WebhookExporter(MetricExporter):
    """Webhook notifications exporter."""

    def __init__(self, config: WebhookConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self._last_export_time = 0.0
        self._last_events_count = 0

    def export_metrics(self, store: MetricStore) -> None:
        """Export metrics via webhook notifications."""
        if not self.config.enabled:
            return

        # Get recent events that match our severity levels
        recent_events = store.get_recent_events()
        
        # Filter events since last export
        new_events = []
        for event in recent_events:
            event_time = event.timestamp.timestamp()
            if event_time > self._last_export_time:
                new_events.append(event)

        if not new_events:
            return

        # Create payload
        payload = self._create_payload(new_events, store)
        
        # Send webhook
        if payload:
            self._send_webhook(payload)
            self._last_export_time = time.time()
            self._last_events_count = len(recent_events)

    def _create_payload(
        self, events: list[MetricEvent], store: MetricStore
    ) -> Optional[dict[str, Any]]:
        """Create webhook payload from metrics."""
        if not events:
            return None

        # Calculate summary statistics
        counters = store.get_counters()
        gauges = store.get_gauges()
        
        error_events = [
            e for e in events 
            if any(
                severity in e.labels.get("status", "").lower() 
                for severity in ["error", "critical", "failed"]
            )
        ]
        
        # Only send if we have alerts or significant events
        should_alert = (
            len(error_events) > 0 or
            any("error" in severity for severity in self.config.severity_levels) and
            any("error" in str(e.labels).lower() for e in events)
        )
        
        if not should_alert:
            return None

        payload = {
            "timestamp": time.time(),
            "service": "cloakpivot",
            "alert_type": "metrics",
            "severity": self._calculate_severity(events),
            "summary": {
                "total_events": len(events),
                "error_events": len(error_events),
                "time_range": {
                    "start": min(e.timestamp.isoformat() for e in events),
                    "end": max(e.timestamp.isoformat() for e in events),
                },
            },
            "metrics": {
                "counters_count": len(counters),
                "gauges_count": len(gauges),
            },
            "details": {
                "events": [
                    {
                        "name": event.name,
                        "value": event.value,
                        "type": event.metric_type.value,
                        "labels": event.labels,
                        "timestamp": event.timestamp.isoformat(),
                    }
                    for event in error_events[:10]  # Limit to first 10 error events
                ],
            },
        }

        # Add specific error information
        if error_events:
            error_types = {}
            for event in error_events:
                error_type = event.labels.get("error_type", "unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            payload["details"]["error_summary"] = error_types

        return payload

    def _calculate_severity(self, events: list[MetricEvent]) -> str:
        """Calculate alert severity based on events."""
        error_count = sum(
            1 for e in events 
            if any(
                severity in e.labels.get("status", "").lower() 
                for severity in ["error", "critical", "failed"]
            )
        )
        
        if error_count > 10:
            return "critical"
        elif error_count > 5:
            return "error"
        elif error_count > 0:
            return "warning"
        else:
            return "info"

    def _send_webhook(self, payload: dict[str, Any]) -> None:
        """Send webhook notification."""
        for attempt in range(self.config.retry_count + 1):
            try:
                data = json.dumps(payload).encode("utf-8")
                
                request = Request(
                    self.config.url,
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "CloakPivot-Observer/1.0",
                    },
                )
                
                with urlopen(request, timeout=self.config.timeout) as response:
                    if response.status == 200:
                        self.logger.info(f"Webhook notification sent successfully")
                        return
                    else:
                        self.logger.warning(
                            f"Webhook returned status {response.status}"
                        )
                        
            except HTTPError as e:
                self.logger.error(f"HTTP error sending webhook: {e.code} {e.reason}")
            except URLError as e:
                self.logger.error(f"URL error sending webhook: {e.reason}")
            except Exception as e:
                self.logger.error(f"Unexpected error sending webhook: {e}")

            if attempt < self.config.retry_count:
                wait_time = 2 ** attempt  # Exponential backoff
                self.logger.info(f"Retrying webhook in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                self.logger.error("Failed to send webhook after all retries")

    def close(self) -> None:
        """Close the exporter."""
        self.logger.info("Webhook exporter closed")