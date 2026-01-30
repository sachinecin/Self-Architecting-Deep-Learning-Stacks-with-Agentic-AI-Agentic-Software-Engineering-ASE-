"""
Structured telemetry logging for KineticStack metrics.

Provides structured logging of telemetry data for analysis and debugging.
Can output to console, files, or telemetry backends.
"""

from typing import Dict, Any, Optional, TextIO
import json
import logging
import sys

logger = logging.getLogger(__name__)


class TelemetryLogger:
    """
    Structured logger for telemetry data.

    Logs telemetry metrics in a structured format for post-hoc analysis.
    Supports JSON output to console or files.
    """

    def __init__(
        self,
        output: Optional[TextIO] = None,
        json_format: bool = True,
    ) -> None:
        """
        Initialize the telemetry logger.

        Args:
            output: Output stream for telemetry (default: sys.stdout)
            json_format: If True, output as JSON lines; if False, use human-readable format
        """
        self.output = output or sys.stdout
        self.json_format = json_format
        logger.info(f"TelemetryLogger initialized (json_format={json_format})")

    def log(self, telemetry: Dict[str, Any], prefix: str = "TELEMETRY") -> None:
        """
        Log telemetry data.

        Args:
            telemetry: Telemetry dictionary to log
            prefix: Prefix for log messages (default: "TELEMETRY")
        """
        if self.json_format:
            self._log_json(telemetry, prefix)
        else:
            self._log_human(telemetry, prefix)

    def _log_json(self, telemetry: Dict[str, Any], prefix: str) -> None:
        """Log telemetry as JSON line."""
        log_entry = {"type": prefix, "data": telemetry}
        try:
            self.output.write(json.dumps(log_entry) + "\n")
            self.output.flush()
        except Exception as e:
            logger.error(f"Failed to write JSON telemetry: {e}")

    def _log_human(self, telemetry: Dict[str, Any], prefix: str) -> None:
        """Log telemetry in human-readable format."""
        try:
            parts = [f"{prefix}:"]
            for key, value in telemetry.items():
                if isinstance(value, float):
                    if "utilization" in key:
                        parts.append(f"{key}={value:.2%}")
                    else:
                        parts.append(f"{key}={value:.2f}")
                else:
                    parts.append(f"{key}={value}")

            self.output.write(" ".join(parts) + "\n")
            self.output.flush()
        except Exception as e:
            logger.error(f"Failed to write human-readable telemetry: {e}")

    def log_event(
        self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a structured event.

        Args:
            event_type: Type of event (e.g., "sculptor_activation", "invariant_check")
            message: Human-readable message
            data: Optional additional data
        """
        event = {
            "event_type": event_type,
            "message": message,
        }
        if data:
            event["data"] = data

        if self.json_format:
            try:
                self.output.write(json.dumps(event) + "\n")
                self.output.flush()
            except Exception as e:
                logger.error(f"Failed to write event: {e}")
        else:
            try:
                parts = [f"EVENT[{event_type}]:", message]
                if data:
                    parts.append(f"data={data}")
                self.output.write(" ".join(parts) + "\n")
                self.output.flush()
            except Exception as e:
                logger.error(f"Failed to write event: {e}")

    def close(self) -> None:
        """Close the telemetry logger and flush output."""
        try:
            self.output.flush()
            if self.output != sys.stdout and self.output != sys.stderr:
                self.output.close()
        except Exception as e:
            logger.warning(f"Error closing telemetry logger: {e}")
