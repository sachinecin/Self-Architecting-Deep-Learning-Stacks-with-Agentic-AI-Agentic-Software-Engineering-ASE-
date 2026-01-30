"""
Telemetry logger for KineticStack.

Provides structured logging of telemetry data for analysis and debugging.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class TelemetryLogger:
    """
    Logger for telemetry data with support for JSON and text formats.
    """

    def __init__(self, log_path: Optional[str] = None, format: str = "json"):
        """
        Initialize the telemetry logger.

        Args:
            log_path: Path to log file. If None, logs to stdout.
            format: Log format ('json' or 'text').
        """
        self.log_path = Path(log_path) if log_path else None
        self.format = format
        self._file_handle = None

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self.log_path, "a")

    def log(self, event_type: str, data: Dict[str, Any]):
        """
        Log a telemetry event.

        Args:
            event_type: Type of event (e.g., 'sample', 'optimization', 'error').
            data: Event data dictionary.
        """
        timestamp = time.time()
        entry = {"timestamp": timestamp, "event_type": event_type, "data": data}

        if self.format == "json":
            log_line = json.dumps(entry)
        else:
            log_line = f"[{timestamp}] {event_type}: {data}"

        if self._file_handle:
            self._file_handle.write(log_line + "\n")
            self._file_handle.flush()
        else:
            print(log_line)

    def close(self):
        """Close the log file if open."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
