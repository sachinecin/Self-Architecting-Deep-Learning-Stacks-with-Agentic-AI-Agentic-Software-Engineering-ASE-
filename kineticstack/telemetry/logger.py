"""
Telemetry logger for recording metrics.

Provides utilities for logging and persisting telemetry data.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class TelemetryLogger:
    """Logger for telemetry data."""

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the TelemetryLogger.
        
        Args:
            log_file: Optional path to log file for persisting data
        """
        self.log_file = Path(log_file) if log_file else None
        self.logs: List[Dict] = []

    def log(self, data: Dict):
        """
        Log a telemetry data point.
        
        Args:
            data: Dictionary of telemetry metrics
        """
        self.logs.append(data)
        
        # Persist to file if configured
        if self.log_file:
            self._persist()

    def _persist(self):
        """Persist logs to file."""
        if not self.log_file:
            return
        
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

    def get_logs(self) -> List[Dict]:
        """Get all logged data."""
        return self.logs.copy()

    def clear(self):
        """Clear all logs."""
        self.logs.clear()
        if self.log_file and self.log_file.exists():
            self.log_file.unlink()
