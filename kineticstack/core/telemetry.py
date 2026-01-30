"""
Telemetry monitoring for KineticStack.

Provides real-time GPU telemetry monitoring using pynvml to track memory usage,
GPU utilization, and determine when refactoring should be triggered.
"""

import time
from typing import Dict, Optional

try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class TelemetryMonitor:
    """
    Monitor GPU telemetry using NVML and determine when to trigger refactoring.

    This class provides methods to sample GPU metrics and apply heuristics to
    determine when the agentic sculptor should be invoked for optimization.
    """

    def __init__(
        self,
        device_index: int = 0,
        memory_threshold: float = 0.85,
        utilization_threshold: float = 0.90,
    ):
        """
        Initialize the telemetry monitor.

        Args:
            device_index: GPU device index to monitor.
            memory_threshold: Memory usage threshold (0-1) for triggering refactor.
            utilization_threshold: GPU utilization threshold (0-1) for triggering refactor.
        """
        self.device_index = device_index
        self.memory_threshold = memory_threshold
        self.utilization_threshold = utilization_threshold
        self._handle = None
        self._initialized = False

        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self._initialized = True
            except Exception as e:
                print(f"Warning: Failed to initialize NVML: {e}")
                self._initialized = False
        else:
            print("Warning: pynvml not available, telemetry disabled")

    def sample_once(self) -> Dict[str, float]:
        """
        Sample GPU metrics once.

        Returns:
            Dictionary with keys: 'memory_used_mb', 'memory_total_mb',
            'memory_utilization', 'gpu_utilization'.
        """
        if not self._initialized:
            return {
                "memory_used_mb": 0.0,
                "memory_total_mb": 0.0,
                "memory_utilization": 0.0,
                "gpu_utilization": 0.0,
            }

        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self._handle)

            return {
                "memory_used_mb": memory_info.used / (1024**2),
                "memory_total_mb": memory_info.total / (1024**2),
                "memory_utilization": memory_info.used / memory_info.total,
                "gpu_utilization": utilization.gpu / 100.0,
            }
        except Exception as e:
            print(f"Warning: Failed to sample telemetry: {e}")
            return {
                "memory_used_mb": 0.0,
                "memory_total_mb": 0.0,
                "memory_utilization": 0.0,
                "gpu_utilization": 0.0,
            }

    def stream(self, interval_seconds: float = 1.0, duration_seconds: Optional[float] = None):
        """
        Stream telemetry samples continuously.

        Args:
            interval_seconds: Time between samples.
            duration_seconds: Total duration to stream. If None, streams indefinitely.

        Yields:
            Telemetry sample dictionaries.
        """
        start_time = time.time()
        while True:
            if duration_seconds is not None:
                if time.time() - start_time >= duration_seconds:
                    break

            yield self.sample_once()
            time.sleep(interval_seconds)

    def should_trigger_refactor(self) -> bool:
        """
        Determine if refactoring should be triggered based on current telemetry.

        Returns:
            True if either memory or GPU utilization exceeds thresholds.
        """
        sample = self.sample_once()
        return (
            sample["memory_utilization"] >= self.memory_threshold
            or sample["gpu_utilization"] >= self.utilization_threshold
        )

    def shutdown(self):
        """Shutdown NVML and release resources."""
        if self._initialized and NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
                self._initialized = False
            except Exception:
                pass

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()
