"""
Telemetry collection for GPU metrics using NVML.

This module provides real-time collection of GPU metrics including memory utilization,
temperature, and power consumption. Supports mocking for CI/testing environments.
"""

from typing import Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)


class TelemetryCollector:
    """
    Collects real-time telemetry from GPU hardware via NVML.

    Monitors HBM utilization, temperature, and other metrics to drive
    optimization decisions. Can be mocked for testing on non-GPU systems.
    """

    def __init__(self, mock_mode: bool = False) -> None:
        """
        Initialize the telemetry collector.

        Args:
            mock_mode: If True, use mock data instead of real NVML calls
        """
        self.mock_mode = mock_mode
        self._initialized = False
        self._start_time = time.time()

        if not mock_mode:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml = pynvml
                self._initialized = True
                logger.info("NVML initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}. Using mock mode.")
                self.mock_mode = True
        else:
            logger.info("TelemetryCollector running in mock mode")

    def collect(self) -> Dict[str, Any]:
        """
        Collect current telemetry metrics.

        Returns:
            Dictionary containing GPU metrics including:
                - hbm_used_gb: HBM memory used in GB
                - hbm_total_gb: Total HBM memory in GB
                - hbm_utilization: Memory utilization ratio (0-1)
                - temperature_c: GPU temperature in Celsius
                - power_usage_w: Current power usage in Watts
        """
        if self.mock_mode:
            return self._collect_mock()

        try:
            handle = self._nvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = self._nvml.nvmlDeviceGetMemoryInfo(handle)
            temperature = self._nvml.nvmlDeviceGetTemperature(
                handle, self._nvml.NVML_TEMPERATURE_GPU
            )
            power = self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W

            return {
                "hbm_used_gb": mem_info.used / (1024**3),
                "hbm_total_gb": mem_info.total / (1024**3),
                "hbm_utilization": mem_info.used / mem_info.total,
                "temperature_c": temperature,
                "power_usage_w": power,
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"Error collecting telemetry: {e}")
            return self._collect_mock()

    def _collect_mock(self) -> Dict[str, Any]:
        """Generate mock telemetry data for testing."""
        # Simulate increasing memory pressure over time
        elapsed = time.time() - self._start_time
        base_utilization = 0.7 + (elapsed % 20) * 0.02  # Ramps from 0.7 to 1.0 over 20s

        return {
            "hbm_used_gb": base_utilization * 40.0,
            "hbm_total_gb": 40.0,
            "hbm_utilization": base_utilization,
            "temperature_c": 65.0 + (elapsed % 10) * 2.0,
            "power_usage_w": 250.0 + (elapsed % 5) * 10.0,
            "timestamp": time.time(),
            "mock": True,
        }

    def shutdown(self) -> None:
        """Clean up NVML resources."""
        if self._initialized and not self.mock_mode:
            try:
                self._nvml.nvmlShutdown()
                logger.info("NVML shutdown successfully")
            except Exception as e:
                logger.warning(f"Error during NVML shutdown: {e}")
