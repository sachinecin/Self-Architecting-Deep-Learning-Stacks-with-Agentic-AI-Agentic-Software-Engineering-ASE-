"""
Telemetry monitoring using NVML (pynvml).

Provides hardware metrics like HBM utilization, SM occupancy, and power draw,
with logic to trigger refactoring when memory pressure is sustained.
"""

import time
from typing import Dict, List, Optional
import warnings

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None  # type: ignore
    PYNVML_AVAILABLE = False
    warnings.warn("pynvml not available, telemetry will return mock data")


class TelemetryMonitor:
    """
    Hardware telemetry monitor using NVML.
    
    Tracks HBM utilization, SM occupancy, and power draw.
    Provides should_trigger_refactor() that returns True when HBM >= 85% 
    for a sustained window (default 5s).
    """

    def __init__(
        self,
        device_index: int = 0,
        window_size: float = 5.0,
        hbm_threshold: float = 85.0
    ):
        """
        Initialize the TelemetryMonitor.
        
        Args:
            device_index: GPU device index to monitor
            window_size: Time window in seconds for sustained high memory usage
            hbm_threshold: HBM utilization threshold percentage (0-100)
        """
        self.device_index = device_index
        self.window_size = window_size
        self.hbm_threshold = hbm_threshold
        self.handle = None
        self.history: List[Dict[str, float]] = []
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception as e:
                warnings.warn(f"Failed to initialize NVML: {e}")
                self.handle = None

    def sample_once(self) -> Dict[str, float]:
        """
        Sample hardware metrics once.
        
        Returns:
            Dictionary with keys: HBM_utilization, SM_occupancy, power_draw, timestamp
        """
        timestamp = time.time()
        
        if not PYNVML_AVAILABLE or self.handle is None:
            # Return mock data when NVML is not available
            return {
                "HBM_utilization": 50.0,
                "SM_occupancy": 60.0,
                "power_draw": 200.0,
                "timestamp": timestamp
            }
        
        try:
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            hbm_utilization = (mem_info.used / mem_info.total) * 100.0
            
            # Get utilization rates
            util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            sm_occupancy = float(util_rates.gpu)
            
            # Get power draw (in watts)
            power_draw = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            
            return {
                "HBM_utilization": hbm_utilization,
                "SM_occupancy": sm_occupancy,
                "power_draw": power_draw,
                "timestamp": timestamp
            }
        except Exception as e:
            warnings.warn(f"Failed to sample telemetry: {e}")
            return {
                "HBM_utilization": 0.0,
                "SM_occupancy": 0.0,
                "power_draw": 0.0,
                "timestamp": timestamp
            }

    def stream(self, duration: float = 1.0, interval: float = 0.1):
        """
        Stream telemetry samples for a given duration.
        
        Args:
            duration: How long to stream in seconds
            interval: Time between samples in seconds
            
        Yields:
            Telemetry samples as dictionaries
        """
        end_time = time.time() + duration
        while time.time() < end_time:
            sample = self.sample_once()
            self.history.append(sample)
            yield sample
            time.sleep(interval)

    def should_trigger_refactor(self) -> bool:
        """
        Determine if refactoring should be triggered based on sustained high HBM usage.
        
        Returns True when HBM >= threshold for the entire window_size duration.
        """
        if not self.history:
            return False
        
        current_time = time.time()
        window_start = current_time - self.window_size
        
        # Get samples within the window
        recent_samples = [
            s for s in self.history 
            if s["timestamp"] >= window_start
        ]
        
        if not recent_samples:
            return False
        
        # Check if all recent samples exceed threshold
        all_high = all(
            s["HBM_utilization"] >= self.hbm_threshold 
            for s in recent_samples
        )
        
        # Ensure we have enough history to cover the window
        time_span = current_time - recent_samples[0]["timestamp"]
        has_sufficient_history = time_span >= self.window_size * 0.8  # 80% of window
        
        return all_high and has_sufficient_history

    def clear_history(self):
        """Clear the telemetry history."""
        self.history.clear()

    def __del__(self):
        """Cleanup NVML resources."""
        if PYNVML_AVAILABLE and self.handle is not None:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
