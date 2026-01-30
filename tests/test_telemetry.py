"""Tests for telemetry monitoring with mocked pynvml."""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
import time

from kineticstack.core.telemetry import TelemetryMonitor


class MockMemoryInfo:
    """Mock for pynvml memory info."""
    def __init__(self, used=8589934592, total=10737418240):  # 8GB / 10GB = 80%
        self.used = used
        self.total = total


class MockUtilizationRates:
    """Mock for pynvml utilization rates."""
    def __init__(self, gpu=75):
        self.gpu = gpu


@pytest.fixture
def mock_pynvml():
    """Fixture to mock pynvml."""
    # Create mock module
    mock_nvml = MagicMock()
    
    # Setup mock handle
    mock_handle = Mock()
    mock_nvml.nvmlInit.return_value = None
    mock_nvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
    mock_nvml.nvmlDeviceGetMemoryInfo.return_value = MockMemoryInfo()
    mock_nvml.nvmlDeviceGetUtilizationRates.return_value = MockUtilizationRates()
    mock_nvml.nvmlDeviceGetPowerUsage.return_value = 250000  # 250W
    mock_nvml.nvmlShutdown.return_value = None
    
    # Patch the module
    with patch.dict('sys.modules', {'pynvml': mock_nvml}), \
         patch('kineticstack.core.telemetry.PYNVML_AVAILABLE', True), \
         patch('kineticstack.core.telemetry.pynvml', mock_nvml):
        yield mock_nvml


def test_telemetry_monitor_init(mock_pynvml):
    """Test TelemetryMonitor initialization."""
    monitor = TelemetryMonitor(device_index=0)
    assert monitor.device_index == 0
    assert monitor.window_size == 5.0
    assert monitor.hbm_threshold == 85.0
    mock_pynvml.nvmlInit.assert_called_once()


def test_sample_once(mock_pynvml):
    """Test sampling telemetry once."""
    monitor = TelemetryMonitor()
    sample = monitor.sample_once()
    
    assert "HBM_utilization" in sample
    assert "SM_occupancy" in sample
    assert "power_draw" in sample
    assert "timestamp" in sample
    
    # Check values from mock
    assert sample["HBM_utilization"] == pytest.approx(80.0, rel=0.1)
    assert sample["SM_occupancy"] == 75.0
    assert sample["power_draw"] == 250.0


def test_sample_once_without_nvml():
    """Test sampling when NVML is not available."""
    with patch('kineticstack.core.telemetry.PYNVML_AVAILABLE', False):
        monitor = TelemetryMonitor()
        sample = monitor.sample_once()
        
        # Should return mock data
        assert sample["HBM_utilization"] == 50.0
        assert sample["SM_occupancy"] == 60.0
        assert sample["power_draw"] == 200.0


def test_should_trigger_refactor_true(mock_pynvml):
    """Test refactor trigger when memory is consistently high."""
    monitor = TelemetryMonitor(window_size=1.0, hbm_threshold=75.0)
    
    # Set high memory usage
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = MockMemoryInfo(
        used=9000000000, total=10000000000  # 90%
    )
    
    # Collect samples over the window using stream() which populates history
    list(monitor.stream(duration=1.2, interval=0.1))
    
    assert monitor.should_trigger_refactor() is True


def test_should_trigger_refactor_false(mock_pynvml):
    """Test refactor trigger when memory is not consistently high."""
    monitor = TelemetryMonitor(window_size=1.0, hbm_threshold=85.0)
    
    # Set moderate memory usage (80% < 85% threshold)
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = MockMemoryInfo(
        used=8000000000, total=10000000000  # 80%
    )
    
    # Collect samples using stream()
    list(monitor.stream(duration=1.2, interval=0.1))
    
    assert monitor.should_trigger_refactor() is False


def test_stream():
    """Test streaming telemetry."""
    with patch('kineticstack.core.telemetry.PYNVML_AVAILABLE', False):
        monitor = TelemetryMonitor()
        
        samples = list(monitor.stream(duration=0.3, interval=0.1))
        
        # Should have approximately 3 samples
        assert len(samples) >= 2
        assert all("HBM_utilization" in s for s in samples)


def test_clear_history(mock_pynvml):
    """Test clearing telemetry history."""
    monitor = TelemetryMonitor()
    
    # Collect some samples using stream() which populates history
    list(monitor.stream(duration=0.5, interval=0.1))
    
    assert len(monitor.history) >= 4  # Should have collected several samples
    
    monitor.clear_history()
    assert len(monitor.history) == 0
