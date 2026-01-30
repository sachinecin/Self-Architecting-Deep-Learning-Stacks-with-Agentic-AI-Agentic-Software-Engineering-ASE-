"""
Tests for telemetry monitoring.

Uses mocked pynvml to test telemetry functionality without requiring GPU hardware.
"""

import pytest
from unittest.mock import MagicMock, patch


class MockMemoryInfo:
    """Mock NVML memory info."""

    def __init__(self, used, total):
        self.used = used
        self.total = total


class MockUtilization:
    """Mock NVML utilization."""

    def __init__(self, gpu):
        self.gpu = gpu


@pytest.fixture
def mock_nvml():
    """Fixture that mocks pynvml."""
    with patch("kineticstack.core.telemetry.NVML_AVAILABLE", True):
        with patch("kineticstack.core.telemetry.pynvml") as mock:
            # Setup mock returns
            mock.nvmlInit = MagicMock()
            mock.nvmlDeviceGetHandleByIndex = MagicMock(return_value="mock_handle")
            mock.nvmlDeviceGetMemoryInfo = MagicMock(
                return_value=MockMemoryInfo(used=8 * 1024**3, total=16 * 1024**3)
            )
            mock.nvmlDeviceGetUtilizationRates = MagicMock(return_value=MockUtilization(gpu=75))
            mock.nvmlShutdown = MagicMock()
            yield mock


def test_telemetry_monitor_init(mock_nvml):
    """Test TelemetryMonitor initialization."""
    from kineticstack.core.telemetry import TelemetryMonitor

    monitor = TelemetryMonitor(device_index=0)
    assert monitor.device_index == 0
    assert monitor.memory_threshold == 0.85
    assert monitor.utilization_threshold == 0.90
    mock_nvml.nvmlInit.assert_called_once()


def test_telemetry_sample_once(mock_nvml):
    """Test single telemetry sample."""
    from kineticstack.core.telemetry import TelemetryMonitor

    monitor = TelemetryMonitor(device_index=0)
    sample = monitor.sample_once()

    assert "memory_used_mb" in sample
    assert "memory_total_mb" in sample
    assert "memory_utilization" in sample
    assert "gpu_utilization" in sample

    # Check values match our mock (8GB used / 16GB total = 0.5)
    assert sample["memory_utilization"] == pytest.approx(0.5, rel=0.01)
    assert sample["gpu_utilization"] == pytest.approx(0.75, rel=0.01)


def test_telemetry_should_trigger_refactor(mock_nvml):
    """Test refactor trigger logic."""
    from kineticstack.core.telemetry import TelemetryMonitor

    # Test with low thresholds (should trigger)
    monitor = TelemetryMonitor(device_index=0, memory_threshold=0.3, utilization_threshold=0.3)
    assert monitor.should_trigger_refactor()

    # Test with high thresholds (should not trigger)
    monitor = TelemetryMonitor(device_index=0, memory_threshold=0.95, utilization_threshold=0.95)
    assert not monitor.should_trigger_refactor()


def test_telemetry_stream(mock_nvml):
    """Test telemetry streaming."""
    from kineticstack.core.telemetry import TelemetryMonitor

    monitor = TelemetryMonitor(device_index=0)
    samples = list(monitor.stream(interval_seconds=0.01, duration_seconds=0.05))

    # Should get at least a few samples
    assert len(samples) >= 2
    for sample in samples:
        assert "memory_utilization" in sample
        assert "gpu_utilization" in sample


def test_telemetry_without_nvml():
    """Test TelemetryMonitor when pynvml is not available."""
    with patch("kineticstack.core.telemetry.NVML_AVAILABLE", False):
        from kineticstack.core.telemetry import TelemetryMonitor

        monitor = TelemetryMonitor(device_index=0)
        sample = monitor.sample_once()

        # Should return zeros when NVML is unavailable
        assert sample["memory_utilization"] == 0.0
        assert sample["gpu_utilization"] == 0.0
        assert not monitor.should_trigger_refactor()


def test_telemetry_shutdown(mock_nvml):
    """Test telemetry shutdown."""
    from kineticstack.core.telemetry import TelemetryMonitor

    monitor = TelemetryMonitor(device_index=0)
    monitor.shutdown()
    mock_nvml.nvmlShutdown.assert_called_once()
