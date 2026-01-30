"""Tests for telemetry collection with mocked NVML."""

import pytest
from kineticstack.core.telemetry import TelemetryCollector
from kineticstack.telemetry.logger import TelemetryLogger
import io


def test_telemetry_collector_mock_mode() -> None:
    """Test that TelemetryCollector works in mock mode for CI."""
    collector = TelemetryCollector(mock_mode=True)
    telemetry = collector.collect()

    # Verify all expected fields are present
    assert "hbm_used_gb" in telemetry
    assert "hbm_total_gb" in telemetry
    assert "hbm_utilization" in telemetry
    assert "temperature_c" in telemetry
    assert "power_usage_w" in telemetry
    assert "timestamp" in telemetry
    assert telemetry.get("mock") is True

    # Verify values are in reasonable ranges
    assert 0 <= telemetry["hbm_utilization"] <= 1.0
    assert telemetry["hbm_total_gb"] > 0
    assert telemetry["temperature_c"] > 0
    assert telemetry["power_usage_w"] > 0

    collector.shutdown()


def test_telemetry_collector_multiple_samples() -> None:
    """Test collecting multiple telemetry samples."""
    collector = TelemetryCollector(mock_mode=True)

    samples = [collector.collect() for _ in range(5)]

    assert len(samples) == 5
    # Timestamps should be increasing
    for i in range(1, len(samples)):
        assert samples[i]["timestamp"] >= samples[i - 1]["timestamp"]

    collector.shutdown()


def test_telemetry_logger_json_format() -> None:
    """Test TelemetryLogger with JSON output."""
    output = io.StringIO()
    logger = TelemetryLogger(output=output, json_format=True)

    test_data = {
        "hbm_utilization": 0.85,
        "temperature_c": 70.5,
        "power_usage_w": 275.0,
    }

    logger.log(test_data)
    logger.close()

    result = output.getvalue()
    assert "TELEMETRY" in result
    assert "0.85" in result or "hbm_utilization" in result


def test_telemetry_logger_human_format() -> None:
    """Test TelemetryLogger with human-readable output."""
    output = io.StringIO()
    logger = TelemetryLogger(output=output, json_format=False)

    test_data = {
        "hbm_utilization": 0.85,
        "temperature_c": 70.5,
    }

    logger.log(test_data)
    logger.close()

    result = output.getvalue()
    assert "TELEMETRY" in result
    assert "hbm_utilization" in result


def test_telemetry_logger_event() -> None:
    """Test logging structured events."""
    output = io.StringIO()
    logger = TelemetryLogger(output=output, json_format=True)

    logger.log_event(
        "test_event",
        "Test message",
        {"key": "value"},
    )
    logger.close()

    result = output.getvalue()
    assert "test_event" in result
    assert "Test message" in result


def test_telemetry_collector_fallback_to_mock() -> None:
    """Test that collector falls back to mock mode when NVML fails."""
    # Initialize without mock mode - should fallback if NVML not available
    collector = TelemetryCollector(mock_mode=False)
    telemetry = collector.collect()

    # Should get valid data regardless
    assert "hbm_utilization" in telemetry
    assert "timestamp" in telemetry

    collector.shutdown()
