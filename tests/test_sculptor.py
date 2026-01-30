"""Tests for SculptorExecutor agent."""

import pytest
from kineticstack.agents.sculptor import SculptorExecutor
import time


def test_sculptor_initialization() -> None:
    """Test SculptorExecutor initialization."""
    sculptor = SculptorExecutor(hbm_threshold=0.85, sustained_duration=5.0)

    assert sculptor.hbm_threshold == 0.85
    assert sculptor.sustained_duration == 5.0
    assert sculptor.name == "SculptorExecutor"


def test_sculptor_activation_low_memory() -> None:
    """Test that sculptor doesn't activate with low memory usage."""
    sculptor = SculptorExecutor(hbm_threshold=0.85, sustained_duration=1.0)

    telemetry = {
        "hbm_utilization": 0.70,
        "timestamp": time.time(),
    }

    assert not sculptor.should_activate(telemetry)


def test_sculptor_activation_sustained_high_memory() -> None:
    """Test sculptor activates after sustained high memory usage."""
    sculptor = SculptorExecutor(hbm_threshold=0.85, sustained_duration=1.0)

    start_time = time.time()

    # First sample - high utilization
    telemetry1 = {
        "hbm_utilization": 0.90,
        "timestamp": start_time,
    }
    assert not sculptor.should_activate(telemetry1)

    # Second sample after sustained duration - should activate
    telemetry2 = {
        "hbm_utilization": 0.90,
        "timestamp": start_time + 1.5,
    }
    assert sculptor.should_activate(telemetry2)


def test_sculptor_resets_on_low_memory() -> None:
    """Test sculptor resets when memory returns to normal."""
    sculptor = SculptorExecutor(hbm_threshold=0.85, sustained_duration=1.0)

    start_time = time.time()

    # High utilization
    telemetry1 = {
        "hbm_utilization": 0.90,
        "timestamp": start_time,
    }
    sculptor.should_activate(telemetry1)

    # Returns to normal
    telemetry2 = {
        "hbm_utilization": 0.70,
        "timestamp": start_time + 0.5,
    }
    sculptor.should_activate(telemetry2)

    # High again - should need to sustain again
    telemetry3 = {
        "hbm_utilization": 0.90,
        "timestamp": start_time + 1.0,
    }
    assert not sculptor.should_activate(telemetry3)


def test_sculptor_execute() -> None:
    """Test sculptor execute method."""
    sculptor = SculptorExecutor(hbm_threshold=0.85, sustained_duration=0.1)

    start_time = time.time()

    # Low memory - should not execute
    telemetry1 = {
        "hbm_utilization": 0.70,
        "timestamp": start_time,
    }
    assert not sculptor.execute(telemetry1)

    # High memory but not sustained
    telemetry2 = {
        "hbm_utilization": 0.90,
        "timestamp": start_time,
    }
    assert not sculptor.execute(telemetry2)

    # Sustained high memory - should execute
    telemetry3 = {
        "hbm_utilization": 0.90,
        "timestamp": start_time + 0.2,
    }
    assert sculptor.execute(telemetry3)

    # Already enabled - should return True without re-applying
    assert sculptor.execute(telemetry3)


def test_sculptor_disable_checkpointing() -> None:
    """Test disabling checkpointing."""
    sculptor = SculptorExecutor(hbm_threshold=0.85, sustained_duration=0.1)

    start_time = time.time()

    # Enable checkpointing
    telemetry = {
        "hbm_utilization": 0.90,
        "timestamp": start_time + 0.2,
    }
    sculptor.execute(telemetry)

    status = sculptor.get_status()
    assert status["checkpointing_enabled"] is True

    # Disable
    sculptor.disable_checkpointing()

    status = sculptor.get_status()
    assert status["checkpointing_enabled"] is False
    assert status["high_util_start"] is None


def test_sculptor_status() -> None:
    """Test getting sculptor status."""
    sculptor = SculptorExecutor(hbm_threshold=0.80, sustained_duration=3.0)

    status = sculptor.get_status()

    assert status["checkpointing_enabled"] is False
    assert status["hbm_threshold"] == 0.80
    assert status["sustained_duration"] == 3.0
    assert status["high_util_start"] is None


def test_sculptor_with_model_context() -> None:
    """Test sculptor with model in context."""
    sculptor = SculptorExecutor(hbm_threshold=0.85, sustained_duration=0.1)

    # Mock model object
    class MockModel:
        pass

    model = MockModel()
    context = {"model": model}

    start_time = time.time()
    telemetry = {
        "hbm_utilization": 0.90,
        "timestamp": start_time + 0.2,
    }

    # Should execute successfully even with model
    assert sculptor.execute(telemetry, context=context)
