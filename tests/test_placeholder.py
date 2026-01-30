"""
Placeholder test to ensure basic imports work.

This serves as a smoke test for the package structure.
"""


def test_imports():
    """Test that all main modules can be imported."""
    import kineticstack  # noqa: F401
    from kineticstack import (  # noqa: F401
        InvariantViolationError,
        KineticStackManager,
        TelemetryMonitor,
        verify_invariant,
    )
    from kineticstack.agents import BaseAgent, SculptorExecutor  # noqa: F401
    from kineticstack.telemetry import TelemetryLogger  # noqa: F401

    # Verify version is set
    assert hasattr(kineticstack, "__version__")
    assert kineticstack.__version__ == "0.1.0"


def test_core_imports():
    """Test core module imports."""
    from kineticstack.core import (
        InvariantViolationError,
        KineticStackManager,
        TelemetryMonitor,
        verify_invariant,
    )

    assert InvariantViolationError is not None
    assert KineticStackManager is not None
    assert TelemetryMonitor is not None
    assert verify_invariant is not None


def test_agents_imports():
    """Test agents module imports."""
    from kineticstack.agents import BaseAgent, SculptorExecutor

    assert BaseAgent is not None
    assert SculptorExecutor is not None


def test_telemetry_imports():
    """Test telemetry module imports."""
    from kineticstack.telemetry import TelemetryLogger

    assert TelemetryLogger is not None
