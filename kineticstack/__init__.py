"""
KineticStack: Self-Architecting Deep Learning Stacks with Agentic AI

An experimental ASE framework that treats the deep learning stack as a living organism.
"""

__version__ = "0.1.0"

from kineticstack.agents.sculptor import SculptorReasoner, SculptorExecutor
from kineticstack.core.telemetry import TelemetryMonitor
from kineticstack.core.invariants import verify_invariant, InvariantViolationError

__all__ = [
    "SculptorReasoner",
    "SculptorExecutor",
    "TelemetryMonitor",
    "verify_invariant",
    "InvariantViolationError",
]
