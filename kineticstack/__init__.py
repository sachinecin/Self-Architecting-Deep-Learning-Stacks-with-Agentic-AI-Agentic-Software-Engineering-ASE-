"""
KineticStack: Self-Architecting Deep Learning Stack with Agentic AI

An experimental Agentic Software Engineering (ASE) framework that treats the deep learning
stack as a living organism, continuously refactoring autograd graphs and optimizing based
on real-time telemetry.
"""

__version__ = "0.1.0"

from kineticstack.core.invariants import InvariantViolationError, verify_invariant
from kineticstack.core.manager import KineticStackManager
from kineticstack.core.telemetry import TelemetryMonitor

__all__ = [
    "KineticStackManager",
    "TelemetryMonitor",
    "verify_invariant",
    "InvariantViolationError",
]
