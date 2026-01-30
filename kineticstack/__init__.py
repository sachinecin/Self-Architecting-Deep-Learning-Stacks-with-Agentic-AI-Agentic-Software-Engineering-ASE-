"""
KineticStack: Telemetry-driven agentic deep learning stack optimization.

An experimental ASE framework that treats the deep learning stack as a living organism,
continuously refactoring Autograd graphs and synthesizing hardware-specific kernels
based on real-time telemetry.
"""

__version__ = "0.1.0"

from kineticstack.core.manager import StackManager
from kineticstack.core.telemetry import TelemetryCollector
from kineticstack.agents.sculptor import SculptorExecutor

__all__ = ["StackManager", "TelemetryCollector", "SculptorExecutor", "__version__"]
