"""
KineticStack: The Agentic AI Runtime
A self-evolving DL stack using Agentic Software Engineering (ASE)
"""

from .core.runtime import KineticRuntime
from .agents.sculptor import SculptorAgent
from .agents.kernel_synthesizer import KernelSynthesizerAgent
from .agents.invariant_guard import InvariantGuard

__version__ = "0.1.0"
__all__ = [
    "KineticRuntime",
    "SculptorAgent",
    "KernelSynthesizerAgent",
    "InvariantGuard",
]
