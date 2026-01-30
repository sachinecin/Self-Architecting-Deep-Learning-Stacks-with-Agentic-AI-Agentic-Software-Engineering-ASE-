"""
Base agent interface for KineticStack optimization agents.

Defines the common interface that all optimization agents must implement
to participate in the telemetry-driven agentic loop.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for KineticStack optimization agents.

    All agents must implement the execute method to apply their
    specific optimizations based on telemetry data.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the base agent.

        Args:
            name: Human-readable name for this agent
        """
        self.name = name
        logger.info(f"Agent '{name}' initialized")

    @abstractmethod
    def execute(self, telemetry: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Execute the agent's optimization logic.

        Args:
            telemetry: Current telemetry data from TelemetryCollector
            context: Optional execution context with additional information

        Returns:
            True if optimization was applied successfully, False otherwise
        """
        pass

    def should_activate(self, telemetry: Dict[str, Any]) -> bool:
        """
        Determine if this agent should activate based on telemetry.

        Default implementation always returns True. Subclasses should
        override to implement activation criteria.

        Args:
            telemetry: Current telemetry data

        Returns:
            True if agent should execute, False otherwise
        """
        return True

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}')"
