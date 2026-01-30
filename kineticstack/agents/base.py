"""Base agent class for KineticStack agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Base class for all KineticStack agents."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the agent with optional configuration."""
        self.config = config or {}

    @abstractmethod
    def reason(self, *args, **kwargs) -> Any:
        """Reason about the given input and return insights."""
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute an action based on reasoning."""
        pass
