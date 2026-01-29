"Example base agent interface for KineticStack."

from typing import Any, Dict


class BaseAgent:
    """Minimal example base class for agents.

    Subclass and implement `step` and `reset`.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.state: Dict[str, Any] = {}

    def reset(self) -> None:
        """Reset internal state."""
        self.state.clear()

    def step(self, observation: Any) -> Any:
        """Perform a single agent step. Override in subclasses."""
        raise NotImplementedError("Agent must implement step()")
