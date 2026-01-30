"""
Manager for KineticStack orchestration.

Coordinates telemetry monitoring, agentic sculptor execution, and invariant verification.
"""

from typing import Callable, Optional

from kineticstack.agents.sculptor import SculptorExecutor
from kineticstack.core.telemetry import TelemetryMonitor


class KineticStackManager:
    """
    Central manager that orchestrates the KineticStack ASE framework.

    Coordinates telemetry monitoring, triggers the agentic sculptor when needed,
    and ensures all changes meet invariant requirements.
    """

    def __init__(
        self,
        telemetry_monitor: Optional[TelemetryMonitor] = None,
        sculptor_executor: Optional[SculptorExecutor] = None,
        auto_refactor: bool = False,
    ):
        """
        Initialize the KineticStack manager.

        Args:
            telemetry_monitor: TelemetryMonitor instance. If None, creates default.
            sculptor_executor: SculptorExecutor instance. If None, creates default.
            auto_refactor: If True, automatically applies refactoring when triggered.
        """
        self.telemetry_monitor = telemetry_monitor or TelemetryMonitor()
        self.sculptor_executor = sculptor_executor or SculptorExecutor()
        self.auto_refactor = auto_refactor
        self._original_modules = {}

    def monitor_and_optimize(
        self,
        model: Callable,
        optimization_callback: Optional[Callable] = None,
        check_interval_seconds: float = 5.0,
    ):
        """
        Monitor telemetry and trigger optimization when thresholds are exceeded.

        Args:
            model: PyTorch model or callable to potentially optimize.
            optimization_callback: Optional callback when optimization is triggered.
            check_interval_seconds: How often to check telemetry.
        """
        for sample in self.telemetry_monitor.stream(
            interval_seconds=check_interval_seconds, duration_seconds=60.0
        ):
            if self.telemetry_monitor.should_trigger_refactor():
                if self.auto_refactor:
                    optimized_model = self.sculptor_executor.apply_kinetic_optimization(model)
                    if optimization_callback:
                        optimization_callback(optimized_model, sample)
                    return optimized_model
                else:
                    if optimization_callback:
                        optimization_callback(None, sample)

        return model

    def apply_optimization(self, model: Callable) -> Callable:
        """
        Manually apply kinetic optimization to a model.

        Args:
            model: PyTorch model or callable to optimize.

        Returns:
            Optimized model.
        """
        return self.sculptor_executor.apply_kinetic_optimization(model)

    def revert_optimization(self, model: Callable) -> Callable:
        """
        Revert kinetic optimization on a model.

        Args:
            model: Optimized model to revert.

        Returns:
            Original model.
        """
        return self.sculptor_executor.revert_optimization(model)

    def shutdown(self):
        """Shutdown manager and release resources."""
        if self.telemetry_monitor:
            self.telemetry_monitor.shutdown()
