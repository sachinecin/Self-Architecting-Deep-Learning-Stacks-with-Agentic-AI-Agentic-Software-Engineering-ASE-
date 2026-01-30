"""
Sculptor agent for KineticStack.

Analyzes model structure and selectively injects activation checkpointing
to optimize memory usage. Uses heuristics based on module types to identify
optimization opportunities.
"""

from typing import Any, Callable, Dict

try:
    import torch
    import torch.nn as nn
    from torch.utils.checkpoint import checkpoint

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from kineticstack.agents.base import BaseAgent


class SculptorExecutor(BaseAgent):
    """
    Agentic sculptor that analyzes model structure and applies
    selective activation checkpointing.

    This agent uses heuristics based on module types to identify where
    checkpointing should be applied, targeting transformer blocks and
    large linear layers to reduce memory usage while maintaining
    acceptable performance.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the sculptor executor.

        Args:
            config: Configuration for sculptor behavior.
        """
        super().__init__(config)
        self._original_forwards = {}
        self._checkpointed_modules = set()

    def reason(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a model and determine where to apply checkpointing.

        Args:
            inputs: Dictionary with 'model' key containing the model to analyze.

        Returns:
            Dictionary with 'checkpoint_layers' list of module names to checkpoint.
        """
        model = inputs.get("model")
        if model is None or not TORCH_AVAILABLE:
            return {"checkpoint_layers": []}

        checkpoint_candidates = []

        # Simple heuristic: checkpoint transformer blocks and large linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.TransformerEncoderLayer):
                checkpoint_candidates.append(name)
            elif isinstance(module, nn.TransformerDecoderLayer):
                checkpoint_candidates.append(name)
            elif isinstance(module, nn.Linear) and hasattr(module, "in_features"):
                # Checkpoint large linear layers (>1024 features)
                if module.in_features > 1024 or module.out_features > 1024:
                    checkpoint_candidates.append(name)

        return {"checkpoint_layers": checkpoint_candidates}

    def execute(self, plan: Dict[str, Any]) -> Any:
        """
        Apply checkpointing based on the reasoning plan.

        Args:
            plan: Dictionary with 'model' and 'checkpoint_layers' keys.

        Returns:
            Modified model with checkpointing applied.
        """
        model = plan.get("model")
        checkpoint_layers = plan.get("checkpoint_layers", [])

        if not TORCH_AVAILABLE or model is None:
            return model

        for layer_name in checkpoint_layers:
            self._apply_checkpoint_to_module(model, layer_name)

        return model

    def _apply_checkpoint_to_module(self, model: nn.Module, module_name: str):
        """
        Apply gradient checkpointing to a specific module.

        Args:
            model: The model containing the module.
            module_name: Dotted path to the module.
        """
        if not TORCH_AVAILABLE:
            return

        try:
            # Navigate to the parent module and the target module
            parts = module_name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)

            target_name = parts[-1]
            if not hasattr(parent, target_name):
                return

            target_module = getattr(parent, target_name)

            # Store original forward if not already stored
            if module_name not in self._original_forwards:
                self._original_forwards[module_name] = target_module.forward

            # Create checkpointed forward pass
            original_forward = self._original_forwards[module_name]

            def checkpointed_forward(*args, **kwargs):
                # Conservative: only checkpoint if args are tensors and no kwargs
                if kwargs or not args:
                    return original_forward(*args, **kwargs)

                # Check if all args are tensors
                if all(isinstance(arg, torch.Tensor) for arg in args):
                    return checkpoint(original_forward, *args, use_reentrant=False)
                else:
                    return original_forward(*args, **kwargs)

            # Replace forward method
            target_module.forward = checkpointed_forward
            self._checkpointed_modules.add(module_name)
        except (AttributeError, TypeError):
            # If module path is invalid or module doesn't exist, skip silently
            pass

    def apply_kinetic_optimization(self, model: Callable) -> Callable:
        """
        High-level API to apply kinetic optimization to a model.

        Args:
            model: PyTorch model to optimize.

        Returns:
            Optimized model with checkpointing applied.
        """
        if not TORCH_AVAILABLE:
            return model

        # Reason about the model
        reasoning = self.reason({"model": model})

        # Execute the optimization plan
        plan = {"model": model, "checkpoint_layers": reasoning["checkpoint_layers"]}
        return self.execute(plan)

    def revert_optimization(self, model: Callable) -> Callable:
        """
        Revert checkpointing optimizations on a model.

        Args:
            model: Model to revert.

        Returns:
            Model with original forward methods restored.
        """
        if not TORCH_AVAILABLE:
            return model

        for module_name in list(self._checkpointed_modules):
            parts = module_name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part, None)
                if parent is None:
                    break

            if parent is None:
                continue

            target_name = parts[-1]
            if hasattr(parent, target_name):
                target_module = getattr(parent, target_name)
                if module_name in self._original_forwards:
                    target_module.forward = self._original_forwards[module_name]

        self._checkpointed_modules.clear()
        self._original_forwards.clear()

        return model
