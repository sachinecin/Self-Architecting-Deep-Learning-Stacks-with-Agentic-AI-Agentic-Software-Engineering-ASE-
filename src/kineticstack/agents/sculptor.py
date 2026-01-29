"""
The Sculptor Agent: Autograd Refactoring for selective checkpointing
Re-writes gradient tapes for selective checkpointing based on memory pressure.
"""

import torch
import torch.fx as fx
from typing import List, Dict, Set, Optional, Callable
from dataclasses import dataclass, field
from ..core.autograd_sculpting import AutogradSculptor, MemoryProfile


@dataclass
class CheckpointStrategy:
    """Strategy for gradient checkpointing"""
    nodes_to_checkpoint: Set[str] = field(default_factory=set)
    recompute_on_backward: bool = True
    memory_savings_bytes: int = 0
    recomputation_overhead: float = 0.0  # Fraction of forward time
    

class SculptorAgent:
    """
    The Sculptor Agent: Autograd Refactoring
    
    This agent continuously monitors gradient computation and rewrites gradient tapes
    for selective checkpointing based on memory pressure and hardware characteristics.
    """
    
    def __init__(
        self,
        memory_budget_gb: float = 80.0,
        checkpoint_threshold: float = 0.3,
    ):
        """
        Initialize the Sculptor Agent.
        
        Args:
            memory_budget_gb: Total memory budget in GB
            checkpoint_threshold: Checkpoint if operation uses > threshold * budget
        """
        self.sculptor = AutogradSculptor(target_memory_gb=memory_budget_gb)
        self.checkpoint_threshold = checkpoint_threshold
        self.checkpoint_strategy = CheckpointStrategy()
        self.gradient_tapes: Dict[str, fx.GraphModule] = {}
        
    def analyze_gradient_tape(
        self,
        model: torch.nn.Module,
        example_inputs: tuple,
    ) -> Dict[str, any]:
        """
        Analyze the gradient tape to identify optimization opportunities.
        
        Args:
            model: PyTorch model to analyze
            example_inputs: Example inputs for tracing
            
        Returns:
            Dictionary containing analysis results
        """
        # Trace the model
        traced = self.sculptor.trace_model(model, example_inputs)
        
        # Analyze memory pressure
        memory_map = self.sculptor.analyze_memory_pressure(traced.graph)
        
        # Identify operations with high memory usage
        high_memory_ops = []
        for node_name, memory in memory_map.items():
            if memory > self.sculptor.target_memory_bytes * self.checkpoint_threshold:
                high_memory_ops.append(node_name)
        
        return {
            'traced_graph': traced,
            'memory_map': memory_map,
            'high_memory_ops': high_memory_ops,
            'total_memory': max(memory_map.values()) if memory_map else 0,
        }
    
    def rewrite_gradient_tape(
        self,
        model: torch.nn.Module,
        example_inputs: tuple,
    ) -> fx.GraphModule:
        """
        Rewrite gradient tape with selective checkpointing.
        
        Args:
            model: PyTorch model to optimize
            example_inputs: Example inputs for tracing
            
        Returns:
            Optimized graph module with checkpointing annotations
        """
        # Analyze the gradient tape
        analysis = self.analyze_gradient_tape(model, example_inputs)
        traced = analysis['traced_graph']
        
        # Apply selective checkpointing
        graph = traced.graph
        nodes_checkpointed = set()
        
        for node in graph.nodes:
            if node.name in analysis['high_memory_ops']:
                # Mark node for checkpointing
                node.meta['checkpoint'] = True
                node.meta['recompute_on_backward'] = True
                nodes_checkpointed.add(node.name)
        
        # Update checkpoint strategy
        self.checkpoint_strategy = CheckpointStrategy(
            nodes_to_checkpoint=nodes_checkpointed,
            recompute_on_backward=True,
            memory_savings_bytes=int(analysis['total_memory'] * 0.4),  # Estimate 40% savings
            recomputation_overhead=len(nodes_checkpointed) * 0.1,  # Estimate 10% overhead per checkpoint
        )
        
        # Recompile the graph
        traced.recompile()
        
        # Store the gradient tape
        tape_id = id(model)
        self.gradient_tapes[str(tape_id)] = traced
        
        return traced
    
    def apply_checkpointing(
        self,
        model: torch.nn.Module,
        checkpoint_layers: Optional[List[str]] = None,
    ) -> torch.nn.Module:
        """
        Apply gradient checkpointing to specific layers of the model.
        
        Args:
            model: PyTorch model
            checkpoint_layers: List of layer names to checkpoint (None = auto-detect)
            
        Returns:
            Model with checkpointing applied
        """
        if checkpoint_layers is None:
            checkpoint_layers = list(self.checkpoint_strategy.nodes_to_checkpoint)
        
        # Wrap specified layers with checkpointing
        for name, module in model.named_modules():
            if name in checkpoint_layers:
                # Apply gradient checkpointing
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()
                else:
                    # Wrap forward pass for manual checkpointing
                    original_forward = module.forward
                    
                    def checkpointed_forward(*args, **kwargs):
                        return torch.utils.checkpoint.checkpoint(
                            original_forward,
                            *args,
                            **kwargs,
                            use_reentrant=False,
                        )
                    
                    module.forward = checkpointed_forward
        
        return model
    
    def optimize_model(
        self,
        model: torch.nn.Module,
        example_inputs: tuple,
        apply_checkpointing: bool = True,
    ) -> torch.nn.Module:
        """
        Full optimization pipeline: analyze, rewrite, and apply checkpointing.
        
        Args:
            model: PyTorch model to optimize
            example_inputs: Example inputs for tracing
            apply_checkpointing: Whether to apply checkpointing (default: True)
            
        Returns:
            Optimized model
        """
        # Rewrite gradient tape
        traced = self.rewrite_gradient_tape(model, example_inputs)
        
        # Apply checkpointing if requested
        if apply_checkpointing:
            # Get module names from checkpoint strategy
            checkpoint_modules = []
            for name, _ in model.named_modules():
                # Simple heuristic: checkpoint large layers
                if any(layer_type in name.lower() for layer_type in ['transformer', 'attention', 'block']):
                    checkpoint_modules.append(name)
            
            if checkpoint_modules:
                model = self.apply_checkpointing(model, checkpoint_modules)
        
        return model
    
    def get_checkpoint_strategy(self) -> CheckpointStrategy:
        """Get the current checkpointing strategy."""
        return self.checkpoint_strategy
    
    def get_memory_profile(self) -> MemoryProfile:
        """Get the current memory profile."""
        return self.sculptor.get_memory_profile()
    
    def estimate_savings(self) -> Dict[str, float]:
        """
        Estimate memory savings and computational overhead from checkpointing.
        
        Returns:
            Dictionary with 'memory_savings_gb' and 'time_overhead_pct'
        """
        return {
            'memory_savings_gb': self.checkpoint_strategy.memory_savings_bytes / (1024**3),
            'time_overhead_pct': self.checkpoint_strategy.recomputation_overhead * 100,
            'nodes_checkpointed': len(self.checkpoint_strategy.nodes_to_checkpoint),
        }
