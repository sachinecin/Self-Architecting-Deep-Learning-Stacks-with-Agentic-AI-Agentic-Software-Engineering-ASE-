"""
KineticRuntime: The main runtime for the self-evolving DL stack.
Coordinates all agents and provides the main API for using KineticStack.
"""

import torch
import torch.fx as fx
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time

from .autograd_sculpting import AutogradSculptor, MemoryProfile
from .kernel_synthesis import KernelSynthesizer, HardwareProfile, KernelSpec
from ..agents.sculptor import SculptorAgent, CheckpointStrategy
from ..agents.kernel_synthesizer import KernelSynthesizerAgent, CompilationResult
from ..agents.invariant_guard import InvariantGuard, InvariantViolation


@dataclass
class RuntimeConfig:
    """Configuration for KineticRuntime"""
    memory_budget_gb: float = 80.0
    hardware_profile: Optional[HardwareProfile] = None
    enable_autograd_sculpting: bool = True
    enable_kernel_synthesis: bool = True
    enable_invariant_checking: bool = True
    auto_checkpoint: bool = True
    strict_validation: bool = False


@dataclass
class OptimizationResult:
    """Result of model optimization"""
    optimized_model: Any
    memory_profile: MemoryProfile
    checkpoint_strategy: CheckpointStrategy
    compiled_kernels: List[KernelSpec] = field(default_factory=list)
    violations: List[InvariantViolation] = field(default_factory=list)
    optimization_time_ms: float = 0.0


class KineticRuntime:
    """
    KineticStack: The Agentic AI Runtime
    
    Main runtime that coordinates all agents for self-evolving deep learning stack.
    Features:
    - Autograd Sculpting: On-the-fly torch.fx graph refactoring for HBM3e
    - Kernel Synthesis: Hardware-aware Triton generation
    - The Sculptor Agent: Autograd refactoring with selective checkpointing
    - Kernel Synthesizer: Hardware-specific JIT compilation
    - Invariant Guard: Validation and safety checks
    """
    
    def __init__(self, config: Optional[RuntimeConfig] = None):
        """
        Initialize the KineticRuntime.
        
        Args:
            config: Runtime configuration (uses defaults if None)
        """
        self.config = config or RuntimeConfig()
        
        # Initialize agents
        self.sculptor_agent = SculptorAgent(
            memory_budget_gb=self.config.memory_budget_gb,
        ) if self.config.enable_autograd_sculpting else None
        
        self.kernel_synthesizer_agent = KernelSynthesizerAgent(
            hardware_profile=self.config.hardware_profile,
        ) if self.config.enable_kernel_synthesis else None
        
        self.invariant_guard = InvariantGuard(
            strict_mode=self.config.strict_validation,
        ) if self.config.enable_invariant_checking else None
        
        # Runtime state
        self.optimized_models: Dict[str, OptimizationResult] = {}
        self.telemetry: Dict[str, List[float]] = {
            'optimization_time': [],
            'memory_savings': [],
        }
        
    def optimize_model(
        self,
        model: torch.nn.Module,
        example_inputs: tuple,
        model_name: Optional[str] = None,
    ) -> OptimizationResult:
        """
        Optimize a PyTorch model using all available agents.
        
        Args:
            model: PyTorch model to optimize
            example_inputs: Example inputs for tracing and profiling
            model_name: Optional name for caching
            
        Returns:
            OptimizationResult containing optimized model and metadata
        """
        start_time = time.time()
        
        # Step 1: Autograd Sculpting and Selective Checkpointing
        optimized_model = model
        checkpoint_strategy = CheckpointStrategy()
        memory_profile = MemoryProfile(0, 0, 0.0)
        
        if self.sculptor_agent and self.config.enable_autograd_sculpting:
            print("Running Sculptor Agent for autograd refactoring...")
            optimized_model = self.sculptor_agent.optimize_model(
                model,
                example_inputs,
                apply_checkpointing=self.config.auto_checkpoint,
            )
            checkpoint_strategy = self.sculptor_agent.get_checkpoint_strategy()
            memory_profile = self.sculptor_agent.get_memory_profile()
        
        # Step 2: Kernel Synthesis
        compiled_kernels = []
        if self.kernel_synthesizer_agent and self.config.enable_kernel_synthesis:
            print("Running Kernel Synthesizer Agent...")
            # Identify operations that would benefit from custom kernels
            operations = self._identify_kernel_candidates(model, example_inputs)
            
            for op_type, shapes in operations:
                result = self.kernel_synthesizer_agent.jit_compile(
                    operation=op_type,
                    input_shapes=shapes['inputs'],
                    output_shape=shapes['output'],
                )
                compiled_kernels.append(result.kernel_spec)
        
        # Step 3: Invariant Validation
        violations = []
        if self.invariant_guard and self.config.enable_invariant_checking:
            print("Running Invariant Guard validation...")
            
            # Validate the transformation
            try:
                with torch.no_grad():
                    original_output = model(*example_inputs)
                    optimized_output = optimized_model(*example_inputs)
                
                is_valid = self.invariant_guard.validate_transformation(
                    original_output,
                    optimized_output,
                    memory_used=memory_profile.peak_memory,
                    memory_budget=int(self.config.memory_budget_gb * 1024**3),
                )
                
                if not is_valid:
                    print("Warning: Invariant violations detected")
                    violations = self.invariant_guard.get_violations()
            except Exception as e:
                print(f"Warning: Could not validate transformation: {e}")
        
        # Calculate optimization time
        optimization_time = (time.time() - start_time) * 1000  # ms
        
        # Create result
        result = OptimizationResult(
            optimized_model=optimized_model,
            memory_profile=memory_profile,
            checkpoint_strategy=checkpoint_strategy,
            compiled_kernels=compiled_kernels,
            violations=violations,
            optimization_time_ms=optimization_time,
        )
        
        # Cache the result
        if model_name:
            self.optimized_models[model_name] = result
        
        # Update telemetry
        self.telemetry['optimization_time'].append(optimization_time)
        if checkpoint_strategy.memory_savings_bytes > 0:
            self.telemetry['memory_savings'].append(
                checkpoint_strategy.memory_savings_bytes / 1024**3
            )
        
        print(f"Optimization complete in {optimization_time:.2f}ms")
        return result
    
    def _identify_kernel_candidates(
        self,
        model: torch.nn.Module,
        example_inputs: tuple,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Identify operations that would benefit from custom kernels.
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs
            
        Returns:
            List of (operation_type, shapes) tuples
        """
        candidates = []
        
        try:
            # Try to trace the model
            traced = fx.symbolic_trace(model)
            
            for node in traced.graph.nodes:
                if node.op == 'call_function' or node.op == 'call_method':
                    target_str = str(node.target).lower()
                    
                    # Identify common operations
                    if 'matmul' in target_str or 'linear' in target_str:
                        candidates.append((
                            'matmul',
                            {'inputs': [(128, 512), (512, 256)], 'output': (128, 256)}
                        ))
                    elif 'attention' in target_str:
                        candidates.append((
                            'attention',
                            {'inputs': [(32, 512, 512)], 'output': (32, 512, 512)}
                        ))
        except Exception:
            # If tracing fails, use heuristics based on model structure
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    in_features = module.in_features
                    out_features = module.out_features
                    candidates.append((
                        'matmul',
                        {'inputs': [(1, in_features), (in_features, out_features)], 
                         'output': (1, out_features)}
                    ))
        
        return candidates
    
    def get_telemetry(self) -> Dict[str, Any]:
        """
        Get runtime telemetry data.
        
        Returns:
            Dictionary containing telemetry from all agents
        """
        telemetry = {
            'runtime': {
                'avg_optimization_time_ms': (
                    sum(self.telemetry['optimization_time']) / len(self.telemetry['optimization_time'])
                    if self.telemetry['optimization_time'] else 0
                ),
                'total_optimizations': len(self.telemetry['optimization_time']),
            }
        }
        
        if self.sculptor_agent:
            telemetry['sculptor'] = self.sculptor_agent.estimate_savings()
        
        if self.kernel_synthesizer_agent:
            telemetry['kernel_synthesizer'] = self.kernel_synthesizer_agent.get_telemetry()
        
        if self.invariant_guard:
            telemetry['invariant_guard'] = self.invariant_guard.get_summary()
        
        return telemetry
    
    def get_optimization_result(self, model_name: str) -> Optional[OptimizationResult]:
        """
        Get cached optimization result by name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            OptimizationResult if found, None otherwise
        """
        return self.optimized_models.get(model_name)
    
    def clear_cache(self):
        """Clear all cached optimization results."""
        self.optimized_models.clear()
        if self.kernel_synthesizer_agent:
            self.kernel_synthesizer_agent.clear_cache()
    
    def print_status(self):
        """Print current runtime status and statistics."""
        print("=" * 60)
        print("KineticStack Runtime Status")
        print("=" * 60)
        print(f"Memory Budget: {self.config.memory_budget_gb} GB")
        print(f"Autograd Sculpting: {'Enabled' if self.config.enable_autograd_sculpting else 'Disabled'}")
        print(f"Kernel Synthesis: {'Enabled' if self.config.enable_kernel_synthesis else 'Disabled'}")
        print(f"Invariant Checking: {'Enabled' if self.config.enable_invariant_checking else 'Disabled'}")
        print()
        
        telemetry = self.get_telemetry()
        
        if 'sculptor' in telemetry:
            print("Sculptor Agent:")
            print(f"  Memory Savings: {telemetry['sculptor']['memory_savings_gb']:.2f} GB")
            print(f"  Time Overhead: {telemetry['sculptor']['time_overhead_pct']:.1f}%")
            print(f"  Checkpointed Nodes: {telemetry['sculptor']['nodes_checkpointed']}")
            print()
        
        if 'kernel_synthesizer' in telemetry:
            print("Kernel Synthesizer Agent:")
            print(f"  Avg Register Usage: {telemetry['kernel_synthesizer']['avg_register_usage']:.0f}")
            print(f"  Avg Compilation Time: {telemetry['kernel_synthesizer']['avg_compilation_time_ms']:.2f} ms")
            print(f"  Total Kernels: {telemetry['kernel_synthesizer']['total_kernels_compiled']}")
            print()
        
        if 'invariant_guard' in telemetry:
            print("Invariant Guard:")
            print(f"  Critical Violations: {telemetry['invariant_guard']['critical']}")
            print(f"  Error Violations: {telemetry['invariant_guard']['error']}")
            print(f"  Warning Violations: {telemetry['invariant_guard']['warning']}")
            print()
        
        print("=" * 60)
