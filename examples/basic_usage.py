"""
Example usage of KineticStack: Self-Evolving Deep Learning Stack
"""

import torch
import torch.nn as nn
from kineticstack import KineticRuntime, SculptorAgent, KernelSynthesizerAgent, InvariantGuard
from kineticstack.core.runtime import RuntimeConfig
from kineticstack.core.kernel_synthesis import HardwareProfile


# Example 1: Simple model optimization
def example_simple_optimization():
    """Demonstrate basic model optimization with KineticStack."""
    print("\n" + "=" * 60)
    print("Example 1: Simple Model Optimization")
    print("=" * 60)
    
    # Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, 256)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Create model and example input
    model = SimpleModel()
    example_input = (torch.randn(32, 512),)
    
    # Initialize KineticRuntime with default config
    runtime = KineticRuntime()
    
    # Optimize the model
    result = runtime.optimize_model(model, example_input, model_name="simple_model")
    
    # Print results
    print(f"\nOptimization completed in {result.optimization_time_ms:.2f}ms")
    print(f"Peak memory: {result.memory_profile.peak_memory / 1e6:.2f} MB")
    print(f"Memory bandwidth: {result.memory_profile.memory_bandwidth:.2f} GB/s")
    print(f"HBM3e optimized: {result.memory_profile.hbm3e_optimized}")
    print(f"Compiled kernels: {len(result.compiled_kernels)}")
    
    # Test the optimized model
    with torch.no_grad():
        output = result.optimized_model(*example_input)
        print(f"Output shape: {output.shape}")
    
    return runtime, result


# Example 2: Using individual agents
def example_individual_agents():
    """Demonstrate using individual agents separately."""
    print("\n" + "=" * 60)
    print("Example 2: Using Individual Agents")
    print("=" * 60)
    
    # Define a larger model
    class TransformerBlock(nn.Module):
        def __init__(self, hidden_dim=512):
            super().__init__()
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, hidden_dim)
            )
            self.norm2 = nn.LayerNorm(hidden_dim)
        
        def forward(self, x):
            # Self-attention
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)
            
            # Feed-forward
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            
            return x
    
    model = TransformerBlock()
    example_input = (torch.randn(32, 128, 512),)  # (seq_len, batch, hidden_dim)
    
    # 1. Use Sculptor Agent for gradient checkpointing
    print("\n1. Running Sculptor Agent...")
    sculptor = SculptorAgent(memory_budget_gb=40.0)
    optimized_model = sculptor.optimize_model(model, example_input, apply_checkpointing=True)
    savings = sculptor.estimate_savings()
    print(f"   Memory savings: {savings['memory_savings_gb']:.2f} GB")
    print(f"   Time overhead: {savings['time_overhead_pct']:.1f}%")
    
    # 2. Use Kernel Synthesizer for custom kernels
    print("\n2. Running Kernel Synthesizer Agent...")
    kernel_agent = KernelSynthesizerAgent(hardware_profile=HardwareProfile.h100_profile())
    
    # Compile a matmul kernel
    matmul_result = kernel_agent.jit_compile(
        operation='matmul',
        input_shapes=[(128, 512), (512, 2048)],
        output_shape=(128, 2048)
    )
    print(f"   Kernel: {matmul_result.kernel_spec.name}")
    print(f"   Register pressure: {matmul_result.kernel_spec.register_pressure}")
    print(f"   Compilation time: {matmul_result.compilation_time_ms:.2f}ms")
    print(f"   Optimized: {matmul_result.optimized}")
    
    # Compile an attention kernel
    attention_result = kernel_agent.jit_compile(
        operation='attention',
        input_shapes=[(128, 512)],
        output_shape=(128, 512)
    )
    print(f"   Attention kernel: {attention_result.kernel_spec.name}")
    
    # 3. Use Invariant Guard for validation
    print("\n3. Running Invariant Guard...")
    guard = InvariantGuard(strict_mode=False)
    
    with torch.no_grad():
        original_output = model(*example_input)
        optimized_output = optimized_model(*example_input)
    
    is_valid = guard.validate_transformation(original_output, optimized_output)
    print(f"   Validation passed: {is_valid}")
    
    summary = guard.get_summary()
    print(f"   Total violations: {summary['total']}")
    print(f"   Critical: {summary['critical']}, Errors: {summary['error']}, Warnings: {summary['warning']}")
    
    return sculptor, kernel_agent, guard


# Example 3: Custom configuration
def example_custom_config():
    """Demonstrate using custom runtime configuration."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Runtime Configuration")
    print("=" * 60)
    
    # Create custom configuration
    config = RuntimeConfig(
        memory_budget_gb=64.0,
        hardware_profile=HardwareProfile.a100_profile(),  # Target A100 instead of H100
        enable_autograd_sculpting=True,
        enable_kernel_synthesis=True,
        enable_invariant_checking=True,
        auto_checkpoint=True,
        strict_validation=False,
    )
    
    # Initialize runtime with custom config
    runtime = KineticRuntime(config)
    
    # Define a model
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
    )
    example_input = (torch.randn(64, 1024),)
    
    # Optimize
    result = runtime.optimize_model(model, example_input)
    
    # Print status
    runtime.print_status()
    
    return runtime, result


# Example 4: Accessing telemetry
def example_telemetry():
    """Demonstrate accessing runtime telemetry."""
    print("\n" + "=" * 60)
    print("Example 4: Runtime Telemetry")
    print("=" * 60)
    
    runtime = KineticRuntime()
    
    # Optimize multiple models
    for i in range(3):
        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        example_input = (torch.randn(16, 256),)
        runtime.optimize_model(model, example_input, model_name=f"model_{i}")
    
    # Get telemetry
    telemetry = runtime.get_telemetry()
    
    print("\nTelemetry Summary:")
    print(f"Total optimizations: {telemetry['runtime']['total_optimizations']}")
    print(f"Avg optimization time: {telemetry['runtime']['avg_optimization_time_ms']:.2f}ms")
    
    if 'sculptor' in telemetry:
        print(f"\nSculptor Agent:")
        for key, value in telemetry['sculptor'].items():
            print(f"  {key}: {value}")
    
    if 'kernel_synthesizer' in telemetry:
        print(f"\nKernel Synthesizer Agent:")
        for key, value in telemetry['kernel_synthesizer'].items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
    
    return runtime


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("KineticStack: The Agentic AI Runtime")
    print("Self-Evolving Deep Learning Stack Examples")
    print("=" * 60)
    
    # Run examples
    try:
        example_simple_optimization()
    except Exception as e:
        print(f"Example 1 error: {e}")
    
    try:
        example_individual_agents()
    except Exception as e:
        print(f"Example 2 error: {e}")
    
    try:
        example_custom_config()
    except Exception as e:
        print(f"Example 3 error: {e}")
    
    try:
        example_telemetry()
    except Exception as e:
        print(f"Example 4 error: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
