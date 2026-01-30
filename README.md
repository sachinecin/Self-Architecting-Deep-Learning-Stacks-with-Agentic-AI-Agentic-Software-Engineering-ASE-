# KineticStack

**Self-Architecting Deep Learning Stacks with Agentic AI**

KineticStack is an experimental Agentic Software Engineering (ASE) framework that treats the deep learning stack as a living organism. It replaces static compilation with Agentic Reasoning Loops that continuously refactor Autograd graphs and synthesize hardware-specific kernels based on real-time telemetry.

![KineticStack Architecture](assets/architecture.svg)

## 🚀 Features

### 📊 Hardware Telemetry (NVML/pynvml)
- **TelemetryMonitor** provides real-time hardware metrics:
  - HBM (High Bandwidth Memory) utilization
  - SM (Streaming Multiprocessor) occupancy
  - Power draw in watts
- **Intelligent refactoring triggers**: `should_trigger_refactor()` returns `True` when HBM >= 85% for a sustained 5-second window
- Gracefully handles environments without NVML (returns mock data for CI/testing)

### 🎨 Agentic Sculptor
The Sculptor agent uses **torch.fx** to reason about model architecture and apply optimizations:

#### SculptorReasoner
- Traces models using `torch.fx.symbolic_trace`
- Detects transformer-like blocks through pattern analysis
- Identifies attention and feedforward components

#### SculptorExecutor
- Applies **selective activation checkpointing** (every N-th block)
- Uses `torch.utils.checkpoint` with `use_reentrant=False`
- **Conservative constraints**: Only checkpoints positional tensor arguments (no kwargs)
- Reversible optimizations via `remove_optimizations()`

### ⚡ Hardware Invariants
- **@verify_invariant** decorator measures p99 latency over iterations
- Raises `InvariantViolationError` if p99 exceeds threshold (ms)
- Configurable warmup and measurement iterations
- Essential for ensuring performance constraints on production hardware

## 📦 Installation

### Prerequisites
- Python 3.8+
- (Optional) CUDA-capable GPU with NVML support for telemetry
- Poetry for dependency management

### Quick Install

```bash
# Clone the repository
git clone https://github.com/sachinecin/Self-Architecting-Deep-Learning-Stacks-with-Agentic-AI-Agentic-Software-Engineering-ASE-.git
cd Self-Architecting-Deep-Learning-Stacks-with-Agentic-AI-Agentic-Software-Engineering-ASE-

# Install with Poetry
poetry install

# Or with pip (in a virtual environment)
pip install -e .
```

## 🎯 Quick Start

### Basic Usage

```python
import torch
from kineticstack import TelemetryMonitor, SculptorReasoner, SculptorExecutor

# Initialize telemetry monitor
monitor = TelemetryMonitor(
    device_index=0,
    window_size=5.0,      # 5-second window
    hbm_threshold=85.0    # 85% HBM threshold
)

# Sample hardware metrics
sample = monitor.sample_once()
print(f"HBM: {sample['HBM_utilization']:.1f}%")
print(f"SM Occupancy: {sample['SM_occupancy']:.1f}%")
print(f"Power: {sample['power_draw']:.1f}W")

# Initialize sculptor agents
reasoner = SculptorReasoner()
executor = SculptorExecutor(checkpoint_every_n=2)

# Apply optimizations to your model
model = YourTransformerModel()
example_input = torch.randn(1, 512, 768)

traced = reasoner.trace_model(model, example_input)
blocks = reasoner.detect_transformer_blocks(traced)
optimized_model = executor.apply_kinetic_optimization(model, blocks)

# Use the optimized model
output = optimized_model(your_input)
```

### Using Hardware Invariants

```python
from kineticstack import verify_invariant, InvariantViolationError

@verify_invariant(
    p99_threshold_ms=100.0,  # Max 100ms p99 latency
    num_iterations=50,
    warmup_iterations=10
)
def inference_step(model, batch):
    with torch.no_grad():
        return model(batch)

try:
    result = inference_step(model, batch)
except InvariantViolationError as e:
    print(f"Performance constraint violated: {e}")
```

### Monitoring and Adaptation

```python
from kineticstack.core.manager import KineticManager

# Create manager with all components
manager = KineticManager()

# Optimize a model
optimized = manager.optimize_model(model, example_input)

# Monitor and check if adaptation is needed
needs_refactor = manager.monitor_and_adapt(duration=10.0)
if needs_refactor:
    print("High memory pressure detected - consider further optimization")
```

## 🧪 Running Tests

```bash
# Run all tests
poetry run pytest -v

# Run specific test suites
poetry run pytest tests/test_telemetry.py -v
poetry run pytest tests/test_sculptor.py -v
poetry run pytest tests/test_invariants.py -v

# Run with coverage
poetry run pytest --cov=kineticstack --cov-report=html
```

**Note**: Telemetry tests use mocked NVML, so they run successfully on standard CI runners without GPUs.

## 🏗️ Architecture

KineticStack follows a modular architecture:

```
kineticstack/
├── agents/              # Agentic reasoning components
│   ├── base.py         # Base agent class
│   └── sculptor.py     # torch.fx reasoner + checkpointing
├── core/               # Core system components
│   ├── manager.py      # Orchestration layer
│   ├── telemetry.py    # NVML hardware monitoring
│   └── invariants.py   # Performance verification
└── telemetry/          # Extended telemetry utilities
    └── logger.py       # Metrics persistence
```

### Key Design Principles

1. **Living Organism Philosophy**: The stack continuously monitors, reasons, and adapts
2. **Conservative Constraints**: Safety-first approach (e.g., checkpoint only tensor args)
3. **Hardware-Aware**: Real-time telemetry drives optimization decisions
4. **Reversible Operations**: All optimizations can be undone
5. **Production-Ready Testing**: Mocked dependencies for CI, with support for real hardware

## 📊 Architecture Diagram

The system architecture showing the interaction between telemetry, reasoning, and execution layers:

![Architecture Diagram](assets/architecture.svg)

To convert the SVG to PNG for better GitHub preview:

```bash
# Using cairosvg
pip install cairosvg
cairosvg assets/architecture.svg -o assets/architecture.png

# Or using Inkscape
inkscape assets/architecture.svg --export-type=png --export-filename=assets/architecture.png
```

## 🔧 Development

### Code Quality

```bash
# Format code
poetry run black kineticstack tests
poetry run isort kineticstack tests

# Lint
poetry run flake8 kineticstack tests

# Type checking
poetry run mypy kineticstack
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Run manually
poetry run pre-commit run --all-files
```

## 🚨 CI/CD

The project uses GitHub Actions for CI:

- **Lint Job**: Runs black, isort, and flake8
- **Test Job**: Executes pytest with coverage reporting
- **Invariants Job** (Gated): Runs performance microbenchmarks
  - Only triggered on `workflow_dispatch` or commits with `[run-invariants]`
  - Should be run on representative GPU hardware before production deployment

## ⚠️ Important Notes

### Invariants Microbenchmark
The `@verify_invariant` decorator measures p99 latency, which is **hardware-dependent**. The gated CI job runs on standard runners without GPUs and may not reflect production performance.

**Before deploying optimizations to production**:
1. Run invariants tests on your target GPU hardware
2. Adjust thresholds based on actual hardware capabilities
3. Validate that optimizations don't violate latency constraints

### NVML Compatibility
- TelemetryMonitor requires NVIDIA GPUs with NVML support
- Gracefully degrades to mock data when NVML is unavailable
- Perfect for development and CI environments without GPUs

### Sculptor Compatibility
SculptorExecutor applies activation checkpointing with conservative constraints:
- ✅ Works with: Models using only positional tensor arguments
- ❌ May skip: Models with keyword arguments or non-tensor inputs
- 🔄 Always verify: Test your specific model architecture before production use

## 📝 Dependencies

Core dependencies with conservative version pins:
- `torch >= 2.1, < 3.0` - PyTorch for deep learning
- `triton >= 2.0, < 3.0` - GPU kernel compilation
- `pynvml >= 11.0` - NVIDIA Management Library interface

Development dependencies:
- `pytest >= 7.0` - Testing framework
- `black >= 23.0` - Code formatting
- `isort >= 5.0` - Import sorting
- `flake8 >= 6.0` - Linting
- `mypy >= 1.0` - Type checking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Review Checklist
- [ ] Run `pytest` and ensure all tests pass
- [ ] Review SculptorExecutor for checkpointing compatibility with your models
- [ ] Run invariants microbenchmark on representative GPU hardware
- [ ] Confirm README and documentation accuracy
- [ ] Add tests for new features

## 📄 License

This project is experimental and intended for research purposes.

## 🙏 Acknowledgments

- PyTorch team for torch.fx symbolic tracing
- NVIDIA for NVML and hardware telemetry capabilities
- The deep learning community for inspiration

## 📮 Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**Experimental Framework** - This is research software. Use in production at your own risk.
