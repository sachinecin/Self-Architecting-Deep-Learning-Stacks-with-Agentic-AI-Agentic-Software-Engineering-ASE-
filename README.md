# KineticStack

[![Python CI](https://github.com/sachinecin/Self-Architecting-Deep-Learning-Stacks-with-Agentic-AI-Agentic-Software-Engineering-ASE-/actions/workflows/python-ci.yml/badge.svg)](https://github.com/sachinecin/Self-Architecting-Deep-Learning-Stacks-with-Agentic-AI-Agentic-Software-Engineering-ASE-/actions/workflows/python-ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20manager-poetry-blue.svg)](https://python-poetry.org/)

**KineticStack** is an experimental Agentic Software Engineering (ASE) framework that treats the deep learning stack as a living organism. It replaces static compilation with telemetry-driven agentic reasoning loops that continuously refactor Autograd graphs and apply selective optimizations based on real-time hardware metrics.

## 🏗️ Architecture

![KineticStack Architecture](assets/architecture.svg)

### Core Components

1. **TelemetryCollector** (`kineticstack.core.telemetry`)
   - Real-time GPU metrics via NVML (pynvml)
   - Monitors HBM utilization, temperature, power consumption
   - Mock mode for CI/testing on CPU-only runners

2. **SculptorExecutor** (`kineticstack.agents.sculptor`)
   - Agentic optimization agent
   - Applies selective activation checkpointing using torch.fx
   - Triggers when HBM >85% for >5s sustained

3. **InvariantVerifier** (`kineticstack.core.invariants`)
   - P99 latency microbenchmarking
   - Ensures optimizations don't degrade performance >15%
   - **Must run on representative GPU hardware before production**

4. **StackManager** (`kineticstack.core.manager`)
   - Orchestrates the telemetry → decision → optimization loop
   - Coordinates telemetry, sculptor, and invariant verification

### Telemetry-Driven Optimization Loop

```
┌──────────────────┐
│ TelemetryCollect │
│  (NVML metrics)  │
└────────┬─────────┘
         │ HBM, temp, power
         ▼
┌──────────────────┐
│  StackManager    │◄─────┐
│ (decision logic) │      │
└────────┬─────────┘      │
         │                 │
         │ HBM >85% for >5s│
         ▼                 │
┌──────────────────┐      │
│ SculptorExecutor │      │
│ (torch.fx xform) │      │
└────────┬─────────┘      │
         │                 │
         ▼                 │
┌──────────────────┐      │
│InvariantVerifier │──────┘
│ (p99 check <15%) │ Pass/Fail
└──────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/sachinecin/Self-Architecting-Deep-Learning-Stacks-with-Agentic-AI-Agentic-Software-Engineering-ASE-.git
cd Self-Architecting-Deep-Learning-Stacks-with-Agentic-AI-Agentic-Software-Engineering-ASE-

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Basic Usage

```python
from kineticstack import StackManager, TelemetryCollector, SculptorExecutor
from kineticstack.core.invariants import InvariantVerifier

# Initialize components
telemetry = TelemetryCollector(mock_mode=False)  # Use True for testing
sculptor = SculptorExecutor(hbm_threshold=0.85, sustained_duration=5.0)
verifier = InvariantVerifier(p99_tolerance=1.15)
manager = StackManager(telemetry_interval=1.0)

# Start the optimization loop
manager.start()

# In your training loop:
metrics = telemetry.collect()
if sculptor.execute(metrics, context={"model": your_model}):
    print("Activation checkpointing applied")

# Shutdown
manager.stop()
telemetry.shutdown()
```

### Running Tests

```bash
# Run all tests with coverage
poetry run pytest tests/ -v --cov=kineticstack --cov-report=term-missing

# Run specific test modules
poetry run pytest tests/test_telemetry.py -v
poetry run pytest tests/test_sculptor.py -v
poetry run pytest tests/test_invariants.py -v
```

**Note:** Tests use mocked NVML so they can run on CPU-only CI runners.

### Code Quality

```bash
# Format code
poetry run black kineticstack tests

# Lint with ruff
poetry run ruff check kineticstack tests --fix

# Type checking
poetry run mypy kineticstack

# Pre-commit hooks (runs all checks)
poetry run pre-commit run --all-files
```

## 📊 Generating Architecture Diagram

The architecture diagram can be regenerated using any tool that converts SVG. Here's an example using Python:

```python
# Generate architecture.svg (requires graphviz or similar)
# This is a placeholder - actual diagram should be created with your preferred tool

# Example using diagrams library:
# pip install diagrams
from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom

with Diagram("KineticStack Architecture", filename="assets/architecture", show=False):
    # Add your architecture components here
    pass
```

Or use online tools like [draw.io](https://draw.io), [Excalidraw](https://excalidraw.com/), or similar.

## 🔬 Compatibility Requirements

### SculptorExecutor Requirements

- **PyTorch**: >=2.1,<3.0 (for torch.fx support)
- **Triton**: >=2.0,<3.0 (for kernel synthesis)
- **CUDA**: Recommended CUDA 11.8+ or 12.x
- **Model constraints**:
  - Model must be traceable with torch.fx
  - No dynamic control flow in forward pass
  - Activation checkpointing applied at module boundaries

### Invariants Microbenchmarking

⚠️ **IMPORTANT**: The invariants microbenchmark **MUST** be run on representative GPU hardware before enabling optimizations in production.

- Default p99 tolerance: 15% (configurable)
- Requires GPU with similar specs to production
- CI tests are gated - label PR with `run-gpu-tests` to enable

## 🧪 Development

### Project Structure

```
kineticstack/
├── agents/
│   ├── __init__.py
│   ├── base.py          # BaseAgent interface
│   └── sculptor.py      # SculptorExecutor implementation
├── core/
│   ├── __init__.py
│   ├── manager.py       # StackManager orchestration
│   ├── telemetry.py     # TelemetryCollector with NVML
│   └── invariants.py    # InvariantVerifier p99 checks
├── telemetry/
│   ├── __init__.py
│   └── logger.py        # Structured telemetry logging
└── __init__.py

tests/
├── test_placeholder.py  # Basic import tests
├── test_telemetry.py    # Telemetry + logging tests (mocked NVML)
├── test_sculptor.py     # SculptorExecutor tests
└── test_invariants.py   # InvariantVerifier tests

.github/
└── workflows/
    └── python-ci.yml    # CI/CD pipeline
```

### Adding New Agents

1. Subclass `BaseAgent` in `kineticstack/agents/`
2. Implement `execute()` method
3. Override `should_activate()` for custom triggers
4. Add tests in `tests/test_<agent_name>.py`
5. Register in `kineticstack/agents/__init__.py`

Example:

```python
from kineticstack.agents.base import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="MyCustomAgent")

    def should_activate(self, telemetry):
        return telemetry.get("custom_metric", 0) > threshold

    def execute(self, telemetry, context=None):
        if not self.should_activate(telemetry):
            return False

        # Apply your optimization
        return True
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`poetry run pytest && poetry run ruff check`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

This project is experimental and provided as-is for research purposes.

## 🙏 Acknowledgments

Built on PyTorch, Triton, and NVML for telemetry-driven deep learning stack optimization.

---

**Status**: Experimental - Research prototype for Agentic Software Engineering (ASE) exploration.
