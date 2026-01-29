# Contributing to KineticStack

Thank you for your interest in contributing to KineticStack! This guide will help you get started.

## Development Setup

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.0.0
- Triton >= 2.0.0
- NVIDIA GPU with compute capability >= 8.0 (recommended for testing)

### Installation

```bash
# Clone the repository
git clone https://github.com/sachinecin/Self-Architecting-Deep-Learning-Stacks-with-Agentic-AI-Agentic-Software-Engineering-ASE-.git
cd Self-Architecting-Deep-Learning-Stacks-with-Agentic-AI-Agentic-Software-Engineering-ASE-

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Project Structure

```
KineticStack/
├── src/kineticstack/          # Main package
│   ├── core/                  # Core modules
│   │   ├── autograd_sculpting.py
│   │   ├── kernel_synthesis.py
│   │   └── runtime.py
│   └── agents/                # Agent implementations
│       ├── sculptor.py
│       ├── kernel_synthesizer.py
│       └── invariant_guard.py
├── examples/                  # Example scripts
├── tests/                     # Test suite
├── README.md                  # User documentation
├── TECHNICAL.md              # Technical documentation
└── setup.py                   # Package configuration
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow the existing code style
- Add docstrings to all public functions/classes
- Keep changes focused and minimal

### 3. Test Your Changes

```bash
# Run structure validation
python tests/test_structure.py

# Run full tests (requires PyTorch)
python tests/test_kineticstack.py

# Test examples
python examples/basic_usage.py
```

### 4. Commit Changes

```bash
git add .
git commit -m "Brief description of changes"
```

### 5. Submit Pull Request

- Push your branch to GitHub
- Create a pull request with a clear description
- Link any related issues

## Code Style Guidelines

### Python Style

- Follow PEP 8 conventions
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use descriptive variable names

### Docstring Format

```python
def function_name(arg1: type1, arg2: type2) -> return_type:
    """
    Brief description of function.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
    """
    pass
```

### Class Documentation

```python
class ClassName:
    """
    Brief description of class.
    
    This class provides functionality for...
    
    Attributes:
        attr1: Description of attr1
        attr2: Description of attr2
    """
    pass
```

## Adding New Features

### Adding a New Agent

1. Create a new file in `src/kineticstack/agents/`
2. Implement the agent class with clear interface
3. Add to `__init__.py` exports
4. Update `KineticRuntime` to integrate the agent
5. Add tests in `tests/`
6. Update documentation

Example structure:

```python
class NewAgent:
    """Description of agent."""
    
    def __init__(self, config_param: type):
        """Initialize the agent."""
        pass
    
    def main_method(self, inputs: type) -> type:
        """Main functionality."""
        pass
    
    def get_telemetry(self) -> Dict:
        """Return agent telemetry."""
        pass
```

### Adding a New Core Module

1. Create a new file in `src/kineticstack/core/`
2. Implement the module with clear API
3. Add comprehensive docstrings
4. Integrate with existing agents/runtime
5. Add tests and examples
6. Update technical documentation

### Adding Hardware Support

To add a new hardware profile:

```python
from kineticstack.core.kernel_synthesis import HardwareProfile

# In kernel_synthesis.py
@classmethod
def new_gpu_profile(cls) -> 'HardwareProfile':
    """Return hardware profile for NEW GPU."""
    return cls(
        register_count=xxxxx,
        shared_memory_kb=xxx,
        compute_capability=(x, x),
        warp_size=32,
        max_threads_per_block=1024,
    )
```

## Testing Guidelines

### Unit Tests

- Test each component in isolation
- Mock external dependencies
- Cover edge cases and error paths
- Use descriptive test names

### Integration Tests

- Test component interactions
- Test complete optimization pipeline
- Validate outputs and telemetry
- Test with different configurations

### Example Test

```python
def test_feature():
    """Test description."""
    # Setup
    component = Component(config)
    
    # Execute
    result = component.method(input)
    
    # Verify
    assert result.property == expected_value
    assert result.is_valid()
```

## Documentation Guidelines

### README.md

- User-facing documentation
- Quick start guide
- Common usage patterns
- Installation instructions

### TECHNICAL.md

- Architecture details
- Implementation specifics
- Algorithm descriptions
- Performance characteristics

### Code Comments

- Explain "why", not "what"
- Document complex algorithms
- Add references for research papers
- Keep comments up-to-date

## Performance Considerations

### Optimization Checklist

- [ ] Minimize memory allocations
- [ ] Cache computed results
- [ ] Use efficient data structures
- [ ] Profile critical paths
- [ ] Benchmark against baseline

### Profiling

```python
import time

start = time.time()
# Your code here
elapsed = time.time() - start
print(f"Elapsed: {elapsed*1000:.2f}ms")
```

## Common Tasks

### Adding a New Kernel Template

1. Add generation method to `KernelSynthesizer`
2. Implement Triton kernel code
3. Add LLVM IR generation
4. Register in `synthesize()` method
5. Add tests and benchmarks

### Adding a New Validation Rule

1. Add check function to `InvariantGuard`
2. Register rule in `_register_default_rules()`
3. Set appropriate severity level
4. Add test cases
5. Document the rule

### Improving Memory Analysis

1. Modify `analyze_memory_pressure()` in `AutogradSculptor`
2. Update heuristics for operation types
3. Add new operation patterns
4. Validate with real models
5. Update documentation

## Release Process

1. Update version in `setup.py` and `__init__.py`
2. Update CHANGELOG.md with changes
3. Run full test suite
4. Build package: `python setup.py sdist bdist_wheel`
5. Tag release: `git tag v0.x.0`
6. Push to PyPI (if applicable)

## Getting Help

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues and documentation
- Review examples for usage patterns

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Follow project guidelines

## License

By contributing to KineticStack, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Thank you for contributing to advancing deep learning optimization!
