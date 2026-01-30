# Changelog

All notable changes to KineticStack will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-XX

### Added
- Initial KineticStack project structure
- TelemetryMonitor using pynvml for hardware metrics (HBM utilization, SM occupancy, power draw)
  - `stream()` method for continuous monitoring
  - `sample_once()` for single metric samples
  - `should_trigger_refactor()` logic that returns True when HBM >= 85% for sustained 5s window
- SculptorReasoner using torch.fx to detect transformer-like blocks
- SculptorExecutor for applying selective activation checkpointing
  - Every N-th block checkpointing using torch.utils.checkpoint
  - Conservative constraints: positional tensor-only args, no kwargs
  - Reversible optimizations with `remove_optimizations()` method
- @verify_invariant decorator for p99 latency verification
  - Measures p99 over configurable iterations
  - Raises InvariantViolationError if threshold exceeded
- KineticManager for orchestrating agents and telemetry
- TelemetryLogger for persisting metrics
- Comprehensive unit tests with mocked NVML for CI compatibility
- CI/CD pipeline with lint, test, and gated invariants jobs
- Pre-commit hooks for code quality
- Documentation including README and architecture diagram

### Technical Details
- Python package with Poetry for dependency management
- Conservative dependency pins: torch>=2.1,<3.0; triton>=2.0,<3.0; pynvml>=11.0
- Black, isort, flake8 for code formatting and linting
- pytest with coverage for testing
- GitHub Actions CI with lint, test, and gated invariants workflows

[0.1.0]: https://github.com/sachinecin/Self-Architecting-Deep-Learning-Stacks-with-Agentic-AI-Agentic-Software-Engineering-ASE-/releases/tag/v0.1.0
