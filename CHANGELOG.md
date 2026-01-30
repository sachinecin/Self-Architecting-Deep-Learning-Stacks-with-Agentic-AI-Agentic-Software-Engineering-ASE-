# Changelog

All notable changes to KineticStack will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-30

### Added
- Initial KineticStack project structure
- Core telemetry collection with NVML integration and mock mode for CI
- SculptorExecutor agent for selective activation checkpointing
- InvariantVerifier for p99 latency microbenchmarking
- StackManager for coordinating telemetry-driven optimization loop
- TelemetryLogger for structured logging
- Comprehensive test suite with mocked NVML for CPU-only CI
- CI/CD pipeline with Python 3.9-3.11 support
- Pre-commit hooks for code quality (black, ruff, mypy)
- Poetry-based dependency management with conservative version pins

### Architecture
- Telemetry-driven agentic loop that applies selective activation checkpointing
- Triggers on sustained HBM >85% for >5s
- Protected by p99 invariant verification (15% tolerance by default)
- torch.fx-based graph transformation (placeholder implementation)

### Testing
- All tests use mocked NVML for CI on CPU runners
- Invariants microbenchmark gated in CI
- Must be validated on representative GPU hardware before production use
