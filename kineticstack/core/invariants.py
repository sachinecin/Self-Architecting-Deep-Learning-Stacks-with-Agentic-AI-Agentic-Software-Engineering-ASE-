"""
Invariants module for KineticStack.

Provides decorators and utilities for verifying performance invariants before
allowing model modifications to proceed.
"""

import functools
import time
from typing import Callable, Optional


class InvariantViolationError(Exception):
    """Raised when a performance invariant is violated."""

    pass


def verify_invariant(
    p99_threshold_ms: Optional[float] = None,
    num_warmup: int = 5,
    num_samples: int = 100,
):
    """
    Decorator that verifies a function meets p99 latency invariants.

    This decorator runs the function multiple times, measures latency, and ensures
    the p99 latency is below the specified threshold. If the threshold is exceeded,
    an InvariantViolationError is raised.

    Args:
        p99_threshold_ms: Maximum allowed p99 latency in milliseconds. If None, only
                         measures but doesn't enforce.
        num_warmup: Number of warmup iterations before measurement.
        num_samples: Number of samples to collect for p99 calculation.

    Raises:
        InvariantViolationError: If p99 latency exceeds threshold.

    Example:
        @verify_invariant(p99_threshold_ms=10.0)
        def my_function():
            # code here
            pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Warmup phase
            for _ in range(num_warmup):
                result = func(*args, **kwargs)

            # Measurement phase
            latencies = []
            for _ in range(num_samples):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms

            # Calculate p99
            latencies.sort()
            p99_index = int(0.99 * len(latencies))
            p99_latency = latencies[p99_index]

            # Check threshold
            if p99_threshold_ms is not None and p99_latency > p99_threshold_ms:
                raise InvariantViolationError(
                    f"p99 latency {p99_latency:.2f}ms exceeds threshold {p99_threshold_ms}ms"
                )

            return result

        return wrapper

    return decorator
