"""
Hardware invariants verification.

Provides @verify_invariant decorator that measures p99 latency over iterations
and raises InvariantViolationError if it exceeds a threshold.
"""

import time
import functools
from typing import Callable, List
import warnings


class InvariantViolationError(Exception):
    """Raised when a hardware invariant is violated."""
    pass


def verify_invariant(
    p99_threshold_ms: float,
    num_iterations: int = 100,
    warmup_iterations: int = 10
):
    """
    Decorator that verifies p99 latency doesn't exceed a threshold.
    
    Args:
        p99_threshold_ms: Maximum allowed p99 latency in milliseconds
        num_iterations: Number of iterations to measure for p99 calculation
        warmup_iterations: Number of warmup iterations to skip
        
    Raises:
        InvariantViolationError: If p99 latency exceeds the threshold
        
    Example:
        @verify_invariant(p99_threshold_ms=100.0, num_iterations=50)
        def my_function(x):
            return x * 2
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            latencies: List[float] = []
            
            # Warmup iterations
            for _ in range(warmup_iterations):
                func(*args, **kwargs)
            
            # Measurement iterations
            for _ in range(num_iterations):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                latencies.append((end - start) * 1000.0)  # Convert to ms
            
            # Calculate p99
            latencies.sort()
            p99_index = int(num_iterations * 0.99)
            if p99_index >= len(latencies):
                p99_index = len(latencies) - 1
            p99_latency = latencies[p99_index]
            
            # Check threshold
            if p99_latency > p99_threshold_ms:
                raise InvariantViolationError(
                    f"P99 latency {p99_latency:.2f}ms exceeds threshold {p99_threshold_ms}ms"
                )
            
            # Log success
            warnings.warn(
                f"Invariant verified: p99={p99_latency:.2f}ms <= {p99_threshold_ms}ms",
                UserWarning
            )
            
            return result
        
        return wrapper
    return decorator
