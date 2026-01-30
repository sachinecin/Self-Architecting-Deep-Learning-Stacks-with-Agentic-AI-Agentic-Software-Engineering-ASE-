"""
Invariant verification through p99 latency microbenchmarking.

This module verifies that optimizations do not degrade performance by measuring
p99 latency before and after applying transformations. Must be run on representative
GPU hardware before production deployment.
"""

from typing import List, Optional, Callable, Any, Dict
import time
import logging
import statistics

logger = logging.getLogger(__name__)


class InvariantVerifier:
    """
    Verifies performance invariants through microbenchmarking.

    Ensures that optimizations like activation checkpointing do not
    introduce unacceptable performance regressions by comparing p99
    latency measurements.
    """

    def __init__(
        self,
        p99_tolerance: float = 1.15,
        num_warmup: int = 10,
        num_trials: int = 100,
    ) -> None:
        """
        Initialize the invariant verifier.

        Args:
            p99_tolerance: Maximum allowed p99 latency multiplier (e.g., 1.15 = 15% slower)
            num_warmup: Number of warmup iterations before measurement
            num_trials: Number of measurement trials for p99 calculation
        """
        self.p99_tolerance = p99_tolerance
        self.num_warmup = num_warmup
        self.num_trials = num_trials
        logger.info(
            f"InvariantVerifier initialized with p99_tolerance={p99_tolerance}, "
            f"warmup={num_warmup}, trials={num_trials}"
        )

    def measure_latency(self, fn: Callable[[], Any]) -> float:
        """
        Measure p99 latency of a callable.

        Args:
            fn: Function to benchmark

        Returns:
            p99 latency in seconds
        """
        # Warmup phase
        for _ in range(self.num_warmup):
            fn()

        # Measurement phase
        latencies: List[float] = []
        for _ in range(self.num_trials):
            start = time.perf_counter()
            fn()
            end = time.perf_counter()
            latencies.append(end - start)

        p99 = self._calculate_percentile(latencies, 99)
        logger.debug(f"Measured p99 latency: {p99*1000:.2f}ms")
        return p99

    def verify(
        self,
        baseline_fn: Callable[[], Any],
        optimized_fn: Callable[[], Any],
    ) -> bool:
        """
        Verify that optimized function meets p99 invariant.

        Args:
            baseline_fn: Original unoptimized function
            optimized_fn: Optimized function to verify

        Returns:
            True if invariant is satisfied, False otherwise
        """
        logger.info("Starting invariant verification...")

        baseline_p99 = self.measure_latency(baseline_fn)
        optimized_p99 = self.measure_latency(optimized_fn)

        ratio = optimized_p99 / baseline_p99 if baseline_p99 > 0 else float("inf")

        passed = ratio <= self.p99_tolerance

        log_msg = (
            f"Invariant verification: baseline_p99={baseline_p99*1000:.2f}ms, "
            f"optimized_p99={optimized_p99*1000:.2f}ms, "
            f"ratio={ratio:.3f}, tolerance={self.p99_tolerance}, "
            f"result={'PASS' if passed else 'FAIL'}"
        )

        if passed:
            logger.info(log_msg)
        else:
            logger.warning(log_msg)

        return passed

    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile from a list of values."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * (percentile / 100.0))
        return sorted_data[min(index, len(sorted_data) - 1)]

    def get_report(
        self, baseline_p99: float, optimized_p99: float
    ) -> Dict[str, Any]:
        """
        Generate a verification report.

        Args:
            baseline_p99: Baseline p99 latency
            optimized_p99: Optimized p99 latency

        Returns:
            Dictionary containing verification metrics
        """
        ratio = optimized_p99 / baseline_p99 if baseline_p99 > 0 else float("inf")
        passed = ratio <= self.p99_tolerance

        return {
            "baseline_p99_ms": baseline_p99 * 1000,
            "optimized_p99_ms": optimized_p99 * 1000,
            "ratio": ratio,
            "tolerance": self.p99_tolerance,
            "passed": passed,
            "speedup": 1.0 / ratio if ratio > 0 else 0.0,
        }
