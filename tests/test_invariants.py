"""Tests for InvariantVerifier with p99 latency checking."""

import pytest
from kineticstack.core.invariants import InvariantVerifier
import time


def test_invariant_verifier_initialization() -> None:
    """Test InvariantVerifier initialization."""
    verifier = InvariantVerifier(
        p99_tolerance=1.15,
        num_warmup=5,
        num_trials=50,
    )

    assert verifier.p99_tolerance == 1.15
    assert verifier.num_warmup == 5
    assert verifier.num_trials == 50


def test_measure_latency() -> None:
    """Test latency measurement."""
    verifier = InvariantVerifier(num_warmup=2, num_trials=10)

    def test_fn() -> None:
        time.sleep(0.001)  # 1ms

    p99 = verifier.measure_latency(test_fn)

    # Should be close to 1ms, allow for overhead
    assert 0.0005 < p99 < 0.01  # Between 0.5ms and 10ms


def test_verify_passes_with_similar_performance() -> None:
    """Test verification passes when performance is similar."""
    verifier = InvariantVerifier(
        p99_tolerance=1.15,
        num_warmup=2,
        num_trials=10,
    )

    def baseline_fn() -> None:
        time.sleep(0.001)

    def optimized_fn() -> None:
        time.sleep(0.001)

    result = verifier.verify(baseline_fn, optimized_fn)
    assert result is True


def test_verify_passes_with_faster_optimized() -> None:
    """Test verification passes when optimized is faster."""
    verifier = InvariantVerifier(
        p99_tolerance=1.15,
        num_warmup=2,
        num_trials=10,
    )

    def baseline_fn() -> None:
        time.sleep(0.002)

    def optimized_fn() -> None:
        time.sleep(0.001)

    result = verifier.verify(baseline_fn, optimized_fn)
    assert result is True


def test_verify_fails_with_slow_optimized() -> None:
    """Test verification fails when optimized is too slow."""
    verifier = InvariantVerifier(
        p99_tolerance=1.15,
        num_warmup=2,
        num_trials=10,
    )

    def baseline_fn() -> None:
        time.sleep(0.001)

    def optimized_fn() -> None:
        time.sleep(0.003)  # 3x slower

    result = verifier.verify(baseline_fn, optimized_fn)
    assert result is False


def test_verify_edge_case_tolerance() -> None:
    """Test verification with edge case near tolerance boundary."""
    verifier = InvariantVerifier(
        p99_tolerance=1.20,
        num_warmup=2,
        num_trials=10,
    )

    def baseline_fn() -> None:
        time.sleep(0.001)

    def optimized_fn() -> None:
        # Slightly slower but within tolerance
        time.sleep(0.0011)

    result = verifier.verify(baseline_fn, optimized_fn)
    # Should pass since it's within 20% tolerance
    assert result is True


def test_get_report() -> None:
    """Test generating verification report."""
    verifier = InvariantVerifier(p99_tolerance=1.15)

    baseline_p99 = 0.001  # 1ms
    optimized_p99 = 0.0011  # 1.1ms

    report = verifier.get_report(baseline_p99, optimized_p99)

    assert report["baseline_p99_ms"] == 1.0
    assert report["optimized_p99_ms"] == 1.1
    assert 1.0 < report["ratio"] < 1.2
    assert report["tolerance"] == 1.15
    assert report["passed"] is True
    assert report["speedup"] > 0


def test_get_report_fails() -> None:
    """Test report when verification fails."""
    verifier = InvariantVerifier(p99_tolerance=1.15)

    baseline_p99 = 0.001  # 1ms
    optimized_p99 = 0.002  # 2ms (2x slower)

    report = verifier.get_report(baseline_p99, optimized_p99)

    assert report["ratio"] > 1.15
    assert report["passed"] is False


def test_percentile_calculation() -> None:
    """Test internal percentile calculation."""
    verifier = InvariantVerifier()

    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    p50 = verifier._calculate_percentile(data, 50)
    p99 = verifier._calculate_percentile(data, 99)

    assert 4.0 <= p50 <= 6.0
    assert 9.0 <= p99 <= 10.0


def test_percentile_empty_data() -> None:
    """Test percentile calculation with empty data."""
    verifier = InvariantVerifier()

    result = verifier._calculate_percentile([], 99)
    assert result == 0.0


def test_measure_latency_no_sleep() -> None:
    """Test measuring latency of fast functions."""
    verifier = InvariantVerifier(num_warmup=2, num_trials=10)

    def fast_fn() -> int:
        return sum(range(100))

    p99 = verifier.measure_latency(fast_fn)

    # Should be very fast but still measurable
    assert p99 > 0
    assert p99 < 0.01  # Less than 10ms
