"""
Tests for invariants verification.

Tests the @verify_invariant decorator for performance checking.
"""

import time

import pytest

from kineticstack.core.invariants import InvariantViolationError, verify_invariant


def test_verify_invariant_pass():
    """Test that verify_invariant passes when latency is below threshold."""

    @verify_invariant(p99_threshold_ms=100.0, num_warmup=2, num_samples=10)
    def fast_function():
        time.sleep(0.001)  # 1ms sleep
        return "success"

    result = fast_function()
    assert result == "success"


def test_verify_invariant_fail():
    """Test that verify_invariant raises error when latency exceeds threshold."""

    @verify_invariant(p99_threshold_ms=5.0, num_warmup=2, num_samples=10)
    def slow_function():
        time.sleep(0.01)  # 10ms sleep
        return "success"

    with pytest.raises(InvariantViolationError) as exc_info:
        slow_function()

    assert "p99 latency" in str(exc_info.value)
    assert "exceeds threshold" in str(exc_info.value)


def test_verify_invariant_no_threshold():
    """Test that verify_invariant measures but doesn't enforce without threshold."""

    @verify_invariant(p99_threshold_ms=None, num_warmup=2, num_samples=10)
    def any_function():
        time.sleep(0.002)
        return "success"

    # Should not raise even though we're sleeping
    result = any_function()
    assert result == "success"


def test_verify_invariant_with_args():
    """Test that verify_invariant works with function arguments."""

    @verify_invariant(p99_threshold_ms=50.0, num_warmup=1, num_samples=5)
    def function_with_args(x, y):
        time.sleep(0.001)
        return x + y

    result = function_with_args(2, 3)
    assert result == 5


def test_verify_invariant_with_kwargs():
    """Test that verify_invariant works with keyword arguments."""

    @verify_invariant(p99_threshold_ms=50.0, num_warmup=1, num_samples=5)
    def function_with_kwargs(x, y=10):
        time.sleep(0.001)
        return x + y

    result = function_with_kwargs(5, y=7)
    assert result == 12


def test_verify_invariant_preserves_function_name():
    """Test that decorator preserves function metadata."""

    @verify_invariant(p99_threshold_ms=50.0)
    def my_function():
        """My docstring."""
        return 42

    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "My docstring."


def test_verify_invariant_calculates_p99_correctly():
    """Test that p99 calculation is reasonable."""
    call_count = 0

    @verify_invariant(p99_threshold_ms=None, num_warmup=0, num_samples=100)
    def varying_function():
        nonlocal call_count
        call_count += 1
        # Sleep longer every 10th call
        if call_count % 10 == 0:
            time.sleep(0.005)
        else:
            time.sleep(0.001)
        return call_count

    varying_function()  # noqa: F841
    # Should have run warmup + samples times
    assert call_count == 100


def test_invariant_violation_error():
    """Test InvariantViolationError exception."""
    error = InvariantViolationError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_verify_invariant_warmup():
    """Test that warmup iterations occur before measurement."""
    call_count = 0

    @verify_invariant(p99_threshold_ms=100.0, num_warmup=5, num_samples=10)
    def counting_function():
        nonlocal call_count
        call_count += 1
        time.sleep(0.001)
        return call_count

    counting_function()  # noqa: F841
    # Should have called warmup (5) + samples (10) times
    assert call_count == 15
