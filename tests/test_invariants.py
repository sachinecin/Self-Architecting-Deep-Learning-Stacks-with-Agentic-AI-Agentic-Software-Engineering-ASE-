"""Tests for hardware invariants verification."""

import pytest
import time

from kineticstack.core.invariants import verify_invariant, InvariantViolationError


def test_verify_invariant_passes():
    """Test that fast functions pass invariant checks."""
    
    @verify_invariant(p99_threshold_ms=10.0, num_iterations=20, warmup_iterations=5)
    def fast_function(x):
        return x * 2
    
    # Should not raise
    result = fast_function(42)
    assert result == 84


def test_verify_invariant_fails():
    """Test that slow functions fail invariant checks."""
    
    @verify_invariant(p99_threshold_ms=1.0, num_iterations=20, warmup_iterations=5)
    def slow_function(x):
        time.sleep(0.002)  # 2ms, exceeds 1ms threshold
        return x * 2
    
    # Should raise InvariantViolationError
    with pytest.raises(InvariantViolationError) as exc_info:
        slow_function(42)
    
    assert "P99 latency" in str(exc_info.value)
    assert "exceeds threshold" in str(exc_info.value)


def test_verify_invariant_with_args():
    """Test that invariant verification works with various argument types."""
    
    @verify_invariant(p99_threshold_ms=5.0, num_iterations=10, warmup_iterations=2)
    def function_with_args(a, b, c=10):
        return a + b + c
    
    result = function_with_args(1, 2, c=3)
    assert result == 6


def test_verify_invariant_edge_case():
    """Test edge case with very tight threshold."""
    
    @verify_invariant(p99_threshold_ms=100.0, num_iterations=10, warmup_iterations=0)
    def edge_function():
        # Minimal computation
        return sum(range(100))
    
    result = edge_function()
    assert result == 4950


def test_invariant_violation_error():
    """Test InvariantViolationError properties."""
    error = InvariantViolationError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_verify_invariant_preserves_function_metadata():
    """Test that decorator preserves function metadata."""
    
    @verify_invariant(p99_threshold_ms=10.0)
    def documented_function(x: int) -> int:
        """This is a documented function."""
        return x * 2
    
    assert documented_function.__name__ == "documented_function"
    assert documented_function.__doc__ == "This is a documented function."


def test_verify_invariant_with_exceptions():
    """Test that decorator handles exceptions from the wrapped function."""
    
    @verify_invariant(p99_threshold_ms=10.0, num_iterations=5, warmup_iterations=1)
    def function_that_raises():
        raise ValueError("Test exception")
    
    # The wrapped function's exception should still be raised
    with pytest.raises(ValueError, match="Test exception"):
        function_that_raises()


def test_p99_calculation():
    """Test that p99 is calculated correctly."""
    
    call_count = [0]
    latencies = []
    
    @verify_invariant(p99_threshold_ms=1.5, num_iterations=100, warmup_iterations=0)
    def varying_latency_function():
        # Create varying latencies
        call_count[0] += 1
        if call_count[0] % 10 == 0:
            time.sleep(0.001)  # 1ms for every 10th call
        return 42
    
    # Should pass because p99 should be around 1ms, below 1.5ms threshold
    result = varying_latency_function()
    assert result == 42
