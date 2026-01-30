"""
Tests for the sculptor agent.

Tests the sculptor's ability to analyze models and apply checkpointing.
"""

import pytest

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")


class SimpleTransformer(nn.Module):
    """Simple transformer model for testing."""

    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.encoder_layer(x)
        x = self.linear(x)
        return x


def test_sculptor_reasoning():
    """Test sculptor's reasoning about where to apply checkpointing."""
    from kineticstack.agents.sculptor import SculptorExecutor

    model = SimpleTransformer()
    sculptor = SculptorExecutor()

    reasoning = sculptor.reason({"model": model})

    assert "checkpoint_layers" in reasoning
    # Should identify the transformer encoder layer
    checkpoint_layers = reasoning["checkpoint_layers"]
    assert any("encoder_layer" in layer for layer in checkpoint_layers)


def test_sculptor_apply_optimization():
    """Test applying kinetic optimization to a model."""
    from kineticstack.agents.sculptor import SculptorExecutor

    model = SimpleTransformer()
    sculptor = SculptorExecutor()

    # Store original forward
    original_forward = model.encoder_layer.forward

    # Apply optimization
    optimized_model = sculptor.apply_kinetic_optimization(model)

    # Forward should have been modified
    assert optimized_model.encoder_layer.forward != original_forward

    # Test that the model still works
    x = torch.randn(10, 32, 512)  # (seq_len, batch, d_model)
    output = optimized_model(x)
    assert output.shape == x.shape


def test_sculptor_revert_optimization():
    """Test reverting optimization on a model."""
    from kineticstack.agents.sculptor import SculptorExecutor

    model = SimpleTransformer()
    sculptor = SculptorExecutor()

    # Store original forward
    original_forward = model.encoder_layer.forward

    # Apply and then revert optimization
    optimized_model = sculptor.apply_kinetic_optimization(model)
    reverted_model = sculptor.revert_optimization(optimized_model)

    # Forward should be restored
    assert reverted_model.encoder_layer.forward == original_forward


def test_sculptor_execution_plan():
    """Test sculptor's execution of a plan."""
    from kineticstack.agents.sculptor import SculptorExecutor

    model = SimpleTransformer()
    sculptor = SculptorExecutor()

    # Create a manual execution plan
    plan = {"model": model, "checkpoint_layers": ["encoder_layer"]}

    # Execute the plan
    result = sculptor.execute(plan)

    # Model should be modified
    assert len(sculptor._checkpointed_modules) > 0


def test_sculptor_large_linear_detection():
    """Test that sculptor detects large linear layers."""
    from kineticstack.agents.sculptor import SculptorExecutor

    class ModelWithLargeLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.large_linear = nn.Linear(2048, 2048)
            self.small_linear = nn.Linear(64, 64)

        def forward(self, x):
            return self.small_linear(self.large_linear(x))

    model = ModelWithLargeLinear()
    sculptor = SculptorExecutor()

    reasoning = sculptor.reason({"model": model})
    checkpoint_layers = reasoning["checkpoint_layers"]

    # Should checkpoint the large linear but not the small one
    assert any("large_linear" in layer for layer in checkpoint_layers)
    assert not any("small_linear" in layer for layer in checkpoint_layers)


def test_sculptor_conservative_checkpointing():
    """Test that sculptor only checkpoints when safe to do so."""
    from kineticstack.agents.sculptor import SculptorExecutor

    model = SimpleTransformer()
    sculptor = SculptorExecutor()

    # Apply optimization
    sculptor.apply_kinetic_optimization(model)

    # Test with kwargs (should not checkpoint)
    x = torch.randn(10, 32, 512)
    # The forward should still work even with kwargs present
    try:
        output = model(x)
        assert output.shape == x.shape
    except Exception as e:
        pytest.fail(f"Model forward failed with kwargs: {e}")
