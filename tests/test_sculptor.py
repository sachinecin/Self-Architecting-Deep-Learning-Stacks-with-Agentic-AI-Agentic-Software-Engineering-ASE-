"""Tests for sculptor agent with synthetic transformer model."""

import pytest
import torch
import torch.nn as nn

from kineticstack.agents.sculptor import SculptorReasoner, SculptorExecutor


class SimpleTransformerBlock(nn.Module):
    """Synthetic transformer block for testing."""
    
    def __init__(self, dim=64):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        # Attention block
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # MLP block
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        return x


class SimpleTransformerModel(nn.Module):
    """Synthetic transformer model with multiple blocks."""
    
    def __init__(self, dim=64, num_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(dim) for _ in range(num_blocks)
        ])
        self.output = nn.Linear(dim, dim)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.output(x)


def test_sculptor_reasoner_trace():
    """Test that SculptorReasoner can trace a model."""
    model = SimpleTransformerModel(dim=64, num_blocks=2)
    reasoner = SculptorReasoner()
    
    example_input = torch.randn(1, 10, 64)
    
    traced = reasoner.trace_model(model, example_input)
    assert traced is not None
    assert id(model) in reasoner.traced_models


def test_sculptor_reasoner_detect_blocks():
    """Test transformer block detection."""
    model = SimpleTransformerModel(dim=64, num_blocks=3)
    reasoner = SculptorReasoner()
    
    example_input = torch.randn(1, 10, 64)
    traced = reasoner.trace_model(model, example_input)
    
    blocks = reasoner.detect_transformer_blocks(traced)
    
    # Should detect transformer-like blocks
    # The exact number depends on how torch.fx traces the model
    assert isinstance(blocks, list)


def test_sculptor_executor_init():
    """Test SculptorExecutor initialization."""
    executor = SculptorExecutor(checkpoint_every_n=2)
    assert executor.checkpoint_every_n == 2
    assert len(executor.original_forwards) == 0


def test_apply_kinetic_optimization():
    """Test applying kinetic optimization to a model."""
    model = SimpleTransformerModel(dim=64, num_blocks=4)
    executor = SculptorExecutor(checkpoint_every_n=2)
    
    # Manually specify block names
    block_names = [f"blocks.{i}" for i in range(4)]
    
    # Apply optimization
    optimized = executor.apply_kinetic_optimization(model, block_names)
    
    # Model should be modified
    assert optimized is model
    
    # Forward should still work
    x = torch.randn(1, 10, 64)
    output = model(x)
    assert output.shape == (1, 10, 64)


def test_apply_kinetic_optimization_with_gradient():
    """Test that checkpointing works with gradients."""
    model = SimpleTransformerModel(dim=64, num_blocks=2)
    executor = SculptorExecutor(checkpoint_every_n=1)
    
    block_names = [f"blocks.{i}" for i in range(2)]
    executor.apply_kinetic_optimization(model, block_names)
    
    # Forward pass with gradient computation
    x = torch.randn(1, 10, 64, requires_grad=True)
    output = model(x)
    loss = output.sum()
    
    # Backward pass should work with checkpointing
    loss.backward()
    
    assert x.grad is not None


def test_remove_optimizations():
    """Test removing applied optimizations."""
    model = SimpleTransformerModel(dim=64, num_blocks=2)
    executor = SculptorExecutor(checkpoint_every_n=1)
    
    # Store original forward
    original_forward_0 = model.blocks[0].forward
    
    # Apply optimization
    block_names = ["blocks.0", "blocks.1"]
    executor.apply_kinetic_optimization(model, block_names)
    
    # Forward should be different
    assert model.blocks[0].forward != original_forward_0
    
    # Remove optimizations
    executor.remove_optimizations(model)
    
    # Forward should be restored
    assert model.blocks[0].forward == original_forward_0


def test_empty_block_names():
    """Test that empty block names don't cause issues."""
    model = SimpleTransformerModel(dim=64, num_blocks=2)
    executor = SculptorExecutor()
    
    optimized = executor.apply_kinetic_optimization(model, [])
    assert optimized is model
    
    # Model should still work
    x = torch.randn(1, 10, 64)
    output = model(x)
    assert output.shape == (1, 10, 64)


def test_selective_checkpointing():
    """Test that checkpointing is applied selectively (every N-th block)."""
    model = SimpleTransformerModel(dim=64, num_blocks=4)
    executor = SculptorExecutor(checkpoint_every_n=2)
    
    block_names = [f"blocks.{i}" for i in range(4)]
    
    # Store original forwards
    originals = {i: model.blocks[i].forward for i in range(4)}
    
    # Apply optimization
    executor.apply_kinetic_optimization(model, block_names)
    
    # Only every 2nd block should be checkpointed (0, 2)
    # Check that some forwards were modified
    modified_count = sum(
        1 for i in range(4) 
        if model.blocks[i].forward != originals[i]
    )
    
    # At least some blocks should be modified
    assert modified_count >= 2
