"""
Kernel Synthesis: Hardware-aware Triton kernel generation
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib


@dataclass
class HardwareProfile:
    """Hardware characteristics for kernel optimization"""
    register_count: int  # Available registers
    shared_memory_kb: int  # Shared memory in KB
    compute_capability: Tuple[int, int]  # e.g., (8, 0) for H100
    warp_size: int = 32
    max_threads_per_block: int = 1024
    
    @classmethod
    def h100_profile(cls) -> 'HardwareProfile':
        """Return hardware profile for NVIDIA H100"""
        return cls(
            register_count=65536,
            shared_memory_kb=228,  # 228KB shared memory per SM
            compute_capability=(9, 0),
            warp_size=32,
            max_threads_per_block=1024
        )
    
    @classmethod
    def a100_profile(cls) -> 'HardwareProfile':
        """Return hardware profile for NVIDIA A100"""
        return cls(
            register_count=65536,
            shared_memory_kb=164,  # 164KB shared memory per SM
            compute_capability=(8, 0),
            warp_size=32,
            max_threads_per_block=1024
        )


@dataclass
class KernelSpec:
    """Specification for a synthesized kernel"""
    name: str
    operation: str  # e.g., "matmul", "conv2d", "attention"
    input_shapes: List[Tuple[int, ...]]
    output_shape: Tuple[int, ...]
    triton_code: str
    register_pressure: int  # Estimated register usage
    block_size: Tuple[int, ...]  # Thread block dimensions


class KernelSynthesizer:
    """
    Hardware-aware Triton kernel generator.
    Synthesizes optimized kernels based on hardware characteristics and operation patterns.
    """
    
    def __init__(self, hardware_profile: Optional[HardwareProfile] = None):
        """
        Initialize the Kernel Synthesizer.
        
        Args:
            hardware_profile: Target hardware profile (defaults to H100)
        """
        self.hardware_profile = hardware_profile or HardwareProfile.h100_profile()
        self.kernel_cache: Dict[str, KernelSpec] = {}
        
    def estimate_register_pressure(self, operation: str, input_shapes: List[Tuple[int, ...]]) -> int:
        """
        Estimate register pressure for an operation.
        
        Args:
            operation: Type of operation
            input_shapes: Input tensor shapes
            
        Returns:
            Estimated number of registers needed
        """
        base_pressure = 32  # Baseline register usage
        
        # Adjust based on operation complexity
        if operation in ['matmul', 'linear']:
            # Matrix multiplication requires more registers for accumulation
            if input_shapes:
                inner_dim = input_shapes[0][-1] if len(input_shapes[0]) > 1 else 1
                base_pressure += min(inner_dim // 16, 64)
        elif operation == 'attention':
            # Attention mechanism is register-intensive
            base_pressure += 128
        elif operation in ['conv2d', 'conv3d']:
            # Convolutions need registers for filter values
            base_pressure += 64
        
        return min(base_pressure, self.hardware_profile.register_count // 32)
    
    def generate_matmul_kernel(self, M: int, N: int, K: int) -> str:
        """
        Generate a Triton kernel for matrix multiplication.
        
        Args:
            M, N, K: Matrix dimensions (M x K) @ (K x N) = (M x N)
            
        Returns:
            Triton kernel code as string
        """
        kernel_code = f'''
import triton
import triton.language as tl

@triton.jit
def matmul_kernel_{M}x{N}x{K}(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized matrix multiplication kernel for HBM3e."""
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Block IDs
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Pointers
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Inner loop
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        accumulator += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)
    
    # Write back
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
'''
        return kernel_code
    
    def generate_attention_kernel(self, seq_len: int, hidden_dim: int) -> str:
        """
        Generate a Triton kernel for scaled dot-product attention.
        
        Args:
            seq_len: Sequence length
            hidden_dim: Hidden dimension size
            
        Returns:
            Triton kernel code as string
        """
        kernel_code = f'''
import triton
import triton.language as tl

@triton.jit
def attention_kernel_{seq_len}x{hidden_dim}(
    Q, K, V, Out,
    seq_len, hidden_dim,
    stride_qm, stride_qk,
    stride_km, stride_kk,
    stride_vm, stride_vk,
    stride_om, stride_ok,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized attention kernel with register pressure management."""
    # Get position
    row_idx = tl.program_id(0)
    
    # Load Q row
    offs_k = tl.arange(0, BLOCK_SIZE)
    q_ptrs = Q + row_idx * stride_qm + offs_k * stride_qk
    q = tl.load(q_ptrs, mask=offs_k < hidden_dim, other=0.0)
    
    # Initialize output accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Compute attention scores
    for col_idx in range(0, seq_len, BLOCK_SIZE):
        offs_n = col_idx + tl.arange(0, BLOCK_SIZE)
        k_ptrs = K + offs_n[:, None] * stride_km + offs_k[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < seq_len) & (offs_k[None, :] < hidden_dim), other=0.0)
        
        # Compute scores with scaling
        qk = tl.sum(q[None, :] * k, axis=1)
        qk = qk / tl.sqrt(hidden_dim.to(tl.float32))
        
        # Apply softmax (simplified)
        scores = tl.exp(qk)
        
        # Load values and accumulate
        v_ptrs = V + offs_n[:, None] * stride_vm + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < seq_len) & (offs_k[None, :] < hidden_dim), other=0.0)
        
        acc += tl.sum(scores[:, None] * v, axis=0)
    
    # Write output
    out_ptrs = Out + row_idx * stride_om + offs_k * stride_ok
    tl.store(out_ptrs, acc, mask=offs_k < hidden_dim)
'''
        return kernel_code
    
    def synthesize(
        self,
        operation: str,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
    ) -> KernelSpec:
        """
        Synthesize a hardware-aware kernel for the given operation.
        
        Args:
            operation: Type of operation (e.g., "matmul", "attention")
            input_shapes: Input tensor shapes
            output_shape: Output tensor shape
            
        Returns:
            KernelSpec containing the generated kernel
        """
        # Create cache key
        cache_key = hashlib.md5(
            f"{operation}_{input_shapes}_{output_shape}".encode()
        ).hexdigest()
        
        if cache_key in self.kernel_cache:
            return self.kernel_cache[cache_key]
        
        # Estimate register pressure
        reg_pressure = self.estimate_register_pressure(operation, input_shapes)
        
        # Generate kernel based on operation type
        if operation in ['matmul', 'linear']:
            if len(input_shapes) >= 2:
                M, K = input_shapes[0][-2:] if len(input_shapes[0]) >= 2 else (1, input_shapes[0][-1])
                K2, N = input_shapes[1][-2:] if len(input_shapes[1]) >= 2 else (input_shapes[1][-1], 1)
                triton_code = self.generate_matmul_kernel(M, N, K)
                block_size = (128, 128, 32)  # Optimized for HBM3e
            else:
                triton_code = "# Placeholder for dynamic matmul"
                block_size = (128, 128, 32)
        elif operation == 'attention':
            seq_len = input_shapes[0][-2] if len(input_shapes) > 0 else 512
            hidden_dim = input_shapes[0][-1] if len(input_shapes) > 0 else 512
            triton_code = self.generate_attention_kernel(seq_len, hidden_dim)
            block_size = (64, 64)  # Balanced for attention
        else:
            # Generic kernel template
            triton_code = f"# Hardware-aware kernel for {operation}"
            block_size = (256,)
        
        # Create kernel spec
        kernel_spec = KernelSpec(
            name=f"{operation}_{cache_key[:8]}",
            operation=operation,
            input_shapes=input_shapes,
            output_shape=output_shape,
            triton_code=triton_code,
            register_pressure=reg_pressure,
            block_size=block_size,
        )
        
        self.kernel_cache[cache_key] = kernel_spec
        return kernel_spec
    
    def get_hardware_profile(self) -> HardwareProfile:
        """Get current hardware profile."""
        return self.hardware_profile
