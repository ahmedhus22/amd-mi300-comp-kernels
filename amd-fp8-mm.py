#!POPCORN leaderboard amd-fp8-mm

# This is a submission template for popcorn leaderboard 'amd-fp8-mm'.
# Your task is as follows:
# > 
# > You will implement a custom fp8-blockwise matmul kernel optimized for MI300.
# > You will be given single-precision scaling factors for your matrices.
# > The shapes of all outer and inner dimensions of tensors are from DeepSeek-R1.
# > To be explicit, you will be given a tuple of tensors:
# > ```
# > (a, b, a_scale, b_scale, c)
# > ```
# > where `a` and `b` are the input matrices, `a_scale` and `b_scale` are the scaling factors for `a` and `b` respectively,
# > and `c` is the output matrix:
# > * `a` is M x K in column-major order in e4m3fnuz
# > * `b` is N x K in column-major order in e4m3fnuz
# > * `a_scale` is M x K in column-major order in fp32
# > * `b_scale` is N x K in column-major order in fp32
# > * `c` is M x N in ROW-major order in bf16
# > 
# > Matrix sizes `m` and `n` are divisible by 64, `k` is divisible by 128.
# > 
# > The ranking criteria is the geometric mean of the benchmark results.
# > 
# > For the grand price, your kernel will be evaluated against the speed of light analysis
# > and the solution closest to the speed of light will be awarded the grand price.
# > ```
# > The speed of light analysis is:
# >  M       N       K     time[us]
# > 1024    1536    7168      8.63
# > 1024    4608    7168     25.89
# > 6144    1536    7168     51.78
# > 6144    4608    7168    155.30
# > 1024    7168     256      3.17
# > 6144    7168     256     17.27
# > ```
# The deadline for this leaderboard is 2025-05-27 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!

import torch
import triton
import triton.language as tl
from task import input_t, output_t

# BLOCK_SIZE_K should be less than 128, (only 1 block scales are loaded)
def get_hip_autotune_config():
    return [
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 4},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 8},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
    ]



@triton.autotune(
    configs=get_hip_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def fp8_mm_kernel(
               a_ptr, b_ptr,
               a_scale_ptr, b_scale_ptr,
               c_ptr,
               # matrix dimensions
               M, N, K,
               # strides
               stride_am, stride_ak,
               stride_bn, stride_bk, # tranpose b
               stride_cm, stride_cn,
               stride_am_scale, stride_ak_scale,
               stride_bn_scale, stride_bk_scale,
               # meta-param
               BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
               GROUP_SIZE_M: tl.constexpr,
               # NOTE: `constexpr` so it can be used as a shape value.
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # each block computes [BLOCK_SIZE_M, BLOCK_SIZE_N] elements for c
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    BLOCK_SIZE_SCALE = 128
    n_block_idx = (pid_n * BLOCK_SIZE_N) // BLOCK_SIZE_SCALE
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        k_block_idx = (k * BLOCK_SIZE_K) // BLOCK_SIZE_SCALE
        b_scale = tl.load(b_scale_ptr + n_block_idx * stride_bn_scale + k_block_idx * stride_bk_scale)
        a_scale_ptrs = a_scale_ptr + (offs_am[:] * stride_am_scale + k_block_idx * stride_ak_scale)
        a_scale = tl.load(a_scale_ptrs)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b) * b_scale.to(tl.float32) * a_scale.to(tl.float32).expand_dims(1).broadcast_to((BLOCK_SIZE_M, BLOCK_SIZE_N))
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.bfloat16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)



def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of block-scale fp8 gemm 
    Args:
        data: Tuple that expands to:
            a: torch.Tensor[float8_e4m3fnuz] of shape [m, k], 
            b: torch.Tensor[float8_e4m3fnuz] of shape [n, k], 
            a_scale: torch.Tensor[float32] of shape [m, k // 128], 
            b_scale: torch.Tensor[float32] of shape [n // 128, k // 128], 
            c: torch.Tensor[bfloat16] of shape [m, n]
    Returns:
        Tensor containing output in bf16
    """
    # c: [m, n] is pre-allocated memory to avoid timing allocation overhead.
    a, b, a_scale, b_scale, c = data

    # constants
    m = a.shape[0]
    n = b.shape[0]
    k = a.shape[1]

    # block multiply and scale
    grid = lambda META: (triton.cdiv(m, META['BLOCK_SIZE_M']) * triton.cdiv(n, META['BLOCK_SIZE_N']), )
    fp8_mm_kernel[grid](a, b, a_scale, b_scale, c, m, n, k,
                    a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                    a_scale.stride(0), a_scale.stride(1), b_scale.stride(0), b_scale.stride(1))

    return c

