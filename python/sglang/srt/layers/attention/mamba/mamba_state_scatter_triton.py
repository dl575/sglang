"""
Fused Triton kernel for Mamba state scatter operations.

This kernel replaces the expensive advanced indexing operations in
`update_mamba_state_after_mtp_verify` with a single fused gather-scatter kernel,
avoiding multiple `index_elementwise_kernel` launches.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def track_mamba_state_if_needed_kernel(
    conv_states_ptr,
    ssm_states_ptr,
    cache_indices_ptr,
    mamba_track_mask_ptr,
    mamba_track_indices_ptr,
    conv_state_stride_0,  # stride for first dimension (batch/pool index)
    ssm_state_stride_0,  # stride for first dimension (batch/pool index)
    conv_state_numel_per_row: tl.constexpr,  # total elements per row
    ssm_state_numel_per_row: tl.constexpr,  # total elements per row
    BLOCK_SIZE: tl.constexpr,
):
    """
    Track conv_states and ssm_states rows based on track mask.

    This kernel replaces a Python loop that copies state tensors for mamba attention.
    For each batch element, if the track mask is True, it copies the entire row from
    the source index (cache_indices[i]) to the destination index (mamba_track_indices[i]).

    Grid: (batch_size,)
    Each block handles one batch element, using multiple threads to copy data in parallel.
    """
    batch_idx = tl.program_id(0)

    # Load the copy mask for this batch element
    track_mask = tl.load(mamba_track_mask_ptr + batch_idx)

    # Early exit if we don't need to track
    if not track_mask:
        return

    # Load source and destination indices
    src_idx = tl.load(cache_indices_ptr + batch_idx)
    dst_idx = tl.load(mamba_track_indices_ptr + batch_idx)

    # Copy conv_states
    # Each thread handles BLOCK_SIZE elements
    for offset in range(0, conv_state_numel_per_row, BLOCK_SIZE):
        element_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = element_indices < conv_state_numel_per_row

        src_ptr = conv_states_ptr + src_idx * conv_state_stride_0 + element_indices
        dst_ptr = conv_states_ptr + dst_idx * conv_state_stride_0 + element_indices

        data = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_ptr, data, mask=mask)

    # Copy ssm_states
    for offset in range(0, ssm_state_numel_per_row, BLOCK_SIZE):
        element_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = element_indices < ssm_state_numel_per_row

        src_ptr = ssm_states_ptr + src_idx * ssm_state_stride_0 + element_indices
        dst_ptr = ssm_states_ptr + dst_idx * ssm_state_stride_0 + element_indices

        data = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_ptr, data, mask=mask)


def track_mamba_states_if_needed(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    mamba_track_mask: torch.Tensor,
    mamba_track_indices: torch.Tensor,
    batch_size: int,
):
    """
    Track mamba states using Triton kernel for better performance.

    Args:
        conv_states: Convolution states tensor [pool_size, ...]
        ssm_states: SSM states tensor [pool_size, ...]
        cache_indices: Source indices for each batch element [batch_size]
        mamba_track_mask: Boolean mask indicating which elements to track [batch_size]
        mamba_track_indices: Indices to track for each batch element [batch_size]
        batch_size: Number of batch elements
    """
    conv_state_numel_per_row = conv_states[0].numel()
    ssm_state_numel_per_row = ssm_states[0].numel()

    # Choose BLOCK_SIZE based on the size of the data
    BLOCK_SIZE = 1024

    # Launch kernel with batch_size blocks
    grid = (batch_size,)
    track_mamba_state_if_needed_kernel[grid](
        conv_states,
        ssm_states,
        cache_indices,
        mamba_track_mask,
        mamba_track_indices,
        conv_states.stride(0),
        ssm_states.stride(0),
        conv_state_numel_per_row,
        ssm_state_numel_per_row,
        BLOCK_SIZE,
    )


# ---------------------------------------------------------------------------
# fp8 (E4M3) quantizing scatter for the GDN MTP commit (Phase B / S-4).
#
# When the destination SSM cache is fp8, the accepted FP32 intermediate snapshot
# is quantized to E4M3 with a per-(HV,V) row scale (amax / 448, block = K) and,
# optionally, stochastic rounding via the hardware ``cvt.rs.satfinite.e4m3x4``
# instruction. The companion fp32 scale pool is written in lockstep so the next
# load-dequant (fp8 * scale) is correct. Matches the decode fp8 store convention.
# ---------------------------------------------------------------------------
_E4M3_MAX = tl.constexpr(448.0)  # Triton @jit can only read constexpr globals


@triton.jit
def _cvt_rs_e4m3x4(x, rand):
    """fp32 -> e4m3 stochastic rounding (hardware cvt.rs, pack=4, SM100+)."""
    return tl.inline_asm_elementwise(
        asm="{ cvt.rs.satfinite.e4m3x4.f32 $0, {$4, $3, $2, $1}, $5; }",
        constraints="=r,r,r,r,r,r,r,r,r",
        args=(x, rand),
        dtype=tl.float8e4nv,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _cvt_rs_narrowx2(x, rand, IS_BF16: tl.constexpr):
    """fp32 -> bf16 or fp16 stochastic rounding (hardware cvt.rs, pack=2, SM100+)."""
    if IS_BF16:
        return tl.inline_asm_elementwise(
            asm="{ cvt.rs.bf16x2.f32 $0, {$2, $1}, $3; }",
            constraints="=r,r,r,r",
            args=(x, rand),
            dtype=tl.bfloat16,
            is_pure=True,
            pack=2,
        )
    else:
        return tl.inline_asm_elementwise(
            asm="{ cvt.rs.f16x2.f32 $0, {$2, $1}, $3; }",
            constraints="=r,r,r,r",
            args=(x, rand),
            dtype=tl.float16,
            is_pure=True,
            pack=2,
        )


@triton.jit
def _scatter_extend_state_kernel(
    src_ptr,    # fp32 [B, HV*V, K]
    dst_ptr,    # bf16/fp16/fp8 [pool, HV*V, K]
    scale_ptr,  # fp32 [pool, HV*V] — only used for fp8, else None
    idx_ptr,    # int64 [B] — indices into dst pool
    seed_ptr,   # int32 [1] — Philox seed
    B, HV_V, K: tl.constexpr,
    pool_HV_V_K_stride,   # dst stride for pool dim = HV*V*K
    USE_SR: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    IS_FP8: tl.constexpr,
    IS_BF16: tl.constexpr,  # IO dtype (ignored for fp8)
):
    """One program per (batch, HV*V row); scatter fp32 → narrow dtype with optional SR.

    Grid: (B * HV_V,).  Each program handles one K-element row.
    Dispatches to fp8 (per-row amax scale + cvt.rs e4m3x4) or bf16/fp16 (cvt.rs narrowx2).
    """
    pid = tl.program_id(0)
    b = pid // HV_V
    row = pid % HV_V  # flat (hv, v) index

    dst_idx = tl.load(idx_ptr + b)

    k = tl.arange(0, K)
    src_off = b * HV_V * K + row * K + k
    dst_off = dst_idx * pool_HV_V_K_stride + row * K + k
    src = tl.load(src_ptr + src_off)  # fp32 [K]

    if IS_FP8:
        amax = tl.max(tl.abs(src))
        scale = tl.maximum(amax / _E4M3_MAX, 1e-8)
        y = src / scale
        if USE_SR:
            rand = tl.randint(
                tl.load(seed_ptr), (pid * K + k).to(tl.int32), PHILOX_ROUNDS
            )
            q = _cvt_rs_e4m3x4(y, rand)
        else:
            q = y.to(tl.float8e4nv)
        tl.store(dst_ptr + dst_off + k, q)
        scale_off = dst_idx * HV_V + row
        tl.store(scale_ptr + scale_off, scale)
    else:
        if USE_SR:
            rand = tl.randint(
                tl.load(seed_ptr), (pid * K + k).to(tl.int32), PHILOX_ROUNDS
            )
            q = _cvt_rs_narrowx2(src, rand, IS_BF16)
        else:
            if IS_BF16:
                q = src.to(tl.bfloat16)
            else:
                q = src.to(tl.float16)
        tl.store(dst_ptr + dst_off + k, q)


def scatter_extend_state(
    src: torch.Tensor,          # fp32 [B, HV, V, K]
    dst: torch.Tensor,          # bf16/fp16/fp8 [pool, HV, V, K]
    dst_scale: torch.Tensor,    # fp32 [pool, HV, V] or None
    indices: torch.Tensor,      # int64 [B]
    use_sr: bool = False,
    philox_rounds: int = 10,
):
    """Scatter fp32 prefill output state to the narrow-dtype SSM pool with optional SR.

    Replaces the plain `index_copy_(... .to(dtype))` in the extend writeback so that
    SR (when enabled) is applied consistently at the prefill→decode boundary, matching
    the per-step decode commit. Supports fp8 (per-row amax/448 scale), bf16, and fp16.
    """
    B, HV, V, K = src.shape
    HV_V = HV * V
    assert src.dtype == torch.float32
    assert dst.shape == (dst.shape[0], HV, V, K)
    assert indices.shape == (B,)
    is_fp8 = dst.dtype == torch.float8_e4m3fn
    is_bf16 = dst.dtype == torch.bfloat16
    if is_fp8 and dst_scale is None:
        raise ValueError("fp8 dst requires dst_scale")

    # Flatten HV, V into one dim for the kernel
    src_flat = src.reshape(B, HV_V, K).contiguous()
    dst_flat = dst.reshape(dst.shape[0], HV_V, K)
    scale_flat = dst_scale.reshape(dst_scale.shape[0], HV_V) if dst_scale is not None else None

    seed = torch.randint(0, 2**31 - 1, (1,), device=src.device, dtype=torch.int32)
    grid = (B * HV_V,)
    _scatter_extend_state_kernel[grid](
        src_flat, dst_flat,
        scale_flat if scale_flat is not None else src_flat,  # dummy for non-fp8
        indices,
        seed,
        B, HV_V, K,
        dst_flat.stride(0),  # pool_HV_V_K_stride
        USE_SR=use_sr,
        PHILOX_ROUNDS=philox_rounds,
        IS_FP8=is_fp8,
        IS_BF16=is_bf16,
    )


@triton.jit
def _scatter_quant_fp8_kernel(
    src_ptr,  # fp32 [layers, spec, draft, HV, V, K] (contiguous)
    dst_ptr,  # e4m3 [layers, cache, HV, V, K]
    scale_ptr,  # fp32 [layers, cache, HV, V]
    dst_indices_raw_ptr,
    step_indices_raw_ptr,
    seed_ptr,
    src_layer_stride,
    src_req_stride,
    src_step_stride,
    dst_layer_stride,
    dst_req_stride,
    scale_layer_stride,
    scale_req_stride,
    src_req_size,
    src_step_size,
    dst_req_size,
    USE_SR: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    K: tl.constexpr,
):
    """One program per (request, layer, (HV,V)-row); each block is one K-row.

    Quantizes the accepted FP32 snapshot row to E4M3 with a per-row scale and
    writes both the fp8 state and the fp32 scale. Grid: (requests, layers, HV*V).
    """
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1).to(tl.int64)
    pid_row = tl.program_id(2).to(tl.int64)

    step_idx = tl.load(step_indices_raw_ptr + pid_req).to(tl.int64)
    if step_idx < 0:
        return
    dst_idx = tl.load(dst_indices_raw_ptr + pid_req).to(tl.int64)
    src_idx = pid_req
    if not (
        (dst_idx >= 0)
        & (dst_idx < dst_req_size)
        & (src_idx < src_req_size)
        & (step_idx < src_step_size)
    ):
        return

    k = tl.arange(0, K)
    src_off = (
        pid_layer * src_layer_stride
        + src_idx * src_req_stride
        + step_idx * src_step_stride
        + pid_row * K
    )
    dst_off = pid_layer * dst_layer_stride + dst_idx * dst_req_stride + pid_row * K

    row = tl.load(src_ptr + src_off + k)  # fp32 [K]
    amax = tl.max(tl.abs(row))
    scale = tl.maximum(amax / _E4M3_MAX, 1e-8)
    y = row / scale  # in [-448, 448]
    if USE_SR:
        # Per-element Philox offset = within-entry flat index; the per-call seed
        # varies the randomness across commits (matches the decode store).
        rand = tl.randint(
            tl.load(seed_ptr), (pid_row * K + k).to(tl.int32), PHILOX_ROUNDS
        )
        q = _cvt_rs_e4m3x4(y, rand)
    else:
        q = y.to(tl.float8e4nv)  # round-to-nearest
    tl.store(dst_ptr + dst_off + k, q)
    scale_off = pid_layer * scale_layer_stride + dst_idx * scale_req_stride + pid_row
    tl.store(scale_ptr + scale_off, scale)


@triton.jit
def _fused_mamba_state_scatter_with_mask_kernel(
    src_ptr,
    dst_ptr,
    # Raw index arrays (before index_select)
    dst_indices_raw_ptr,  # [total_requests] - state_indices_tensor
    step_indices_raw_ptr,  # [total_requests] - last_correct_step_indices or mamba_steps_to_track
    elem_per_entry: tl.constexpr,
    src_layer_stride,
    src_req_stride,
    src_step_stride,
    dst_layer_stride,
    dst_req_stride,
    src_req_size,
    src_step_size,
    dst_req_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused gather-scatter kernel with built-in masking.

    This kernel fuses the index_select operations by:
    1. Iterating over all requests (pid_req from 0 to total_requests-1)
    2. Checking if step_indices_raw[pid_req] >= 0 (valid mask)
    3. If valid, performing the scatter:
       dst[l, dst_indices_raw[pid_req], :] = src[l, pid_req, step_indices_raw[pid_req], :]

    Grid: (total_requests, num_layers, ceil(elem_per_entry / BLOCK_SIZE))
    """
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1).to(tl.int64)
    pid_block = tl.program_id(2).to(tl.int64)

    # Load step index to check validity (step >= 0 means valid)
    step_idx = tl.load(step_indices_raw_ptr + pid_req).to(tl.int64)

    # Early exit if this request is not valid (step < 0)
    if step_idx < 0:
        return

    # Load destination index
    dst_idx = tl.load(dst_indices_raw_ptr + pid_req).to(tl.int64)

    # Source index is just the request index itself
    src_idx = pid_req

    # Bounds check to avoid illegal memory access
    if not (
        (dst_idx >= 0)
        & (dst_idx < dst_req_size)
        & (src_idx >= 0)
        & (src_idx < src_req_size)
        & (step_idx < src_step_size)
    ):
        return

    # Compute base offsets
    src_offset = (
        pid_layer * src_layer_stride
        + src_idx * src_req_stride
        + step_idx * src_step_stride
    )
    dst_offset = pid_layer * dst_layer_stride + dst_idx * dst_req_stride

    # Compute element range for this block
    start = pid_block * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elem_per_entry

    # Load from source and store to destination
    data = tl.load(src_ptr + src_offset + offsets, mask=mask)
    tl.store(dst_ptr + dst_offset + offsets, data, mask=mask)


def fused_mamba_state_scatter_with_mask(
    dst: torch.Tensor,  # [num_layers, cache_size, *state_shape]
    src: torch.Tensor,  # [num_layers, spec_size, draft_tokens, *state_shape]
    dst_indices_raw: torch.Tensor,  # [total_requests] - raw indices (e.g., state_indices_tensor)
    step_indices_raw: torch.Tensor,  # [total_requests] - raw step indices (step >= 0 means valid)
    dst_scale: torch.Tensor = None,  # fp32 [num_layers, cache_size, HV, V] when dst is fp8
    use_sr: bool = False,  # stochastic rounding for the fp8 quantization
    philox_rounds: int = 10,
):
    """
    Fully fused gather-scatter with built-in masking for mamba state updates.

    This function fuses the following operations into a single kernel:
    1. valid_mask = step_indices_raw >= 0
    2. valid_indices = valid_mask.nonzero()
    3. dst_indices = dst_indices_raw[valid_indices]  (index_select)
    4. step_indices = step_indices_raw[valid_indices]  (index_select)
    5. for each valid i: dst[:, dst_indices[i], :] = src[:, i, step_indices[i], :]

    Args:
        dst: Destination tensor [num_layers, cache_size, *state_shape]
        src: Source tensor [num_layers, spec_size, draft_tokens, *state_shape]
        dst_indices_raw: Raw destination indices for all requests [total_requests]
        step_indices_raw: Raw step indices; entry >= 0 means valid [total_requests]
    """
    total_requests = step_indices_raw.shape[0]
    if total_requests == 0:
        return

    if dst.device != src.device:
        raise ValueError(
            f"dst and src must be on the same device. {dst.device=} {src.device=}"
        )
    if not dst.is_cuda or not src.is_cuda:
        raise ValueError(
            "fused_mamba_state_scatter_with_mask only supports CUDA tensors."
        )
    if dst.ndim < 2 or src.ndim < 3:
        raise ValueError(f"Unexpected tensor ranks: {dst.ndim=} {src.ndim=}")
    if dst.shape[0] != src.shape[0]:
        raise ValueError(
            f"Layer dimension mismatch: {dst.shape[0]=} vs {src.shape[0]=}"
        )
    if dst.shape[2:] != src.shape[3:]:
        raise ValueError(
            f"Trailing dims mismatch: {dst.shape[2:]=} vs {src.shape[3:]=}"
        )
    if dst_indices_raw.ndim != 1 or step_indices_raw.ndim != 1:
        raise ValueError(
            f"indices must be 1D: {dst_indices_raw.shape=} {step_indices_raw.shape=}"
        )
    if dst_indices_raw.shape[0] != step_indices_raw.shape[0]:
        raise ValueError(
            f"indices length mismatch: {dst_indices_raw.shape[0]=} vs {step_indices_raw.shape[0]=}"
        )

    num_layers = dst.shape[0]
    src_req_size = src.shape[1]
    src_step_size = src.shape[2]
    dst_req_size = dst.shape[1]

    # Flatten trailing dimensions: number of elements per (layer, cache_line) entry.
    elem_per_entry = dst.numel() // (dst.shape[0] * dst.shape[1])

    # Get strides (in elements, not bytes)
    src_layer_stride = src.stride(0)
    src_req_stride = src.stride(1)
    src_step_stride = src.stride(2)
    dst_layer_stride = dst.stride(0)
    dst_req_stride = dst.stride(1)

    # Ensure indices are int32 and contiguous
    dst_indices_raw = dst_indices_raw.to(torch.int32).contiguous()
    step_indices_raw = step_indices_raw.to(torch.int32).contiguous()

    # Ensure tensors are contiguous
    if not dst.is_contiguous():
        raise ValueError("dst tensor must be contiguous")
    if not src.is_contiguous():
        raise ValueError("src tensor must be contiguous")

    # fp8 destination (GDN MTP commit): quantize the accepted FP32 snapshot to
    # E4M3 with a per-(HV,V) row scale (block = K), optionally with hardware SR,
    # and write the companion fp32 scale pool in lockstep. Keyed on dst.dtype so
    # both SSM scatter calls (commit + prefix-cache track) pick it up; the conv
    # scatters (bf16 dst) fall through to the plain copy below.
    if dst.dtype == torch.float8_e4m3fn:
        if dst_scale is None:
            raise ValueError("fp8 scatter requires dst_scale (the temporal_scale pool)")
        if src.dtype != torch.float32:
            raise ValueError(f"fp8 scatter expects fp32 src, got {src.dtype}")
        if not dst_scale.is_contiguous():
            raise ValueError("dst_scale must be contiguous")
        K = dst.shape[-1]
        if elem_per_entry % K != 0:
            raise ValueError(f"elem_per_entry {elem_per_entry} not divisible by K {K}")
        num_rows = elem_per_entry // K  # HV * V
        seed = torch.randint(0, 2**31 - 1, (1,), device=dst.device, dtype=torch.int32)
        grid_fp8 = (total_requests, num_layers, num_rows)
        _scatter_quant_fp8_kernel[grid_fp8](
            src,
            dst,
            dst_scale,
            dst_indices_raw,
            step_indices_raw,
            seed,
            src_layer_stride,
            src_req_stride,
            src_step_stride,
            dst_layer_stride,
            dst_req_stride,
            dst_scale.stride(0),
            dst_scale.stride(1),
            src_req_size,
            src_step_size,
            dst_req_size,
            USE_SR=use_sr,
            PHILOX_ROUNDS=philox_rounds,
            K=K,
        )
        return

    # Block size for copying elements
    BLOCK_SIZE = 1024

    # Grid over all requests - invalid ones will early-exit in the kernel
    grid = (total_requests, num_layers, triton.cdiv(elem_per_entry, BLOCK_SIZE))

    _fused_mamba_state_scatter_with_mask_kernel[grid](
        src,
        dst,
        dst_indices_raw,
        step_indices_raw,
        elem_per_entry,
        src_layer_stride,
        src_req_stride,
        src_step_stride,
        dst_layer_stride,
        dst_req_stride,
        src_req_size,
        src_step_size,
        dst_req_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
