"""fp8 (E4M3) GDN MTP commit-scatter (Phase B / S-4).

``fused_mamba_state_scatter_with_mask`` quantizes the accepted FP32 intermediate
snapshot to E4M3 with a per-(HV,V) row scale (amax / 448, block = K), optionally
with hardware stochastic rounding (``cvt.rs.satfinite.e4m3x4``), and writes the
companion fp32 scale pool in lockstep. Requires SM100+ (Blackwell).

Loads the scatter module from a sibling file when present (standalone cluster
run), else from the sglang package (in-repo).
"""
import importlib.util
import os

import pytest
import torch

_sib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mamba_state_scatter_triton.py")
if os.path.exists(_sib):
    _spec = importlib.util.spec_from_file_location("mamba_state_scatter_triton", _sib)
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    fused_mamba_state_scatter_with_mask = _m.fused_mamba_state_scatter_with_mask
else:
    from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
        fused_mamba_state_scatter_with_mask,
    )


def _skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("fp8 cvt.rs requires SM100+")


@pytest.mark.parametrize("use_sr", [False, True])
def test_fp8_commit_scatter_quant(use_sr):
    """Per-row scale == amax/448; dequant(dst)*scale tracks the source row."""
    _skip()
    torch.manual_seed(0)
    L, SPEC, DRAFT, HV, V, K, CACHE = 2, 4, 3, 4, 128, 128, 8
    src = torch.randn(L, SPEC, DRAFT, HV, V, K, device="cuda", dtype=torch.float32) * 0.3
    dst = torch.zeros(L, CACHE, HV, V, K, device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.zeros(L, CACHE, HV, V, device="cuda", dtype=torch.float32)
    dst_indices = torch.tensor([1, 4, 6], device="cuda", dtype=torch.int32)
    step_indices = torch.tensor([0, 2, 1], device="cuda", dtype=torch.int32)
    fused_mamba_state_scatter_with_mask(
        dst, src, dst_indices, step_indices,
        dst_scale=scale, use_sr=use_sr, philox_rounds=10,
    )
    for r in range(dst_indices.numel()):
        di, si = dst_indices[r].item(), step_indices[r].item()
        ref = src[:, r, si]  # [L, HV, V, K]
        exp_scale = (ref.abs().amax(dim=-1) / 448.0).clamp(min=1e-8)
        assert torch.allclose(scale[:, di], exp_scale, rtol=1e-3, atol=1e-6), \
            f"scale mismatch req {r}"
        deq = dst[:, di].float() * scale[:, di][..., None]
        rel = (deq - ref).abs().mean().item() / (ref.abs().mean().item() + 1e-9)
        assert rel < 0.12, f"dequant rel err {rel} (req {r})"


def test_fp8_commit_scatter_unbiased():
    """SR commit is unbiased: mean over many draws ~ the input."""
    _skip()
    torch.manual_seed(1)
    L, SPEC, DRAFT, HV, V, K, CACHE = 1, 1, 1, 2, 128, 128, 2
    src = torch.randn(L, SPEC, DRAFT, HV, V, K, device="cuda", dtype=torch.float32) * 0.3
    dst_indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    step_indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    N = 64
    acc = torch.zeros(L, HV, V, K, device="cuda", dtype=torch.float32)
    for _ in range(N):
        dst = torch.zeros(L, CACHE, HV, V, K, device="cuda", dtype=torch.float8_e4m3fn)
        scale = torch.zeros(L, CACHE, HV, V, device="cuda", dtype=torch.float32)
        fused_mamba_state_scatter_with_mask(
            dst, src, dst_indices, step_indices, dst_scale=scale, use_sr=True, philox_rounds=10,
        )
        acc += dst[:, 0].float() * scale[:, 0][..., None]
    rel = (acc / N - src[:, 0, 0]).abs().mean().item() / (src[:, 0, 0].abs().mean().item() + 1e-9)
    assert rel < 0.05, f"SR commit biased: rel {rel}"


def test_fp8_commit_scatter_mask():
    """step_indices < 0 means skip — must not write that dst slot's scale."""
    _skip()
    L, SPEC, DRAFT, HV, V, K, CACHE = 2, 4, 3, 4, 128, 128, 8
    src = torch.randn(L, SPEC, DRAFT, HV, V, K, device="cuda", dtype=torch.float32) * 0.3
    dst = torch.zeros(L, CACHE, HV, V, K, device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.zeros(L, CACHE, HV, V, device="cuda", dtype=torch.float32)
    dst_indices = torch.tensor([2], device="cuda", dtype=torch.int32)
    step_indices = torch.tensor([-1], device="cuda", dtype=torch.int32)  # invalid -> skip
    fused_mamba_state_scatter_with_mask(dst, src, dst_indices, step_indices, dst_scale=scale)
    assert (scale[:, 2] == 0).all(), "masked (step<0) request wrote a scale"
