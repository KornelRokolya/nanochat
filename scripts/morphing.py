"""Model morphing utilities for Nanochat experiments.

Two strategies are intentionally provided:

1. ``duplicate``
   Direct dN -> d2N model-to-model morph. Width/head count/depth must all double.
   The old residual representation is duplicated, matrices use averaging copies,
   and every copied tensor receives Frobenius-normalized noise.

2. ``new_capacity``
   Embed the trained model into the first half of the wider model and initialize
   only the newly-added portions with tiny Frobenius-normalized noise. Newly
   added transformer layers keep Nanochat's normal random initialization.

Optimizer state migration is deliberately out of scope: callers rebuild the
optimizer after morphing.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch


@dataclass
class MorphReport:
    strategy: str
    noise_gain: float
    old_layers: int
    new_layers: int
    old_embd: int
    new_embd: int
    old_heads: int
    new_heads: int
    old_params: int
    new_params: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def _noise_with_norm(reference: torch.Tensor, gain: float, generator: torch.Generator) -> torch.Tensor:
    """Return Gaussian noise with ||noise||_F = gain * ||reference||_F."""
    ref = reference.detach().float()
    noise = torch.randn(ref.shape, device=ref.device, dtype=torch.float32, generator=generator)
    ref_norm = torch.linalg.vector_norm(ref)
    noise_norm = torch.linalg.vector_norm(noise).clamp_min(1e-12)
    # Zero source tensors are common for Nanochat output projections. In that
    # case there is intentionally no perturbation: the requested norm is zero.
    return noise * (gain * ref_norm / noise_norm)


@torch.no_grad()
def _copy_with_relative_noise(dst: torch.Tensor, src: torch.Tensor, gain: float, generator: torch.Generator) -> None:
    src32 = src.detach().float()
    out = src32 + _noise_with_norm(src32, gain, generator)
    dst.copy_(out.to(dtype=dst.dtype))


@torch.no_grad()
def _duplicate_vector(src: torch.Tensor, gain: float, generator: torch.Generator) -> torch.Tensor:
    base = torch.cat((src.detach().float(), src.detach().float()), dim=0)
    return base + _noise_with_norm(base, gain, generator)


@torch.no_grad()
def _duplicate_last_dim(src: torch.Tensor, gain: float, generator: torch.Generator) -> torch.Tensor:
    base = torch.cat((src.detach().float(), src.detach().float()), dim=-1)
    return base + _noise_with_norm(base, gain, generator)


@torch.no_grad()
def _average_double_matrix(src: torch.Tensor, gain: float, generator: torch.Generator) -> torch.Tensor:
    """W[m,n] -> .5 [[W,W],[W,W]] + noise.

    For duplicated x'=[x,x], the noiseless transform produces [Wx,Wx].
    Works for square matrices and Nanochat's rectangular MLP matrices because
    both dimensions double in a dN -> d2N morph.
    """
    w = src.detach().float()
    base = 0.5 * torch.cat((torch.cat((w, w), dim=1), torch.cat((w, w), dim=1)), dim=0)
    return base + _noise_with_norm(base, gain, generator)


@torch.no_grad()
def _average_input_only(src: torch.Tensor, gain: float, generator: torch.Generator) -> torch.Tensor:
    """W[m,n] -> [.5W,.5W] + noise, used by the vocabulary output head."""
    w = src.detach().float()
    base = 0.5 * torch.cat((w, w), dim=1)
    return base + _noise_with_norm(base, gain, generator)


@torch.no_grad()
def _embed_matrix_top_left(dst: torch.Tensor, src: torch.Tensor, gain: float, generator: torch.Generator) -> None:
    """Put src in the top-left of dst; newly-added entries get tiny random values."""
    src32 = src.detach().float()
    out = torch.zeros(dst.shape, device=src.device, dtype=torch.float32)
    rows, cols = src32.shape
    out[:rows, :cols] = src32

    mask = torch.ones(dst.shape, device=src.device, dtype=torch.bool)
    mask[:rows, :cols] = False
    n_new = int(mask.sum().item())
    if n_new:
        rnd = torch.randn(n_new, device=src.device, dtype=torch.float32, generator=generator)
        rnd = rnd / torch.linalg.vector_norm(rnd).clamp_min(1e-12)
        rnd = rnd * (gain * torch.linalg.vector_norm(src32))
        out[mask] = rnd
    dst.copy_(out.to(dtype=dst.dtype))


@torch.no_grad()
def _embed_last_dim(dst: torch.Tensor, src: torch.Tensor, gain: float, generator: torch.Generator) -> None:
    """Copy existing last-dimension channels; initialize the new channels tiny."""
    src32 = src.detach().float()
    out = torch.zeros(dst.shape, device=src.device, dtype=torch.float32)
    old_d = src32.shape[-1]
    out[..., :old_d] = src32
    new_slice = out[..., old_d:]
    if new_slice.numel():
        rnd = torch.randn(new_slice.shape, device=src.device, dtype=torch.float32, generator=generator)
        rnd = rnd / torch.linalg.vector_norm(rnd).clamp_min(1e-12)
        rnd = rnd * (gain * torch.linalg.vector_norm(src32))
        out[..., old_d:] = rnd
    dst.copy_(out.to(dtype=dst.dtype))


def _assert_direct_double(old_model, new_model) -> None:
    o, n = old_model.config, new_model.config
    assert n.n_layer == 2 * o.n_layer, f"duplicate strategy requires layer doubling: {o.n_layer} -> {n.n_layer}"
    assert n.n_embd == 2 * o.n_embd, f"duplicate strategy requires width doubling: {o.n_embd} -> {n.n_embd}"
    assert n.n_head == 2 * o.n_head, f"duplicate strategy requires head doubling: {o.n_head} -> {n.n_head}"
    assert n.n_kv_head == 2 * o.n_kv_head, f"duplicate strategy requires KV-head doubling: {o.n_kv_head} -> {n.n_kv_head}"
    assert n.n_embd // n.n_head == o.n_embd // o.n_head, "head_dim must remain unchanged"


@torch.no_grad()
def _morph_duplicate(old_model, new_model, gain: float, generator: torch.Generator) -> None:
    """Approximate function-preserving dN -> d2N duplication morph."""
    _assert_direct_double(old_model, new_model)

    # Residual stream enters as [x, x].
    new_model.transformer.wte.weight.copy_(
        _duplicate_last_dim(old_model.transformer.wte.weight, gain, generator).to(new_model.transformer.wte.weight.dtype)
    )
    # Output logits average both residual halves.
    new_model.lm_head.weight.copy_(
        _average_input_only(old_model.lm_head.weight, gain, generator).to(new_model.lm_head.weight.dtype)
    )

    # Global smear/backout path. smear_gate only looks at the first 24 channels,
    # so its shape does not grow.
    _copy_with_relative_noise(new_model.smear_gate.weight, old_model.smear_gate.weight, gain, generator)
    _copy_with_relative_noise(new_model.smear_lambda, old_model.smear_lambda, gain, generator)
    _copy_with_relative_noise(new_model.backout_lambda, old_model.backout_lambda, gain, generator)

    old_layers = old_model.config.n_layer
    for old_i, old_block in enumerate(old_model.transformer.h):
        for copy_idx in (0, 1):
            new_i = 2 * old_i + copy_idx
            new_block = new_model.transformer.h[new_i]

            for attr in ("c_q", "c_k", "c_v"):
                src = getattr(old_block.attn, attr).weight
                dst = getattr(new_block.attn, attr).weight
                dst.copy_(_average_double_matrix(src, gain, generator).to(dst.dtype))

            # Two copied residual blocks replace one old block. Halving each
            # residual-branch output is the least intrusive split for Nanochat's
            # pre-norm residual architecture (there is no post-MLP norm).
            for module_name, attr in (("attn", "c_proj"), ("mlp", "c_proj")):
                src = getattr(getattr(old_block, module_name), attr).weight
                dst = getattr(getattr(new_block, module_name), attr).weight
                widened = 0.5 * _average_double_matrix(src, gain, generator)
                dst.copy_(widened.to(dst.dtype))

            src = old_block.mlp.c_fc.weight
            dst = new_block.mlp.c_fc.weight
            dst.copy_(_average_double_matrix(src, gain, generator).to(dst.dtype))

            if old_block.attn.ve_gate is not None and new_block.attn.ve_gate is not None:
                gate_base = torch.cat((old_block.attn.ve_gate.weight.detach().float(),) * 2, dim=0)
                gate_base = gate_base + _noise_with_norm(gate_base, gain, generator)
                new_block.attn.ve_gate.weight.copy_(gate_base.to(new_block.attn.ve_gate.weight.dtype))

        # Avoid applying the learned residual-stream affine twice. The first
        # copy carries the old layer's scalars; the second copy is neutral.
        new_model.resid_lambdas[2 * old_i].copy_(old_model.resid_lambdas[old_i])
        new_model.x0_lambdas[2 * old_i].copy_(old_model.x0_lambdas[old_i])
        new_model.resid_lambdas[2 * old_i + 1].fill_(1.0)
        new_model.x0_lambdas[2 * old_i + 1].zero_()

    # Value embeddings follow layer mapping. If a target layer has a VE but its
    # source does not (possible for unusual parity changes), leave Nanochat init.
    for old_i in range(old_layers):
        old_key = str(old_i)
        if old_key not in old_model.value_embeds:
            continue
        for copy_idx in (0, 1):
            new_key = str(2 * old_i + copy_idx)
            if new_key in new_model.value_embeds:
                src = old_model.value_embeds[old_key].weight
                dst = new_model.value_embeds[new_key].weight
                dst.copy_(_duplicate_last_dim(src, gain, generator).to(dst.dtype))


@torch.no_grad()
def _morph_new_capacity(old_model, new_model, gain: float, generator: torch.Generator) -> None:
    """Embed old capacity; keep newly-added layers at normal Nanochat init."""
    o, n = old_model.config, new_model.config
    assert n.n_layer >= o.n_layer and n.n_embd >= o.n_embd
    assert n.n_embd // n.n_head == o.n_embd // o.n_head, "head_dim must remain unchanged"

    _embed_last_dim(new_model.transformer.wte.weight, old_model.transformer.wte.weight, gain, generator)
    _embed_last_dim(new_model.lm_head.weight, old_model.lm_head.weight, gain, generator)
    _copy_with_relative_noise(new_model.smear_gate.weight, old_model.smear_gate.weight, gain, generator)
    _copy_with_relative_noise(new_model.smear_lambda, old_model.smear_lambda, gain, generator)
    _copy_with_relative_noise(new_model.backout_lambda, old_model.backout_lambda, gain, generator)

    # Existing layers occupy the first old_n layers. New layers remain exactly
    # as produced by GPT.init_weights(), per the requested strategy.
    for i, old_block in enumerate(old_model.transformer.h):
        new_block = new_model.transformer.h[i]
        for attr in ("c_q", "c_k", "c_v", "c_proj"):
            _embed_matrix_top_left(getattr(new_block.attn, attr).weight, getattr(old_block.attn, attr).weight, gain, generator)
        for attr in ("c_fc", "c_proj"):
            _embed_matrix_top_left(getattr(new_block.mlp, attr).weight, getattr(old_block.mlp, attr).weight, gain, generator)

        if old_block.attn.ve_gate is not None and new_block.attn.ve_gate is not None:
            old_gate = old_block.attn.ve_gate.weight
            dst_gate = new_block.attn.ve_gate.weight
            # Gate input width is fixed (12), only head rows grow.
            out = torch.zeros_like(dst_gate, dtype=torch.float32)
            rows = old_gate.shape[0]
            out[:rows] = old_gate.detach().float()
            if out.shape[0] > rows:
                rnd = torch.randn(out[rows:].shape, device=out.device, dtype=torch.float32, generator=generator)
                rnd = rnd / torch.linalg.vector_norm(rnd).clamp_min(1e-12)
                rnd *= gain * torch.linalg.vector_norm(old_gate.detach().float())
                out[rows:] = rnd
            dst_gate.copy_(out.to(dst_gate.dtype))

        new_model.resid_lambdas[i].copy_(old_model.resid_lambdas[i])
        new_model.x0_lambdas[i].copy_(old_model.x0_lambdas[i])

        key = str(i)
        if key in old_model.value_embeds and key in new_model.value_embeds:
            _embed_last_dim(new_model.value_embeds[key].weight, old_model.value_embeds[key].weight, gain, generator)


def morph_model(old_model, new_model, strategy: str = "duplicate", noise_gain: float = 1e-3, seed: int = 12345) -> MorphReport:
    """Morph ``old_model`` into already initialized ``new_model`` in-place."""
    if noise_gain < 0:
        raise ValueError("noise_gain must be >= 0")
    generator = torch.Generator(device=old_model.get_device())
    generator.manual_seed(seed)

    strategy = strategy.lower()
    if strategy == "duplicate":
        _morph_duplicate(old_model, new_model, noise_gain, generator)
    elif strategy == "new_capacity":
        _morph_new_capacity(old_model, new_model, noise_gain, generator)
    else:
        raise ValueError(f"Unknown morph strategy: {strategy}")

    return MorphReport(
        strategy=strategy,
        noise_gain=noise_gain,
        old_layers=old_model.config.n_layer,
        new_layers=new_model.config.n_layer,
        old_embd=old_model.config.n_embd,
        new_embd=new_model.config.n_embd,
        old_heads=old_model.config.n_head,
        new_heads=new_model.config.n_head,
        old_params=sum(p.numel() for p in old_model.parameters()),
        new_params=sum(p.numel() for p in new_model.parameters()),
    )
