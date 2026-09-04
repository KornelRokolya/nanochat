"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
import csv
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
parser.add_argument("--max-wall-clock-time", type=float, default=-1.0, help="hard stop after this many accumulated training seconds; step 0 time is excluded (-1 = disable)")
parser.add_argument("--resize-wall-clock-time", type=float, default=-1.0, help="at this accumulated training time (seconds), simultaneously double total batch size and model width/head count once; negative disables")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
print0(f"Resize wall-clock time: {args.resize_wall_clock_time}s (negative disables)")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3: efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta


# -----------------------------------------------------------------------------
# Online model-width morphing helpers
#
# The first implementation deliberately keeps n_layer fixed and doubles n_embd,
# n_head and n_kv_head. This preserves head_dim and lets us use an exact
# duplicated-residual construction. Depth growth is a separate problem.
#
# Random perturbations are normalized to the Frobenius norm of the source
# matrix, then multiplied by this hard-coded relative gain.
MORPH_RANDOM_GAIN = 0.10
MORPH_RANDOM_SEED = 12345


@dataclass
class TrainingModelState:
    orig_model: object
    model: object
    optimizer: object
    config: GPTConfig
    num_params: int
    num_scaling_params: int
    num_flops_per_token: float
    target_tokens: float
    weight_decay_scaled: float


def _safe_param_filename(name):
    # Keep names Fiji/filesystem friendly while preserving the parameter path.
    return name.replace(".", "_").replace("/", "_").replace("\\", "_")


@torch.no_grad()
def SaveModelAsRawParams(model_to_save, output_directory):
    """Write every parameter as a linearized little-endian float32 .raw file.

    2-D tensors are named:
        {parameterName}_f32_dim0xdim1.raw
    1-D tensors are represented as Nx1 for easy Fiji Raw Import.
    Higher-dimensional tensors (if ever added) use all dimensions joined by 'x'.

    PyTorch/NumPy C-order is used: the last dimension varies fastest.
    """
    if not master_process:
        return

    import numpy as np

    os.makedirs(output_directory, exist_ok=True)
    manifest_path = os.path.join(output_directory, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as manifest_file:
        manifest = csv.writer(manifest_file, delimiter=";")
        manifest.writerow(["parameter_name", "dtype", "shape", "raw_file"])

        for name, parameter in model_to_save.named_parameters():
            tensor = parameter.detach().to(device="cpu", dtype=torch.float32).contiguous()
            shape = tuple(tensor.shape)
            file_shape = shape if len(shape) != 1 else (shape[0], 1)
            dims = "x".join(str(d) for d in file_shape)
            filename = f"{_safe_param_filename(name)}_f32_{dims}.raw"
            filepath = os.path.join(output_directory, filename)

            array = tensor.numpy().astype("<f4", copy=False)
            array.tofile(filepath)
            manifest.writerow([name, "float32_le", "x".join(map(str, shape)), filename])

    print0(f"Saved raw float32 parameters to: {output_directory}")


@torch.no_grad()
def _rand_relative_to(reference, gain, generator):
    """Random tensor with Frobenius norm = gain * ||reference||_F."""
    ref32 = reference.detach().float()

    rnd = torch.randn(
        ref32.shape,
        device=ref32.device,
        dtype=torch.float32,
        generator=generator,
    )

    rnd_norm = torch.linalg.vector_norm(rnd)
    ref_norm = torch.linalg.vector_norm(ref32)

    target_norm = gain * (
        ref_norm
        if ref_norm > 0
        else torch.tensor(1.0, device=ref32.device)
    )

    rnd = rnd / rnd_norm.clamp_min(1e-12) * target_norm
    return rnd


@torch.no_grad()
def _rand_like_shape(shape, device, target_norm, generator):
    """Generate random float32 tensor of given shape with chosen Frobenius norm."""
    rnd = torch.randn(
        shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    rnd_norm = torch.linalg.vector_norm(rnd)

    return rnd / rnd_norm.clamp_min(1e-12) * target_norm


@torch.no_grad()
def _morph_rectangular_double(
    weight,
    gain,
    generator,
    output_noise_gain=0.01,
):
    """For W:[m,n], create a [2m,2n] widened matrix.

    Main symmetry-breaking construction:

        [[0.5W + A, 0.5W - A],
         [0.5W + B, 0.5W - B]]

    Then add a much smaller unrestricted perturbation E, so the mapping is
    intentionally no longer exactly function preserving.

    For duplicated input [x,x], without E the result would be [Wx,Wx].
    """

    W = weight.detach().float()

    A = _rand_relative_to(W, gain, generator)
    B = _rand_relative_to(W, gain, generator)

    half_W = 0.5 * W

    top = torch.cat([
        half_W + A,
        half_W - A,
    ], dim=1)

    bottom = torch.cat([
        half_W + B,
        half_W - B,
    ], dim=1)

    widened = torch.cat([top, bottom], dim=0)

    # Small non-canceling perturbation.
    E = _rand_relative_to(
        widened,
        output_noise_gain,
        generator,
    )

    return widened + E


@torch.no_grad()
def _morph_qkv_double_heads(
    weight,
    gain,
    generator,
    output_noise_gain=0.01,
):
    """Double Q/K/V width and head count.

    Uses the structured widening construction plus a small unrestricted
    perturbation so both old and new heads begin slightly displaced from the
    original function.
    """

    return _morph_rectangular_double(
        weight,
        gain,
        generator,
        output_noise_gain=output_noise_gain,
    )


@torch.no_grad()
def morph_model_width_2x(
    old_model,
    new_model,
    gain=MORPH_RANDOM_GAIN,
    output_noise_gain=0.01,
):
    """Widen nanochat model by 2x with deliberate small function perturbations.

    Strategy:
      * token embedding: duplicate old embedding + small random perturbation
      * Q/K/V: structured 2x widening + small non-canceling perturbation
      * W_O: preserve old heads approximately, initialize new-head contribution
             with a small random matrix instead of exact zero
      * MLP c_fc/c_proj: structured widening + non-canceling perturbation
      * lm_head: average the duplicated residual halves + small random perturbation
      * value embeddings: copy old channels + random new channels, and slightly
                          perturb old channels too
      * value gates: copy old gate rows with small noise; initialize new rows
                     near zero with small random values
      * scalar parameters: copy + very small perturbation

    This is intentionally NOT exactly function preserving.
    """

    old_cfg = old_model.config
    new_cfg = new_model.config

    assert new_cfg.n_layer == old_cfg.n_layer, \
        "This morph keeps depth fixed."

    assert new_cfg.n_embd == 2 * old_cfg.n_embd
    assert new_cfg.n_head == 2 * old_cfg.n_head
    assert new_cfg.n_kv_head == 2 * old_cfg.n_kv_head

    assert (
        new_cfg.n_embd // new_cfg.n_head
        ==
        old_cfg.n_embd // old_cfg.n_head
    )

    generator = torch.Generator(device=old_model.get_device())
    generator.manual_seed(MORPH_RANDOM_SEED)

    # -------------------------------------------------------------------------
    # Token embedding
    # -------------------------------------------------------------------------

    old_wte = old_model.transformer.wte.weight.detach().float()

    widened_wte = torch.cat([
        old_wte,
        old_wte,
    ], dim=1)

    widened_wte += _rand_relative_to(
        widened_wte,
        output_noise_gain,
        generator,
    )

    new_model.transformer.wte.weight.copy_(
        widened_wte.to(new_model.transformer.wte.weight.dtype)
    )

    # -------------------------------------------------------------------------
    # Output head
    #
    # Base mapping averages the duplicated residual halves:
    #
    #     [0.5 W, 0.5 W]
    #
    # Then perturb it slightly so logits are not exactly preserved.
    # -------------------------------------------------------------------------

    old_head = old_model.lm_head.weight.detach().float()

    widened_head = torch.cat([
        0.5 * old_head,
        0.5 * old_head,
    ], dim=1)

    widened_head += _rand_relative_to(
        widened_head,
        output_noise_gain,
        generator,
    )

    new_model.lm_head.weight.copy_(
        widened_head.to(new_model.lm_head.weight.dtype)
    )

    # -------------------------------------------------------------------------
    # Transformer blocks
    # -------------------------------------------------------------------------

    for old_block, new_block in zip(
        old_model.transformer.h,
        new_model.transformer.h,
    ):

        # ---------------------------------------------------------------------
        # Q / K / V
        # ---------------------------------------------------------------------

        for attr in ("c_q", "c_k", "c_v"):

            old_w = getattr(
                old_block.attn,
                attr,
            ).weight

            new_w = _morph_qkv_double_heads(
                old_w,
                gain,
                generator,
                output_noise_gain=output_noise_gain,
            )

            getattr(
                new_block.attn,
                attr,
            ).weight.copy_(
                new_w.to(
                    getattr(
                        new_block.attn,
                        attr,
                    ).weight.dtype
                )
            )

        # ---------------------------------------------------------------------
        # Attention output projection W_O
        #
        # Previously:
        #
        #     [[W, 0],
        #      [W, 0]]
        #
        # which completely blocked the new heads.
        #
        # Now the new-head columns receive a small random matrix, while the
        # old-head path also gets a tiny perturbation.
        # ---------------------------------------------------------------------

        old_wo = old_block.attn.c_proj.weight.detach().float()

        d_out, d_in = old_wo.shape

        new_wo = torch.zeros(
            (2 * d_out, 2 * d_in),
            device=old_wo.device,
            dtype=torch.float32,
        )

        # Main old-head contribution.
        new_wo[:d_out, :d_in] = old_wo
        new_wo[d_out:, :d_in] = old_wo

        old_norm = torch.linalg.vector_norm(old_wo)

        # New heads get a small but nonzero projection immediately.
        new_head_noise = _rand_like_shape(
            (2 * d_out, d_in),
            device=old_wo.device,
            target_norm=output_noise_gain * old_norm,
            generator=generator,
        )

        new_wo[:, d_in:] = new_head_noise

        # Also perturb the old-head pathway slightly.
        new_wo[:, :d_in] += _rand_like_shape(
            (2 * d_out, d_in),
            device=old_wo.device,
            target_norm=output_noise_gain * old_norm,
            generator=generator,
        )

        new_block.attn.c_proj.weight.copy_(
            new_wo.to(
                new_block.attn.c_proj.weight.dtype
            )
        )

        # ---------------------------------------------------------------------
        # Value embedding gate
        # ---------------------------------------------------------------------

        if old_block.attn.ve_gate is not None:

            old_gate = (
                old_block.attn.ve_gate.weight
                .detach()
                .float()
            )

            new_gate = torch.zeros_like(
                new_block.attn.ve_gate.weight,
                dtype=torch.float32,
            )

            old_rows = old_gate.shape[0]

            # Old gates copied with tiny perturbation.
            old_gate_noisy = old_gate + _rand_relative_to(
                old_gate,
                output_noise_gain,
                generator,
            )

            new_gate[:old_rows, :] = old_gate_noisy

            # New heads start near zero, but not identically zero.
            # Since nanochat applies sigmoid afterward, these are still around
            # sigmoid(0)=0.5 in gate-space. W_O is what keeps their initial
            # contribution small.
            new_rows = new_gate.shape[0] - old_rows

            if new_rows > 0:
                gate_scale = (
                    torch.linalg.vector_norm(old_gate)
                    / max(old_gate.numel() ** 0.5, 1.0)
                )

                new_gate[old_rows:, :] = torch.randn(
                    new_gate[old_rows:, :].shape,
                    device=new_gate.device,
                    dtype=torch.float32,
                    generator=generator,
                ) * (
                    output_noise_gain
                    * gate_scale
                )

            new_block.attn.ve_gate.weight.copy_(
                new_gate.to(
                    new_block.attn.ve_gate.weight.dtype
                )
            )

        # ---------------------------------------------------------------------
        # MLP
        # ---------------------------------------------------------------------

        for attr in ("c_fc", "c_proj"):

            old_w = getattr(
                old_block.mlp,
                attr,
            ).weight

            new_w = _morph_rectangular_double(
                old_w,
                gain,
                generator,
                output_noise_gain=output_noise_gain,
            )

            getattr(
                new_block.mlp,
                attr,
            ).weight.copy_(
                new_w.to(
                    getattr(
                        new_block.mlp,
                        attr,
                    ).weight.dtype
                )
            )

    # -------------------------------------------------------------------------
    # Learned scalar parameters
    #
    # Preserve them approximately, but introduce tiny symmetry-breaking noise.
    # -------------------------------------------------------------------------

    def copy_scalar_with_noise(dst, src):
        src32 = src.detach().float()

        if src32.numel() == 0:
            dst.copy_(src)
            return

        noisy = src32 + _rand_relative_to(
            src32,
            output_noise_gain,
            generator,
        )

        dst.copy_(noisy.to(dst.dtype))

    copy_scalar_with_noise(
        new_model.resid_lambdas,
        old_model.resid_lambdas,
    )

    copy_scalar_with_noise(
        new_model.x0_lambdas,
        old_model.x0_lambdas,
    )

    copy_scalar_with_noise(
        new_model.smear_lambda,
        old_model.smear_lambda,
    )

    copy_scalar_with_noise(
        new_model.backout_lambda,
        old_model.backout_lambda,
    )

    # smear_gate is a matrix-like parameter.
    old_smear_gate = (
        old_model.smear_gate.weight
        .detach()
        .float()
    )

    new_smear_gate = old_smear_gate + _rand_relative_to(
        old_smear_gate,
        output_noise_gain,
        generator,
    )

    new_model.smear_gate.weight.copy_(
        new_smear_gate.to(
            new_model.smear_gate.weight.dtype
        )
    )

    # -------------------------------------------------------------------------
    # Value embeddings
    #
    # Old part is copied with small perturbation.
    # New part is random with a norm proportional to the old table.
    # -------------------------------------------------------------------------

    for key, old_ve in old_model.value_embeds.items():

        new_ve = new_model.value_embeds[key]

        old_table = (
            old_ve.weight
            .detach()
            .float()
        )

        old_table_noisy = old_table + _rand_relative_to(
            old_table,
            output_noise_gain,
            generator,
        )

        random_table = _rand_relative_to(
            old_table,
            gain,
            generator,
        )

        morphed_table = torch.cat([
            old_table_noisy,
            random_table,
        ], dim=1)

        new_ve.weight.copy_(
            morphed_table.to(
                new_ve.weight.dtype
            )
        )


# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1

# Per-step semicolon-separated training log. Only rank 0 writes it.
# New runs overwrite the file; resumed runs append to it.
training_csv_path = os.path.join(checkpoint_dir, "training_log.csv")
training_csv_file = None
training_csv_writer = None
if master_process:
    os.makedirs(checkpoint_dir, exist_ok=True)
    csv_mode = "a" if resuming and os.path.exists(training_csv_path) else "w"
    training_csv_file = open(training_csv_path, csv_mode, newline="", encoding="utf-8", buffering=1)
    training_csv_writer = csv.writer(training_csv_file, delimiter=";")
    if csv_mode == "w":
        training_csv_writer.writerow([
            "step",
            "num_iterations",
            "pct_done",
            "training_loss",
            "lrm",
            "total_batch_size",
            "grad_accum_steps",
            "dt_ms",
            "tok_per_sec",
            "bf16_mfu",
            "epoch",
            "pq_idx",
            "rg_idx",
            "total_training_time_s",
            "total_training_time_ms",
            "eta_s",
            "eta_min",
            "training_tokens_so_far",
            "total_training_flops",
            "model_n_layer",
            "model_n_embd",
            "model_n_head",
            "model_num_params",
        ])
    print0(f"Training CSV: {training_csv_path}")
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data
    # A scheduled run may have changed the total batch size before the checkpoint.
    # Restore that runtime value so gradient accumulation and later LR rescaling stay consistent.
    checkpoint_total_batch_size = meta_data.get("total_batch_size", total_batch_size)
    if checkpoint_total_batch_size != total_batch_size:
        print0(
            f"Restoring total batch size from checkpoint: "
            f"{total_batch_size:,} -> {checkpoint_total_batch_size:,}"
        )
    total_batch_size = checkpoint_total_batch_size
    # weight_decay_scaled was initially computed from the CLI/start batch. Recompute it
    # for the checkpoint's actual scheduled batch before the per-step WD scheduler uses it.
    weight_decay_scaled = (
        args.weight_decay
        * math.sqrt(total_batch_size / B_REF)
        * (D_REF / target_tokens)
    )

# setup_optimizer() applies the batch LR scale to every normal AdamW/Muon group,
# except the special smear group. Tag groups for online schedule bookkeeping.
def _tag_initial_optimizer_groups(model_for_optimizer, optimizer_for_model):
    smear_param_ids = {
        id(model_for_optimizer.smear_gate.weight),
        id(model_for_optimizer.smear_lambda),
        id(model_for_optimizer.backout_lambda),
    }
    for group in optimizer_for_model.param_groups:
        is_smear_group = any(id(p) in smear_param_ids for p in group["params"])
        group["_batch_lr_scaled"] = not is_smear_group

_tag_initial_optimizer_groups(orig_model, optimizer)

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")

if resuming and "num_iterations" in meta_data:
    checkpoint_num_iterations = meta_data["num_iterations"]
    if checkpoint_num_iterations != num_iterations:
        print0(
            f"Restoring num_iterations from checkpoint: "
            f"{num_iterations:,} -> {checkpoint_num_iterations:,}"
        )
    num_iterations = checkpoint_num_iterations

total_tokens = total_batch_size * num_iterations
print0(f"Initial-batch token estimate: {total_tokens:,} (changes if the online batch schedule triggers)")
print0(f"Initial-batch Tokens : Scaling params ratio: {total_tokens / num_scaling_params:.2f}")
print0(f"Initial-batch training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
    training_tokens_so_far = 0 # actual tokens consumed; required once batch size can change
    training_flops_so_far = 0.0 # actual cumulative FLOPs; model FLOPs/token can change online
    growth_event_done = False
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]
    # New scheduled checkpoints contain these fields. Fall back gracefully for old checkpoints.
    training_tokens_so_far = loop_state.get("training_tokens_so_far", total_batch_size * step)
    training_flops_so_far = loop_state.get(
        "training_flops_so_far",
        num_flops_per_token * training_tokens_so_far,
    )
    growth_event_done = loop_state.get("growth_event_done", False)

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step.
# IMPORTANT: the dataloader's device batch size stays fixed. Online total-batch growth is implemented
# purely by changing gradient accumulation, so x/y tensor shapes do not change and torch.compile does
# not need to recompile the model.
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per micro-step for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # tokens per micro-step across all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0, (
    f"total_batch_size ({total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
)
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

def _tag_batch_scaled_optimizer_groups(model_for_optimizer, optimizer_for_model):
    """Mark the special smear group, whose LR is intentionally not batch-scaled."""
    smear_param_ids = {
        id(model_for_optimizer.smear_gate.weight),
        id(model_for_optimizer.smear_lambda),
        id(model_for_optimizer.backout_lambda),
    }
    for group in optimizer_for_model.param_groups:
        is_smear_group = any(id(p) in smear_param_ids for p in group["params"])
        group["_batch_lr_scaled"] = not is_smear_group


# Wrap the live model/optimizer and model-dependent accounting in one object.
training_state = TrainingModelState(
    orig_model=orig_model,
    model=model,
    optimizer=optimizer,
    config=model_config,
    num_params=num_params,
    num_scaling_params=num_scaling_params,
    num_flops_per_token=num_flops_per_token,
    target_tokens=target_tokens,
    weight_decay_scaled=weight_decay_scaled,
)


def grow_batch_and_model_width_2x(current_step):
    """Atomically double total batch size and model width/head count once.

    The LR *schedule multiplier* is untouched. A new optimizer is constructed
    because the parameter tensors have new shapes. Its base LRs are recalculated
    from the new batch size and new d_model; the current global lrm is then
    applied. Optimizer moments are intentionally restarted for this first test.
    """
    global training_state
    global orig_model, model, optimizer, model_config, model_config_kwargs
    global num_params, num_scaling_params, num_flops_per_token, target_tokens
    global total_batch_size, grad_accum_steps, weight_decay_scaled

    if args.fp8:
        raise NotImplementedError(
            "Online width morphing is intentionally disabled for --fp8 in this first implementation."
        )

    old_state = training_state
    old_cfg = old_state.config
    old_batch = total_batch_size
    new_batch = old_batch * 1 # * 2

    if new_batch % world_tokens_per_fwdbwd != 0:
        raise ValueError(
            f"new total batch size {new_batch:,} must be a multiple of "
            f"{world_tokens_per_fwdbwd:,} tokens/micro-step"
        )

    new_cfg = GPTConfig(
        sequence_len=old_cfg.sequence_len,
        vocab_size=old_cfg.vocab_size,
        n_layer=old_cfg.n_layer,
        n_head=old_cfg.n_head * 2,
        n_kv_head=old_cfg.n_kv_head * 2,
        n_embd=old_cfg.n_embd * 2,
        window_pattern=old_cfg.window_pattern,
    )

    print0("=" * 80)
    print0(
        f"GROWTH EVENT at training time {total_training_time:.3f}s, step {current_step}: "
        f"B {old_batch:,}->{new_batch:,}; "
        f"d_model {old_cfg.n_embd}->{new_cfg.n_embd}; "
        f"heads {old_cfg.n_head}->{new_cfg.n_head}; "
        f"layers stay {old_cfg.n_layer}"
    )

    # Save the exact pre-morph parameters for Fiji inspection.
    raw_root = os.path.join(checkpoint_dir, "raw_params")
    SaveModelAsRawParams(
        old_state.orig_model,
        os.path.join(raw_root, f"step_{current_step:05d}_pre_morph"),
    )

    # Construct and initialize the wider model. init_weights() also materializes
    # rotary buffers correctly; all learned parameters are overwritten below.
    with torch.device("meta"):
        new_orig_model = GPT(new_cfg)
    new_orig_model.to_empty(device=device)
    new_orig_model.init_weights()

    morph_model_width_2x(
        old_state.orig_model,
        new_orig_model,
        gain=MORPH_RANDOM_GAIN,
    )

    # Cheap function-preservation sanity check on a small prefix.
    # This is outside total_training_time by design (only optimizer-step dt is accumulated).
    check_T = min(256, x.shape[1])
    with torch.no_grad():
        old_check_loss = old_state.orig_model(x[:1, :check_T], y[:1, :check_T]).float().item()
        new_check_loss = new_orig_model(x[:1, :check_T], y[:1, :check_T]).float().item()
    print0(
        f"Morph check loss on 1x{check_T}: "
        f"old={old_check_loss:.9f}, new={new_check_loss:.9f}, "
        f"abs_diff={abs(new_check_loss - old_check_loss):.3e}"
    )

    # Save the morphed wider model too.
    SaveModelAsRawParams(
        new_orig_model,
        os.path.join(raw_root, f"step_{current_step:05d}_post_morph"),
    )

    # Recompute model-dependent scaling quantities.
    new_param_counts = new_orig_model.num_scaling_params()
    new_num_params = new_param_counts["total"]
    new_num_scaling_params = get_scaling_params(new_orig_model)
    new_num_flops_per_token = new_orig_model.estimate_flops()
    new_target_tokens = int(args.target_param_data_ratio * new_num_scaling_params)

    new_batch_lr_scale = math.sqrt(new_batch / B_REF)
    new_weight_decay_scaled = (
        args.weight_decay
        * math.sqrt(new_batch / B_REF)
        * (D_REF / new_target_tokens)
    )

    # Old optimizer state points at old tensors and is intentionally discarded.
    old_state.optimizer = None
    old_state.model = None
    gc.collect()
    if device_type == "cuda":
        torch.cuda.empty_cache()

    new_optimizer = new_orig_model.setup_optimizer(
        unembedding_lr=args.unembedding_lr * new_batch_lr_scale,
        embedding_lr=args.embedding_lr * new_batch_lr_scale,
        scalar_lr=args.scalar_lr * new_batch_lr_scale,
        matrix_lr=args.matrix_lr * new_batch_lr_scale,
        weight_decay=new_weight_decay_scaled,
    )
    _tag_batch_scaled_optimizer_groups(new_orig_model, new_optimizer)

    # Keep the existing LR schedule phase. Do not restart or alter warmup/warmdown.
    current_lrm = get_lr_multiplier(current_step)
    current_muon_momentum = get_muon_momentum(current_step)
    current_muon_weight_decay = (
        new_weight_decay_scaled
        * 0.5
        * (1 + math.cos(math.pi * current_step / num_iterations))
    )
    for group in new_optimizer.param_groups:
        group["lr"] = group["initial_lr"] * current_lrm
        if group["kind"] == "muon":
            group["momentum"] = current_muon_momentum
            group["weight_decay"] = current_muon_weight_decay

    new_model = torch.compile(new_orig_model, dynamic=False)

    # Publish the new live state atomically.
    total_batch_size = new_batch
    grad_accum_steps = new_batch // world_tokens_per_fwdbwd

    orig_model = new_orig_model
    model = new_model
    optimizer = new_optimizer
    model_config = new_cfg
    model_config_kwargs = asdict(new_cfg)
    num_params = new_num_params
    num_scaling_params = new_num_scaling_params
    num_flops_per_token = new_num_flops_per_token
    target_tokens = new_target_tokens
    weight_decay_scaled = new_weight_decay_scaled

    training_state = TrainingModelState(
        orig_model=orig_model,
        model=model,
        optimizer=optimizer,
        config=model_config,
        num_params=num_params,
        num_scaling_params=num_scaling_params,
        num_flops_per_token=num_flops_per_token,
        target_tokens=target_tokens,
        weight_decay_scaled=weight_decay_scaled,
    )

    # Release the old model only after morphing and optimizer replacement are done.
    old_state.orig_model = None
    del old_state
    gc.collect()
    if device_type == "cuda":
        torch.cuda.empty_cache()

    print0(
        f"Growth complete | params={num_params:,} | "
        f"FLOPs/token={num_flops_per_token:.6e} | "
        f"B={total_batch_size:,} | accum={grad_accum_steps} | "
        f"batch_lr_scale={new_batch_lr_scale:.6f} | "
        f"current_lrm={current_lrm:.6f}"
    )
    print0("=" * 80)


# Go!
while True:
    wall_clock_limit_reached = (
        args.max_wall_clock_time >= 0
        and total_training_time >= args.max_wall_clock_time
    )
    # The loop normally runs num_iterations+1 times so that we can eval/save at the end.
    # A wall-clock stop behaves like an early final step: no more training is performed,
    # but the normal end-of-run evaluation/checkpoint path is still executed.
    last_step = (step == num_iterations) or wall_clock_limit_reached

    # -------------------------------------------------------------------------
    # One-time joint growth event, keyed to accumulated *training-step* wall time.
    # Step 0 is excluded from that accumulator, matching --max-wall-clock-time.
    if (
        not last_step
        and args.resize_wall_clock_time >= 0
        and not growth_event_done
        and total_training_time >= args.resize_wall_clock_time
    ):
        grow_batch_and_model_width_2x(current_step=step)
        growth_event_done = True

    flops_so_far = training_flops_so_far

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or (step > 0 and step % args.eval_every == 0)):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "num_iterations": num_iterations,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                    "training_tokens_so_far": training_tokens_so_far,
                    "training_flops_so_far": training_flops_so_far,
                    "growth_event_done": growth_event_done,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        if wall_clock_limit_reached:
            print0(
                f"Wall-clock limit reached: {total_training_time:.2f}s "
                f">= {args.max_wall_clock_time:.2f}s. Stopping training."
            )
            # Save the morphed wider model too.
            raw_root = os.path.join(checkpoint_dir, "raw_params")
            SaveModelAsRawParams(
                model,
                os.path.join(raw_root, f"step_{current_step:05d}_final"),
            )
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    train_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        # Log the mean loss over the entire optimizer-step batch, not merely the final
        # micro-batch. This becomes important once grad_accum_steps changes online.
        train_loss_sum += loss.detach().float()
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = (train_loss_sum / grad_accum_steps).item() # one CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # This optimizer step consumed exactly total_batch_size tokens. Track the real
    # cumulative amount because total_batch_size can now change during the run.
    training_tokens_so_far += total_batch_size
    training_flops_so_far += num_flops_per_token * total_batch_size
    flops_so_far = training_flops_so_far
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    # Step 0 often contains compile/startup overhead, so exclude only that one step.
    # Every later optimizer-step duration contributes to the wall-clock hard-stop counter.
    if step > 0:
        total_training_time += dt

    # Calculate ETA from counted training steps. This does not alter the LR schedule.
    steps_done = step
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_seconds = None
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | B: {total_batch_size:,} | accum: {grad_accum_steps} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")

    # Mirror every repeated training-status print into a machine-friendly CSV.
    if master_process:
        training_csv_writer.writerow([
            step,
            num_iterations,
            f"{pct_done:.6f}",
            f"{debiased_smooth_loss:.9f}",
            f"{lrm:.9f}",
            total_batch_size,
            grad_accum_steps,
            f"{dt * 1000:.6f}",
            tok_per_sec,
            f"{mfu:.6f}",
            dataloader_state_dict["epoch"],
            dataloader_state_dict["pq_idx"],
            dataloader_state_dict["rg_idx"],
            f"{total_training_time:.9f}",
            f"{total_training_time * 1000:.6f}",
            "" if eta_seconds is None else f"{eta_seconds:.9f}",
            "" if eta_seconds is None else f"{eta_seconds / 60:.9f}",
            training_tokens_so_far,
            f"{flops_so_far:.9e}",
            model_config.n_layer,
            model_config.n_embd,
            model_config.n_head,
            num_params,
        ])

    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/total_batch_size": total_batch_size,
            "train/grad_accum_steps": grad_accum_steps,
            "train/tokens_so_far": training_tokens_so_far,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# cleanup
if training_csv_file is not None:
    training_csv_file.close()
wandb_run.finish() # wandb run finish
compute_cleanup()
