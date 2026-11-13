# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics of the DSpine code bridge, written in plain PyTorch.

DSpine is a DFlash draft plus two extra pieces (see docs/design/dspine-vllm-port.md
section 3):

* an *adjacent causal code bridge*: every layer writes a compact code of each
  position into a shared stream and the next position reads its predecessor's
  write, which is the only token dependency a single parallel pass can carry;
* a *last-write refinement*: the final layer's write is redone with the exact
  token code the previous position actually chose, and only the ``K`` draft
  candidates of each position are scored, which yields the ``[rows, K, K]``
  table the shared candidate walk already consumes.

This module holds the arithmetic only, dtype-agnostically and on any device, so
the tests can check it against a float64 reference. ``qwen3_dspine.py`` owns the
vLLM modules, weight loading, and tensor-parallel gathers.

The rounding points follow Tencent's fused SGLang kernels
(``sglang_dspine/kernels/dspine.py``): every reduction runs in FP32 and stores
back in the runtime dtype, one rounding per stored value.
"""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

# Lower bound on a row mean-square. Keeps an all-zero row's reciprocal finite and
# matches the constant baked into the reference kernel.
_MIN_MEAN_SQUARE = torch.finfo(torch.float32).tiny

# Rows refined per chunk. The refinement keeps [rows, K, hidden] FP32
# intermediates, so bound them for large batch sizes.
REFINE_ROW_CHUNK = 64


@dataclass(frozen=True)
class CodeWrite:
    """What one adjacent code-bridge write produces for one draft forward.

    normed: normalized post-write stream; the input of the next layer, or of the
        draft head after the last layer.
    stream: post-write residual stream, before normalization.
    raw: pre-write residual stream, kept for the last-write refinement.
    rms: per-row write scale, the RMS of ``raw``.
    packed: ``codes @ [gate; predecessor_gate; prev].T``; the receiver query of
        each row is ``packed[:, :rank]`` and the sender of the next row is the
        same row's columns ``rank:2 * rank``. Kept for the refinement.
    """

    normed: torch.Tensor
    stream: torch.Tensor
    raw: torch.Tensor
    rms: torch.Tensor
    packed: torch.Tensor


def _row_rms(rows: torch.Tensor) -> torch.Tensor:
    """Per-row root mean square in FP32, the reduction the kernel performs."""
    return rows.float().square().mean(-1).clamp_min(_MIN_MEAN_SQUARE).sqrt()


def _rms_norm_after_rounding(
    rows: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """RMSNorm that rounds after the normalization and again after the weight,
    as the reference ``feature_finish`` kernel does."""
    normalized = rows.float() * torch.rsqrt(
        rows.float().square().mean(-1, keepdim=True) + eps
    )
    return (normalized.to(rows.dtype).float() * weight.float()).to(rows.dtype)


def code_bridge_write(
    mlp_out: torch.Tensor,
    residual: torch.Tensor | None,
    anchor_codes: torch.Tensor,
    *,
    read_weight: torch.Tensor,
    packed_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    write_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    block_size: int,
    code_size: int,
    eps: float,
    norm_eps: float,
) -> CodeWrite:
    """One code-bridge write after a draft layer.

    mlp_out: [rows, hidden] this layer's attention-plus-MLP output.
    residual: [rows, hidden] incoming residual stream, or None when the layer
        contributed the whole stream.
    anchor_codes: [rows, code_size] codebook row of the anchor (bonus) token of
        the block each row belongs to, repeated over that block. Anchor rows keep
        the exact token code instead of a code predicted from features, and never
        send a message.

    The rows are whole request blocks: row 0 of every block is its anchor. Only
    the block size locates those rows, and only on tensor *values*. The row count
    stays symbolic because this runs inside the draft model's compiled forward,
    where solving a marked-dynamic shape into a constant is an error, not a
    recompile. The eager `refine_last_write` takes the same rows and checks them.
    """
    num_rows = mlp_out.shape[0]
    row = torch.arange(num_rows, device=mlp_out.device)
    # Row 0 of a block is its anchor; every other row reads its predecessor. The
    # clamp only keeps the gather in range: anchor messages are masked away below.
    anchor_row = row - row % block_size
    is_anchor = anchor_row == row
    rank = write_weight.shape[1]
    dtype = mlp_out.dtype
    code_scale = code_size**0.5

    # --- Read: pre-write stream, its RMS, and the projected code ---
    raw = (
        mlp_out
        if residual is None
        else (mlp_out.float() + residual.float()).to(dtype)
    )
    rms = _row_rms(raw)
    normalized = (raw.float() / (rms[:, None] + eps)).to(dtype)
    projected = F.linear(normalized, read_weight).float()

    # --- Codes: unit-direction prediction rows, exact token codes on anchors ---
    predicted = code_scale * projected / (projected.norm(dim=-1, keepdim=True) + eps)
    codes = torch.where(
        is_anchor[:, None], anchor_codes.float() * code_scale, predicted
    ).to(dtype)

    # --- Message: the predecessor's gated value, none on an anchor row ---
    packed = F.linear(codes, packed_weight)
    query = packed[:, :rank]
    sender = packed[(row - 1).clamp_min(0), rank : 2 * rank]
    value = packed[(row - 1).clamp_min(0), 2 * rank :]
    message = torch.sigmoid(query.float() + sender.float() + gate_bias.float())
    message = torch.where(is_anchor[:, None], 0, message * value.float()).to(dtype)

    # --- Write: scale the message into the residual stream, then normalize ---
    delta = F.linear(message, write_weight)
    stream = (raw.float() + rms[:, None] * delta.float()).to(dtype)
    normed = _rms_norm_after_rounding(stream, norm_weight, norm_eps)

    return CodeWrite(
        normed=normed, stream=stream, raw=raw, rms=rms, packed=packed
    )


def refine_last_write(
    raw: torch.Tensor,
    rms: torch.Tensor,
    packed: torch.Tensor,
    anchor_ids: torch.Tensor,
    candidates: torch.Tensor,
    *,
    exact_code: Callable[[torch.Tensor], torch.Tensor],
    head_row: Callable[[torch.Tensor], torch.Tensor],
    predecessor_gate_weight: torch.Tensor,
    prev_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    write_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    code_size: int,
    eps: float,
    block_size: int,
) -> torch.Tensor:
    """Redo the last code write with the exact sender code and score the candidates.

    raw, rms, packed: the last write's ``CodeWrite`` state for the query rows.
    anchor_ids: [blocks] every block's anchor (bonus) token.
    candidates: [predicted_rows, top_k] candidate ids of every prediction row,
        block-major; ``predicted_rows`` is ``blocks * (block_size - 1)``.
    exact_code: maps token ids to codebook rows.
    head_row: maps token ids to draft language-model-head rows.

    Returns scores [predicted_rows, top_k, top_k] in FP32, where
    ``scores[n, j, c]`` scores candidate ``c`` of row ``n`` when the previous
    row chose lane ``j``. Row 0 of every block scores identically for every
    lane, because its sender is the anchor token whatever the previous choice.
    """
    num_rows = raw.shape[0]
    if num_rows % block_size:
        raise ValueError(
            f"Refinement needs whole request blocks, got {num_rows} rows for a "
            f"block size of {block_size}."
        )
    blocks = num_rows // block_size
    rows_per_block = block_size - 1
    num_pred_rows, top_k = candidates.shape
    if num_pred_rows != blocks * rows_per_block:
        raise ValueError(
            f"Expected {blocks * rows_per_block} prediction rows for {blocks} "
            f"blocks of size {block_size}, got {num_pred_rows}."
        )
    rank = write_weight.shape[1]
    code_scale = code_size**0.5

    scores = torch.empty(
        (num_pred_rows, top_k, top_k), dtype=torch.float32, device=raw.device
    )
    row_index = torch.arange(num_pred_rows, device=raw.device)
    for start in range(0, num_pred_rows, REFINE_ROW_CHUNK):
        rows = row_index[start : start + REFINE_ROW_CHUNK]
        block = rows // rows_per_block
        offset = rows % rows_per_block

        # Data preparation: the row to redo, and who sends its code.
        query_rows = block * block_size + 1 + offset
        lane_ids = rows[:, None].expand(-1, top_k)
        prev_candidates = candidates[(rows - 1).clamp_min(0)]
        sender_ids = torch.where(
            offset[:, None] == 0, anchor_ids[block][:, None].expand(-1, top_k),
            prev_candidates,
        ).reshape(-1)

        # Core result: exact sender code, refined write, candidate scores.
        exact = (exact_code(sender_ids).view(-1, top_k, code_size).float()
                 * code_scale)
        query = (
            packed.view(num_rows, -1)[:, :rank]
            .index_select(0, query_rows)
            .float()
        )
        gate = torch.sigmoid(
            query[:, None, :]
            + exact @ predecessor_gate_weight.float().t()
            + gate_bias.float()
        )
        message = gate * (exact @ prev_weight.float().t())
        base = raw.index_select(0, query_rows).float()
        scale = rms.index_select(0, query_rows).float()
        refined = base[:, None, :] + scale[:, None, None] * (
            message @ write_weight.float().t()
        )
        normalized = (
            refined / (refined.square().mean(-1, keepdim=True) + eps).sqrt()
        ) * norm_weight.float()
        hidden = base.shape[-1]
        head = (
            head_row(candidates[rows].reshape(-1))
            .view(rows.numel(), top_k, hidden)
            .float()
        ).float()
        scores[start : start + rows.numel()] = torch.einsum(
            "njh,nch->njc", normalized, head
        )
    return scores
