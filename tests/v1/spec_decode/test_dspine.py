# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DSpine drafter: its checkpoint contract and its numerics.

DSpine is a DFlash draft with an adjacent causal code bridge and a last-write
refinement (docs/design/dspine-vllm-port.md section 3). The references here are
independent: explicit loops in float64, plus a plain-torch port of the eager
path of Tencent's reference plugin, which fixes the rounding points its fused
kernels use. See `upstream-reference/DSpine/tests/ref_utils.py` (commit
c66bf44) for the upstream originals of `ref_eager_bridge_write` and
`ref_refine_table`.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.utils import counters
from transformers import Qwen3Config

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.models.dspine_bridge import (
    CodeWrite,
    code_bridge_write,
    refine_last_write,
)
from vllm.model_executor.models.dspine_contract import (
    DSPINE_DRAFT_ARCHITECTURE,
    is_dspine_draft,
    resolve_dspine_contract,
    validate_served_block_size,
)
from vllm.model_executor.models.qwen3_dspine import (
    _stores_plain_rows,
    read_rows_through_head,
)

MIN_MEAN_SQUARE = torch.finfo(torch.float32).tiny


def make_bridge_case(
    blocks: int,
    block_size: int,
    *,
    hidden: int = 64,
    code_size: int = 32,
    rank: int = 16,
    vocab: int = 97,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> dict:
    """Random weights and inputs for one draft forward of `blocks` request blocks."""
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape, scale: float = 1.0) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * scale

    rows = blocks * block_size
    return dict(
        hidden=hidden,
        mlp_out=randn(rows, hidden),
        residual=randn(rows, hidden),
        anchor_ids=torch.randint(vocab, (blocks,), generator=generator),
        codebook=randn(vocab, code_size),
        read_weight=randn(code_size, hidden, scale=hidden**-0.5),
        gate_weight=randn(rank, code_size, scale=code_size**-0.5),
        predecessor_gate_weight=randn(rank, code_size, scale=code_size**-0.5),
        prev_weight=randn(rank, code_size, scale=code_size**-0.5),
        gate_bias=randn(rank),
        write_weight=randn(hidden, rank, scale=rank**-0.5),
        norm_weight=1.0 + randn(hidden, scale=0.1),
        block_size=block_size,
        code_size=code_size,
        eps=1e-6,
        norm_eps=1e-6,
        dtype=dtype,
    )


def row_anchor_codes(case: dict) -> torch.Tensor:
    """Codebook row of each row's block anchor, repeated over that block.

    `code_bridge_write` takes the anchor of the block a row belongs to on every
    row, because the write finds the anchor rows itself and must not divide the
    row count to do it.
    """
    return case["codebook"][case["anchor_ids"]].repeat_interleave(
        case["block_size"], dim=0
    )


def cast_bridge_case(case: dict) -> dict:
    """Cast the tensors of a case to its `dtype` and return the FP64 copies."""
    dtype = case["dtype"]
    tensors = {
        key: (
            value.to(dtype)
            if isinstance(value, torch.Tensor) and value.is_floating_point()
            else value
        )
        for key, value in case.items()
    }
    return tensors


def ref_code_bridge_write(case: dict) -> CodeWrite:
    """Float64 reference of one code-bridge write, written as explicit blocks."""
    mlp_out = case["mlp_out"].double()
    residual = case["residual"].double()
    codebook = case["codebook"].double()
    read_weight = case["read_weight"].double()
    write_weight = case["write_weight"].double()
    gate_bias = case["gate_bias"].double()
    norm_weight = case["norm_weight"].double()
    block_size = case["block_size"]
    code_size = case["code_size"]
    rank = write_weight.shape[1]
    eps = case["eps"]
    rows = mlp_out.shape[0]
    blocks = rows // block_size
    packed_weight = torch.cat(
        (
            case["gate_weight"].double(),
            case["predecessor_gate_weight"].double(),
            case["prev_weight"].double(),
        ),
        dim=0,
    )

    raw = mlp_out + residual
    rms = raw.square().mean(-1).clamp_min(MIN_MEAN_SQUARE).sqrt()
    projected = (raw / (rms[:, None] + eps)) @ read_weight.t()

    codes = torch.zeros(blocks, block_size, code_size, dtype=torch.float64)
    codes[:, 0] = codebook[case["anchor_ids"]] * math.sqrt(code_size)
    predicted = projected.view(blocks, block_size, code_size)[:, 1:]
    codes[:, 1:] = (
        math.sqrt(code_size)
        * predicted
        / (predicted.norm(dim=-1, keepdim=True) + eps)
    )

    packed = codes.reshape(rows, code_size) @ packed_weight.t()
    message = torch.zeros(blocks, block_size, rank, dtype=torch.float64)
    gate = torch.sigmoid(
        packed.view(blocks, block_size, 3 * rank)[:, 1:, :rank]
        + packed.view(blocks, block_size, 3 * rank)[:, :-1, rank : 2 * rank]
        + gate_bias
    )
    message[:, 1:] = gate * packed.view(blocks, block_size, 3 * rank)[
        :, :-1, 2 * rank :
    ]

    stream = raw + rms[:, None] * (
        message.reshape(rows, rank) @ write_weight.t()
    )
    normed = stream * torch.rsqrt(
        stream.square().mean(-1, keepdim=True) + case["norm_eps"]
    )
    return CodeWrite(
        normed=normed * norm_weight,
        stream=stream,
        raw=raw,
        rms=rms,
        packed=packed.reshape(rows, 3 * rank),
    )


def ref_eager_bridge_write(case: dict):
    """Plain-torch port of the upstream eager bridge path, rounding points kept.

    Returns (normed, raw, rms, packed) in the case dtype. This is the contract
    the upstream fused Triton kernels are tested against, so matching it fixes
    our arithmetic at the same precision as the published checkpoint.
    """
    block_size = case["block_size"]
    code_size = case["code_size"]
    rank = case["write_weight"].shape[1]
    eps = case["eps"]
    dtype = case["dtype"]

    raw = case["mlp_out"] + case["residual"]
    raw_f = raw.float()
    rms = raw_f.square().mean(-1).clamp_min(MIN_MEAN_SQUARE).sqrt()
    normalized = (raw_f / (rms[:, None] + eps)).to(dtype)
    projected = F.linear(normalized, case["read_weight"]).float()
    codes = math.sqrt(code_size) * projected / (
        projected.norm(dim=-1, keepdim=True) + eps
    )
    anchors = case["anchor_ids"].view(-1)
    anchor_codes = math.sqrt(code_size) * F.embedding(anchors, case["codebook"]).float()
    codes = codes.view(-1, block_size, code_size)
    codes = torch.cat((anchor_codes[:, None, :], codes[:, 1:]), dim=1).view(
        -1, code_size
    )
    packed_weight = torch.cat(
        (
            case["gate_weight"],
            case["predecessor_gate_weight"],
            case["prev_weight"],
        ),
        dim=0,
    )
    packed = F.linear(codes.to(dtype), packed_weight)
    query, sender, value = (
        packed.float().view(-1, block_size, 3 * rank).split(rank, dim=-1)
    )
    gate = torch.sigmoid(query[:, 1:] + sender[:, :-1] + case["gate_bias"].float())
    message = torch.cat(
        (torch.zeros_like(query[:, :1]), gate * value[:, :-1]), dim=1
    ).to(dtype)
    delta = F.linear(message.view(-1, rank), case["write_weight"]).float()
    row = torch.arange(raw.shape[0], device=raw.device)
    not_anchor = ((row % block_size != 0).float()[:, None])
    out = (raw_f + rms[:, None] * delta * not_anchor).to(dtype)
    x = out.float()
    normalized_out = (
        x * torch.rsqrt(x.square().mean(-1, keepdim=True) + case["norm_eps"])
    ).to(dtype)
    normed = (normalized_out.float() * case["norm_weight"].float()).to(dtype)
    return normed, raw, rms, packed


@pytest.mark.parametrize("block_size", [8, 16])
def test_code_bridge_write_matches_float64_reference(block_size: int):
    case = cast_bridge_case(make_bridge_case(3, block_size, dtype=torch.float32))
    expected = ref_code_bridge_write({**case, "dtype": torch.float64})

    actual = code_bridge_write(
        case["mlp_out"],
        case["residual"],
        row_anchor_codes(case),
        read_weight=case["read_weight"],
        packed_weight=torch.cat(
            (case["gate_weight"], case["predecessor_gate_weight"], case["prev_weight"]),
            dim=0,
        ),
        gate_bias=case["gate_bias"],
        write_weight=case["write_weight"],
        norm_weight=case["norm_weight"],
        block_size=case["block_size"],
        code_size=case["code_size"],
        eps=case["eps"],
        norm_eps=case["norm_eps"],
    )

    # The implementation works in FP32, the reference in FP64.
    for name in ("normed", "stream", "raw", "rms", "packed"):
        torch.testing.assert_close(
            getattr(actual, name).double(), getattr(expected, name),
            rtol=1e-4, atol=1e-5,
        )


@pytest.mark.parametrize("block_size", [2, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_code_bridge_write_matches_the_upstream_eager_path(block_size: int, dtype):
    """The fused path must agree with the eager path at the same rounding points.

    Compared bit-exactly in both precisions: the FP16 worker of an SM75 route
    shifts every rounding, so a drift here would be invisible in acceptance
    length but wrong.
    """
    case = cast_bridge_case(make_bridge_case(2, block_size, dtype=dtype))
    expected_normed, expected_raw, expected_rms, expected_packed = (
        ref_eager_bridge_write(case)
    )

    actual = code_bridge_write(
        case["mlp_out"],
        case["residual"],
        row_anchor_codes(case),
        read_weight=case["read_weight"],
        packed_weight=torch.cat(
            (case["gate_weight"], case["predecessor_gate_weight"], case["prev_weight"]),
            dim=0,
        ),
        gate_bias=case["gate_bias"],
        write_weight=case["write_weight"],
        norm_weight=case["norm_weight"],
        block_size=case["block_size"],
        code_size=case["code_size"],
        eps=case["eps"],
        norm_eps=case["norm_eps"],
    )

    torch.testing.assert_close(actual.normed, expected_normed, rtol=0, atol=0)
    torch.testing.assert_close(actual.raw, expected_raw, rtol=0, atol=0)
    torch.testing.assert_close(actual.rms, expected_rms, rtol=0, atol=0)
    torch.testing.assert_close(actual.packed, expected_packed, rtol=0, atol=0)


@pytest.mark.parametrize("block_size", [8, 16])
def test_code_bridge_anchor_rows_keep_their_own_stream(block_size: int):
    """An anchor row gets no message, so its write is the identity on the stream.

    This is what makes the anchor's drafted token (the bonus token the target
    already accepted) free of any draft-side prediction error.
    """
    case = cast_bridge_case(make_bridge_case(3, block_size))
    anchor_codes = row_anchor_codes(case)
    write = lambda codes: code_bridge_write(  # noqa: E731
        case["mlp_out"],
        case["residual"],
        codes,
        read_weight=case["read_weight"],
        packed_weight=torch.cat(
            (case["gate_weight"], case["predecessor_gate_weight"], case["prev_weight"]),
            dim=0,
        ),
        gate_bias=case["gate_bias"],
        write_weight=case["write_weight"],
        norm_weight=case["norm_weight"],
        block_size=case["block_size"],
        code_size=case["code_size"],
        eps=case["eps"],
        norm_eps=case["norm_eps"],
    )

    actual = write(anchor_codes)
    anchors = torch.arange(0, actual.stream.shape[0], block_size)
    torch.testing.assert_close(
        actual.stream[anchors], actual.raw[anchors], rtol=0, atol=0
    )

    # The anchor code enters the stream only through its neighbours, so replacing
    # the anchor codes cannot change an anchor row either.
    perturbed = write(torch.zeros_like(anchor_codes))
    torch.testing.assert_close(perturbed.stream[anchors], actual.stream[anchors])


@pytest.mark.parametrize("block_size", [8, 16])
@pytest.mark.parametrize("offset", [0, 1, 5, 7, 15])
def test_code_bridge_write_stays_inside_its_block(block_size: int, offset: int):
    """A row's write only ever reads its own row and its predecessor's.

    The bridge reads the predecessor row by row index, so a drafter bug that let
    a block read across its boundary would silently leak one request's codes
    into the next request's draft.
    """
    if offset >= block_size:
        pytest.skip(f"block size {block_size} has no offset {offset}")
    case = cast_bridge_case(make_bridge_case(3, block_size))
    kwargs = dict(
        read_weight=case["read_weight"],
        packed_weight=torch.cat(
            (case["gate_weight"], case["predecessor_gate_weight"], case["prev_weight"]),
            dim=0,
        ),
        gate_bias=case["gate_bias"],
        write_weight=case["write_weight"],
        norm_weight=case["norm_weight"],
        block_size=case["block_size"],
        code_size=case["code_size"],
        eps=case["eps"],
        norm_eps=case["norm_eps"],
    )
    anchor_codes = row_anchor_codes(case)
    baseline = code_bridge_write(
        case["mlp_out"], case["residual"], anchor_codes, **kwargs
    )

    perturbed_mlp_out = case["mlp_out"].clone()
    row = block_size + offset
    perturbed_mlp_out[row] += 4.0
    perturbed = code_bridge_write(
        perturbed_mlp_out, case["residual"], anchor_codes, **kwargs
    )

    # Nothing outside the perturbed block moves, not even the next block's
    # anchor row, which is where a cross-block leak would show up.
    outside = torch.ones(3 * block_size, dtype=torch.bool)
    outside[block_size : 2 * block_size] = False
    torch.testing.assert_close(
        perturbed.stream[outside], baseline.stream[outside], rtol=0, atol=0
    )

    block = slice(block_size, 2 * block_size)
    changed = ~torch.isclose(
        perturbed.stream[block], baseline.stream[block]
    ).all(dim=-1)
    # A row's write reads its predecessor's code, so a change reaches row + 1.
    # An anchor row is the exception: it emits the exact codebook row of its
    # token, so its own features never reach a neighbour.
    reaches_neighbour = offset != 0
    expected = [
        row == block_size + i or (reaches_neighbour and row + 1 == block_size + i)
        for i in range(block_size)
    ]
    assert changed.tolist() == expected


def bridge_write_kwargs(case: dict) -> dict:
    """The keyword half of a `code_bridge_write` call, from a bridge case."""
    return dict(
        read_weight=case["read_weight"],
        packed_weight=torch.cat(
            (case["gate_weight"], case["predecessor_gate_weight"], case["prev_weight"]),
            dim=0,
        ),
        gate_bias=case["gate_bias"],
        write_weight=case["write_weight"],
        norm_weight=case["norm_weight"],
        block_size=case["block_size"],
        code_size=case["code_size"],
        eps=case["eps"],
        norm_eps=case["norm_eps"],
    )


def test_code_bridge_write_keeps_the_row_dimension_dynamic():
    """One compiled graph serves every block count the draft runs with.

    vLLM marks the token dimension of the draft model dynamic, and the write runs
    inside that compiled forward. Geometry that divides the row count solves the
    symbol into a constant, and on a marked dimension that is an error rather
    than a recompile: the engine fails to start instead of falling back.
    """
    torch._dynamo.reset()
    counters.clear()
    written = torch.compile(code_bridge_write, dynamic=True)

    for blocks in (1, 2, 3):
        case = cast_bridge_case(make_bridge_case(blocks, 8))
        args = (case["mlp_out"], case["residual"], row_anchor_codes(case))
        for rows in args:
            torch._dynamo.mark_dynamic(rows, 0)
        actual = written(*args, **bridge_write_kwargs(case))
        expected = code_bridge_write(*args, **bridge_write_kwargs(case))
        # Inductor reorders the reductions, so this compares at the tolerance the
        # FP32 reference tests use rather than bit for bit.
        torch.testing.assert_close(
            actual.stream, expected.stream, rtol=1e-4, atol=1e-5
        )

    graphs = counters["stats"]["unique_graphs"]
    assert graphs == 1, f"the write needed {graphs} graphs for 3 block counts"


def ref_refine_table(case: dict, raw, rms, packed, candidates) -> torch.Tensor:
    """Float64 [rows, K, K] table of the redone last write, explicit loops."""
    block_size = case["block_size"]
    code_size = case["code_size"]
    eps = case["eps"]
    write_weight = case["write_weight"].double()
    predecessor_gate = case["predecessor_gate_weight"].double()
    prev_weight = case["prev_weight"].double()
    gate_bias = case["gate_bias"].double()
    norm_weight = case["norm_weight"].double()
    codebook = case["codebook"].double()
    head = case["head"].double()
    anchors = case["anchor_ids"]
    rows_per_block = block_size - 1
    num_pred_rows, top_k = candidates.shape
    rank = write_weight.shape[1]

    table = torch.zeros(num_pred_rows, top_k, top_k, dtype=torch.float64)
    for n in range(num_pred_rows):
        block, offset = divmod(n, rows_per_block)
        query_row = block * block_size + 1 + offset
        senders = (
            [int(anchors[block])] * top_k
            if offset == 0
            else candidates[n - 1].tolist()
        )
        query = packed[query_row, :rank].double()
        for lane, sender in enumerate(senders):
            exact = math.sqrt(code_size) * codebook[sender]
            gate = torch.sigmoid(
                query + predecessor_gate @ exact + gate_bias
            )
            message = gate * (prev_weight @ exact)
            refined = raw[query_row].double() + rms[query_row].double() * (
                write_weight @ message
            )
            normalized = (
                refined / torch.sqrt(refined.square().mean() + eps) * norm_weight
            )
            table[n, lane] = head[candidates[n]] @ normalized
    return table


def make_refinement_case(blocks: int, block_size: int, top_k: int, seed: int = 7):
    case = make_bridge_case(blocks, block_size, seed=seed, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed + 1)
    vocab = case["codebook"].shape[0]
    case["head"] = torch.randn(vocab, case["hidden"], generator=generator) * 0.2
    case["candidates"] = torch.stack(
        [
            torch.randperm(vocab, generator=generator)[:top_k].sort().values
            for _ in range(blocks * (block_size - 1))
        ]
    )
    return case


def run_refine(case: dict, write: CodeWrite) -> torch.Tensor:
    return refine_last_write(
        write.raw,
        write.rms,
        write.packed,
        case["anchor_ids"],
        case["candidates"],
        exact_code=lambda ids: case["codebook"][ids],
        head_row=lambda ids: case["head"][ids],
        predecessor_gate_weight=case["predecessor_gate_weight"],
        prev_weight=case["prev_weight"],
        gate_bias=case["gate_bias"],
        write_weight=case["write_weight"],
        norm_weight=case["norm_weight"],
        code_size=case["code_size"],
        eps=case["eps"],
        block_size=case["block_size"],
    )


def last_write(case: dict) -> CodeWrite:
    return code_bridge_write(
        case["mlp_out"],
        case["residual"],
        row_anchor_codes(case),
        read_weight=case["read_weight"],
        packed_weight=torch.cat(
            (case["gate_weight"], case["predecessor_gate_weight"], case["prev_weight"]),
            dim=0,
        ),
        gate_bias=case["gate_bias"],
        write_weight=case["write_weight"],
        norm_weight=case["norm_weight"],
        block_size=case["block_size"],
        code_size=case["code_size"],
        eps=case["eps"],
        norm_eps=case["norm_eps"],
    )


@pytest.mark.parametrize("block_size", [8, 16])
def test_refine_last_write_matches_float64_reference(block_size: int):
    case = make_refinement_case(2, block_size, top_k=4)
    write = last_write(case)

    actual = run_refine(case, write)
    expected = ref_refine_table(
        case,
        write.raw.double(),
        write.rms.double(),
        write.packed.double(),
        case["candidates"],
    )

    assert actual.dtype is torch.float32
    assert actual.shape == (2 * (block_size - 1), 4, 4)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("block_size", [8, 16])
def test_refine_first_prediction_row_is_lane_independent(block_size: int):
    """Every lane of a block's first prediction row sends the anchor's code."""
    case = make_refinement_case(2, block_size, top_k=4)
    table = run_refine(case, last_write(case))

    rows_per_block = block_size - 1
    for block in range(2):
        first_row = block * rows_per_block
        for lane in range(1, table.shape[2]):
            torch.testing.assert_close(
                table[first_row, lane], table[first_row, 0], rtol=0, atol=0
            )


def test_refine_rejects_a_partial_block():
    case = make_refinement_case(1, 8, top_k=4)
    write = last_write(case)
    with pytest.raises(ValueError, match="whole request blocks"):
        refine_last_write(
            write.raw[:7],
            write.rms[:7],
            write.packed[:7],
            case["anchor_ids"],
            case["candidates"],
            exact_code=lambda ids: case["codebook"][ids],
            head_row=lambda ids: case["head"][ids],
            predecessor_gate_weight=case["predecessor_gate_weight"],
            prev_weight=case["prev_weight"],
            gate_bias=case["gate_bias"],
            write_weight=case["write_weight"],
            norm_weight=case["norm_weight"],
            code_size=case["code_size"],
            eps=case["eps"],
            block_size=8,
        )


def ref_candidate_walk(
    scores: torch.Tensor, candidate_ids: torch.Tensor, num_steps: int
) -> list[int]:
    """Greedy serial chain: the lane chosen at a row is the sender of the next."""
    _, _, top_k, _ = scores.shape
    lane = 0
    tokens = []
    for step in range(num_steps):
        index = int(scores[0, step, lane].argmax())
        tokens.append(int(candidate_ids[0, step, index]))
        lane = index
    return tokens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_candidate_walk_follows_the_serial_chain():
    from vllm.v1.worker.gpu.spec_decode.utils import walk_candidate_scores

    torch.manual_seed(3)
    num_reqs, num_steps, top_k = 2, 3, 4
    # scores[n, j, :] peaks at the lane this cell wants to hand to the next row,
    # so the walk is a forced path rather than a search the test has to predict.
    lanes = (torch.arange(top_k) + 1) % top_k
    scores = torch.zeros(num_reqs, num_steps, top_k, top_k)
    for step in range(num_steps):
        for lane in range(top_k):
            scores[:, step, lane, lanes[(lane + step) % top_k]] = 1.0
    scores += torch.rand(num_reqs, num_steps, top_k, top_k) * 1e-3
    candidate_ids = torch.arange(num_reqs * num_steps * top_k).view(
        num_reqs, num_steps, top_k
    )

    device = "cuda"
    # The second request is inert: its sample rows are unmapped, and the walk
    # must leave its draft tokens alone instead of scattering into slot -1.
    sample_idx_mapping = torch.tensor([0] * num_steps + [-1] * num_steps, device=device)
    draft_tokens = torch.full(
        (num_reqs, num_steps), -7, dtype=torch.int64, device=device
    )
    realized_scores = torch.zeros(
        num_reqs, num_steps, top_k, dtype=torch.float32, device=device
    )
    walk_candidate_scores(
        scores.to(device),
        candidate_ids.to(device),
        top_k=top_k,
        num_steps=num_steps,
        num_reqs=num_reqs,
        sample_pos=torch.zeros(num_reqs * num_steps, dtype=torch.int64, device=device),
        sample_idx_mapping=sample_idx_mapping,
        temperature=torch.zeros(num_reqs, dtype=torch.float32, device=device),
        seeds=torch.zeros(num_reqs, dtype=torch.int64, device=device),
        draft_tokens=draft_tokens,
        realized_scores=realized_scores,
        sample_probabilistic=False,
        use_fp64=False,
    )

    expected = ref_candidate_walk(scores, candidate_ids, num_steps)
    assert draft_tokens[0].tolist() == expected
    assert draft_tokens[1].tolist() == [-7] * num_steps
    lane = 0
    for step in range(num_steps):
        index = int(scores[0, step, lane].argmax())
        torch.testing.assert_close(
            realized_scores[0, step].cpu(), scores[0, step, lane], rtol=0, atol=0
        )
        lane = index


# --- Checkpoint contract -----------------------------------------------------

# The shape of tencent/Qwen3.8-27B-DSpine-b16, shrunk to dimensions a unit test
# can afford. The field names and the relations between them are what the host
# reads, so they are what a checkpoint must be able to state.
def dspine_config(**overrides) -> Qwen3Config:
    dflash_config = {
        "draft_block_causal": True,
        "mask_token_id": 248070,
        "selected_prefix_code_bridge": {
            "code_size": 32,
            "eps": 1e-6,
            "kappa": 10.0,
            "rank": 16,
        },
        "selected_prefix_decode_mode": "serial",
        "selected_prefix_feature_layers": [1, 2, 3, 4, 5],
        "selected_prefix_feature_write_mode": "independent",
        "selected_prefix_feature_writes": True,
        "selected_prefix_last_write_refine": {
            "head_topk": 16,
            "kind": "exact_sender",
            "version": 1,
            "weight": 1.0,
        },
        "selected_prefix_terminal": False,
        "target_layer_ids": [1, 16, 31, 46, 61],
    }
    second_pass = {"mode": "serial_table", "first_exact": 4, "head_topk": 16}
    fields = dict(
        model_type="qwen3",
        architectures=["DFlashDraftModel"],
        hidden_size=64,
        vocab_size=97,
        draft_vocab_size=97,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        block_size=16,
        is_causal=True,
        code_bridge_second_pass=second_pass,
        dflash_config=dflash_config,
    )
    for key, value in overrides.items():
        nested = key in {"dflash_config", "code_bridge_second_pass"}
        if nested and isinstance(value, dict):
            merged = dict(fields[key])
            merged.update(value)
            fields[key] = merged
        else:
            fields[key] = value
    return Qwen3Config(**fields)


def plain_dflash_config() -> Qwen3Config:
    """A DFlash draft with no code bridge: the same family, a different drafter."""
    return Qwen3Config(
        model_type="qwen3",
        architectures=["DFlashDraftModel"],
        hidden_size=64,
        vocab_size=97,
        num_hidden_layers=5,
        block_size=8,
        dflash_config={"mask_token_id": 248070},
    )


def test_dspine_contract_reads_the_published_checkpoint_shape():
    contract = resolve_dspine_contract(dspine_config())

    assert contract.code_size == 32
    assert contract.rank == 16
    assert contract.head_topk == 16
    assert contract.num_layers == 5
    assert contract.trained_block_size == 16
    # The config names layers 1..5; the host works in zero-based indices.
    assert contract.active_layers == frozenset(range(5))


def test_dspine_identity_lives_in_the_bridge_not_the_architecture():
    """A DSpine checkpoint declares itself a plain DFlash draft.

    This is why vllm/config/speculative.py has to rewrite the architecture to
    `DSPINE_DRAFT_ARCHITECTURE`: routing on `architectures` alone would load the
    bridge weights into a model that ignores them.
    """
    config = dspine_config()

    assert config.architectures == ["DFlashDraftModel"]
    assert is_dspine_draft(config)
    assert DSPINE_DRAFT_ARCHITECTURE == "DSpineDraftModel"
    assert not is_dspine_draft(plain_dflash_config())


def _rejects(config: Qwen3Config, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_dspine_contract(config)


def test_dspine_contract_rejects_a_terminal_enabled_bridge():
    _rejects(
        dspine_config(dflash_config={"selected_prefix_terminal": True}),
        "terminal-disabled",
    )


def test_dspine_contract_rejects_a_shared_write_mode():
    _rejects(
        dspine_config(
            dflash_config={"selected_prefix_feature_write_mode": "shared"}
        ),
        "independent learned writes",
    )


def test_dspine_contract_rejects_a_non_causal_draft():
    """The bridge hands a message from the predecessor, so a bidirectional
    block would let every position read the token it is supposed to predict."""
    _rejects(
        dspine_config(dflash_config={"draft_block_causal": False}),
        "draft_block_causal",
    )
    _rejects(dspine_config(is_causal=False), "is_causal")


def test_dspine_contract_rejects_an_incomplete_write_schedule():
    _rejects(
        dspine_config(dflash_config={"selected_prefix_feature_layers": [1, 2, 3, 4]}),
        "must list all of them",
    )


def test_dspine_contract_rejects_a_refinement_that_is_not_the_last_write():
    _rejects(
        dspine_config(code_bridge_second_pass={"first_exact": 2}),
        "first_exact",
    )


def test_dspine_contract_rejects_refinement_widths_that_disagree():
    _rejects(
        dspine_config(
            code_bridge_second_pass={"head_topk": 8},
            dflash_config={
                "selected_prefix_last_write_refine": {
                    "kind": "exact_sender",
                    "head_topk": 16,
                }
            },
        ),
        "disagree",
    )


def test_dspine_contract_rejects_a_refinement_width_the_walk_cannot_lane():
    _rejects(
        dspine_config(
            code_bridge_second_pass={"head_topk": 12},
            dflash_config={
                "selected_prefix_last_write_refine": {
                    "kind": "exact_sender",
                    "head_topk": 12,
                }
            },
        ),
        "power of two",
    )


def test_dspine_contract_rejects_a_draft_that_is_not_plain_dflash():
    """Convolutions and a candidate selector belong to other drafters."""
    _rejects(
        dspine_config(dflash_config={"conv_kernel_size": 2}),
        "unsupported draft config keys",
    )


def test_dspine_contract_rejects_a_truncated_vocabulary():
    """The codebook is indexed by token id in two places, so a draft vocabulary
    smaller than the target's would index the wrong rows without any error."""
    _rejects(dspine_config(draft_vocab_size=64), "vocabularies must match")


def test_dspine_serves_a_smaller_block_than_trained_but_never_a_larger_one():
    contract = resolve_dspine_contract(dspine_config())

    assert validate_served_block_size(contract, 8) == 8
    assert validate_served_block_size(contract, 16) == 16
    with pytest.raises(ValueError, match="2..16"):
        validate_served_block_size(contract, 32)
    with pytest.raises(ValueError, match="2..16"):
        validate_served_block_size(contract, 1)


# --- Output-matrix rows for the refinement -----------------------------------


def test_head_rows_read_back_the_matrix_they_came_from():
    """Reading an output matrix with the identity recovers its rows.

    The chunk width deliberately does not divide the hidden size, so the last
    partial read is checked too.
    """
    matrix = torch.randn(37, 12)

    rows = read_rows_through_head(
        lambda block: block @ matrix.t(),
        hidden_size=12,
        dtype=matrix.dtype,
        device=matrix.device,
        chunk_rows=5,
    )

    torch.testing.assert_close(rows, matrix, rtol=1e-5, atol=1e-6)


def test_head_rows_read_back_survive_a_repacked_head():
    """The read reaches a matrix a gather could not: it goes through the head.

    A quantized target stores its output matrix packed for its own gemm, and a
    DFlash draft inherits that matrix, so the rows are not reachable as a tensor
    the host could index. Reading through a head that keeps its weights
    column-major shows the recovery does not care how they are stored.
    """
    matrix = torch.randn(37, 12)
    packed = matrix.t().contiguous()

    rows = read_rows_through_head(
        lambda block: block @ packed,
        hidden_size=12,
        dtype=matrix.dtype,
        device=matrix.device,
        chunk_rows=12,
    )

    torch.testing.assert_close(rows, matrix, rtol=1e-5, atol=1e-6)


def _head_like(quant_method, weight) -> SimpleNamespace:
    return SimpleNamespace(quant_method=quant_method, weight=weight)


def test_plain_row_table_is_recognized_only_for_a_plain_row_table():
    """A head needs no copy exactly when its weight is rows of features."""
    hidden = 12
    plain = _head_like(UnquantizedEmbeddingMethod(), torch.zeros(37, hidden))
    transposed = _head_like(UnquantizedEmbeddingMethod(), torch.zeros(hidden, 37))
    linear = _head_like(UnquantizedLinearMethod(), torch.zeros(37, hidden))
    packed = _head_like(object(), torch.zeros(hidden // 16, 37, dtype=torch.int32))

    assert _stores_plain_rows(plain, hidden)
    assert _stores_plain_rows(linear, hidden)
    assert not _stores_plain_rows(transposed, hidden)
    assert not _stores_plain_rows(packed, hidden)
