# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpine draft model: a DFlash Qwen3 drafter with an adjacent causal code bridge.

Tencent ships DSpine as an SGLang plugin only. This is the vLLM host of the same
checkpoint: the DFlash context fusion and block drafting are inherited from
``qwen3_dflash.py``, and this file adds the two pieces DSpine brings (see
docs/design/dspine-vllm-port.md section 3):

* a code write after every layer, so each position reads its predecessor's
  prediction; the arithmetic lives in ``dspine_bridge.py``;
* a last-write refinement that rescores the ``head_topk`` candidates of every
  position with the exact code of the token its predecessor chose, producing the
  ``[rows, K, K]`` table the shared candidate walk consumes.

Weights keep their checkpoint names. The two version scalars are consumed at load
time and the terminal projection is checked to stay zero, because both mark a
checkpoint this implementation must refuse rather than mis-serve.
"""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)

from .dspine_bridge import CodeWrite, code_bridge_write, refine_last_write
from .dspine_contract import (
    DSpineContract,
    resolve_dspine_contract,
    validate_served_block_size,
)
from .qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from .utils import WeightsMapper, maybe_prefix

logger = init_logger(__name__)


def gather_sharded_rows(
    table: torch.Tensor,
    shard_indices: Any,
    tp_size: int,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Rows of a vocabulary-sharded table, summed over the tensor-parallel shards.

    The output matrix refuses the embedding-style forward that would do this,
    because its weights belong to the sampler, so the lookup an embedding layer
    performs is repeated here on a plain row table.
    """
    if tp_size == 1:
        return F.embedding(token_ids.long(), table)

    masked_ids, unowned = get_masked_input_and_mask(
        token_ids,
        shard_indices.org_vocab_start_index,
        shard_indices.org_vocab_end_index,
        shard_indices.num_org_vocab_padding,
        shard_indices.added_vocab_start_index,
        shard_indices.added_vocab_end_index,
    )
    rows = F.embedding(masked_ids.long(), table)
    rows.masked_fill_(unowned.unsqueeze(-1), 0)
    return tensor_model_parallel_all_reduce(rows)


def _stores_plain_rows(head: nn.Module, hidden_size: int) -> bool:
    """Whether a head's weight already holds one plain row per vocabulary entry.

    A quantized head keeps its weights packed for its own gemm instead, and a
    DFlash draft inherits the target's head, so this has to be asked rather than
    assumed. Accept both unquantized methods, as `LogitsProcessor._apply_head`
    does: a quant config that excludes the head hands out the linear method.
    """
    if not isinstance(
        head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
    ):
        return False
    weight = head.weight
    return weight.dim() == 2 and weight.shape[1] == hidden_size


class _ReplicatedProjection(nn.Module):
    """One code-bridge projection, kept whole on every rank.

    Named like the checkpoint's ``nn.Linear`` so the bridge weights load without
    renaming. These projections are a few MB each and are combined into a single
    packed GEMM by the bridge, so tensor parallelism would cost more than it
    saves.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        params_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=params_dtype),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=params_dtype), False)
            if bias
            else None
        )


class DSpineCodeBridge(nn.Module):
    """The adjacent causal code writes shared by every active draft layer."""

    def __init__(
        self,
        contract: DSpineContract,
        served_block_size: int,
        params_dtype: torch.dtype,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.contract = contract
        self.block_size = served_block_size
        hidden, code, rank = contract.hidden_size, contract.code_size, contract.rank
        vocab, layers = contract.vocab_size, contract.num_layers

        # The codebook is indexed by token id twice per step: for the anchor code
        # of every block, and for the exact sender codes of the refinement. It is
        # the bridge's only large tensor, so it is sharded like a vocabulary.
        self.codebook = VocabParallelEmbedding(
            vocab, code, prefix=maybe_prefix(prefix, "codebook")
        )
        self.read_weight = nn.Parameter(
            torch.empty(code, hidden, dtype=params_dtype), requires_grad=False
        )
        # Layers 0..num_layers-2 read their code through their own weight; the
        # last layer uses `read_weight`.
        self.early_read_weights = nn.Parameter(
            torch.empty(max(layers - 1, 0), code, hidden, dtype=params_dtype),
            requires_grad=False,
        )
        self.feature_write_weights = nn.Parameter(
            torch.empty(layers, hidden, rank, dtype=params_dtype), requires_grad=False
        )
        self.gate = _ReplicatedProjection(code, rank, True, params_dtype)
        self.predecessor_gate = _ReplicatedProjection(code, rank, False, params_dtype)
        self.prev = _ReplicatedProjection(code, rank, False, params_dtype)
        # Terminal projection. A terminal-disabled checkpoint stores zeros here,
        # and `prepare` refuses anything else.
        self.out = _ReplicatedProjection(rank, code, False, params_dtype)
        for name in ("independent_write_version", "terminal_disabled_version"):
            self.register_buffer(name, torch.zeros((), dtype=torch.int64))
        # [gate; predecessor_gate; prev] rows, concatenated once after loading so
        # the bridge needs one GEMM per layer instead of three.
        self.register_buffer("packed_weight", None, persistent=False)
        # Set by `prepare`, which must run after the weights land.
        self.ready = False

    def prepare(self) -> None:
        """Fold the three sender projections into one and check the checkpoint."""
        if self.out.weight.count_nonzero().item():
            raise ValueError(
                "A terminal-disabled DSpine checkpoint must store a zero "
                "selected_prefix_bridge.out.weight; this one does not."
            )
        for name in ("independent_write_version", "terminal_disabled_version"):
            if getattr(self, name).item() != 1:
                raise ValueError(
                    f"Unsupported DSpine checkpoint: {name} must be 1, got "
                    f"{getattr(self, name).item()}."
                )
        packed = torch.cat(
            (
                self.gate.weight,
                self.predecessor_gate.weight,
                self.prev.weight,
            ),
            dim=0,
        ).contiguous()
        if self.packed_weight is None:
            self.packed_weight = packed
        else:
            self.packed_weight.copy_(packed)
        self.ready = True

    def layer_read_weight(self, layer: int) -> torch.Tensor:
        contract = self.contract
        if layer == contract.num_layers - 1:
            return self.read_weight
        return self.early_read_weights[layer]

    def exact_code(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Codebook rows of `token_ids`, gathered across the vocabulary shards."""
        return self.codebook(token_ids)

    def forward(
        self,
        mlp_out: torch.Tensor,
        residual: torch.Tensor,
        anchor_ids: torch.Tensor,
        layer: int,
        next_norm: nn.Module,
    ) -> CodeWrite:
        """One code write after `layer`, fused with the next layer's input norm."""
        if not self.ready:
            raise RuntimeError(
                "The DSpine code bridge was used before its weights were prepared."
            )
        contract = self.contract
        return code_bridge_write(
            mlp_out,
            residual,
            self.exact_code(anchor_ids),
            read_weight=self.layer_read_weight(layer),
            packed_weight=self.packed_weight,
            gate_bias=self.gate.bias,
            write_weight=self.feature_write_weights[layer],
            norm_weight=next_norm.weight,
            block_size=self.block_size,
            code_size=contract.code_size,
            eps=contract.eps,
            norm_eps=next_norm.variance_epsilon,
        )


class DSpineQwen3DecoderLayer(DFlashQwen3DecoderLayer):
    """DFlash layer that can take an input a fused code write already normed."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        input_normed: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_normed:
            # The preceding code write applied `input_layernorm` itself, as part
            # of the same kernel the reference uses. Applying it twice would
            # quietly change the draft's activations.
            assert residual is not None
        elif residual is not None:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DSpineQwen3Model(DFlashQwen3Model):
    decoder_layer_cls = DSpineQwen3DecoderLayer

    hf_to_vllm_mapper = DFlashQwen3Model.hf_to_vllm_mapper | WeightsMapper(
        # The checkpoint stores the codebook itself; the bridge holds it in a
        # vocab-sharded embedding, which adds the `.weight` suffix.
        orig_to_new_substr={
            "selected_prefix_bridge.codebook": "selected_prefix_bridge.codebook.weight"
        },
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.contract = resolve_dspine_contract(self.config)
        self.served_block_size = validate_served_block_size(
            self.contract, 1 + speculative_config.num_speculative_tokens
        )
        self.selected_prefix_bridge = DSpineCodeBridge(
            self.contract,
            self.served_block_size,
            vllm_config.model_config.dtype,
            prefix=maybe_prefix(prefix, "selected_prefix_bridge"),
        )
        # Version scalars, consumed by `load_weights` of the draft head.
        for name in ("draft_block_causal_version", "last_write_refine_version"):
            self.register_buffer(name, torch.zeros((), dtype=torch.int64))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draft the block, and keep the last write open for the refinement.

        Returns (normed hidden states, pre-write stream, write scale, packed
        codes). The last three are what the last-write refinement needs to redo
        the final write with the exact code of each position's predecessor.
        """
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        bridge = self.selected_prefix_bridge
        block_size = bridge.block_size
        anchor_ids = input_ids[::block_size]
        last_layer = len(self.layers) - 1

        hidden_states = input_embeds
        residual: torch.Tensor | None = None
        write: CodeWrite | None = None
        for index, layer in enumerate(self.layers):
            # Every write applies the next layer's input norm, so only the first
            # layer normalizes its own input.
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                input_normed=index > 0,
            )
            next_norm = (
                self.layers[index + 1].input_layernorm
                if index < last_layer
                else self.norm
            )
            write = bridge(hidden_states, residual, anchor_ids, index, next_norm)
            hidden_states, residual = write.normed, write.stream

        assert write is not None, "the contract guarantees at least one draft layer"
        return hidden_states, write.raw, write.rms, write.packed


class DSpineQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DSpineQwen3Model

    # Rows of the identity matrix read through the head per step of
    # materialize_head_rows, which bounds the logits buffer it builds.
    HEAD_ROWS_READ_CHUNK = 64

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.head_topk = self.model.contract.head_topk
        # One plain row per vocabulary entry of this rank's shard, set by
        # materialize_head_rows. Either the head's own weight or a copy read
        # back through it; see that method for when a copy is needed.
        self._head_rows: torch.Tensor | None = None

    def materialize_head_rows(self) -> None:
        """Give the refinement the candidate rows of the output matrix.

        The refinement rescores every candidate of every position, which needs
        the candidates' *rows*, not only their logits; the reference gathers
        `lm_head_w [V, H]` by row. A DFlash draft does not own its output matrix
        (the spec-decode loader hands it the target's `lm_head`), and a
        quantized target keeps that weight packed for its own gemm, where no row
        lookup here could follow it. So the rows are read back through the head:
        with the identity matrix as input, the head's output is its own column,
        and a one-hot input survives any activation quantization exactly. The
        copy therefore holds the values the target itself scores with.

        Called once by the speculator after the head swap, so it runs before the
        KV cache is sized and before any cudagraph is captured.
        """
        head = self.lm_head
        hidden_size = self.config.hidden_size
        if _stores_plain_rows(head, hidden_size):
            self._head_rows = head.weight
            logger.info(
                "DSpine refinement: gathering rows from the unquantized head."
            )
            return

        rows: torch.Tensor | None = None
        # One-hot inputs are what make the read exact, so the head is fed the
        # identity matrix and its output is handed back as its own columns.
        eye = torch.eye(
            hidden_size,
            dtype=self.model.norm.weight.dtype,
            device=head.weight.device,
        )
        for start in range(0, hidden_size, self.HEAD_ROWS_READ_CHUNK):
            block = eye[start : start + self.HEAD_ROWS_READ_CHUNK]
            # The same projection get_top_k_tokens uses, so the rows and the
            # candidate logits a draft step reports come from one numeric path.
            logits = self.logits_processor._apply_head(head, block, None)
            if rows is None:
                rows = torch.empty(
                    logits.shape[1],
                    hidden_size,
                    dtype=logits.dtype,
                    device=block.device,
                )
            rows[:, start : start + block.shape[0]] = logits.t().to(rows.dtype)
        del eye

        assert rows is not None
        self._head_rows = rows
        logger.info(
            "DSpine refinement: read %d output-matrix rows (%.2f GiB) back "
            "through the %s head for the code bridge.",
            rows.shape[0],
            rows.numel() * rows.element_size() / 2**30,
            type(head.quant_method).__name__,
        )

    def head_rows(self, token_ids: torch.Tensor) -> torch.Tensor:
        """The rows of the output matrix that `token_ids` select, TP-correct."""
        if self._head_rows is None:
            raise RuntimeError(
                "DSpine refinement needs the output-matrix rows, but "
                "materialize_head_rows has not run."
            )
        return gather_sharded_rows(
            self._head_rows,
            self.lm_head.shard_indices,
            self.lm_head.tp_size,
            token_ids,
        )

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The `head_topk` candidates of every prediction row, TP-correct."""
        return self.logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.head_topk
        )

    def refine_edges(
        self,
        raw: torch.Tensor,
        rms: torch.Tensor,
        packed: torch.Tensor,
        anchor_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Rescore every candidate with its predecessor's exact code.

        Returns the [rows, head_topk, head_topk] table the candidate walk
        consumes. Unlike the published greedy path, this table is exact and in
        FP32 (the reference's `serial_rs_table`), so the sampled and greedy
        drafts agree; see docs/design/dspine-vllm-port.md section 7.7.
        """
        contract = self.model.contract
        bridge = self.model.selected_prefix_bridge
        return refine_last_write(
            raw,
            rms,
            packed,
            anchor_ids,
            candidate_ids,
            exact_code=bridge.exact_code,
            head_row=self.head_rows,
            predecessor_gate_weight=bridge.predecessor_gate.weight,
            prev_weight=bridge.prev.weight,
            gate_bias=bridge.gate.bias,
            write_weight=bridge.feature_write_weights[-1],
            norm_weight=self.model.norm.weight,
            code_size=contract.code_size,
            eps=contract.eps,
            block_size=self.model.served_block_size,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        model = self.model
        for name in ("draft_block_causal_version", "last_write_refine_version"):
            value = getattr(model, name).item()
            if value not in (0, 1):
                raise ValueError(f"Unsupported DSpine checkpoint: {name} is {value}.")
        model.selected_prefix_bridge.prepare()
        contract = model.contract
        logger.info(
            "Loaded DSpine draft: layers=%d active_layers=%s code_size=%d "
            "rank=%d head_topk=%d trained_block=%d served_block=%d",
            contract.num_layers,
            sorted(index + 1 for index in contract.active_layers),
            contract.code_size,
            contract.rank,
            contract.head_topk,
            contract.trained_block_size,
            model.served_block_size,
        )
        return loaded


EntryClass = DSpineQwen3ForCausalLM
