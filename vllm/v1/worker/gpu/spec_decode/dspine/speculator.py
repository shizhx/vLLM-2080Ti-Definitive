# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpine speculator: DFlash block drafting with a code-bridge refinement tail.

The context fusion, the context KV precompute, and the uniform one-query-block
forward all come from `DFlashSpeculator`. What DSpine changes is how the draft
decides its tokens: the last code write is redone with the exact code of each
position's predecessor, which yields a [rows, K, K] score table, and the shared
candidate walk then serialises that table into one token per position.
"""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.models.dspine_contract import resolve_dspine_contract
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import (
    cache_candidate_logits,
    walk_candidate_scores,
)


class DSpineSpeculator(DFlashSpeculator):
    _speculator_name = "DSpine"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        # Read from the config rather than the model: the buffers below exist
        # before the draft weights land.
        self.head_topk = resolve_dspine_contract(
            self.draft_model_config.hf_config
        ).head_topk
        self._anchor_indices = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        self._walk_scores = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            self.head_topk,
            dtype=torch.float32,
            device=device,
        )
        self._cached_candidate_ids = torch.zeros(
            self._walk_scores.shape, dtype=torch.int64, device=device
        )

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        # The draft hands its output matrix to the target's inside the call just
        # made, so the rows recovered here are the rows the target scores with.
        self.model.materialize_head_rows()

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # fp32 so the walk and the rejection that checks it read the same
        # distribution; -inf because only the head_topk candidates get a score.
        return torch.float32, -float("inf")

    def _run_draft_block(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draft one block per request, and keep the last code write open.

        Returns the normalized draft features plus the pre-write stream, the
        write scale, and the packed codes the refinement needs. `DFlashSpeculator`
        types its `_run_model` as one tensor, so the wider contract gets its own
        call rather than an override with a different return type.
        """
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            return self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
            )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        draft_hidden, raw, rms, packed = self._run_draft_block(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        hidden_states = draft_hidden[self.sample_indices[:num_sample]].view(
            num_reqs, self.num_speculative_steps, -1
        )
        candidate_ids, _ = self.model.compute_candidates(hidden_states.flatten(0, 1))

        # The refinement only needs the live rows: the padded tail of a captured
        # graph holds other requests' data, and a partial block would break the
        # block layout the refinement walks.
        block_rows = num_reqs * self.num_query_per_req
        anchor_ids = self.input_buffers.input_ids[self._anchor_indices[:num_reqs]]
        scores = self.model.refine_edges(
            raw[:block_rows],
            rms[:block_rows],
            packed[:block_rows],
            anchor_ids,
            candidate_ids,
        )

        walk_candidate_scores(
            scores,
            candidate_ids,
            top_k=self.head_topk,
            num_steps=self.num_speculative_steps,
            num_reqs=num_reqs,
            sample_pos=self.sample_pos,
            sample_idx_mapping=self.sample_idx_mapping,
            temperature=self.temperature,
            seeds=self.seeds,
            draft_tokens=self.draft_tokens,
            realized_scores=self._walk_scores,
            sample_probabilistic=self.draft_logits is not None,
            use_fp64=self.use_fp64_gumbel,
        )
        if self.draft_logits is not None:
            cache_candidate_logits(
                self.draft_logits,
                self._cached_candidate_ids,
                candidate_ids,
                self._walk_scores,
                top_k=self.head_topk,
                num_steps=self.num_speculative_steps,
                sample_idx_mapping=self.sample_idx_mapping,
                num_sample=num_sample,
            )
