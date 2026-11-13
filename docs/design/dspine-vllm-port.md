# DSpine Draft Model on vLLM 2080 Ti Definitive Edition

Status: research complete, decisions recorded in §12, implementation not started.
One design note, English only; no Chinese twin because this is an internal design
note, not user documentation.

## 1. Question

Tencent released DSpine, a parallel drafter that reports higher acceptance
length and throughput than DFlash2. Tencent ships an SGLang plugin only
(<https://github.com/Tencent/DSpine>). Can this fork run DSpine as a
speculative-decoding method, and what must change to reach a validated 4xT10
route for Qwen3.8-27B?

Short answer: yes, with a limited amount of new code. DSpine is a DFlash draft
(a five-layer Qwen3 block drafter with target-feature context fusion) plus two
extra pieces that this fork does not have:

1. a per-layer **code bridge** that passes each position's predicted feature to
   the next position inside the block, and
2. a **last-write refinement** that re-does the final code write with the exact
   predecessor token and scores only the K top candidates per position.

The fork already hosts both structural halves of this pattern for DFlash2: a
draft model subclass that emits K candidates, and a speculator that walks a
`[rows, K, K]` score table and caches sparse draft logits. The port is therefore
mostly a new model file, four small Triton kernels, and one attention-graph
gate change.

## 2. Primary sources

All claims below trace to these sources. Line numbers are from the local trees.

| ID | Source | Role |
| --- | --- | --- |
| S1 | `Qwen3.8-27B-DSpine-b16/config.json`, `model.safetensors` (local copy, 71 tensors, 4.052 GB) | Checkpoint contract |
| S2 | `Qwen3.8-27B-DSpine-b16/README.md` | Published results, SGLang launch line |
| S3 | Tencent/DSpine, commit `c66bf44`, kept for reference at `/home/hsz/Workspace/vLLM-DSpine/upstream-reference/DSpine` (outside this repository, not committed): `sglang_dspine/models/dspine.py`, `runtime/worker.py`, `kernels/dspine.py`, `spec.py`, `layers/attention.py`, `verification/*`, `tests/ref_utils.py` | Reference implementation |
| S4 | SGLang `python/sglang/srt/models/dflash.py` (fetched from `main`) | Host behaviour DSpine inherits |
| S5 | This fork: `vllm/model_executor/models/qwen3_dflash.py`, `qwen3_dflash2.py`, `dflash2_sm75.py`, `vllm/v1/worker/gpu/spec_decode/{__init__,dflash/*,dflash2/*}.py`, `vllm/config/speculative.py`, `vllm/v1/attention/backends/flashinfer.py`, `launcher.sh`, `profiles/4xT10/*`, `AGENTS.md` | Target integration surface |
| S6 | `Qwen3.8-27B-NVFP4/config.json`, `Qwen3.8-27B-DFlash2/config.json` | Target and baseline draft configs |

## 3. What DSpine is

### 3.1 Draft backbone (identical to DFlash)

From S1 and S3 (`models/dspine.py:388-437`):

- 5 Qwen3 decoder layers, `hidden_size` 5120, `intermediate_size` 17408,
  `head_dim` 128, 32 query heads, 8 KV heads, all layers
  `sliding_attention` with `sliding_window` 8192.
- `is_causal: true` and `dflash_config.draft_block_causal: true`, so block
  attention is causal. DFlash2 sets `is_causal: false`
  (S5 `qwen3_dflash.py:58-87`, S6). This difference matters for attention
  backend selection; see §7.4.
- `block_size: 16`: one anchor row plus 15 mask rows per request. DFlash2 uses
  block 8 (S6).
- Target features: `num_target_layers` 64, `target_layer_ids`
  `[1, 16, 31, 46, 61]`, projected by `fc.weight [5120, 25600]` and
  `hidden_norm.weight`, exactly the DFlash context-fusion path
  (S1, S4 `DFlashDraftModel.project_target_hidden`, S5
  `qwen3_dflash.py:582-652`).
- The draft has no `embed_tokens` and no `lm_head`; both are shared with the
  target (S1 weight list, S5 `dflash/utils.py:maybe_share_target_embed`,
  `get_target_lm_head`).
- `draft_vocab_size` equals `vocab_size` (248320), so no draft-to-target id map
  is needed (S1).
- Mask token 248070, same as DFlash2 (S1, S6).

Context rows never see the code bridge. The bridge runs only inside the draft
forward over whole request blocks, and SGLang raises if the row count is not a
whole number of blocks (S3 `models/dspine.py:271-279`, S4 confirms context K/V
enters the cache through `kv_proj_only`). The fork's DFlash
`precompute_and_store_context_kv` therefore needs no change.

### 3.2 Code bridge: adjacent causal injection

The paper calls this "adjacent causal injection": each layer writes a compact
feature of every position into a shared code stream, and the next position reads
its predecessor's write. Depth carries the token dependency that a single
parallel pass cannot see (S2, S3 `kernels/dspine.py:7-135`).

Bridge tensors (S1, all under `selected_prefix_bridge.`):

| Tensor | Shape | Use |
| --- | --- | --- |
| `codebook` | `[248320, 1024]` bf16 | Exact token code, used for anchor rows and for the refinement senders |
| `read_weight` | `[1024, 5120]` | Code read for the last layer |
| `early_read_weights` | `[4, 1024, 5120]` | Code read for layers 0-3 |
| `feature_write_weights` | `[5, 5120, 512]` | Per-layer write into the residual stream |
| `gate.weight` / `gate.bias` | `[512, 1024]`, `[512]` | Receiver query and gate bias |
| `predecessor_gate.weight` | `[512, 1024]` | Sender half of the gate |
| `prev.weight` | `[512, 1024]` | Sender value |
| `out.weight` | `[1024, 512]` | Terminal projection; the checkpoint stores zeros and SGLang asserts it stays zero (S3 `models/dspine.py:205-207`) |
| `independent_write_version`, `terminal_disabled_version` | scalar int64 | Format flags, both must equal 1 (S3 `models/dspine.py:209-212`) |

Bridge math for one layer, run after the layer's MLP residual branch, on `m`
rows laid out as `bs` blocks of `T` rows (`T = block_size`):

```
raw      = round_to_dtype(attn_mlp_out + residual)          # pre-write stream
rms      = sqrt(mean(raw^2))                                # per row, no eps
normed   = raw / (rms + eps)
code     = normed @ read_weight.T                           # [m, C=1024], C=1024
         # read_weight = bridge.read_weight for the last layer,
         #               bridge.early_read_weights[layer] for layers 0-3
code     = sqrt(C) * codebook[anchor(b)]      if row is a block anchor
         = sqrt(C) * code / (||code|| + eps)  otherwise
packed   = code @ [gate.w; pred_gate.w; prev.w].T           # [m, 3R], R=512
query    = packed[row, 0:R]        sender = packed[row-1, R:2R]
value    = packed[row-1, 2R:3R]    bias   = gate.bias
message  = 0                        if row is a block anchor
         = sigmoid(query + sender + bias) * value           otherwise
delta    = message @ feature_write_weights[layer].T         # [m, H]
out      = round_to_dtype(raw + rms * delta)                # anchors keep raw
out      = RMSNorm(out, next_layer.input_layernorm)         # fused, feeds next layer
```

All reductions run in fp32 inside the kernel and store back in the runtime
dtype (S3 `kernels/dspine.py:7-135`: `prepare_read`, `normalize_codes`,
`feature_message`, `feature_finish`). The next layer must skip its own input
norm because the write already applied it (S3 `models/dspine.py:355-386`).
The last layer's write uses `self.norm`, so the model output is the normalised,
post-write stream (S3 `models/dspine.py:464-509`).

Bridge writes happen after every layer: `selected_prefix_feature_layers` is
`[1,2,3,4,5]` (S1). The reference loop is S3 `models/dspine.py:464-509` and the
pre-normed layer is `models/dspine.py:355-386`.

### 3.3 Last-write refinement (`code_bridge_second_pass`)

`code_bridge_second_pass = {mode: serial_table, first_exact: 4, head_topk: 16}`
means: redo **only the last layer's write** (`first_exact == num_layers - 1`,
validated in S3 `models/dspine.py:resolve_dspine_head_topk`) with the **exact**
sender code of the token the previous position actually chose, and score only
the K = 16 top candidates of each position.

Procedure per draft step (S3 `kernels/dspine.py:serial_table`,
`serial_rs_table`, `serial_chain`, `serial_sample_chain`; worker
`runtime/worker.py:216-341`):

1. `rows` = output hidden of block rows 1..T-1 (the 15 prediction rows).
   `cand[n] = top_K(lm_head @ rows[n])`, K = 16.
2. Keep the last write's stash: `raw`, `rms`, `packed` (receiver query in
   `packed[:, :R]`).
3. For row `n` in block `b` at offset `r` (1-based), and for every candidate
   lane `j` of the previous row, the sender id is
   `anchor(b)` when `r == 1`, else `cand[n-1, j]`. Then

```
exact   = sqrt(C) * codebook[sender]
msg(j)  = sigmoid(q_n + exact @ pred_gate.T + gate.bias) * (exact @ prev.T)
refined = raw_n + rms_n * (msg(j) @ W_last.T)
table[n, j, c] = (refined / rms(refined) * norm.weight) · lm_head[cand[n, c]]
```

   giving `table` of shape `[n, K, K]`, exactly the shape DFlash2's selector
   already produces in this fork (S5 `qwen3_dflash2.py:351-372 _score_edges`).
4. Walk rows in order: lane chosen at row `n` selects the sender for row `n+1`.
   SGLang walks greedily from a bf16 table that **omits** the RMS renormalisation
   (`serial_table`, `serial_chain`), then re-walks sampled requests from an
   exact fp32 table (`serial_rs_table`, `serial_sample_chain`).
   vLLM's DFlash2 path keeps one fp32 table and samples or argmaxes from it with
   Gumbel noise (S5 `dflash2/speculator.py:_selector_walk_kernel`), which is
   both simpler and exact. The plan follows the vLLM convention.

The refinement is a draft-quality step only. Target verification still decides
acceptance, so an approximate or missing refinement costs acceptance length,
never output correctness.

### 3.4 Verification and optional engine features

SGLang verifies the dense block with a bonus token, as DFlash does, so vLLM's
existing DFlash rejection-sampling path applies unchanged
(S3 `verification/caps.py:_accept_capped` commits a prefix of the block).

SGLang adds two features the plan treats as out of scope for the first release:

- **Dynamic verification (SPS)**: per-batch-size verification widths tuned per
  hardware (`verification/policy.py`, `scripts/tune_sps.py`, default
  `uniform_widths = (16,)*7`, i.e. off). The fork has a DSpark-specific
  analogue in `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`.
- **ReplaySSM** for the target's linear-attention states
  (`runtime/replayssm/`), tied to SGLang's Mamba pool layout. The fork already
  solves the same problem its own way for DFlash2 and DSpark
  (`CHANGELOG.md` v0.2.2-post2, `vllm/v1/attention/backends/gdn_attn.py`).

SGLang also rejects FP4 KV for DSpine because a prefix commit of a dense block
needs a plain KV layout (`spec.py:handle_dspine_server_args`), and restricts
DSpine to `tp_size == 1` and the `fa3` draft backend. Neither restriction is a
property of the algorithm; both are SGLang-side scope limits. §8 covers the vLLM
position on TP.

## 4. Published gain to reproduce

From S2, Qwen3.8-27B, SGLang 0.5.22, BF16, FA3, concurrency 8, block 16 versus
DFlash2 block 8:

| Task (T=0) | DFlash2 | DSpine |
| --- | ---: | ---: |
| GSM8K | 5.69 | 7.20 |
| MATH-500 | 5.49 | 6.49 |
| HumanEval | 4.87 | 5.71 |
| MBPP | 5.04 | 6.14 |
| MT-Bench | 4.11 | 4.48 |

Agent replay throughput improved 8-20 % over DFlash2 (for example WorkBuddy
Office xhigh: 662 to 764 tok/s). Acceptance length is the primary metric to
reproduce; the fork's own acceptance harness is
`tests/v1/e2e/spec_decode/acceptance_rates/dflash/test_dflash.py`.

These numbers come from SGLang with BF16 and FlashAttention 3, at concurrency
8; the model card does not name the GPU, though an FA3 backend implies a
Hopper-class card. They are a target, not a claim for this fork. Per `AGENTS.md`,
no throughput or capacity claim may be published without this repository's own
audit evidence.

## 5. How the fork hosts DFlash today

The relevant chain for Model Runner V2 (`launcher.sh:4835` sets
`VLLM_USE_V2_MODEL_RUNNER=1` for DFlash routes):

| Step | File and line | Notes |
| --- | --- | --- |
| Method select and draft config wrap | `vllm/config/speculative.py:1307-1400` | `method` may stay `dflash`; the draft `hf_config` is wrapped in `EAGLEConfig`; `dspark` shows the precedent of rewriting `architectures` (`:1403-1411`) |
| Extra drafting slots | `vllm/config/speculative.py:1866-1868` | DFlash reserves `K+1` slots |
| Speculator dispatch | `vllm/v1/worker/gpu/spec_decode/__init__.py:15-31` | `method == "dflash"`, then architecture picks `DFlash2Speculator` or `DFlashSpeculator` |
| Draft model load, shared embed and head | `vllm/v1/worker/gpu/spec_decode/dflash/utils.py` | `maybe_share_target_embed`, `get_target_lm_head` |
| Draft model | `vllm/model_executor/models/qwen3_dflash.py` | layers `:246-361`, context KV precompute `:582-652`, block forward `:655-673`, `get_draft_attn_causal` `:751` |
| DFlash2 variant | `vllm/model_executor/models/qwen3_dflash2.py:273-500` | layer subclass, `CandidateSelector` `:375`, `compute_candidates` `:490` via `LogitsProcessor.get_top_k_tokens` (`vllm/model_executor/layers/logits_processor.py:253`) |
| Draft step | `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:329-540` | prepare inputs, precompute context KV, one uniform query forward of `1+K` rows per request, full-graph dispatch |
| Query row layout | same file `:646-690` | row 0 of each block holds the sampled (bonus) token, rows 1..K hold the mask token; sampling rows are 1..K |
| Candidate walk and sparse logits | `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:106-217` | `_selector_walk_kernel` consumes `[rows, K, K]` fp32 scores plus `[rows, K]` ids; `_cache_draft_logits_kernel` writes them into `draft_logits` for the rejection sampler |
| Draft logits buffer | `vllm/v1/worker/gpu/spec_decode/speculator.py:186-197` | `[max_num_reqs, steps, vocab]`, allocated only for `draft_sample_method=probabilistic` |
| Aux feature layers | `vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:100-137` | `dflash_config.target_layer_ids` are used with `+1` (capture-after semantics) |
| Model Runner V2 requirement | `vllm/config/vllm.py:2887, 2901-2909` | Candidate-table drafts are listed as V1-unsupported, which forces `use_v2_model_runner`; `launcher.sh:4835` also sets `VLLM_USE_V2_MODEL_RUNNER=1` for DFlash routes |
| Reference tests | `tests/v1/spec_decode/test_dflash2.py:367` (`_score_edges` versus a sequential reference), `tests/v1/spec_decode/test_dflash_causality.py`, `test_dflash_prefix_cache_masking.py` | Pattern to follow |
| Acceptance eval | `tests/v1/e2e/spec_decode/acceptance_rates/dflash/test_dflash.py` | GSM8K acceptance length and accuracy gate |

Consequence: `method="dflash"` is the key that unlocks about forty
specialisations across the scheduler, KV-cache grouping, prefix-cache masking,
hybrid-Mamba state restore, capture sizing, and the SM75 attention gates. DSpine
must keep that key. See §8.

## 6. Gap analysis

| Need | Fork state | Work |
| --- | --- | --- |
| Draft model with per-layer bridge | absent (`grep -r "code_bridge\|selected_prefix" vllm/` is empty) | new model file |
| Refinement table | shape and consumer exist for DFlash2 | new table builder, reuse the walk and cache kernels |
| Causal block attention with window 8192 | supported (`_resolve_layer_attention` honours `is_causal`) | config only |
| `block_size` 16 (K=15) | no config cap found; SM75 graph gate blocks it | §7.4 |
| Bridge kernels | absent | port 4 Triton kernels |
| bf16-trained draft on an fp16 SM75 worker | solved once for DFlash2 (`dflash2_sm75.py`, toggle `VLLM_DFLASH2_SM75_TRANSPORT`) | measure, reuse if needed |
| Launcher and profiles for DSpine | `normalize_speculative_method_value` maps only `dflash|dflash2` (`launcher.sh:355-365`) | add alias, default tokens, labels, profile files |
| Tests | pattern exists | new unit and acceptance tests |

## 7. Hard constraints found during the survey

### 7.1 The checkpoint declares itself a plain DFlash draft

`architectures` is `["DFlashDraftModel"]` and the bridge is announced only by
`dflash_config.selected_prefix_code_bridge` (S1). SGLang rewrites the
architecture to `DSpineDraftModel` when that key is present and refuses a
DSpine checkpoint served as DFlash, and the reverse
(S3 `spec.py:map_dspine_draft_architecture`). The fork must do the same rewrite.
The failure mode without it is loud rather than silent, because
`AutoWeightsLoader` raises on unexpected weight names
(`vllm/model_executor/models/utils.py:298-305`): the checkpoint's two version
scalars and every `selected_prefix_bridge.*` tensor are unknown to the plain
DFlash model. The rewrite is still needed to give a clear message and to route
the draft to the right speculator.

### 7.2 The bridge assumes whole blocks and a runtime-dtype residual stream

SGLang raises unless the row count is a multiple of the block size and unless
activations are bf16 (S3 `models/dspine.py:271-279`). vLLM on SM75 runs the
draft in fp16, so the port must accept the runtime dtype. The rounding point
inside `prepare_read` is already generic
(`.to(X.dtype.element_ty)`), so a fp16 worker matches the same structure with a
different rounding width. Dynamic range is the real risk; §9 R2.

### 7.3 Serving a block smaller than trained is legitimate

SGLang exposes `set_block_size` for `2 <= T <= trained` and keeps the bridge and
convolutions consistent (S3 `models/dspine.py:439-461`). So DSpine can serve
`K=7` (block 8) as an interim route. This keeps every SM75 shape identical to
the validated DFlash2 route, at some acceptance cost.

### 7.4 SM75 full CUDA graphs stop at `K = 7`

`_sm75_spec_prefill_graph_query_len` (`vllm/v1/attention/backends/flashinfer.py:106-157`)
returns the graph query length only when the device is SM75, dtype is fp16,
attention is causal, KV is not NVFP4, head size is 128 or 256, capture sizes are
exactly `{(K+1) * b}`, and `1 <= num_speculative_tokens <= 7`. On SM75
`supports_trtllm_attention` is false (`vllm/utils/flashinfer.py:566-585`), so
this gate is the only path to `AttentionCGSupport.UNIFORM_BATCH`
(`flashinfer.py:1128-1155`). Below the gate, `get_cudagraph_support` returns
`UNIFORM_SINGLE_TOKEN_DECODE`, which means a `K+1` query decode cannot use a
full graph.

Two consequences:

- The **target** verify forward for `K=15` loses full graphs. This is the
  expensive forward, so a naive `K=15` port can be slower than the DFlash2/7
  baseline even with a better drafter.
- The **draft** forward for DFlash2 today is non-causal, so it is already below
  this gate (`attn_vllm_config` sets `use_non_causal`, gate line 117 rejects it)
  and runs eagerly. DSpine's draft is causal, so at `K=7` both the draft and the
  target verify forward qualify for the stable FA2 wrapper, and at `K=15` both
  need only the width gate widened. That is an extra win, not a cost.

The commit that added the gate (`d187dfdde0`) says: "Expand the shape gate only
with a new CUDA correctness audit." The plan schedules exactly that audit.

### 7.5 Two vocab-indexed gathers must stay TP-correct

The refinement needs `lm_head[cand]` and `codebook[sender]`, both indexed into a
248320-row table. `LogitsProcessor.get_top_k_tokens` already reduces top-k
across tensor-parallel head shards and returns global ids
(`vllm/model_executor/layers/logits_processor.py:253-283`). For the row gathers,
`ParallelLMHead` is a `VocabParallelEmbedding`, so calling it with candidate ids
performs the masked lookup plus all-reduce. The codebook should be loaded as a
replicated buffer (508 MB) or as a `VocabParallelEmbedding` if memory demands.
Do not hand-roll row slicing on a sharded head.

### 7.6 `draft_logits` memory scales with `K`

`[max_num_reqs, K, 248320]` fp32 (`speculator.py:186-197`): 14.9 MB per request
slot at `K=15`, versus 6.9 MB at `K=7`. `MAX_NUM_SEQS=1` profiles make this
noise; a four-way concurrency profile would pay 60 MB.

### 7.7 Reference numerics differ from SGLang's greedy path

SGLang's greedy refinement skips the RMS renormalisation of the refined vector
(`kernels/dspine.py:_serial_table`, valid for a full-vocab argmax but not for a
per-candidate ranking) and works from bf16 tables, while the sampled path uses
the exact fp32 table. The vLLM port uses the exact fp32 table for both, which is
the behaviour of `serial_rs_table`. Expect small acceptance differences from the
published greedy numbers and record the choice in the code.

### 7.8 Aux layer ids use `+1` capture-after semantics

`get_eagle3_aux_layers_from_config` maps `target_layer_ids` to `id + 1`
(`eagle/eagle3_utils.py:108-111`), matching DFlash2. If DSpine's ids were wrong
by one, acceptance would collapse to about 1.0 with no error. The first end-to-end
run must therefore print acceptance length before anything else.

### 7.9 The compiled draft forward cannot divide its own row count

vLLM compiles the draft model with its token dimension marked dynamic, so one
graph serves the prefill-draft pass and every block-draft batch. Solving a marked
dimension into a constant is a hard failure there, not a recompile:

```
ConstraintViolationError: Constraints violated (L['input_ids'].size()[0],
  L['positions'].size()[0])!
  - You marked L['input_ids'].size()[0] as dynamic but your code specialized it
    to be a constant (8).
```

The first draft of `code_bridge_write` found each block's anchor row the way the
upstream kernel does: `num_rows % block_size` to validate, `num_rows // block_size`
to reshape into blocks, and `input_ids[::block_size]` for the anchor tokens. With
a block size of 8 and one request, that pins the token dimension to 8 and the
route never starts. The write now finds the same rows with elementwise arithmetic
on a row index (`dspine_bridge.py:77`), which keeps the row count symbolic and
changes no number: the FP64 reference and the upstream eager path still match bit
for bit. Geometry that *may* divide the rows belongs to the eager caller
(`refine_last_write`, `dspine_bridge.py:152`), where the shape is already known.

`test_code_bridge_write_keeps_the_row_dimension_dynamic`
(`tests/v1/spec_decode/test_dspine.py:416`) holds the line: it compiles the write
once, feeds it one, two, and three blocks, and requires one graph rather than two.
Shape arithmetic shows up there as a second graph, which is what the engine pays
for as a failed start.

### 7.10 A quantized target head has no rows to gather

The refinement scores candidates with rows of the draft's output matrix, and the
draft shares that matrix with the target. Gathering `lm_head.weight` rows assumes
the head stores rows plainly. On the NVFP4 target it does not: the head is stored
as packed INT4 with a per-output-channel scale
(`weight (320, 248320) int32` on a TP4 rank, `CompressedTensorsLinearMethod`), so
a row gather reads packed blocks and fails on the device.

The rows are instead read back *through* the head (`read_rows_through_head`,
`qwen3_dspine.py:361`): feeding the head one-hot inputs makes it emit its own
columns, and transposing the logits gives the row table. One-hot inputs are exact
under any activation quantization, and the numbers come from the same
`LogitsProcessor._apply_head` call the draft's own top-k uses, so the refinement
scores with the head the target actually has. It costs 0.59 GiB per rank here
(62,080 rows in FP16), read once at load (`materialize_head_rows`,
`qwen3_dspine.py:410`). A head that does store plain rows skips all of this.

### 7.11 A draft with KV of its own needs DFlash2's native KV layout

DSpine drafts keep attention KV like DFlash2 drafts do. Under the shared DFlash
layout that KV is overlaid with the target's pages, and a long route gives the
manager a block larger than the Triton draft kernel's 64-token block, so kernel
blocks have to subdivide a manager block. Subdivision needs dense, unpadded pages
(`kv_cache_interface.py:368-380`), and an overlaid page is not dense:

```
ValueError: The resolved KV cache layout (BLHNC) does not store blocks as dense,
unpadded pages (block stride 13631488 != page 851968), so a manager block cannot
be split into 13 kernel blocks of 64 tokens.
```

Forcing `--disable-hybrid-kv-cache-manager` instead fails elsewhere ("Failed to
promote local KV cache specs to one unified type"). The fix is the one DFlash2
already uses: `_uses_native_dflash2` (`kv_cache_utils.py:2415`) also recognises
`DSpineDraftModel`, which gives every KV group its own block-ID pool. Short
context hides this: at 4 K tokens the manager block equals the kernel block, so
the first bring-up runs passed without it.

## 8. Design

### 8.1 Method identity

Keep `speculative_config.method == "dflash"` and dispatch on the draft
architecture, exactly like DFlash2 (`__init__.py:18`). Reasons: the DFlash
invariants listed in §5 are keyed on `use_dflash()`, and a new method string
would silently miss several of them (slot reservation, capture sizing,
prefix-cache masking, hybrid-Mamba state restore). DSpine stays visible where
users meet it: log line, speculator name, launcher value `dspine`, profile file
names, and a distinct model class.

Detection, in `vllm/config/speculative.py` next to the existing DFlash block:

- Auto-detect method `dflash` when the draft path contains `dspine`
  (today `:1319` matches only `dflash`), and always when
  `dflash_config.selected_prefix_code_bridge` exists.
- Validate the DSpine contract and raise on mismatch, mirroring S3
  `models/dspine.py:DSpineCodeBridge.__init__` and `resolve_dspine_head_topk`:
  `draft_block_causal` true, `is_causal` not false,
  `selected_prefix_feature_writes` true, write mode `independent`,
  `selected_prefix_terminal` false, last layer active,
  `code_bridge_second_pass.mode == "serial_table"`,
  `first_exact == num_hidden_layers - 1`, `head_topk` a power of two in
  `[2, 64]` and agreed by `selected_prefix_last_write_refine.head_topk`, and
  reject `conv_kernel_size`, `selector_rank`, `selector_top_k`,
  `attention_sink_bias`, `partial_rotary_factor != 1`.
- Rewrite `architectures[0]` to `DSpineDraftModel` and call `update_arch_()`,
  the pattern at `:1403-1411`. Two traps here:
  - Ordering. For `method="dflash"` the draft `hf_config` is wrapped in
    `EAGLEConfig`, which rewrites `architectures` by prefixing `DFlash` to any
    name that neither starts nor ends with `DFlash`
    (`vllm/transformers_utils/configs/eagle.py:65-73`). `DSpineDraftModel`
    would become `DFlashDSpineDraftModel`. Do the rewrite *after* the wrap block
    (`:1383-1401`), which is what the DSpark branch does.
  - `EAGLEConfig` copies the inner config's fields onto itself
    (`eagle.py:80-83`), which is why `dflash_config` stays readable on the
    wrapper.
- Related: a first-class `method="dspine"` string is not free. `EAGLEConfig`
  raises `Invalid method` for anything outside `eagle`, `eagle3`, `dflash`
  (`eagle.py:69-72`), so that route also needs an `EAGLEConfig` change.
  `SpeculatorsConfig` is the other accepted wrapper type.
- Keep the draft on Model Runner V2. The fork already forces V2 for
  candidate-table drafts: `_is_dflash2_draft()` feeds
  `_get_v1_model_runner_unsupported_features()`
  (`vllm/config/vllm.py:2887, 2901-2909`), whose comment explains that on V1 the
  checkpoint "would degrade to DFlash1 silently". Widen that check (or add
  `_is_dspine_draft()` beside it) so a DSpine draft cannot land on the V1
  `DFlashProposer`, which has no candidate path.

### 8.2 New and changed files

New:

| Path | Content |
| --- | --- |
| `vllm/model_executor/models/qwen3_dspine.py` | `DSpineQwen3DecoderLayer` (accepts a pre-normed input), `DSpineCodeBridge` (weights, per-layer write, stash), `DSpineQwen3Model(DFlashQwen3Model)` (loop with bridge writes, last-write stash, `refine_edges`), `DSpineQwen3ForCausalLM(DFlashQwen3ForCausalLM)` (`compute_candidates` reuses `LogitsProcessor.get_top_k_tokens` with `K = head_topk`). `load_weights` must also consume the two top-level version scalars `draft_block_causal_version` and `last_write_refine_version`, each required to equal 1 (S1) |
| `vllm/v1/worker/gpu/spec_decode/dspine/speculator.py` | `DSpineSpeculator(DFlashSpeculator)` with `_speculator_name = "DSpine"`. It needs the same candidate walk and sparse-logit caching as DFlash2, so lift `_selector_walk_kernel`, `_cache_draft_logits_kernel`, `_sample_path` and `_cache_draft_logits` into `vllm/v1/worker/gpu/spec_decode/utils.py` and let both variants use them. Do not subclass `DFlash2Speculator`: its `__init__` reads `dflash_config["selector_top_k"]`, which a DSpine checkpoint does not have (S1) |
| `vllm/model_executor/models/dspine_kernels.py` | Triton ports of `prepare_read`, `normalize_codes`, `feature_message`, `feature_finish`, registered through `direct_register_custom_op` like `qwen3_dflash2.py:139` |
| `tests/v1/spec_decode/test_dspine.py` | Bridge and refinement tests against a float64 reference |
| `profiles/4xT10/qwen27b/{w8a16,w4a16}/dspine*-*.env` | Routes, added only after audit |
| `docs/design/dspine-vllm-port.md` | This note |

Changed:

| Path | Change |
| --- | --- |
| `vllm/model_executor/models/registry.py:636` | `"DSpineDraftModel": ("qwen3_dspine", "DSpineQwen3ForCausalLM")` |
| `vllm/config/speculative.py` | detection, contract validation, architecture rewrite (§8.1) |
| `vllm/v1/worker/gpu/spec_decode/__init__.py:15-31` | `DSpineDraftModel` → `DSpineSpeculator` |
| `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` | lift the block size `T` into one attribute the bridge kernels read (today `num_query_per_req` is the same value; only a named alias is needed) |
| `vllm/v1/attention/backends/flashinfer.py:121` | widen the width gate after the audit in §9 R1 |
| `vllm/config/vllm.py:2901-2909` | Widen `_is_dflash2_draft()` (or add `_is_dspine_draft()`) so a DSpine draft is listed as V1-unsupported and Model Runner V2 is required |
| `launcher.sh` | `dspine\|dspine16) echo dflash`, `infer_speculative_method_from_model_ref` matches `*dspine*`, default `SPECULATIVE_TOKENS` 15 for DSpine (`:461`), route label `dspine{block}/{tokens}`, menu entry, draft auto-fetch route |
| `profiles/4xT10/README*.md`, `profiles/README*.md`, `docs/non-interactive-launch*.md`, `CHANGELOG.md`, `README*.md` | keep English and Chinese in step, mark status `experimental` until promoted |

Deliberately not changed: context-KV precompute, the DFlash prepare kernel, the
rejection sampler, KV-cache grouping, scheduler slot math, and the SGLang-only
SPS and ReplaySSM features.

### 8.3 Data flow per draft step

```
target forward (aux features at layers 2,17,32,47,62 of the capture index)
  -> fc + hidden_norm                      [DFlash, unchanged]
  -> precompute_and_store_context_kv       [DFlash, unchanged]
  -> query block: 1 bonus + 15 mask rows    [DFlash, unchanged]
       per layer: attention + MLP, then bridge write (fused with next input norm)
       last write leaves stash {raw, rms, packed} and the normed output
  -> compute_candidates: top-16 ids + unary logits over the shared lm_head
  -> refine_edges: [n,16,16] fp32 table     [new, DSpine]
  -> _selector_walk_kernel (Gumbel, per-request seed and position)
  -> _cache_draft_logits_kernel -> draft_logits for the rejection sampler
  -> target verifies 16 rows, commits a prefix + bonus token
```

Everything from `compute_candidates` down already exists for DFlash2.

## 9. Risks

| ID | Risk | Severity | Handling |
| --- | --- | --- | --- |
| R1 | `K=15` loses SM75 full graphs (§7.4), so the route may be slower than DFlash2/7 | High | Phase 2 ships `K=7` inside validated shapes; Phase 4 widens the gate with a graph-versus-eager equality audit over sequence lengths, batch sizes, and both attention roles |
| R2 | fp16 SM75 worker against a bf16-trained bridge: overflow or precision loss in the residual stream (DFlash2 needed `dflash2_sm75.py` for exactly this) | High | Measure acceptance length with and without the SM75 transport; if it drops, add row-scaled residual handling for the bridge path before promoting any profile |
| R3 | Wrong aux-layer mapping (`target_layer_ids` `+1`) silently gives acceptance about 1.0 | Medium | First smoke run must print acceptance length; compare with the DFlash2 baseline |
| R4 | Draft config accepted but routed to the plain DFlash model, so the bridge weights never load | Medium | §8.1 detection and rewrite, plus a load test that asserts every `selected_prefix_bridge.*` tensor landed and the version scalars read 1 |
| R5 | Memory: codebook 508 MB replicated per rank, `draft_logits` 14.9 MB per request slot, refinement `[n,K,H]` fp32 intermediates | Medium | Chunk the refinement rows like `serial_rs_table` (`chunk_rows` 256), keep `MAX_NUM_SEQS` small in the first profiles, note the cost in the profile README |
| R6 | `torch.compile` tracing the new Triton kernels | Medium | Register each kernel as a custom op, as `qwen3_dflash2.py:128-158` does; keep every captured shape static |
| R7 | Structured output, watermarking, logprobs and other paths that assume dense draft logits | Low | Assert unsupported combinations in the speculator constructor; the sparse-logit convention is inherited from DFlash2 |
| R8 | Two GPU models on one host (T10 and 2080 Ti, both SM75) | Low | `--gpu-devices 0,1,2,3` selects the four T10 devices in both `PCI_BUS_ID` and default order; never pass 4 or 5 |

## 10. Phased plan

Each phase ends in a check that can be run and read.

**Phase 0 — Reference tests, no engine change.**
Port the independent references in S3 `tests/ref_utils.py`
(`ref_refine_table`, `ref_sample_chain`) and write
`tests/v1/spec_decode/test_dspine.py`:
bridge write versus a float64 reference (anchor and non-anchor rows, block 8 and
16), refinement table versus the float64 reference, and the serial chain walk.
Write them as ordinary pytest tests in the style of
`tests/v1/spec_decode/test_dflash2.py` (`pytest.mark.parametrize`,
`pytest.mark.skipif(not torch.cuda.is_available(), ...)`) so they merge to
mainline unchanged.
Verify: `.venv/bin/python -m pytest tests/v1/spec_decode/test_dspine.py -q` is
green on CPU. The test tools are already installed; see §12 decision 2 for the
command and for the local pass baseline.

**Phase 1 — Model, config, weights.**
Implement §8.1 and the model file. Verify:
1. `python3 -m py_compile` on changed files.
2. A load test on one T10 device: build the draft from
   `Qwen3.8-27B-DSpine-b16` with `--dummy-weights` off, assert all 71 tensors
   land, `out.weight` is all zero, both version tensors equal 1, no NaN remains.
3. A fixed-input forward of one block against a straightforward torch reference
   of §3.2 plus §3.3 on GPU, tolerance matched to fp16.

**Phase 2 — End to end at `K=7`.**
`DSpineSpeculator`, refinement table, walk, sparse logits. Verify:
1. Serve on four T10 devices against the NVFP4 target (decision 4), using
   `w4a16/dflash2-fp16kv-1x262K-text-only.env` as the template with
   `SPECULATIVE_TOKENS=7`, and generate text; output must match the same prompt
   with `SPECULATIVE_METHOD=none` at `temperature=0`.
2. Acceptance length on GSM8K must beat the DFlash2/7 baseline in the fork's
   harness, not merely run.
3. Log lines must show the DSpine speculator and the causal draft attention.

**Phase 3 — Numerical fidelity decision.**
Compare acceptance length and greedy output equality with the SM75 transport
enabled and disabled (`VLLM_DFLASH2_SM75_TRANSPORT=0`), and record which setting
the DSpine route uses and why. If the bridge needs it, design the narrowest
codec that fixes the observed loss. Verify by the acceptance delta alone.

**Phase 4 — Full block 16.**
Widen the SM75 gate to `K=15`, with the audit R1 demands: draft and target
outputs must match eager execution across sequence lengths and batch sizes, and
the capture-size gate must be satisfied by the profile's compilation config.
Verify: `launcher.sh --profile ... --print-config` shows the expected capture
sizes; the draft logs full-graph capture; GSM8K acceptance length moves toward
the published DSpine/16 range; 4K/128 and 32K/512 decode throughput is at or
above the DFlash2/7 profile in `profiles/4xT10/README.md`.

**Phase 5 — Release to DFlash2 parity, only after Phases 2-4 pass.**
Deliver every route in the §12 decision-5 parity table: three `w4a16` routes
(FP16 KV single, FP8 KV two-way, TurboQuant k8v4 four-way) and three `w8a16`
routes (FP16 KV single text-only, FP16 KV single text+image, FP8 KV two-way).
Each needs startup, 4K/128, 32K/512, concurrency and, for the image route,
image-semantics evidence, in the repository's documented benchmark口径, before
`profiles/4xT10/README*.md` tables, `CHANGELOG.md`, launcher menu and draft-fetch
routing, and English and Chinese documentation stay in step. Until that evidence
exists, keep the wording `experimental` (`AGENTS.md`). Optional follow-ups:
dynamic verification through `adaptive_verification.py`, codebook sharding for
larger `MAX_NUM_SEQS`, and 2x2080Ti capacity sizing.

## 11. Commands used in this survey

```bash
# Environment (verified)
.venv/bin/python -m pytest tests/v1/spec_decode/test_dflash2.py -q   # 33 passed with test_dflash_causality.py
nvidia-smi --query-gpu=index,name,compute_cap --format=csv      # 0-3 Tesla T10 sm75, 4-5 RTX 2080 Ti sm75
.venv/bin/python -c "import vllm; print(vllm.__file__)"         # editable install in this working tree

# Route review (T10 only, 2080 Ti untouched). NVFP4 target per decision 4.
# The route comes from the profile; the drafter is not a profile field, so the
# draft is named here. CUDA_DEVICE_ORDER keeps the T10s at indices 0-3: without
# it this host numbers a 2080 Ti first.
CUDA_DEVICE_ORDER=PCI_BUS_ID PROFILE_FILE=profiles/4xT10/qwen27b/w4a16/dspine-fp16kv-1x262K-text-only.env \
  MODEL_DIR=/home/hsz/Models/Qwen3.8-27B-NVFP4 \
  SPECULATIVE_MODEL=/home/hsz/Workspace/vLLM-DSpine/Qwen3.8-27B-DSpine-b16 \
  GPU_DEVICES=0,1,2,3 TP_SIZE=4 PP_SIZE=1 MODE=fast \
  ./launcher.sh --print-config

# Starting it non-interactively also needs the host commit headroom to cover the
# largest checkpoint file (21 GiB here), which is what vm.overcommit_memory=1
# gives. The launcher's mmap preflight refuses to start without it, and raising it
# needs root.

# Before publishing any change (AGENTS.md)
bash -n build.sh launcher.sh tools/validate_profiles.sh
bash tools/validate_profiles.sh
python3 -m py_compile <changed python files>
git diff --check
```

## 12. Decisions taken by the owner (2026-11-13)

1. **`K=7` first.** Phase 2 ships block 8; `K=15` follows in Phase 4 behind the
   SM75 graph audit.
2. **pytest is approved, installed the mainline way.** Tests must be written as
   ordinary pytest files in `tests/`, matching the fork's markers and harnesses,
   so they can merge to mainline. The runtime `.venv` was given the test tools at
   the versions pinned in `requirements/test/cuda.txt`:

   ```bash
   uv pip install --python .venv/bin/python \
     "pytest==9.1.0" "pytest-asyncio==1.4.0" "pytest-forked==1.6.0" \
     "pytest-mock==3.14.0" "pytest-timeout==2.3.1" "pytest-rerunfailures==14.0" \
     "pytest-shard==0.1.2" "tblib==3.1.0"
   ```

   `tblib` is required by `tests/conftest.py:7`. The full `requirements/test/cuda.txt`
   lock was deliberately not installed because it pins torch and would move the
   compiled runtime environment. Local baseline on a clean tree at `93c9013e37`:

   ```text
   tests/v1/spec_decode/test_dflash2.py + test_dflash_causality.py -> 33 passed
   tests/v1/spec_decode/test_dflash_layout.py + test_dflash_prefix_cache_masking.py -> 13 passed
   tests/v1/spec_decode/test_dflash_prepare_inputs.py -> 3 failed, 13 passed
   ```

   The 3 failures are pre-existing and unrelated to this work: Triton 3.7.1
   rejects the prepare kernel (`Unsupported ptr type triton.language.int32`).
   Do not attribute them to DSpine changes and do not fix them here.
3. **`dspine` maps to `method="dflash"`** and dispatches on the draft
   architecture, as §8.1 recommends.
4. **Development target is the NVFP4 checkpoint**
   (`/home/hsz/Models/Qwen3.8-27B-NVFP4`, `w4a16` routes).
5. **No reduced scope for the first release: the route set must match DFlash2.**
   A smoke-only profile is not a release. The parity bar on 4xT10 is every
   published DFlash2 route (`profiles/4xT10/README.md`):

   | Existing DFlash2 route | DSpine equivalent must reach |
   | --- | --- |
   | `w4a16/dflash2-fp16kv-1x262K-text-only` (seqs 1) | same context, same KV, same benchmark口径 |
   | `w4a16/dflash2-fp8kv-2x262K-text-only` (seqs 2) | 2 concurrent 32K requests, exact token counts |
   | `w4a16/dflash2-tqk8v4-4x220K-text-only` (seqs 4, TurboQuant KV) | 4 concurrent, `KV_CACHE_DTYPE=turboquant_k8v4` |
   | `w8a16/dflash2-fp16kv-1x262K-text-only` (seqs 1) | FP8 target route |
   | `w8a16/dflash2-fp16kv-1x240K-text-image` (text+image) | image semantics validated as the existing route does |
   | `w8a16/dflash2-fp8kv-2x220K-text-only` (seqs 2) | 2 concurrent |

   Three consequences the implementer must plan for:
   - Concurrency above 1 exercises the refinement tail at `bs` 2 and 4 and
     `draft_logits` at `[seqs, 15, 248320]` (R5). Measure, do not assume.
   - The FP8 and TurboQuant KV routes must be checked against the DFlash prefix
     commit path, because SGLang refuses FP4 KV for DSpine (§3.4). TurboQuant
     k8v4 is the widest open question; if a DSpine route cannot hold it, say so
     in the profile README rather than shipping an unvalidated route.
   - `MESSAGE_TYPE=text+image` means the vision prefill path plus a text-only
     speculative decode loop, the same shape DFlash2 uses
     (`supports_mm_inputs = False` in `DFlashSpeculator`).

Good news found after the decisions: the launcher already emits
`cudagraph_capture_sizes = {(K+1) * b : b in 1..MAX_NUM_SEQS}` and
`max_cudagraph_capture_size = MAX_NUM_SEQS * (K+1)`
(`launcher.sh:467-480, 5033-5038`), which satisfies the SM75 gate's capture-size
condition for any `K` and any `MAX_NUM_SEQS`. Only the hard width limit at
`flashinfer.py:121` blocks `K=15`, and it blocks every concurrency level equally.

## 13. What the first runs found

### 13.1 Greedy output matches the shipped drafter, not the no-speculative baseline

Five prompts, 128 greedy tokens each, TP4 over four T10 GPUs, eager mode, NVFP4
target with a `K = 7` draft:

| Comparison | Result |
|---|---|
| DSpine `K = 7` vs DFlash2 `K = 7` | identical token ids on all five prompts |
| DSpine `K = 7` vs no speculative decoding | differs on three prompts, at tokens 105, 40, 55 |
| DFlash2 `K = 7` vs no speculative decoding | differs on the same three prompts, at the same tokens, with the same tokens |

The divergences are near ties, not garbled text: `this premise` against
`that premise`, `for the function` against `as the primary function`. Verifying a
block of eight positions changes the target's own numerics, and on a quantized
head a near tie can flip. Every DFlash-family drafter on this route inherits that,
so "greedy output equals the no-speculative baseline" is not a claim this port can
make, and the shipped DFlash2 route does not make it either. What the port can
claim, and measured, is that DSpine behaves exactly like the drafter already
published.

### 13.2 Acceptance on that sample

Same five prompts, greedy, acceptance length as reported by the engine:

| Draft | Acceptance length |
|---|---:|
| DSpine `K = 7` | 3.868 |
| DSpine `K = 3` | 2.981 |
| DFlash2 `K = 7` (control) | 4.365 |

Five prompts of 128 tokens is a bring-up sample, not a benchmark. It says the
bridge works and that DFlash2's selector drafter accepts more on this sample; it
does not rank the two drafters. The GSM8K acceptance test in §10 Phase 6 is what
would settle that, and it is not run yet.

### 13.3 Costs found at run time

- The head row read-back (§7.10) reads 62,080 rows per rank and holds 0.59 GiB of
  FP16 per rank for the life of the process. It happens once, at draft load.
- That reservation is visible in capacity: the `w4a16/dspine-fp16kv-1x262K` route
  gets 395,704 GPU KV tokens where the DFlash2 twin gets 423,080 (§14).
- The refinement adds one all-reduce per draft step for the exact sender codes on
  top of the candidate gather DFlash2 already does.

### 13.4 Checks that run in the suite

`tests/v1/spec_decode/test_dspine.py` holds 41 checks: the bridge against a float64
reference and against a port of the upstream eager path (bit-exact in FP32 and
BF16, block sizes 2/8/16), block-locality of a write, the refinement table against
a float64 loop, the candidate walk against a hand-built chain, the checkpoint
contract including each shape it refuses, the head row read-back through a
repacked head, and the shape-dynamism guard of §7.9. One check skips: served block
8 has no offset 15, so the trained-block locality case cannot run on it.
`tests/v1/spec_decode/test_dflash2.py` and the three DFlash behaviour suites (46
checks) cover the shared walk and cache utilities this port lifted out of DFlash2.

## 14. Measured routes

Launcher `fast` mode, TP4 over four T10 GPUs (PCI indices 0-3), one request at a
time, the reference lane of `profiles/4xT10/README.md`: warm-up excluded, 4K/128 as
the median of three requests, 32K/512 run to completion, Prefix Cache off during
the test. Each DFlash2 row is the same profile file with the DFlash2 draft, run on
this host in this session, so rows pair with the one under them.

They do not pair with the numbers published in `profiles/4xT10/README.md`, which
were taken on another host: the DFlash2 fp16kv route reads 423,080 GPU KV tokens
here against 432,205 there.

| Route | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---:|---:|
| `w4a16/dspine-fp16kv-1x262K-text-only` | 395,704 | 1629.78 / 206.72 | 1532.39 / 183.01 |
| `w4a16/dflash2-fp16kv-1x262K-text-only` | 423,080 | 1628.09 / 223.45 | 1539.66 / 222.57 |
| `w4a16/dspine-fp8kv-2x262K-text-only` | 704,512 | 1389.00 / 209.81 | 1277.10 / 182.55 |
| `w4a16/dflash2-fp8kv-2x262K-text-only` | 755,302 | 1388.23 / 227.23 | 1302.87 / 222.69 |
| `w4a16/dspine-tqk8v4-4x220K-text-only` | 780,000 | 1424.12 / 203.61 | 1207.92 / 127.32 |
| `w4a16/dflash2-tqk8v4-4x220K-text-only` | 851,960 | 1428.97 / 220.34 | 1232.28 / 145.68 |

What the three pairs say:

- **Prefill is a wash**, within a third of a percent in every pair. Speculative
  decoding does not touch prefill, and the draft is not in the prefill path.
- **DSpine holds 6.5-8.4% less KV** than its twin. The refinement's head-row copy
  (§7.10) reserves 0.59 GiB per rank, which is most of the difference; the
  candidate and walk buffers account for the rest.
- **DSpine decodes 7.5-7.7% slower at 4K/128 and 12.6-18.0% slower at 32K/512.**
  Two costs sit there, and neither has been attributed yet: the work the refinement
  adds per draft step (a head-row gather, an all-reduce for the exact sender codes,
  a `[rows, 16, 16]` einsum), and the acceptance difference of §13.2, where the
  DFlash2 drafter accepted more on the sample. The gap widening with context
  suggests per-step overhead rather than acceptance, since acceptance length should
  not depend much on context. That is a hypothesis, not a measurement.
- **The gain published in §4 is not reproduced on this hardware.** Those figures
  come from SGLang on different GPUs and with a different block size. Saying more
  about the difference would need a benchmark this port has not run.

Not measured: the W8A16 profiles (the Qwen3.8-27B-FP8 target is not on this host),
probabilistic draft sampling, and `K = 15`, which §7.4 keeps behind the SM75 graph
gate.
