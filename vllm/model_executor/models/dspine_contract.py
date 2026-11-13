# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The checkpoint contract of a DSpine draft, read from its config.

A DSpine checkpoint declares itself a plain ``DFlashDraftModel``; the code
bridge is announced only by ``dflash_config.selected_prefix_code_bridge``
(see docs/design/dspine-vllm-port.md section 7.1). Serving a bridge the
checkpoint does not have, or serving one the loader only half understands,
costs acceptance length without ever raising an error, so the whole contract is
resolved and checked here, once, from the config.

The rules mirror the reference plugin's ``DSpineCodeBridge.__init__`` and
``resolve_dspine_head_topk`` (upstream-reference/DSpine, commit c66bf44).
"""

from dataclasses import dataclass
from typing import Any

# The architecture the registry maps to the vLLM implementation.
DSPINE_DRAFT_ARCHITECTURE = "DSpineDraftModel"

# Draft features this implementation does not host. A checkpoint that carries
# any of them is a different drafter and would load with silently wrong maths.
_UNSUPPORTED_CONFIG_KEYS = (
    "conv_kernel_size",
    "conv_group_size",
    "selector_rank",
    "selector_top_k",
    "projector_type",
    "attention_sink_bias",
    "attention_value_scale",
    "selected_prefix_terminal_residual_rank",
)


@dataclass(frozen=True)
class DSpineContract:
    """What a DSpine draft config promises the implementation may rely on."""

    hidden_size: int
    vocab_size: int
    num_layers: int
    code_size: int
    rank: int
    eps: float
    norm_eps: float
    head_topk: int
    # Zero-based indices of the layers that end with a code write. This host
    # requires every layer to write, because a write consumes the layer's
    # residual and applies the next layer's input norm in one step.
    active_layers: frozenset[int]
    # Block size the checkpoint was trained at; serving may be smaller.
    trained_block_size: int


def _config_mapping(value: Any) -> dict:
    return dict(value) if isinstance(value, dict) else (value or {})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def is_dspine_draft(hf_config: Any) -> bool:
    """Whether a draft config carries a code bridge, whatever it calls itself."""
    dflash_config = _config_mapping(getattr(hf_config, "dflash_config", None))
    return dflash_config.get("selected_prefix_code_bridge") is not None


def _resolve_head_topk(config: Any, dflash_config: dict, num_layers: int) -> int:
    """The refinement width K, which the config may state in two places."""
    widths: dict[str, int] = {}

    second_pass = _config_mapping(getattr(config, "code_bridge_second_pass", None))
    if second_pass:
        unknown = set(second_pass) - {"mode", "first_exact", "head_topk"}
        _require(
            not unknown,
            f"Unknown code_bridge_second_pass keys: {sorted(unknown)}",
        )
        _require(
            second_pass.get("mode") == "serial_table",
            "DSpine serves the serial last-write refinement only; "
            f"code_bridge_second_pass.mode is {second_pass.get('mode')!r}",
        )
        _require(
            int(second_pass.get("first_exact", -1)) == num_layers - 1,
            "The serial refinement redoes the LAST write only, so "
            f"code_bridge_second_pass.first_exact must be {num_layers - 1}, got "
            f"{second_pass.get('first_exact')!r}",
        )
        widths["code_bridge_second_pass.head_topk"] = int(second_pass["head_topk"])

    refine = dflash_config.get("selected_prefix_last_write_refine")
    if refine is not None:
        refine = _config_mapping(refine)
        _require(
            not set(refine) - {"kind", "version", "head_topk", "weight"}
            and {"kind", "head_topk"} <= set(refine)
            and refine["kind"] == "exact_sender"
            and refine.get("version", 1) == 1
            and refine.get("weight", 1.0) >= 0,
            "DSpine expects selected_prefix_last_write_refine = "
            "{kind: 'exact_sender', version: 1, head_topk, weight >= 0}, "
            f"got {refine!r}",
        )
        widths["selected_prefix_last_write_refine.head_topk"] = int(
            refine["head_topk"]
        )

    _require(
        widths,
        "A DSpine draft must name its refinement width in "
        "code_bridge_second_pass.head_topk or "
        "dflash_config.selected_prefix_last_write_refine.head_topk",
    )
    _require(
        len(set(widths.values())) == 1,
        f"The DSpine refinement widths disagree across the config: {widths}",
    )
    head_topk = next(iter(widths.values()))
    _require(
        head_topk & (head_topk - 1) == 0 and 2 <= head_topk <= 64,
        "DSpine head_topk must be a power of two in [2, 64], got "
        f"{head_topk}: the candidate walk uses one lane per candidate",
    )
    return head_topk


def resolve_dspine_contract(hf_config: Any) -> DSpineContract:
    """Read and check the DSpine contract of a draft config."""
    dflash_config = _config_mapping(getattr(hf_config, "dflash_config", None))
    bridge = dflash_config.get("selected_prefix_code_bridge")
    _require(
        bridge is not None,
        "This draft config has no dflash_config.selected_prefix_code_bridge, "
        "so it is not a DSpine draft.",
    )
    bridge = _config_mapping(bridge)

    _require(
        dflash_config.get("selected_prefix_terminal") is False,
        "DSpine supports terminal-disabled code bridges only; "
        "dflash_config.selected_prefix_terminal must be false",
    )
    _require(
        dflash_config.get("selected_prefix_feature_write_mode") == "independent",
        "DSpine requires independent learned writes: "
        "dflash_config.selected_prefix_feature_write_mode must be 'independent'",
    )
    _require(
        dflash_config.get("selected_prefix_feature_writes") is True,
        "DSpine requires dflash_config.selected_prefix_feature_writes = true",
    )
    _require(
        dflash_config.get("draft_block_causal") is True,
        "DSpine requires causal draft attention: "
        "dflash_config.draft_block_causal must be true",
    )
    _require(
        getattr(hf_config, "is_causal", True) is not False,
        "DSpine requires causal draft attention; the config sets is_causal=false",
    )
    _require(
        dflash_config.get("selected_prefix_decode_mode", "serial") == "serial",
        "DSpine walks the refinement serially; "
        "dflash_config.selected_prefix_decode_mode must be 'serial'",
    )
    _require(
        dflash_config.get("selected_prefix_feature_reliability") is None,
        "Reliability probes are not part of the DSpine serving path",
    )
    _require(
        not dflash_config.get("selected_prefix_feature_next_token_rows", False),
        "The DSpine refinement is defined for anchor-row (DFlash) semantics; "
        "next-token rows are a DSpark layout",
    )
    unsupported = [key for key in _UNSUPPORTED_CONFIG_KEYS if dflash_config.get(key)]
    _require(
        not unsupported,
        "A DSpine draft is a plain DFlash draft plus a code bridge; unsupported "
        f"draft config keys: {unsupported}",
    )
    _require(
        float(getattr(hf_config, "partial_rotary_factor", 1.0) or 1.0) == 1.0,
        "DSpine drafts rotate the whole head; partial_rotary_factor must be 1",
    )

    num_layers = int(hf_config.num_hidden_layers)
    _require(num_layers >= 1, "A DSpine draft needs at least one layer")
    active_layers = frozenset(
        int(layer) - 1 for layer in dflash_config["selected_prefix_feature_layers"]
    )
    _require(
        active_layers <= set(range(num_layers)),
        "dflash_config.selected_prefix_feature_layers is out of range: "
        f"{sorted(active_layers)} for {num_layers} draft layers",
    )
    _require(
        active_layers == set(range(num_layers)),
        "This host writes a code after every draft layer, so "
        "dflash_config.selected_prefix_feature_layers must list all of them; "
        f"it lists {sorted(index + 1 for index in active_layers)} of {num_layers}.",
    )

    vocab_size = int(hf_config.vocab_size)
    draft_vocab_size = int(
        getattr(hf_config, "draft_vocab_size", vocab_size) or vocab_size
    )
    _require(
        draft_vocab_size == vocab_size,
        "The DSpine code bridge indexes the codebook by token id, so the draft "
        f"and target vocabularies must match: {draft_vocab_size} != {vocab_size}",
    )

    trained_block_size = int(hf_config.block_size)
    _require(
        trained_block_size >= 2,
        f"A DSpine draft needs a block size of at least 2, got {trained_block_size}",
    )

    return DSpineContract(
        hidden_size=int(hf_config.hidden_size),
        vocab_size=vocab_size,
        num_layers=num_layers,
        code_size=int(bridge["code_size"]),
        rank=int(bridge["rank"]),
        eps=float(bridge["eps"]),
        norm_eps=float(hf_config.rms_norm_eps),
        head_topk=_resolve_head_topk(hf_config, dflash_config, num_layers),
        active_layers=active_layers,
        trained_block_size=trained_block_size,
    )


def validate_served_block_size(contract: DSpineContract, served_block_size: int) -> int:
    """Check the block size a route serves against the trained one.

    A DSpine draft may serve a smaller block than it was trained on (the bridge
    locates every row inside its own block), never a larger one: rows past the
    trained block were never trained.
    """
    _require(
        2 <= served_block_size <= contract.trained_block_size,
        f"DSpine serves block sizes 2..{contract.trained_block_size} for this "
        f"draft, got {served_block_size} (one bonus token plus the speculative "
        "tokens). Rows past the trained block were never trained.",
    )
    return served_block_size
