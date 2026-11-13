# 4xT10 Profiles

## Qwen3.8-27B-FP8

Tested weight: [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8); DFlash2 draft: [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen27b/w8a16/dflash2-fp16kv-1x262K-text-only.env` | 262K | FP16 | DFlash2/7 | text-only | 300,304 | 1433.91 / 191.89 | 1536.13 / 189.38 |
| `qwen27b/w8a16/dflash2-fp16kv-1x240K-text-image.env` | 240K | FP16 | DFlash2/7 | text+image | 241,215 | 1443.96 / 190.89 | 1532.96 / 187.85 |
| `qwen27b/w8a16/dflash2-fp8kv-2x220K-text-only.env` | 2 x 220K | FP8 | DFlash2/7 | text-only | 445,644 | 1395.16 / 200.56 | 1486.57 / 210.18 |
| `qwen27b/w8a16/mtp4-fp16kv-1x262K-text-image.env` | 262K | FP16 | MTP/4 | text+image | 320,484 | 1484.30 / 106.04 | 1527.71 / 104.25 |
| `qwen27b/w8a16/mtp4-fp8kv-2x262K-text-image.env` | 2 x 262K | FP8 | MTP/4 | text+image | 565,524 | 1459.21 / 102.41 | 1521.55 / 104.32 |

## Qwen3.8-27B-NVFP4

Tested weight: [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4); DFlash2 draft: [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen27b/w4a16/dflash2-fp16kv-1x262K-text-only.env` | 262K | FP16 | DFlash2/7 | text-only | 432,205 | 1435.68 / 220.99 | 1505.42 / 216.84 |
| `qwen27b/w4a16/dflash2-fp8kv-2x262K-text-only.env` | 2 x 262K | FP8 | DFlash2/7 | text-only | 763,494 | 1334.35 / 213.94 | 1348.40 / 228.76 |
| `qwen27b/w4a16/dflash2-tqk8v4-4x220K-text-only.env` | 4 x 220K | TQK8V4 | DFlash2/7 | text-only | 923,137 | 1374.96 / 234.92 | 1380.19 / 154.43 |
| `qwen27b/w4a16/mtp4-fp16kv-1x262K-text-image.env` | 262K | FP16 | MTP/4 | text+image | 436,388 | 1526.32 / 125.70 | 1517.31 / 123.13 |
| `qwen27b/w4a16/mtp4-fp8kv-2x262K-text-image.env` | 2 x 262K | FP8 | MTP/4 | text+image | 823,249 | 1507.94 / 116.86 | 1477.49 / 118.77 |


### DSpine and its DFlash2 twins, measured 2026-11-13

Tested weight: [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4); DSpine draft: `Qwen3.8-27B-DSpine-b16` from Tencent DSpine, local checkpoint; DFlash2 draft: [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen27b/w4a16/dspine-fp16kv-1x262K-text-only.env` | 262K | FP16 | DSpine/7 | text-only | 395,704 | 1629.78 / 206.72 | 1532.39 / 183.01 |
| `qwen27b/w4a16/dspine-fp8kv-2x262K-text-only.env` | 2 x 262K | FP8 | DSpine/7 | text-only | 704,512 | 1389.00 / 209.81 | 1277.10 / 182.55 |
| `qwen27b/w4a16/dspine-tqk8v4-4x220K-text-only.env` | 4 x 220K | TQK8V4 | DSpine/7 | text-only | 780,000 | 1424.12 / 203.61 | 1207.92 / 127.32 |
| `qwen27b/w4a16/dflash2-fp16kv-1x262K-text-only.env` | 262K | FP16 | DFlash2/7 | text-only | 423,080 | 1628.09 / 223.45 | 1539.66 / 222.57 |
| `qwen27b/w4a16/dflash2-fp8kv-2x262K-text-only.env` | 2 x 262K | FP8 | DFlash2/7 | text-only | 755,302 | 1388.23 / 227.23 | 1302.87 / 222.69 |
| `qwen27b/w4a16/dflash2-tqk8v4-4x220K-text-only.env` | 4 x 220K | TQK8V4 | DFlash2/7 | text-only | 851,960 | 1428.97 / 220.34 | 1232.28 / 145.68 |

These rows come from the development host of that date (six GPUs: four Tesla T10 at
PCI indices 0-3 plus two RTX 2080 Ti, 251 GiB RAM), not from the host of note 3, and
they are not comparable with the tables above. Each DSpine row pairs only with the
DFlash2 row beside it, measured in the same session: the same DFlash2 fp16kv route
reads 423,080 GPU KV tokens here against 432,205 on the host of note 3. Within these
pairs DSpine holds 6.5-8.4% less KV, because the last-write refinement keeps a copy
of the output matrix per rank; prefill matches DFlash2 within a third of a percent;
decode is 7.5-7.7% lower at 4K/128 and 12.6-18.0% lower at 32K/512. The DSpine
routes are published as measured routes, not as a faster alternative: the checks
behind them, and what is not yet attributed, are in
`docs/design/dspine-vllm-port.md` sections 13 and 14.

## Notes

1. Profiles are flat under each model/weight directory. Mode is selected by the launcher and defaults to `fast`.
2. Performance uses the launcher's reproducible reference lane: Prefix Cache is disabled only during testing, one text-only request is sent at a time, warm-up is excluded, 4K/128 is the median of three requests, and 32K/512 is run to completion. Multimodal profiles use the same text-only performance lane; image semantics are validated separately. `4K/128` means exactly 4,096 input tokens and 128 output tokens; `32K/512` means exactly 32,768 input tokens and 512 output tokens. Raw logs and request JSON remain in the external audit directory.
3. Test environment: the FP8/W8A16 profiles were validated on 2026-09-19 and the NVFP4/W4A16 profiles on 2026-09-23, both with v0.2.1. The host has two sockets with Intel Xeon E5-2673 v4 CPUs (80 logical CPUs total), 60 GiB RAM and 8 GiB swap. The tested topology uses physical GPUs 0, 2, 3 and 4, four Tesla T10 GPUs with 16,384 MiB each; all four are on the same NUMA node and connected by PCIe PIX links, and the service uses TP4 over this group. The host driver is NVIDIA 595.91.07. The runtime is the repository's `vllm-def-cu130` environment (CUDA 13.0, torch 2.13.0+cu130).
4. The `4x220K` route uses `GPU_UTIL=0.93` to reserve runtime headroom. Four simultaneous 32K requests passed; four simultaneous requests near the 220K limit are not a supported worst-case guarantee and can OOM on DFlash temporary buffers. Multi-concurrency profiles should not receive simultaneous ultra-long requests.
5. All profiles use the launcher's default `MAX_BATCHED_TOKENS=2048`; enable Prefix Cache in production for FP8KV routes. The measurements above deliberately disabled Prefix Cache. The W8A16 FP8KV DFlash2 2x220K route was capacity-sized from 445,644 GPU KV tokens at `GPU_UTIL=0.96`; two simultaneous 32K/512 requests completed with exact token counts and the service remained healthy. Under that concurrent load, decode throughput was 62.88 and 65.70 tok/s per request.
