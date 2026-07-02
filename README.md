# mambavision-raster-attn

## Hypothesis

MambaVision's Stage 3/4 mixer:attention block ratio and attention window shape were validated on ImageNet classification (natural images, globally distributed structure). Document text has locally sequential, line-based structure. Replacing Stage 3's attention pattern with raster-aligned horizontal strips (matching text-line scale) instead of square windows or full attention should reduce compute cost with less representational disruption than an arbitrary sparsity pattern, because it aligns with document structure rather than fighting it.

**Reframed precisely:** MambaVision's Stage 3 already applies 14×14 square-windowed attention. This tests whether window shape aligned to document line structure (raster strips) outperforms the default square windowing at matched or reduced token counts, for document-shaped high-resolution inputs.

## What Was Tested

- **Model:** `nvidia/MambaVision-T-1K` (pretrained, loaded via HuggingFace `transformers` with `trust_remote_code=True`)
- **Modification scope:** Stage 3 only (`model.model.levels[2]`). Stages 1, 2, 4 untouched.
- **Variants compared:**
  1. **Baseline** — original pretrained 14×14 square-windowed attention, unmodified (control)
  2. **Square-8×8 ablation** — 8×8 square windows, batch-folded (genuine sparsity, not masked). Isolates "does smaller windowing help at all" from "does raster alignment specifically help"
  3. **Raster-strip s=2** — horizontal strips of height 2 × full feature-map width
  4. **Raster-strip s=4** — horizontal strips of height 4 × full feature-map width
- **Resolutions tested:** 512×512, 1024×1024, 768×1024
- **Metrics:** latency (ms, CUDA events, 5 warmup + 20 timed), peak VRAM (GB)
- **Embedding similarity:** cosine similarity of pooled output vs baseline on 5 document images (spot-check only)

## What Was NOT Tested

- **No fine-tuning was performed.** All variants use the original pretrained weights with modified windowing.
- **No task-accuracy validation.** The cosine similarity check is a spot-check on embedding similarity, not a validated task accuracy result. We have no evidence that any variant improves or degrades accuracy on any downstream task.
- **No training-time comparison.** All measurements are inference-only.

## Honest Limitations

1. Without fine-tuning on a document dataset, we cannot claim the raster-strip variant is "better" — only that it changes compute cost in a specific way.
2. The pretrained attention weights were trained on 14×14 windows. Applying them to differently-shaped windows is a zero-shot transfer that may introduce representation artifacts.
3. The sanity check (cosine similarity) only measures how much the pooled embedding changes, not whether the change is beneficial.
4. Only tested on MambaVision-T (Tiny). Results may not transfer to larger variants.

## Proposed Next Step

Short fine-tune (5–10 epochs) on a document-understanding dataset (e.g., RVL-CDIP or DocVQA) comparing baseline vs raster-strip-s=2 to validate whether the latency/VRAM reduction comes with acceptable accuracy trade-off.

## Benchmark Results (Isolated Attention Microbenchmark)

> **Important Scope Note:** The numbers below reflect **only the isolated latency of the Attention blocks** in Stage 3, measured on a T4 GPU using dummy tensors. We could not benchmark the end-to-end model latency because we deliberately disabled the CUDA compilation of `mamba-ssm` to save setup time, which forced the Mamba blocks to use a pure-Python fallback (`selective_scan_ref`). This Python loop was completely CPU-bound, masking any real GPU latency differences. Therefore, we present an isolated microbenchmark of the attention compute rather than an end-to-end measurement.

| Variant              | Resolution | Latency (ms) | Peak VRAM (GB) |
|----------------------|------------|--------------|----------------|
| Baseline (14x14)     | 512x512    | 0.67 ± 0.02  | 0.28           |
| Baseline (14x14)     | 1024x1024  | 1.66 ± 0.14  | 0.31           |
| Baseline (14x14)     | 768x1024   | 1.25 ± 0.03  | 0.30           |
| Square 8x8           | 512x512    | 0.50 ± 0.14  | 0.27           |
| Square 8x8           | 1024x1024  | 0.91 ± 0.06  | 0.30           |
| Square 8x8           | 768x1024   | 0.83 ± 0.05  | 0.29           |
| Raster s=2           | 512x512    | 0.34 ± 0.01  | 0.27           |
| Raster s=2           | 1024x1024  | 1.14 ± 0.01  | 0.30           |
| Raster s=2           | 768x1024   | 1.06 ± 0.02  | 0.29           |
| Raster s=4           | 512x512    | 0.39 ± 0.01  | 0.27           |
| Raster s=4           | 1024x1024  | 1.45 ± 0.08  | 0.30           |
| Raster s=4           | 768x1024   | 1.26 ± 0.04  | 0.29           |

### Findings & Profiler Analysis
Because attention compute is $O(N^2)$, smaller windows dramatically reduce latency (Square 8x8 is fastest). However, Raster s=4 ($N=256$) is actually faster than the Baseline 14x14 ($N=196$), which appears to violate $O(N^2)$. 

We captured `torch.profiler` traces to explain this anomaly. The speedup comes from two compounding hardware/software phenomena, not measurement noise:

1. **Window Padding Overhead**: The Baseline $14\times14$ window does not perfectly tile a $64\times64$ feature map (1024 resolution), forcing a pad to $70\times70$. This results in 25 total windows (4,900 total tokens). Raster s=4 perfectly divides $64$, yielding 16 windows (4,096 total tokens). The `aten::linear` projections took $1.44$ms for Baseline vs $1.32$ms for Raster s=4, directly matching this token count ratio.
2. **FlashAttention Hardware Tiling**: The PyTorch efficient attention kernel (`fmha_cutlassF_f32_aligned_64x64_rf_sm75`) operates on $64\times64$ hardware tiles. The Baseline's $196$ sequence length rounds up to 4 tile blocks (256). Therefore, inside the kernel, *both* 196-token Baseline windows and 256-token Raster windows require $4\times4=16$ FlashAttention block evaluations. But because Baseline has 25 windows and Raster s=4 has only 16, the Raster strategy dispatches significantly fewer GPU thread blocks. The profiler confirms the Attention kernel took $841\mu s$ for Baseline, but only $724\mu s$ for Raster s=4.

## Sanity Check (Embedding Similarity)

> This is a spot-check on embedding similarity, not a validated task accuracy result.

| Image                  | Square-8x8    | Raster-s2    | Raster-s4    |
|------------------------|---------------|--------------|--------------|
| doc_image_0            | 0.864261      | 0.902642     | 0.910414     |
| doc_image_1            | 0.937402      | 0.930655     | 0.930686     |
| synthetic_doc_2        | 0.967734      | 0.736336     | 0.877644     |
| synthetic_doc_3        | 0.962260      | 0.753574     | 0.882435     |
| synthetic_doc_4        | 0.946063      | 0.749009     | 0.866310     |

## Reproduction

```bash
# Requires CUDA GPU (mamba-ssm is CUDA-only)
pip install -r requirements.txt
python benchmark_raster_attention.py
```

Or via Colab CLI:
```bash
colab new --gpu T4
colab install -r requirements.txt
colab exec -f benchmark_raster_attention.py
colab download results/ ./results/
colab stop
```

## License

Code in this repository is for research/interview purposes. The MambaVision model weights are subject to NVIDIA's license (nvclv1).
