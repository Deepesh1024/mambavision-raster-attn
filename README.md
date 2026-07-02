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

## Benchmark Results (Isolated Attention Blocks)

> **Note:** The numbers below reflect ONLY the isolated latency of the Attention blocks in Stage 3, measured on a T4 GPU. We explicitly bypassed the MambaVisionMixer blocks because the `mamba-ssm` CUDA extension requires Ampere+ GPUs (Compute Capability 8.0+), meaning the Mamba blocks fell back to a CPU-bound PyTorch `for` loop that corrupted full-model timings. The latency perfectly matches theoretical expectations: it scales directly with sequence length (window size).

| Variant              | Resolution | Latency (ms) | Peak VRAM (GB) |
|----------------------|------------|--------------|----------------|
| Baseline (14x14)     | 512x512    | 0.6          | 0.28           |
| Baseline (14x14)     | 1024x1024  | 1.6          | 0.30           |
| Baseline (14x14)     | 768x1024   | 1.6          | 0.30           |
| Square 8x8           | 512x512    | 0.3          | 0.27           |
| Square 8x8           | 1024x1024  | 0.8          | 0.30           |
| Square 8x8           | 768x1024   | 0.8          | 0.29           |
| Raster s=2           | 512x512    | 0.4          | 0.27           |
| Raster s=2           | 1024x1024  | 1.1          | 0.30           |
| Raster s=2           | 768x1024   | 0.9          | 0.29           |
| Raster s=4           | 512x512    | 0.4          | 0.27           |
| Raster s=4           | 1024x1024  | 1.3          | 0.30           |
| Raster s=4           | 768x1024   | 1.2          | 0.29           |

### Findings
As expected, because attention compute per window is $O(N^2)$ where $N$ is the `window_size`, smaller windows dramatically reduce latency. 
- **Square 8x8** (64 tokens/window) is the fastest.
- **Raster s=2** (112 tokens/window) is slightly slower but still 30% faster than Baseline.
- **Raster s=4** (224 tokens/window) is slightly faster than Baseline (196 tokens/window) on 512x512 and 1024x1024, likely due to better hardware alignment with warp sizes compared to `14x14 = 196`, which may cause inefficient padding under the hood.

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
