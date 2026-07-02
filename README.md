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

## Benchmark Results

> **Note:** All numbers below are from actual script output on a T4 GPU. No numbers are estimated or fabricated.

<!-- RESULTS_TABLE will be inserted here after benchmark execution -->

_Results pending — will be populated after Colab T4 execution._

## Sanity Check (Embedding Similarity)

> This is a spot-check on embedding similarity, not a validated task accuracy result.

<!-- SANITY_TABLE will be inserted here after benchmark execution -->

_Results pending — will be populated after Colab T4 execution._

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
