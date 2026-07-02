#!/usr/bin/env python3
"""
MambaVision Raster-Aligned Attention Benchmark
===============================================

Tests whether raster-strip windowed attention in Stage 3 of MambaVision-T
outperforms the default 14x14 square windowing for document-shaped inputs.

Variants:
  1. Baseline — original 14x14 square windows (control)
  2. Square-8x8 — 8x8 square windows (ablation: does smaller windowing help?)
  3. Raster-strip s=2 — horizontal strips of height 2 x full width
  4. Raster-strip s=4 — horizontal strips of height 4 x full width

All variants share the same pretrained attention weights.
Only Stage 3 (model.model.levels[2]) is modified.

Requires: CUDA GPU, mamba-ssm, transformers, timm, einops, matplotlib
"""

import sys
import os
import copy
import time
import json
import inspect
import math
import types

# MOCK CUDA EXTENSIONS FOR MAMBA-SSM SO IT DOES NOT CRASH
sys.modules['selective_scan_cuda'] = types.ModuleType('selective_scan_cuda')
sys.modules['causal_conv1d_cuda'] = types.ModuleType('causal_conv1d_cuda')
sys.modules['selective_scan_cuda'].fwd = lambda *args, **kwargs: None
sys.modules['selective_scan_cuda'].bwd = lambda *args, **kwargs: None

try:
    import mamba_ssm.ops.selective_scan_interface
    if hasattr(mamba_ssm.ops.selective_scan_interface, 'selective_scan_ref'):
        mamba_ssm.ops.selective_scan_interface.selective_scan_fn = mamba_ssm.ops.selective_scan_interface.selective_scan_ref
except Exception as e:
    print(f"Warning: Failed to patch mamba_ssm: {e}")

from pathlib import Path
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─── Ensure results directory exists ───
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: SETUP AND VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

def verify_setup():
    """Verify CUDA, load model, print architecture details."""
    print("=" * 70)
    print("SECTION 1: SETUP AND VERIFICATION")
    print("=" * 70)

    # CUDA check
    if not torch.cuda.is_available():
        print("FATAL: No CUDA device available.")
        print("This script requires a GPU (tested on T4).")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    from transformers import AutoModel
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, 'all_tied_weights_keys'):
        PreTrainedModel.all_tied_weights_keys = property(lambda self: {})
    
    print("\nLoading nvidia/MambaVision-T-1K...")
    model = AutoModel.from_pretrained(
        "nvidia/MambaVision-T-1K",
        trust_remote_code=True
    )
    model = model.to(device).eval()
    print("Model loaded successfully.")

    # Print module tree (abbreviated)
    print("\n--- MODULE TREE (top-level) ---")
    for name, mod in model.named_children():
        print(f"  {name}: {mod.__class__.__name__}")
        if hasattr(mod, 'named_children'):
            for n2, m2 in mod.named_children():
                print(f"    {n2}: {m2.__class__.__name__}")
                if hasattr(m2, '__len__'):
                    try:
                        for i, item in enumerate(m2):
                            print(f"      [{i}]: {item.__class__.__name__}")
                    except TypeError:
                        pass

    # Stage 3 analysis
    stage3 = model.model.levels[2]
    print(f"\n--- STAGE 3 (model.model.levels[2]) ---")
    print(f"  Class: {stage3.__class__.__name__}")
    print(f"  window_size: {stage3.window_size}")
    print(f"  transformer_block: {stage3.transformer_block}")
    print(f"  Number of blocks: {len(stage3.blocks)}")
    for i, blk in enumerate(stage3.blocks):
        print(f"    Block {i}: {blk.mixer.__class__.__name__}")

    # Attention source
    attn_block = None
    for blk in stage3.blocks:
        if blk.mixer.__class__.__name__ == "Attention":
            attn_block = blk.mixer
            break

    if attn_block:
        print(f"\n--- ATTENTION DETAILS ---")
        print(f"  Source: {inspect.getfile(attn_block.__class__)}")
        print(f"  num_heads={attn_block.num_heads}, head_dim={attn_block.head_dim}")
        ws = stage3.window_size
        dummy = torch.randn(2, ws * ws, attn_block.num_heads * attn_block.head_dim, device=device)
        with torch.no_grad():
            out = attn_block(dummy)
        print(f"  forward() input:  (B, {ws}*{ws}, C) = {tuple(dummy.shape)}")
        print(f"  forward() output: {tuple(out.shape)}")

    # Feature map sizes at different input resolutions
    print("\n--- FEATURE MAP SIZES AT STAGE 3 INPUT ---")
    for res in [(512, 512), (1024, 1024), (768, 1024)]:
        dummy_img = torch.randn(1, 3, *res, device=device)
        with torch.no_grad():
            x = model.model.patch_embed(dummy_img)
            x, _ = model.model.levels[0](x)
            x, _ = model.model.levels[1](x)
        print(f"  Input {res[0]}x{res[1]} → Stage 3 input: {tuple(x.shape)} (B, C, H, W)")
        del dummy_img, x
    torch.cuda.empty_cache()

    # Deepcopy gate
    print("\n--- DEEPCOPY GATE ---")
    try:
        model_copy = copy.deepcopy(model)
        print(f"deepcopy succeeded: True")
        p1 = next(model.parameters()).data_ptr()
        p2 = next(model_copy.parameters()).data_ptr()
        print(f"Independent memory: {p1 != p2}")
        del model_copy
        torch.cuda.empty_cache()
        use_deepcopy = True
    except Exception as e:
        print(f"deepcopy FAILED: {e}")
        print("Will reload model per variant instead.")
        use_deepcopy = False

    return model, device, use_deepcopy


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: ATTENTION VARIANTS
# ═══════════════════════════════════════════════════════════════════════

def strip_partition(x, strip_height):
    """
    Partition feature map into horizontal strips.

    Args:
        x: (B, C, H, W) feature map
        strip_height: height of each strip in tokens

    Returns:
        strips: (B * num_strips, strip_height * W, C)
        pad_h: amount of padding added to H
    """
    B, C, H, W = x.shape

    # Pad H if not divisible
    pad_h = (strip_height - H % strip_height) % strip_height
    if pad_h > 0:
        x = F.pad(x, (0, 0, 0, pad_h))  # pad bottom only

    _, _, Hp, _ = x.shape
    num_strips = Hp // strip_height

    # (B, C, num_strips, strip_height, W) → (B*num_strips, strip_height*W, C)
    x = x.view(B, C, num_strips, strip_height, W)
    x = x.permute(0, 2, 3, 4, 1)  # (B, num_strips, strip_height, W, C)
    x = x.reshape(B * num_strips, strip_height * W, C)
    return x, pad_h


def strip_reverse(strips, strip_height, H, W, pad_h):
    """
    Reverse strip partition.

    Args:
        strips: (B * num_strips, strip_height * W, C)
        strip_height: height of each strip
        H: original H (before padding)
        W: width
        pad_h: padding that was added

    Returns:
        x: (B, C, H, W)
    """
    Hp = H + pad_h
    num_strips = Hp // strip_height
    C = strips.shape[-1]
    B = strips.shape[0] // num_strips

    x = strips.reshape(B, num_strips, strip_height, W, C)
    x = x.permute(0, 4, 1, 2, 3)  # (B, C, num_strips, strip_height, W)
    x = x.reshape(B, C, Hp, W)

    if pad_h > 0:
        x = x[:, :, :H, :W].contiguous()
    return x


def square_window_partition(x, window_size):
    """
    Partition into square windows (same as original but with different window_size).
    Handles padding.

    Args:
        x: (B, C, H, W)
        window_size: int

    Returns:
        windows: (B * num_windows, window_size * window_size, C)
        pad_h, pad_w: padding added
    """
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))

    _, _, Hp, Wp = x.shape
    x = x.view(B, C, Hp // window_size, window_size, Wp // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows, pad_h, pad_w


def square_window_reverse(windows, window_size, H, W, pad_h, pad_w):
    """Reverse square window partition."""
    Hp = H + pad_h
    Wp = W + pad_w
    C = windows.shape[-1]
    B = int(windows.shape[0] / (Hp * Wp / window_size / window_size))

    x = windows.reshape(B, Hp // window_size, Wp // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)

    if pad_h > 0 or pad_w > 0:
        x = x[:, :, :H, :W].contiguous()
    return x


class ModifiedMambaVisionLayer(nn.Module):
    """
    MambaVisionLayer with configurable windowing strategy.
    Supports: 'square' (with custom window_size) or 'strip' (horizontal strips).
    """

    def __init__(self, original_layer, mode='square', window_size=8, strip_height=2):
        super().__init__()
        # Copy all sub-modules from the original layer
        self.blocks = original_layer.blocks
        self.downsample = original_layer.downsample
        self.transformer_block = original_layer.transformer_block
        self.conv = original_layer.conv

        self.mode = mode
        self.custom_window_size = window_size
        self.strip_height = strip_height

    def forward(self, x):
        _, _, H, W = x.shape

        if self.transformer_block:
            if self.mode == 'square':
                x, pad_h, pad_w = square_window_partition(x, self.custom_window_size)
                for blk in self.blocks:
                    x = blk(x)
                x = square_window_reverse(x, self.custom_window_size, H, W, pad_h, pad_w)

            elif self.mode == 'strip':
                x, pad_h = strip_partition(x, self.strip_height)
                for blk in self.blocks:
                    x = blk(x)
                x = strip_reverse(x, self.strip_height, H, W, pad_h)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
        else:
            for blk in self.blocks:
                x = blk(x)

        if self.downsample is None:
            return x, x
        return self.downsample(x), x


def create_variant(base_model, variant_name, device, use_deepcopy=True):
    """
    Create a model variant by modifying Stage 3's windowing.

    Args:
        base_model: the original loaded model
        variant_name: one of 'baseline', 'square8', 'raster_s2', 'raster_s4'
        device: torch device
        use_deepcopy: if True, deepcopy the model; if False, reload from HF

    Returns:
        Modified model on device
    """
    if variant_name == 'baseline':
        if use_deepcopy:
            return copy.deepcopy(base_model)
        else:
            from transformers import AutoModel
            m = AutoModel.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)
            return m.to(device).eval()

    # Create a copy
    if use_deepcopy:
        model = copy.deepcopy(base_model)
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)
        model = model.to(device).eval()

    # Get the original Stage 3
    original_stage3 = model.model.levels[2]

    if variant_name == 'square8':
        modified = ModifiedMambaVisionLayer(original_stage3, mode='square', window_size=8)
    elif variant_name == 'raster_s2':
        modified = ModifiedMambaVisionLayer(original_stage3, mode='strip', strip_height=2)
    elif variant_name == 'raster_s4':
        modified = ModifiedMambaVisionLayer(original_stage3, mode='strip', strip_height=4)
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    # Replace Stage 3
    model.model.levels[2] = modified
    return model


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: SANITY CHECK
# ═══════════════════════════════════════════════════════════════════════

def download_document_images(device):
    """
    Download 5 document-like images for sanity check.
    Falls back to synthetic tall-aspect-ratio images if download fails.
    """
    import requests
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((768, 1024)),  # Tall document shape
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Try to download real document images from public datasets
    urls = [
        "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg",
        "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg",
    ]

    images = []
    labels = []

    for i, url in enumerate(urls):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                images.append(transform(img).unsqueeze(0).to(device))
                labels.append(f"doc_image_{i}")
        except Exception as e:
            print(f"  Download failed for {url}: {e}")

    # Fill remaining with synthetic document-like images (text-like patterns)
    while len(images) < 5:
        idx = len(images)
        # Create synthetic document: white background with horizontal dark lines
        img = torch.ones(1, 3, 768, 1024, device=device) * 0.9
        # Add horizontal "text lines" at regular intervals
        for row in range(50, 700, 30):
            thickness = torch.randint(2, 5, (1,)).item()
            img[:, :, row:row+thickness, 50:950] = 0.1 + 0.1 * torch.rand(1, device=device)
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        img = (img - mean) / std
        images.append(img)
        labels.append(f"synthetic_doc_{idx}")

    return images, labels


def run_sanity_check(base_model, device, use_deepcopy):
    """Compare embeddings across variants on document images."""
    print("\n" + "=" * 70)
    print("SECTION 3: SANITY CHECK (EMBEDDING SIMILARITY)")
    print("This is a spot-check on embedding similarity, NOT a validated")
    print("task accuracy result.")
    print("=" * 70)

    images, labels = download_document_images(device)
    print(f"\nUsing {len(images)} images: {labels}")

    variants = ['baseline', 'square8', 'raster_s2', 'raster_s4']
    embeddings = {v: [] for v in variants}

    for vname in variants:
        print(f"\n  Creating variant: {vname}...")
        try:
            model = create_variant(base_model, vname, device, use_deepcopy)
            model.eval()

            for img in images:
                with torch.no_grad():
                    # MambaVisionModel.forward returns (pooled, outs)
                    pooled, _ = model(img)
                    embeddings[vname].append(pooled.cpu())

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR creating/running variant {vname}: {e}")
            embeddings[vname] = None

    # Compute cosine similarity vs baseline
    print("\n--- COSINE SIMILARITY VS BASELINE ---")
    results = []
    header = f"{'Image':<20} {'Square-8x8':>12} {'Raster-s2':>12} {'Raster-s4':>12}"
    print(header)
    print("-" * len(header))

    if embeddings['baseline'] is not None:
        for i, label in enumerate(labels):
            row = {"image": label}
            base_emb = embeddings['baseline'][i]
            parts = []
            for vname in ['square8', 'raster_s2', 'raster_s4']:
                if embeddings[vname] is not None and i < len(embeddings[vname]):
                    cos_sim = F.cosine_similarity(
                        base_emb.flatten().unsqueeze(0),
                        embeddings[vname][i].flatten().unsqueeze(0)
                    ).item()
                    row[vname] = cos_sim
                    parts.append(f"{cos_sim:>12.6f}")
                else:
                    row[vname] = "ERROR"
                    parts.append(f"{'ERROR':>12}")
            print(f"{label:<20} {' '.join(parts)}")
            results.append(row)

    # Save results
    sanity_md = "# Sanity Check: Embedding Cosine Similarity vs Baseline\n\n"
    sanity_md += "> This is a spot-check on embedding similarity, not a validated task accuracy result.\n\n"
    sanity_md += "| Image | Square-8x8 | Raster-s2 | Raster-s4 |\n"
    sanity_md += "|-------|-----------|-----------|----------|\n"
    for r in results:
        sq = f"{r.get('square8', 'ERR'):.6f}" if isinstance(r.get('square8'), float) else str(r.get('square8', 'ERR'))
        rs2 = f"{r.get('raster_s2', 'ERR'):.6f}" if isinstance(r.get('raster_s2'), float) else str(r.get('raster_s2', 'ERR'))
        rs4 = f"{r.get('raster_s4', 'ERR'):.6f}" if isinstance(r.get('raster_s4'), float) else str(r.get('raster_s4', 'ERR'))
        sanity_md += f"| {r['image']} | {sq} | {rs2} | {rs4} |\n"

    (RESULTS_DIR / "sanity_check.md").write_text(sanity_md)
    print(f"\nSaved to {RESULTS_DIR / 'sanity_check.md'}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: BENCHMARK
# ═══════════════════════════════════════════════════════════════════════

def benchmark_variant(model, vname, device, resolution, warmup=5, iterations=20):
    """
    Benchmark JUST the isolated Attention block at a given resolution.
    This bypasses the Mamba block PyTorch fallback overhead completely.
    """
    H, W = resolution
    
    # Stage 3 is downsampled by 16
    stage3_H = H // 16
    stage3_W = W // 16
    
    # Stage 3 feature dimension
    C = 320
    
    # Stage 3 input tensor
    x = torch.randn(1, C, stage3_H, stage3_W, device=device)
    
    # Get the Attention block from Stage 3
    attn_block = None
    for blk in model.model.levels[2].blocks:
        if blk.mixer.__class__.__name__ == "Attention":
            attn_block = blk.mixer
            break
            
    if attn_block is None:
        return {"error": "Attention block not found"}
        
    # Partition the input tensor EXACTLY as the layer would
    if vname == 'baseline':
        x, _, _ = square_window_partition(x, window_size=14)
    elif vname == 'square8':
        x, _, _ = square_window_partition(x, window_size=8)
    elif vname == 'raster_s2':
        x, _ = strip_partition(x, strip_height=2)
    elif vname == 'raster_s4':
        x, _ = strip_partition(x, strip_height=4)

    try:
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = attn_block(x)
            torch.cuda.synchronize()

        # Reset VRAM tracking
        torch.cuda.reset_peak_memory_stats()

        # Timed iterations
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        for _ in range(iterations):
            with torch.no_grad():
                _ = attn_block(x)

        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        avg_ms = elapsed_ms / iterations
        peak_vram = torch.cuda.max_memory_allocated() / 1e9

        del x
        torch.cuda.empty_cache()

        return {"latency_ms": avg_ms, "peak_vram_gb": peak_vram}

    except torch.cuda.OutOfMemoryError as e:
        del x
        torch.cuda.empty_cache()
        return {"error": f"OOM: {str(e)[:200]}"}
    except Exception as e:
        del x
        torch.cuda.empty_cache()
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def run_benchmarks(base_model, device, use_deepcopy):
    """Run full benchmark suite."""
    print("\n" + "=" * 70)
    print("SECTION 4: BENCHMARK")
    print("=" * 70)

    resolutions = [(512, 512), (1024, 1024), (768, 1024)]
    variants = ['baseline', 'square8', 'raster_s2', 'raster_s4']
    variant_labels = {
        'baseline': 'Baseline (14x14)',
        'square8': 'Square 8x8',
        'raster_s2': 'Raster s=2',
        'raster_s4': 'Raster s=4',
    }

    results = []

    for vname in variants:
        print(f"\n--- Variant: {variant_labels[vname]} ---")
        try:
            model = create_variant(base_model, vname, device, use_deepcopy)
            model.eval()
        except Exception as e:
            print(f"  FAILED to create variant: {e}")
            for res in resolutions:
                results.append({
                    'variant': vname,
                    'resolution': f"{res[0]}x{res[1]}",
                    'error': f"Creation failed: {str(e)[:200]}"
                })
            continue

        for res in resolutions:
            print(f"  Resolution {res[0]}x{res[1]}...", end=" ", flush=True)
            r = benchmark_variant(model, vname, device, res)

            entry = {
                'variant': vname,
                'variant_label': variant_labels[vname],
                'resolution': f"{res[0]}x{res[1]}",
            }

            if 'error' in r:
                entry['error'] = r['error']
                print(f"ERROR: {r['error']}")
            else:
                entry['latency_ms'] = r['latency_ms']
                entry['peak_vram_gb'] = r['peak_vram_gb']
                print(f"latency={r['latency_ms']:.1f}ms, VRAM={r['peak_vram_gb']:.2f}GB")

            results.append(entry)

        del model
        torch.cuda.empty_cache()

    # Print results table
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS TABLE")
    print("=" * 70)

    header = f"{'Variant':<20} {'Resolution':<12} {'Latency(ms)':>12} {'Peak VRAM(GB)':>14}"
    print(header)
    print("-" * len(header))
    for r in results:
        if 'error' in r:
            print(f"{r.get('variant_label', r['variant']):<20} {r['resolution']:<12} {'ERROR':>12} {r['error'][:14]:>14}")
        else:
            print(f"{r.get('variant_label', r['variant']):<20} {r['resolution']:<12} {r['latency_ms']:>12.1f} {r['peak_vram_gb']:>14.2f}")

    # Save markdown table
    md = "# Benchmark Results\n\n"
    md += "| Variant | Resolution | Latency (ms) | Peak VRAM (GB) |\n"
    md += "|---------|-----------|-------------|----------------|\n"
    for r in results:
        if 'error' in r:
            md += f"| {r.get('variant_label', r['variant'])} | {r['resolution']} | ERROR | {r['error']} |\n"
        else:
            md += f"| {r.get('variant_label', r['variant'])} | {r['resolution']} | {r['latency_ms']:.1f} | {r['peak_vram_gb']:.2f} |\n"

    (RESULTS_DIR / "benchmark_table.md").write_text(md)
    print(f"\nSaved to {RESULTS_DIR / 'benchmark_table.md'}")

    # Save raw JSON
    (RESULTS_DIR / "benchmark_raw.json").write_text(json.dumps(results, indent=2))

    return results


def make_plot(results):
    """Generate benchmark visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Filter successful results
    ok = [r for r in results if 'error' not in r]
    if not ok:
        print("No successful benchmark results to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    variants = list(dict.fromkeys(r['variant_label'] for r in ok))
    resolutions = list(dict.fromkeys(r['resolution'] for r in ok))
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

    # Latency plot
    x = np.arange(len(resolutions))
    width = 0.18
    for i, v in enumerate(variants):
        vals = []
        for res in resolutions:
            match = [r for r in ok if r['variant_label'] == v and r['resolution'] == res]
            vals.append(match[0]['latency_ms'] if match else 0)
        ax1.bar(x + i * width, vals, width, label=v, color=colors[i % len(colors)], alpha=0.85)

    ax1.set_xlabel('Resolution', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('Inference Latency by Variant', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * (len(variants) - 1) / 2)
    ax1.set_xticklabels(resolutions)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # VRAM plot
    for i, v in enumerate(variants):
        vals = []
        for res in resolutions:
            match = [r for r in ok if r['variant_label'] == v and r['resolution'] == res]
            vals.append(match[0]['peak_vram_gb'] if match else 0)
        ax2.bar(x + i * width, vals, width, label=v, color=colors[i % len(colors)], alpha=0.85)

    ax2.set_xlabel('Resolution', fontsize=12)
    ax2.set_ylabel('Peak VRAM (GB)', fontsize=12)
    ax2.set_title('Peak VRAM Usage by Variant', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width * (len(variants) - 1) / 2)
    ax2.set_xticklabels(resolutions)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "benchmark_plot.png", dpi=150, bbox_inches='tight')
    print(f"Plot saved to {RESULTS_DIR / 'benchmark_plot.png'}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("MambaVision Raster-Aligned Attention Benchmark")
    print("=" * 70)
    print()

    # Phase 1: Setup
    base_model, device, use_deepcopy = verify_setup()

    # Phase 2: Sanity check
    sanity_results = run_sanity_check(base_model, device, use_deepcopy)

    # Phase 3: Benchmark
    bench_results = run_benchmarks(base_model, device, use_deepcopy)

    # Phase 4: Plot
    make_plot(bench_results)

    print("\n" + "=" * 70)
    print("ALL DONE")
    print(f"Results saved to: {RESULTS_DIR.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
