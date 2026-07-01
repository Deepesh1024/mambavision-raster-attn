#!/usr/bin/env python3
"""
Phase 2: Model exploration and deepcopy gate.
Run on Colab T4 BEFORE Phase 3.

Outputs:
- Full module tree of MambaVision-T
- Stage 3 block types (Attention vs MambaVisionMixer)
- Attention forward() shape contract
- Attention source code location
- CUDA status
- deepcopy success/failure (GATE for Phase 3)
"""

import sys
import os
import inspect
import copy
import torch
from transformers import AutoModel

def main():
    print("=" * 70)
    print("PHASE 2: MODEL EXPLORATION AND DEEPCOPY GATE")
    print("=" * 70)

    # 1. CUDA check
    print("\n--- CUDA STATUS ---")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("ERROR: No CUDA device. This script requires a GPU.")
        print("Stopping. Run this on a Colab T4 instance.")
        sys.exit(1)

    # 2. Load model
    print("\n--- LOADING MODEL ---")
    model = AutoModel.from_pretrained(
        "nvidia/MambaVision-T-1K",
        trust_remote_code=True
    )
    model = model.cuda().eval()
    print("Model loaded successfully.")

    # 3. Print full module tree
    print("\n--- MODULE TREE ---")
    for name, module in model.named_modules():
        indent = "  " * name.count(".")
        print(f"{indent}{name}: {module.__class__.__name__}")

    # 4. Confirm Stage 3
    print("\n--- STAGE 3 ANALYSIS ---")
    stage3 = model.model.levels[2]
    print(f"model.model.levels[2] class: {stage3.__class__.__name__}")
    print(f"  conv (is ConvBlock stage): {stage3.conv}")
    print(f"  transformer_block: {stage3.transformer_block}")
    print(f"  window_size: {stage3.window_size}")
    print(f"  num blocks: {len(stage3.blocks)}")

    for i, blk in enumerate(stage3.blocks):
        mixer_type = blk.mixer.__class__.__name__
        print(f"  Block {i}: mixer = {mixer_type}")

    # 5. Attention forward shape contract
    print("\n--- ATTENTION FORWARD SHAPE CONTRACT ---")
    # Find an attention block in stage 3
    attn_block = None
    for blk in stage3.blocks:
        if blk.mixer.__class__.__name__ == "Attention":
            attn_block = blk.mixer
            break

    if attn_block is not None:
        print(f"Attention class: {attn_block.__class__.__name__}")
        print(f"  num_heads: {attn_block.num_heads}")
        print(f"  head_dim: {attn_block.head_dim}")
        print(f"  fused_attn: {attn_block.fused_attn}")

        # Test with dummy input matching window_size^2 tokens
        ws = stage3.window_size
        dummy = torch.randn(2, ws * ws, attn_block.num_heads * attn_block.head_dim).cuda()
        with torch.no_grad():
            out = attn_block(dummy)
        print(f"  Input shape:  {tuple(dummy.shape)}  (B, window_size^2, C)")
        print(f"  Output shape: {tuple(out.shape)}")
    else:
        print("WARNING: No Attention block found in Stage 3!")

    # 6. Attention source code location
    print("\n--- ATTENTION SOURCE LOCATION ---")
    if attn_block is not None:
        src_file = inspect.getfile(attn_block.__class__)
        print(f"Source file: {src_file}")
        # Print first 50 lines of forward method
        src_lines = inspect.getsource(attn_block.forward)
        print(f"forward() source:\n{src_lines}")

    # 7. Stage 3 dim check
    print("\n--- STAGE 3 DIMENSIONS ---")
    # Run a forward pass through stages 0,1,2 to see actual feature map shape
    dummy_img = torch.randn(1, 3, 512, 512).cuda()
    with torch.no_grad():
        x = model.model.patch_embed(dummy_img)
        print(f"After PatchEmbed (512 input): {tuple(x.shape)}")
        x, _ = model.model.levels[0](x)
        print(f"After Stage 0: {tuple(x.shape)}")
        x, _ = model.model.levels[1](x)
        print(f"After Stage 1: {tuple(x.shape)}")
        # Don't run through stage 2 yet — just check what goes in
        print(f"Stage 2 (Stage 3 in 1-indexed) input would be: {tuple(x.shape)}")

    # 8. DEEPCOPY GATE
    print("\n--- DEEPCOPY GATE ---")
    try:
        model_copy = copy.deepcopy(model)
        print(f"deepcopy succeeded: {model_copy is not None}")
        # Verify it's independent
        param_orig = next(model.parameters()).data_ptr()
        param_copy = next(model_copy.parameters()).data_ptr()
        print(f"Parameters are independent (different memory): {param_orig != param_copy}")
        del model_copy
        torch.cuda.empty_cache()
        print("GATE PASSED: deepcopy works. Phase 3 can use copy.deepcopy().")
    except Exception as e:
        print(f"deepcopy FAILED: {type(e).__name__}: {e}")
        print("FALLBACK: Phase 3 must use AutoModel.from_pretrained() per variant.")
        print("GATE FAILED — report this to decide Phase 3 strategy.")

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
