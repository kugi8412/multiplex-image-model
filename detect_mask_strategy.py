#!/usr/bin/env python
# -*- coding: utf-8 -*-
# detect_mask_strategy.py
#
# Diagnostic tool: inspect a trained checkpoint (or a config YAML) and report
#   1. Which mask strategy is configured (zero / negative / learnable / learnable_full).
#   2. The concrete VALUES of every mask token found in the checkpoint
#      (encoder mask_token for iBOT masking + I-JEPA predictor mask_token),
#      including whether it is a single shared token or a full per-position bank,
#      plus value statistics (min/max/mean/std) and a data-driven inference of
#      the strategy that actually produced those weights.
#
# Usage:
#   python detect_mask_strategy.py checkpoints/ijepa_kronos_vit-...-final.pth
#   python detect_mask_strategy.py checkpoints/ijepa_kronos_vit-...-final.pth --config configs/ijepa_vit.yaml
#   python detect_mask_strategy.py --config configs/ijepa_vit.yaml        # config only, no checkpoint

import argparse
import os

import torch

try:
    from ruamel.yaml import YAML
    _yaml = YAML(typ="safe")
except Exception:  # pragma: no cover - ruamel is a project dependency
    _yaml = None


# ------------------------------------------------------------------
# Value-level inference
# ------------------------------------------------------------------

def _infer_strategy_from_values(t: torch.Tensor, is_parameter: bool) -> str:
    """Infer the mask strategy from the raw tensor values.

    Distinguishes the four supported strategies purely from the weights, so it
    works even when the config is unavailable.
    """
    t = t.detach().float()
    n_positions = t.shape[1] if t.dim() >= 2 else 1

    if torch.allclose(t, torch.zeros_like(t)):
        # An all-zero learnable parameter is an *untrained* zero/learnable init;
        # a registered buffer of zeros is the fixed "zero" strategy.
        return "zero" if not is_parameter else "learnable (untrained / collapsed to 0)"
    if torch.allclose(t, torch.full_like(t, -1.0)):
        return "negative"

    # Non-trivial values ⇒ learnable. Full vs shared depends on #positions.
    if n_positions > 1:
        return "learnable_full (per-position)"
    return "learnable (single shared token)"


def _describe_tensor(name: str, t: torch.Tensor, is_parameter: bool) -> None:
    t = t.detach().float()
    shape = tuple(t.shape)
    n_positions = shape[1] if len(shape) >= 2 else 1
    embed_dim = shape[-1] if len(shape) >= 1 else 0

    kind = "per-position (FULL)" if n_positions > 1 else "single shared"
    storage = "Parameter (learnable)" if is_parameter else "buffer (fixed)"

    print(f"  [{name}]")
    print(f"    shape         : {shape}  ->  {n_positions} position(s) x {embed_dim} dims")
    print(f"    storage       : {storage}")
    print(f"    layout        : {kind} mask token")
    print(f"    values        : min={t.min():.4f}  max={t.max():.4f}  "
          f"mean={t.mean():.4f}  std={t.std():.4f}")
    print(f"    L2 norm       : {t.norm().item():.4f}")

    if n_positions > 1:
        # How different are the per-position tokens from each other?
        per_pos = t.reshape(n_positions, -1)
        centroid = per_pos.mean(dim=0, keepdim=True)
        spread = (per_pos - centroid).norm(dim=1)
        print(f"    per-pos spread: mean={spread.mean():.4f}  max={spread.max():.4f}  "
              f"(0 => positions identical / not really 'full')")

    print(f"    inferred      : {_infer_strategy_from_values(t, is_parameter)}")


# ------------------------------------------------------------------
# Checkpoint / config loading
# ------------------------------------------------------------------

def _load_config(path: str) -> dict:
    if _yaml is None:
        raise RuntimeError("ruamel.yaml not available to read config")
    with open(path, "r") as f:
        return _yaml.load(f)


def _find_mask_tokens(state_dict: dict) -> dict:
    """Return {full.key: tensor} for every entry whose name contains 'mask_token'."""
    return {k: v for k, v in state_dict.items()
            if "mask_token" in k and isinstance(v, torch.Tensor)}


def inspect_checkpoint(ckpt_path: str) -> None:
    print("=" * 70)
    print(f"CHECKPOINT: {ckpt_path}")
    print("=" * 70)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 1. Configured strategy (if the checkpoint embedded its config)
    cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    if cfg is not None:
        configured = cfg.get("mask_strategy", "<not set> (defaults to 'zero')")
        print(f"\nConfigured mask_strategy : {configured}")
    else:
        print("\nConfigured mask_strategy : <no config embedded in checkpoint>")

    # 2. Mask tokens found in each sub-model's state dict
    sub_dicts = {}
    if isinstance(ckpt, dict):
        for key in ("student_state_dict", "teacher_state_dict",
                    "predictor_state_dict", "model_state_dict", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                sub_dicts[key] = ckpt[key]
        if not sub_dicts:
            # Maybe the checkpoint *is* a raw state dict
            if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
                sub_dicts["<root>"] = ckpt

    if not sub_dicts:
        print("No state dicts found in checkpoint.")
        return

    found_any = False
    for sd_name, sd in sub_dicts.items():
        tokens = _find_mask_tokens(sd)
        if not tokens:
            continue
        found_any = True
        print(f"\n--- {sd_name} ---")
        for k, v in tokens.items():
            # Heuristic: encoder mask tokens are buffers only for zero/negative,
            # but in a saved state_dict everything looks identical, so we infer
            # 'is_parameter' from whether values look trainable (non-constant).
            is_param = not (torch.allclose(v.float(), torch.zeros_like(v.float()))
                            or torch.allclose(v.float(), torch.full_like(v.float(), -1.0)))
            _describe_tensor(k, v, is_parameter=is_param)

    if not found_any:
        print("\nNo 'mask_token' entries found in any state dict.")


def inspect_config(cfg_path: str) -> None:
    print("=" * 70)
    print(f"CONFIG: {cfg_path}")
    print("=" * 70)
    cfg = _load_config(cfg_path)
    strategy = cfg.get("mask_strategy", "<not set> (defaults to 'zero')")
    print(f"\nmask_strategy : {strategy}")

    explain = {
        "zero": "Fixed buffer of 0.0 at every masked position (no learning).",
        "negative": "Fixed buffer of -1.0 at every masked position (no learning).",
        "learnable": "One shared learnable vector broadcast to all masked positions.",
        "learnable_full": "Full per-position learnable bank: a distinct learnable "
                          "vector for EVERY patch slot (what you asked to add).",
    }
    key = strategy if strategy in explain else None
    if key:
        print(f"meaning       : {explain[key]}")
    else:
        print("meaning       : <unknown / custom value>")


def main():
    parser = argparse.ArgumentParser(
        description="Detect mask strategy and report mask-token values.")
    parser.add_argument("checkpoint", nargs="?", default=None,
                        help="Path to a .pth checkpoint to inspect.")
    parser.add_argument("--config", default=None,
                        help="Optional config YAML to inspect for mask_strategy.")
    args = parser.parse_args()

    if not args.checkpoint and not args.config:
        parser.error("Provide a checkpoint path and/or --config.")

    if args.config:
        if not os.path.exists(args.config):
            parser.error(f"Config not found: {args.config}")
        inspect_config(args.config)
        print()

    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            parser.error(f"Checkpoint not found: {args.checkpoint}")
        inspect_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
