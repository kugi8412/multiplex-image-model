#!/usr/bin/env python
# -*- coding: utf-8 -*-
# run_experiments.py
#
# Master runner script that orchestrates the full 10-step experiment pipeline.
# Runs steps sequentially; each step can be skipped via --skip flag.
#
# Usage:
#   python run_experiments.py                    # Run all steps
#   python run_experiments.py --steps 1 2 3      # Only run steps 1, 2, 3
#   python run_experiments.py --skip-steps 0 6   # Run all except steps 0, 6
#   python run_experiments.py --dry-run           # Print commands without executing

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

# Step 0: Dataset preparation (Virtues preprocessing)
STEP_0_PREPARE_DATA = {
    "cmd": ["python", "prepare_virtues_data.py",
            "--panel-config", "configs/all_panels_config.yaml",
            "--file-extension", "npy"],
}

# Step 1: Train original KRONOS (DINOv2 and DINOv3)
STEP_1_TRAIN_KRONOS = [
    {"label": "KRONOS DINOv2",
     "cmd": ["python", "train_kronos_dinov2v3.py", "configs/train_kronos_dinov2_config.yaml"]},
    {"label": "KRONOS DINOv3",
     "cmd": ["python", "train_kronos_dinov2v3.py", "configs/train_kronos_dinov3_config.yaml"]},
]

# Step 2: Train ImmuKRONOS on all 3 backbones × 2 DINO versions = 6 runs
STEP_2_TRAIN_IMMUKRONOS = [
    {"label": "ImmuKRONOS ViT × DINOv2",
     "cmd": ["python", "train_immukronos_unified.py", "configs/immukronos_vit_v2.yaml"]},
    {"label": "ImmuKRONOS ViT × DINOv3",
     "cmd": ["python", "train_immukronos_unified.py", "configs/immukronos_vit_v3.yaml"]},
    {"label": "ImmuKRONOS ConvNeXt × DINOv2",
     "cmd": ["python", "train_immukronos_unified.py", "configs/immukronos_convnext_v2.yaml"]},
    {"label": "ImmuKRONOS ConvNeXt × DINOv3",
     "cmd": ["python", "train_immukronos_unified.py", "configs/immukronos_convnext_v3.yaml"]},
    {"label": "ImmuKRONOS Swin × DINOv2",
     "cmd": ["python", "train_immukronos_unified.py", "configs/immukronos_swin_v2.yaml"]},
    {"label": "ImmuKRONOS Swin × DINOv3",
     "cmd": ["python", "train_immukronos_unified.py", "configs/immukronos_swin_v3.yaml"]},
]

# Step 3: Generate embeddings for all trained models
# NOTE: Checkpoint paths must be updated after training completes.
# The helper function below resolves the latest checkpoint from each checkpoints_dir.
STEP_3_MODELS_FOR_EMBEDDINGS = [
    # (model_type, config_or_kronos_config, checkpoint_dir, kronos_version, extra_args)
    ("kronos_dinov2v3", "configs/train_kronos_dinov2_config.yaml",
     "checkpoints/kronos_dinov2", "v2", []),
    ("kronos_dinov2v3", "configs/train_kronos_dinov3_config.yaml",
     "checkpoints/kronos_dinov3", "v3", []),
    ("kronos_immuvis_v2", "configs/immukronos_vit_v2.yaml",
     "checkpoints/immukronos_vit_v2", "v2", []),
    ("kronos_immuvis_v3", "configs/immukronos_vit_v3.yaml",
     "checkpoints/immukronos_vit_v3", "v3", []),
    ("kronos_immuvis_v2", "configs/immukronos_convnext_v2.yaml",
     "checkpoints/immukronos_convnext_v2", "v2", []),
    ("kronos_immuvis_v3", "configs/immukronos_convnext_v3.yaml",
     "checkpoints/immukronos_convnext_v3", "v3", []),
    ("kronos_immuvis_v2", "configs/immukronos_swin_v2.yaml",
     "checkpoints/immukronos_swin_v2", "v2", []),
    ("kronos_immuvis_v3", "configs/immukronos_swin_v3.yaml",
     "checkpoints/immukronos_swin_v3", "v3", []),
]

# Step 4: Finetune encoder-decoder for each ImmuKRONOS model
# (kronos_config, kronos_checkpoint_dir, version, extra_args)
STEP_4_FINETUNE = [
    ("configs/immukronos_vit_v2.yaml", "checkpoints/immukronos_vit_v2", "v2",
     ["--output-activation", "sigmoid", "--checkpoints-dir", "checkpoints/ft_vit_v2"]),
    ("configs/immukronos_vit_v3.yaml", "checkpoints/immukronos_vit_v3", "v3",
     ["--output-activation", "sigmoid", "--checkpoints-dir", "checkpoints/ft_vit_v3"]),
    ("configs/immukronos_convnext_v2.yaml", "checkpoints/immukronos_convnext_v2", "v2",
     ["--output-activation", "sigmoid", "--checkpoints-dir", "checkpoints/ft_convnext_v2"]),
    ("configs/immukronos_convnext_v3.yaml", "checkpoints/immukronos_convnext_v3", "v3",
     ["--output-activation", "sigmoid", "--checkpoints-dir", "checkpoints/ft_convnext_v3"]),
    ("configs/immukronos_swin_v2.yaml", "checkpoints/immukronos_swin_v2", "v2",
     ["--output-activation", "sigmoid", "--checkpoints-dir", "checkpoints/ft_swin_v2"]),
    ("configs/immukronos_swin_v3.yaml", "checkpoints/immukronos_swin_v3", "v3",
     ["--output-activation", "sigmoid", "--checkpoints-dir", "checkpoints/ft_swin_v3"]),
]

# Step 5: Run validation (leave-one-out) for finetuned models
STEP_5_VALIDATE = [
    ("configs/immukronos_vit_v2.yaml", "checkpoints/ft_vit_v2", "v2",
     ["--results-dir", "results/val_vit_v2"]),
    ("configs/immukronos_vit_v3.yaml", "checkpoints/ft_vit_v3", "v3",
     ["--results-dir", "results/val_vit_v3"]),
    ("configs/immukronos_convnext_v2.yaml", "checkpoints/ft_convnext_v2", "v2",
     ["--results-dir", "results/val_convnext_v2"]),
    ("configs/immukronos_convnext_v3.yaml", "checkpoints/ft_convnext_v3", "v3",
     ["--results-dir", "results/val_convnext_v3"]),
    ("configs/immukronos_swin_v2.yaml", "checkpoints/ft_swin_v2", "v2",
     ["--results-dir", "results/val_swin_v2"]),
    ("configs/immukronos_swin_v3.yaml", "checkpoints/ft_swin_v3", "v3",
     ["--results-dir", "results/val_swin_v3"]),
]

# Step 6: Generate clinics panels
STEP_6_CLINICS = [
    ("immukronos_finetuned", "configs/immukronos_vit_v2.yaml",
     "checkpoints/ft_vit_v2", "ImmuKRONOS_ViT_v2", "results/clinics_vit_v2",
     "configs/all_panels_config.yaml", "v2"),
    ("immukronos_finetuned", "configs/immukronos_convnext_v2.yaml",
     "checkpoints/ft_convnext_v2", "ImmuKRONOS_ConvNeXt_v2", "results/clinics_convnext_v2",
     "configs/all_panels_config.yaml", "v2"),
    ("immukronos_finetuned", "configs/immukronos_swin_v2.yaml",
     "checkpoints/ft_swin_v2", "ImmuKRONOS_Swin_v2", "results/clinics_swin_v2",
     "configs/all_panels_config.yaml", "v2"),
]

# Step 7 is covered by step 4 (finetune encoder-decoder uses the same script)

# Step 8: Downstream evaluation
STEP_8_DOWNSTREAM = [
    # Cell typing on embeddings
    {"label": "Cell typing — ViT v2",
     "cmd": ["python", "downstream_eval.py", "cell_typing",
             "--embeddings-dir", "embeddings/immukronos_vit_v2",
             "--labels-csv", "xai/hn_sc_metadata.csv",
             "--output-dir", "results/downstream/cell_typing_vit_v2"]},
    {"label": "Cell typing — ConvNeXt v2",
     "cmd": ["python", "downstream_eval.py", "cell_typing",
             "--embeddings-dir", "embeddings/immukronos_convnext_v2",
             "--labels-csv", "xai/hn_sc_metadata.csv",
             "--output-dir", "results/downstream/cell_typing_convnext_v2"]},
    {"label": "Cell typing — Swin v2",
     "cmd": ["python", "downstream_eval.py", "cell_typing",
             "--embeddings-dir", "embeddings/immukronos_swin_v2",
             "--labels-csv", "xai/hn_sc_metadata.csv",
             "--output-dir", "results/downstream/cell_typing_swin_v2"]},
    # Virtual staining
    {"label": "VStain — ViT v2",
     "cmd": ["python", "downstream_eval.py", "virtual_stain",
             "--embeddings-dir", "embeddings/immukronos_vit_v2",
             "--output-dir", "results/downstream/vstain_vit_v2"]},
    {"label": "VStain — ConvNeXt v2",
     "cmd": ["python", "downstream_eval.py", "virtual_stain",
             "--embeddings-dir", "embeddings/immukronos_convnext_v2",
             "--output-dir", "results/downstream/vstain_convnext_v2"]},
    {"label": "VStain — Swin v2",
     "cmd": ["python", "downstream_eval.py", "virtual_stain",
             "--embeddings-dir", "embeddings/immukronos_swin_v2",
             "--output-dir", "results/downstream/vstain_swin_v2"]},
]

# Step 9: SAE comparison
STEP_9_SAE = [
    {"label": "SAE: ViT-v2 vs ConvNeXt-v2",
     "cmd": ["python", "cross_sae/train_cross_sae.py",
             "--source-a", "immuvis",
             "--config-a", "configs/immukronos_vit_v2.yaml",
             "--checkpoint-a", "<PLACEHOLDER_VIT_V2_CKPT>",
             "--label-a", "ImmuKRONOS-ViT-v2",
             "--source-b", "immuvis",
             "--config-b", "configs/immukronos_convnext_v2.yaml",
             "--checkpoint-b", "<PLACEHOLDER_CONVNEXT_V2_CKPT>",
             "--label-b", "ImmuKRONOS-ConvNeXt-v2",
             "--output-dir", "results/sae/vit_vs_convnext_v2"]},
    {"label": "SAE: ViT-v2 vs Swin-v2",
     "cmd": ["python", "cross_sae/train_cross_sae.py",
             "--source-a", "immuvis",
             "--config-a", "configs/immukronos_vit_v2.yaml",
             "--checkpoint-a", "<PLACEHOLDER_VIT_V2_CKPT>",
             "--label-a", "ImmuKRONOS-ViT-v2",
             "--source-b", "immuvis",
             "--config-b", "configs/immukronos_swin_v2.yaml",
             "--checkpoint-b", "<PLACEHOLDER_SWIN_V2_CKPT>",
             "--label-b", "ImmuKRONOS-Swin-v2",
             "--output-dir", "results/sae/vit_vs_swin_v2"]},
    {"label": "SAE: ConvNeXt-v2 vs Swin-v2",
     "cmd": ["python", "cross_sae/train_cross_sae.py",
             "--source-a", "immuvis",
             "--config-a", "configs/immukronos_convnext_v2.yaml",
             "--checkpoint-a", "<PLACEHOLDER_CONVNEXT_V2_CKPT>",
             "--label-a", "ImmuKRONOS-ConvNeXt-v2",
             "--source-b", "immuvis",
             "--config-b", "configs/immukronos_swin_v2.yaml",
             "--checkpoint-b", "<PLACEHOLDER_SWIN_V2_CKPT>",
             "--label-b", "ImmuKRONOS-Swin-v2",
             "--output-dir", "results/sae/convnext_vs_swin_v2"]},
]

# Step 10: Final comparison (immukronos vs vit_hn baseline)
STEP_10_COMPARISON = [
    {"label": "Compare ImmuKRONOS-ViT vs ViT-HN baseline",
     "cmd": ["python", "compare_immukronos_vs_vit_hn.py",
             "--immukronos-checkpoint", "<PLACEHOLDER_FT_VIT_V2_CKPT>",
             "--kronos-config", "configs/immukronos_vit_v2.yaml",
             "--kronos-version", "v2",
             "--vit-checkpoint", "<PLACEHOLDER_VIT_BASELINE_CKPT>",
             "--vit-config", "configs/exp7c_vit_baseline.yaml",
             "--output-dir", "results/comparison_vit_v2"]},
]


# ============================================================
# HELPERS
# ============================================================

def find_latest_checkpoint(ckpt_dir: str, pattern: str = "*final*.pth") -> str | None:
    """Find the latest checkpoint matching a pattern in a directory."""
    d = Path(ckpt_dir)
    if not d.exists():
        return None
    # Prefer 'final' checkpoints
    finals = list(d.glob(pattern))
    if finals:
        return str(max(finals, key=lambda p: p.stat().st_mtime))
    # Fallback: any .pth
    all_pth = list(d.glob("*.pth"))
    if all_pth:
        return str(max(all_pth, key=lambda p: p.stat().st_mtime))
    return None


def replace_placeholders(cmd: list[str], placeholder_map: dict[str, str]) -> list[str]:
    """Replace <PLACEHOLDER_...> tokens in a command list."""
    result = []
    for arg in cmd:
        for placeholder, value in placeholder_map.items():
            if placeholder in arg:
                arg = arg.replace(placeholder, value)
        result.append(arg)
    return result


def run_command(cmd: list[str], label: str, dry_run: bool = False, log_dir: str = "logs") -> int:
    """Run a command, logging stdout/stderr to a file."""
    os.makedirs(log_dir, exist_ok=True)
    safe_label = label.replace(" ", "_").replace("/", "_").replace("×", "x")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{safe_label}_{ts}.log")

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"  LOG: {log_file}")
    print(f"{'='*70}")

    if dry_run:
        print("  [DRY RUN — skipped]")
        return 0

    with open(log_file, "w") as lf:
        lf.write(f"# {label}\n# {' '.join(cmd)}\n# {ts}\n\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)

    if proc.returncode != 0:
        print(f"  *** FAILED (exit {proc.returncode}) — see {log_file}")
    else:
        print(f"  SUCCESS")
    return proc.returncode


# ============================================================
# STEP RUNNERS
# ============================================================

def step_0_prepare_data(dry_run):
    print("\n\n" + "="*70)
    print(" STEP 0: Prepare Virtues dataset")
    print("="*70)
    return run_command(STEP_0_PREPARE_DATA["cmd"], "Step0_prepare_data", dry_run=dry_run)


def step_1_train_kronos(dry_run):
    print("\n\n" + "="*70)
    print(" STEP 1: Train original KRONOS (DINOv2 & DINOv3)")
    print("="*70)
    for spec in STEP_1_TRAIN_KRONOS:
        rc = run_command(spec["cmd"], f"Step1_{spec['label']}", dry_run=dry_run)
        if rc != 0:
            return rc
    return 0


def step_2_train_immukronos(dry_run):
    print("\n\n" + "="*70)
    print(" STEP 2: Train ImmuKRONOS (3 backbones × 2 DINO versions)")
    print("="*70)
    for spec in STEP_2_TRAIN_IMMUKRONOS:
        rc = run_command(spec["cmd"], f"Step2_{spec['label']}", dry_run=dry_run)
        if rc != 0:
            return rc
    return 0


def step_3_generate_embeddings(dry_run):
    print("\n\n" + "="*70)
    print(" STEP 3: Generate embeddings for all models")
    print("="*70)
    for model_type, config, ckpt_dir, version, extra in STEP_3_MODELS_FOR_EMBEDDINGS:
        ckpt = find_latest_checkpoint(ckpt_dir)
        if ckpt is None:
            print(f"  WARNING: No checkpoint found in {ckpt_dir}, skipping")
            continue
        out_dir = f"embeddings/{Path(ckpt_dir).name}"
        cmd = ["python", "generate_embeddings.py",
               "--model-type", model_type,
               "--config", config, "--checkpoint", ckpt,
               "--output_dir", out_dir,
               "--kronos-config", config,
               "--kronos-version", version] + extra
        rc = run_command(cmd, f"Step3_embed_{Path(ckpt_dir).name}", dry_run=dry_run)
        if rc != 0:
            return rc
    return 0


def step_4_finetune(dry_run):
    print("\n\n" + "="*70)
    print(" STEP 4: Finetune encoder-decoder for each ImmuKRONOS model")
    print("="*70)
    for kronos_cfg, ckpt_dir, version, extra in STEP_4_FINETUNE:
        ckpt = find_latest_checkpoint(ckpt_dir)
        if ckpt is None:
            print(f"  WARNING: No checkpoint found in {ckpt_dir}, skipping")
            continue
        cmd = ["python", "finetune_immukronos_decoder.py",
               "--kronos-config", kronos_cfg,
               "--kronos-checkpoint", ckpt,
               "--version", version] + extra
        rc = run_command(cmd, f"Step4_finetune_{Path(ckpt_dir).name}", dry_run=dry_run)
        if rc != 0:
            return rc
    return 0


def step_5_validate(dry_run):
    print("\n\n" + "="*70)
    print(" STEP 5: Leave-one-out validation for finetuned models")
    print("="*70)
    for kronos_cfg, ckpt_dir, version, extra in STEP_5_VALIDATE:
        ckpt = find_latest_checkpoint(ckpt_dir)
        if ckpt is None:
            print(f"  WARNING: No checkpoint in {ckpt_dir}, skipping")
            continue
        cmd = ["python", "run_validation_leave_one_out.py",
               "--model-type", "immukronos_finetuned",
               "--kronos-config", kronos_cfg,
               "--kronos-version", version,
               "--checkpoint", ckpt,
               "--config", kronos_cfg] + extra
        rc = run_command(cmd, f"Step5_validate_{Path(ckpt_dir).name}", dry_run=dry_run)
        if rc != 0:
            return rc
    return 0


def step_6_clinics(dry_run):
    print("\n\n" + "="*70)
    print(" STEP 6: Generate clinics panels")
    print("="*70)
    for model_type, config, ckpt_dir, model_name, out_dir, panel_cfg, version in STEP_6_CLINICS:
        ckpt = find_latest_checkpoint(ckpt_dir)
        if ckpt is None:
            print(f"  WARNING: No checkpoint in {ckpt_dir}, skipping")
            continue
        cmd = ["python", "generate_clinics.py",
               "--model-type", model_type,
               "--config", config, "--checkpoint", ckpt,
               "--model_name", model_name, "--output_dir", out_dir,
               "--panel_config", panel_cfg,
               "--kronos-config", config, "--kronos-version", version]
        rc = run_command(cmd, f"Step6_clinics_{model_name}", dry_run=dry_run)
        if rc != 0:
            return rc
    return 0


def step_7_downstream(dry_run):
    """Step 7 is the encoder-decoder training from step 4 — different activations.
    Create separate finetune runs with custom output-activation."""
    print("\n\n" + "="*70)
    print(" STEP 7: Baseline encoder-decoder training (same as step 4 w/ variants)")
    print("="*70)
    # This is already covered by step 4. For swishoid activation,
    # the user would add a custom activation class in finetune_immukronos_decoder.py.
    print("  (Covered by step 4 finetuning runs)")
    return 0


def step_8_downstream_eval(dry_run):
    print("\n\n" + "="*70)
    print(" STEP 8: Downstream evaluation (cell typing + virtual staining)")
    print("="*70)
    for spec in STEP_8_DOWNSTREAM:
        rc = run_command(spec["cmd"], f"Step8_{spec['label']}", dry_run=dry_run)
        if rc != 0:
            return rc
    return 0


def step_9_sae(dry_run):
    print("\n\n" + "="*70)
    print(" STEP 9: SAE comparison between models")
    print("="*70)
    # Build placeholder map from actual checkpoints
    placeholder_map = {}
    ckpt_vit_v2 = find_latest_checkpoint("checkpoints/immukronos_vit_v2")
    ckpt_conv_v2 = find_latest_checkpoint("checkpoints/immukronos_convnext_v2")
    ckpt_swin_v2 = find_latest_checkpoint("checkpoints/immukronos_swin_v2")
    if ckpt_vit_v2:
        placeholder_map["<PLACEHOLDER_VIT_V2_CKPT>"] = ckpt_vit_v2
    if ckpt_conv_v2:
        placeholder_map["<PLACEHOLDER_CONVNEXT_V2_CKPT>"] = ckpt_conv_v2
    if ckpt_swin_v2:
        placeholder_map["<PLACEHOLDER_SWIN_V2_CKPT>"] = ckpt_swin_v2

    for spec in STEP_9_SAE:
        cmd = replace_placeholders(spec["cmd"], placeholder_map)
        if any("<PLACEHOLDER" in arg for arg in cmd):
            print(f"  WARNING: Unresolved placeholders in {spec['label']}, skipping")
            continue
        rc = run_command(cmd, f"Step9_{spec['label']}", dry_run=dry_run)
        if rc != 0:
            return rc
    return 0


def step_10_comparison(dry_run):
    print("\n\n" + "="*70)
    print(" STEP 10: Final comparison (ImmuKRONOS vs ViT-HN baseline)")
    print("="*70)
    placeholder_map = {}
    ckpt_ft_vit = find_latest_checkpoint("checkpoints/ft_vit_v2")
    ckpt_vit_bl = find_latest_checkpoint("checkpoints/vit_baseline")
    if ckpt_ft_vit:
        placeholder_map["<PLACEHOLDER_FT_VIT_V2_CKPT>"] = ckpt_ft_vit
    if ckpt_vit_bl:
        placeholder_map["<PLACEHOLDER_VIT_BASELINE_CKPT>"] = ckpt_vit_bl

    for spec in STEP_10_COMPARISON:
        cmd = replace_placeholders(spec["cmd"], placeholder_map)
        if any("<PLACEHOLDER" in arg for arg in cmd):
            print(f"  WARNING: Unresolved placeholders in {spec['label']}, skipping")
            continue
        rc = run_command(cmd, f"Step10_{spec['label']}", dry_run=dry_run)
        if rc != 0:
            return rc
    return 0


# ============================================================
# MAIN
# ============================================================

STEP_FUNCTIONS = {
    0: ("Prepare Virtues dataset", step_0_prepare_data),
    1: ("Train original KRONOS", step_1_train_kronos),
    2: ("Train ImmuKRONOS (multi-backbone)", step_2_train_immukronos),
    3: ("Generate embeddings", step_3_generate_embeddings),
    4: ("Finetune encoder-decoder", step_4_finetune),
    5: ("Validation leave-one-out", step_5_validate),
    6: ("Generate clinics", step_6_clinics),
    7: ("Baseline encoder-decoder variants", step_7_downstream),
    8: ("Downstream eval", step_8_downstream_eval),
    9: ("SAE comparison", step_9_sae),
    10: ("Final comparison", step_10_comparison),
}


def main():
    parser = argparse.ArgumentParser(description="Run the full experiment pipeline")
    parser.add_argument("--steps", nargs="+", type=int, default=None,
                        help="Only run these step numbers (0-10)")
    parser.add_argument("--skip-steps", nargs="+", type=int, default=[],
                        help="Skip these step numbers")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    parser.add_argument("--log-dir", default="logs",
                        help="Directory for log files (default: logs)")
    args = parser.parse_args()

    all_steps = sorted(STEP_FUNCTIONS.keys())
    steps_to_run = args.steps if args.steps is not None else all_steps
    steps_to_run = [s for s in steps_to_run if s not in args.skip_steps]

    print(f"{'='*70}")
    print(f"  EXPERIMENT PIPELINE — {datetime.datetime.now().isoformat()}")
    print(f"  Steps: {steps_to_run}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*70}")

    for step_num in steps_to_run:
        if step_num not in STEP_FUNCTIONS:
            print(f"Unknown step {step_num}, skipping")
            continue
        label, func = STEP_FUNCTIONS[step_num]
        rc = func(args.dry_run)
        if rc != 0:
            print(f"\n*** Step {step_num} ({label}) FAILED with exit code {rc}. Stopping.")
            sys.exit(rc)

    print(f"\n\n{'='*70}")
    print(f"  ALL STEPS COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
