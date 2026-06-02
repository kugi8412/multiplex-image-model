# -*- coding: utf-8 -*-
"""Generate hyperparameter ablation configs for ImmuKRONOS and KRONOS."""

import os
from collections import OrderedDict

BASE_DIR = "configs/ablations"


def yaml_val(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        # scientific notation for small values
        if abs(v) < 0.01 and v != 0:
            return "%.1e" % v
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, list):
        return "[%s]" % ", ".join(yaml_val(x) for x in v)
    if isinstance(v, str):
        return '"%s"' % v
    return str(v)


def write_yaml(path, header, cfg):
    parent = os.path.dirname(path)
    if not os.path.exists(parent):
        os.makedirs(parent)
    with open(path, "w") as f:
        f.write(header)
        for k, v in cfg:
            f.write("%s: %s\n" % (k, yaml_val(v)))
    print("  Created: %s" % path)

# ============================================================
# ImmuKRONOS HP Search - 4 backbones x 3 lr x 2 preproc = 24
# ============================================================

IMMUKRONOS_BASE = OrderedDict([
    ("vit", {
        "backbone": "kronos_vit",
        "embed_dim": 768, "depth": 12, "num_heads": 12,
        "drop_path_rate": 0.3, "num_register_tokens": 0,
    }),
    ("convnext", {
        "backbone": "convnext",
        "embed_dim": 768, "depth": 12, "num_heads": 12,
        "drop_path_rate": 0.1, "num_register_tokens": 0,
    }),
    ("swin", {
        "backbone": "swin",
        "embed_dim": 768, "depth": 12, "num_heads": 12,
        "drop_path_rate": 0.1, "num_register_tokens": 0,
    }),
    ("vim", {
        "backbone": "vim",
        "embed_dim": 384, "depth": 12, "num_heads": 6,
        "drop_path_rate": 0.1, "num_register_tokens": 0,
    }),
])

LR_SWEEP_IMMUKRONOS = OrderedDict([
    ("lr2e3", 2.0e-3),
    ("lr4e3", 4.0e-3),
    ("lr8e3", 8.0e-3),
])

def make_immukronos_config(bb_name, bb_cfg, lr_name, lr_val, virtues=False):
    preproc = "virtues" if virtues else "immuvis"
    suffix = "_virtues" if virtues else ""
    cfg = [
        ("backbone", bb_cfg["backbone"]),
        ("dino_version", "v2"),
        ("preprocessing", preproc),
        ("panel_config_path", "configs/all_panels_config.yaml"),
        ("tokenizer_config_path", "configs/all_markers_tokenizer.yaml"),
        ("global_crops_number", 2),
        ("global_crops_size", 128),
        ("global_crops_scale", [0.48, 1.0]),
        ("local_crops_number", 6),
        ("local_crops_size", 64),
        ("local_crops_scale", [0.16, 0.48]),
        ("num_workers", 8),
        ("patch_size", 8),
        ("out_dim", 65536),
        ("embed_dim", bb_cfg["embed_dim"]),
        ("depth", bb_cfg["depth"]),
        ("num_heads", bb_cfg["num_heads"]),
        ("drop_path_rate", bb_cfg["drop_path_rate"]),
        ("num_register_tokens", bb_cfg["num_register_tokens"]),
        ("device", "cuda"),
        ("epochs", 10),
        ("warmup_epochs", 2),
        ("batch_size", 8),
        ("gradient_accumulation_steps", 128),
        ("lr", lr_val),
        ("final_lr", 1.0e-6),
        ("weight_decay", 0.04),
        ("weight_decay_final", 0.4),
        ("clip_grad", 3.0),
        ("warmup_teacher_temp_epochs", 5),
        ("warmup_teacher_temp", 0.04),
        ("teacher_temp", 0.07),
        ("teacher_momentum", 0.992),
        ("channel_fraction", [0.75, 1.0]),
        ("mask_strategy", "zero"),
        ("checkpoints_dir", "checkpoints/ablations/immukronos_%s_%s%s" % (bb_name, lr_name, suffix)),
        ("save_checkpoint_freq", 5),
        ("tags", ["immukronos", bb_name, "ablation", "hpsearch", lr_name, preproc]),
        ("comet_project", "immu-vis"),
        ("comet_workspace", "kugi8412"),
    ]
    return cfg


def make_immukronos_header(bb_name, lr_name, virtues):
    suffix = "_virtues" if virtues else ""
    preproc_label = "VirTues" if virtues else "ImmuVis"
    return (
        "# HP search: %s | ImmuKRONOS DINOv2\n"
        "# Backbone: %s | Preprocessing: %s | 10 epochs\n"
        "# Train with: python train_immukronos_unified.py "
        "configs/ablations/immukronos_hpsearch/immukronos_%s_%s%s.yaml\n"
        % (lr_name, bb_name, preproc_label, bb_name, lr_name, suffix)
    )


# ============================================================
# KRONOS HP Search - 2 dino x 3 lr x 2 preproc = 12
# ============================================================

KRONOS_DINO_BASES = OrderedDict([
    ("v2", OrderedDict([
        ("training_mode", "dinov2"),
        ("model_name", "vitb16"),
        ("out_dim", 65536),
        ("drop_path_rate", 0.3),
        ("num_register_tokens", 0),
        ("ffn_layer", "mlp"),
        ("batch_size", 8),
        ("gradient_accumulation_steps", 64),
        ("warmup_epochs", 20),
        ("clip_grad", 1.0),
        ("warmup_teacher_temp_epochs", 40),
        ("teacher_momentum", 0.996),
    ])),
    ("v3", OrderedDict([
        ("training_mode", "dinov3"),
        ("model_name", "vitb16"),
        ("out_dim", 65536),
        ("ibot_out_dim", 8192),
        ("drop_path_rate", 0.1),
        ("num_register_tokens", 4),
        ("ffn_layer", "swiglu"),
        ("batch_size", 4),
        ("gradient_accumulation_steps", 128),
        ("warmup_epochs", 15),
        ("clip_grad", 1.0),
        ("warmup_teacher_temp_epochs", 30),
        ("teacher_momentum", 0.996),
        ("ibot_mask_ratio", 0.3),
        ("ibot_loss_weight", 1.0),
        ("koleo_loss_weight", 0.1),
    ])),
])

LR_SWEEP_KRONOS = OrderedDict([
    ("lr1e4", 1.0e-4),
    ("lr3e4", 3.0e-4),
    ("lr1e3", 1.0e-3),
])


def make_kronos_config(dino_ver, dino_cfg, lr_name, lr_val, virtues=False):
    suffix = "_virtues" if virtues else ""

    cfg = [
        ("training_mode", dino_cfg["training_mode"]),
        ("model_name", dino_cfg["model_name"]),
        ("out_dim", dino_cfg["out_dim"]),
        ("drop_path_rate", dino_cfg["drop_path_rate"]),
        ("patch_size", 8),
        ("num_register_tokens", dino_cfg["num_register_tokens"]),
        ("ffn_layer", dino_cfg["ffn_layer"]),
    ]

    if "ibot_out_dim" in dino_cfg:
        cfg.append(("ibot_out_dim", dino_cfg["ibot_out_dim"]))

    cfg.extend([
        ("tokenizer_config", "configs/all_markers_tokenizer.yaml"),
        ("marker_metadata_csv", "configs/marker_metadata.csv"),
        ("panel_config", "configs/all_panels_config.yaml"),
        ("file_extension", "npy"),
        ("num_workers", 8),
        ("channel_fraction", [0.75, 1.0]),
        ("global_crops_number", 2),
        ("global_crops_scale", [0.48, 1.0] if dino_ver == "v2" else [0.4, 1.0]),
        ("global_crops_size", [128, 128]),
        ("local_crops_number", 8),
        ("local_crops_scale", [0.16, 0.48] if dino_ver == "v2" else [0.1, 0.4]),
        ("local_crops_size", [64, 64]),
        ("device", "cuda"),
        ("epochs", 10),
        ("batch_size", dino_cfg["batch_size"]),
        ("gradient_accumulation_steps", dino_cfg["gradient_accumulation_steps"]),
        ("lr", lr_val),
        ("final_lr", 1.0e-6),
        ("warmup_epochs", 2),
        ("weight_decay", 0.04),
        ("weight_decay_final", 0.4),
        ("clip_grad", dino_cfg["clip_grad"]),
        ("warmup_teacher_temp_epochs", 5),
        ("warmup_teacher_temp", 0.04),
        ("teacher_temp", 0.07),
        ("teacher_momentum", dino_cfg["teacher_momentum"]),
    ])

    if "ibot_mask_ratio" in dino_cfg:
        cfg.extend([
            ("ibot_mask_ratio", dino_cfg["ibot_mask_ratio"]),
            ("ibot_loss_weight", dino_cfg["ibot_loss_weight"]),
            ("koleo_loss_weight", dino_cfg["koleo_loss_weight"]),
        ])

    if virtues:
        cfg.append(("preprocessing_mode", "virtues_arcsinh"))

    cfg.extend([
        ("from_checkpoint", None),
        ("checkpoints_dir", "checkpoints/ablations/kronos_dino%s_%s%s" % (dino_ver, lr_name, suffix)),
        ("save_checkpoint_freq", 5),
        ("tags", ["kronos", "dino%s" % dino_ver, "ablation", "hpsearch", lr_name,
                  "virtues" if virtues else "immuvis"]),
        ("comet_project", "immu-vis"),
        ("comet_workspace", "kugi8412"),
    ])

    return cfg


def make_kronos_header(dino_ver, lr_name, virtues):
    suffix = "_virtues" if virtues else ""
    preproc_label = "VirTues" if virtues else "ImmuVis"
    script = "train_kronos_dinov2v3_virtues.py" if virtues else "train_kronos_dinov2v3.py"
    return (
        "# HP search: %s | KRONOS DINO%s\n"
        "# Backbone: ViT-B | Preprocessing: %s | 10 epochs\n"
        "# Train with: python %s "
        "configs/ablations/kronos_hpsearch/kronos_dino%s_%s%s.yaml\n"
        % (lr_name, dino_ver, preproc_label, script, dino_ver, lr_name, suffix)
    )


def main():
    # --- ImmuKRONOS ablation configs ---
    print("=== ImmuKRONOS HP Search ===")
    out_dir = os.path.join(BASE_DIR, "immukronos_hpsearch")
    count = 0
    for bb_name, bb_cfg in IMMUKRONOS_BASE.items():
        for lr_name, lr_val in LR_SWEEP_IMMUKRONOS.items():
            for virtues in [False, True]:
                suffix = "_virtues" if virtues else ""
                fname = "immukronos_%s_%s%s.yaml" % (bb_name, lr_name, suffix)
                path = os.path.join(out_dir, fname)
                header = make_immukronos_header(bb_name, lr_name, virtues)
                cfg = make_immukronos_config(bb_name, bb_cfg, lr_name, lr_val, virtues)
                write_yaml(path, header, cfg)
                count += 1
    print("  Total ImmuKRONOS configs: %d" % count)

    # --- KRONOS ablation configs ---
    print("\n=== KRONOS HP Search ===")
    out_dir = os.path.join(BASE_DIR, "kronos_hpsearch")
    count = 0
    for dino_ver, dino_cfg in KRONOS_DINO_BASES.items():
        for lr_name, lr_val in LR_SWEEP_KRONOS.items():
            for virtues in [False, True]:
                suffix = "_virtues" if virtues else ""
                fname = "kronos_dino%s_%s%s.yaml" % (dino_ver, lr_name, suffix)
                path = os.path.join(out_dir, fname)
                header = make_kronos_header(dino_ver, lr_name, virtues)
                cfg = make_kronos_config(dino_ver, dino_cfg, lr_name, lr_val, virtues)
                write_yaml(path, header, cfg)
                count += 1
    print("  Total KRONOS configs: %d" % count)

    print("\nDone. All configs in %s/" % BASE_DIR)


if __name__ == "__main__":
    main()
