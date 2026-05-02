#!/usr/bin/env python3
# downstream_eval.py
# -*- coding: utf-8 -*-

"""
Downstream evaluation on frozen embeddings produced by generate_embeddings.py.

Supports two tasks:
  1) cell_typing   – classify cell phenotype from CLS / pooled embeddings
  2) virtual_stain – predict held-out marker intensity from embeddings (MLP decoder)

Supports multiple classifiers / heads:
  --method logistic      Logistic Regression (KRONOS-paper default)
  --method mlp           2-layer MLP with ReLU (fine-tuned on frozen embeddings)
  --method knn           k-Nearest Neighbors
  --method linear_probe  Single linear layer (SGD, for large-scale)
  --method random_forest Random Forest (sklearn)

Usage examples:
  # Cell typing with logistic regression (KRONOS default)
  python downstream_eval.py cell_typing \\
      --embeddings-dir embeddings/exp3/ \\
      --labels-csv data/cell_annotations.csv \\
      --output-dir results/exp3_cell_typing/ \\
      --method logistic

  # Cell typing with MLP
  python downstream_eval.py cell_typing \\
      --embeddings-dir embeddings/exp6b/ \\
      --labels-csv data/cell_annotations.csv \\
      --output-dir results/exp6b_cell_typing/ \\
      --method mlp --epochs 50 --lr 1e-3

  # Virtual staining (MLP decoder on frozen DINO embeddings)
  python downstream_eval.py virtual_stain \\
      --embeddings-dir embeddings/exp3/ \\
      --patches-dir /raid_encrypted/immucan/immuvis_split_patches_onlyarcsinh/test/ \\
      --output-dir results/exp3_virtual_stain/ \\
      --method mlp --epochs 100 --lr 1e-3 \\
      --skip-markers DNA1 DNA2

  # Quick k-NN baseline for cell typing
  python downstream_eval.py cell_typing \\
      --embeddings-dir embeddings/exp1/ \\
      --labels-csv data/cell_annotations.csv \\
      --output-dir results/exp1_knn/ \\
      --method knn --knn-k 20
"""

import argparse
import gc
import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")


# ============================================================================
# Data Loading
# ============================================================================

def load_embeddings(embeddings_dir, split="all"):
    """Load embeddings from a directory.

    Supports two formats:
      1) Panel-config mode: embeddings_{split}.npz with 'embeddings', 'paths', 'datasets' arrays
      2) Legacy mode: emb_*.npy files, each with {"embeddings": ..., "labels": ...}

    Returns:
        embeddings: (N, D) array
        labels: (N,) array or None
        file_names: list of source file basenames
    """
    # Try panel-config format first (embeddings_*.npz)
    npz_files = sorted(f for f in os.listdir(embeddings_dir)
                       if f.startswith("embeddings_") and f.endswith(".npz"))
    if npz_files:
        emb_list, name_list, dataset_list = [], [], []
        for fname in npz_files:
            path = os.path.join(embeddings_dir, fname)
            data = np.load(path, allow_pickle=True)
            emb_list.append(data["embeddings"])
            if "paths" in data:
                name_list.extend(data["paths"].tolist())
            else:
                name_list.extend([fname] * len(data["embeddings"]))
            if "datasets" in data:
                dataset_list.extend(data["datasets"].tolist())
        embeddings = np.concatenate(emb_list, axis=0)
        return embeddings, None, name_list

    # Fall back to legacy emb_*.npy format
    emb_list, label_list, name_list = [], [], []

    files = sorted(f for f in os.listdir(embeddings_dir) if f.startswith("emb_") and f.endswith(".npy"))
    if not files:
        raise FileNotFoundError(
            f"No embeddings found in {embeddings_dir}. "
            f"Expected embeddings_*.npz (panel-config mode) or emb_*.npy (legacy mode)."
        )

    for fname in files:
        path = os.path.join(embeddings_dir, fname)
        raw = np.load(path, allow_pickle=True)

        if isinstance(raw, np.ndarray) and raw.ndim == 0:
            data = raw.item()
        else:
            data = raw

        if isinstance(data, dict):
            emb = data["embeddings"]
            lbl = data.get("labels", None)
        else:
            emb = data
            lbl = None

        if emb.ndim == 1:
            emb = emb[np.newaxis]
        emb_list.append(emb)
        if lbl is not None:
            label_list.append(lbl)
        name_list.extend([fname] * len(emb))

    embeddings = np.concatenate(emb_list, axis=0)
    labels = np.concatenate(label_list, axis=0) if label_list else None
    return embeddings, labels, name_list


def load_labels_csv(csv_path, n_embeddings=None):
    """Load a labels CSV with columns: [file_name, sample_idx, label].

    Falls back to just a 'label' column if file info is missing.
    """
    df = pd.read_csv(csv_path)
    if "label" in df.columns:
        return df["label"].values
    elif "cell_type" in df.columns:
        return df["cell_type"].values
    elif "phenotype" in df.columns:
        return df["phenotype"].values
    else:
        raise ValueError(f"Labels CSV must have a 'label', 'cell_type', or 'phenotype' column. "
                         f"Found: {list(df.columns)}")


# ============================================================================
# Classifiers for Cell Typing
# ============================================================================

def train_logistic(X_train, y_train, X_val, y_val, args):
    """Logistic regression with Optuna hyperparameter search (KRONOS default)."""
    import optuna
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    def objective(trial):
        C = trial.suggest_float("C", args.c_low, args.c_high, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        solver = "saga" if penalty == "l1" else "lbfgs"
        clf = LogisticRegression(
            C=C, penalty=penalty, solver=solver,
            max_iter=args.max_iter, multi_class="multinomial", n_jobs=-1,
        )
        clf.fit(X_train_s, y_train)
        return f1_score(y_val, clf.predict(X_val_s), average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    best = study.best_params
    print(f"Best params: {best} (F1={study.best_value:.4f})")

    solver = "saga" if best["penalty"] == "l1" else "lbfgs"
    clf = LogisticRegression(
        C=best["C"], penalty=best["penalty"], solver=solver,
        max_iter=args.max_iter, multi_class="multinomial", n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)
    return clf, scaler


def train_knn(X_train, y_train, X_val, y_val, args):
    """k-NN classifier."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    clf = KNeighborsClassifier(n_neighbors=args.knn_k, metric="cosine", n_jobs=-1)
    clf.fit(X_train_s, y_train)
    return clf, scaler


def train_random_forest(X_train, y_train, X_val, y_val, args):
    """Random Forest classifier."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(
        n_estimators=args.rf_trees, max_depth=args.rf_depth,
        n_jobs=-1, random_state=42,
    )
    clf.fit(X_train_s, y_train)
    return clf, scaler


def train_mlp_classifier(X_train, y_train, X_val, y_val, args):
    """2-layer MLP classifier (PyTorch) trained on frozen embeddings."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    n_classes = len(le.classes_)

    device = torch.device(args.device)
    dim = X_train_s.shape[1]
    hidden = args.mlp_hidden or dim

    model = nn.Sequential(
        nn.Linear(dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(args.mlp_dropout),
        nn.Linear(hidden, hidden // 2),
        nn.BatchNorm1d(hidden // 2),
        nn.ReLU(),
        nn.Dropout(args.mlp_dropout),
        nn.Linear(hidden // 2, n_classes),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(X_train_s).float(),
        torch.from_numpy(y_train_enc).long(),
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    best_f1, best_state = 0.0, None
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_val_s).float().to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
        from sklearn.metrics import f1_score
        f1 = f1_score(y_val_enc, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}  val F1={f1:.4f}  best={best_f1:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    # Wrap for sklearn-like interface
    class MLPWrapper:
        def __init__(self, model, scaler, le, device):
            self.model = model
            self.scaler_ = scaler
            self.le = le
            self.device = device

        def predict(self, X):
            X_s = self.scaler_.transform(X)
            with torch.no_grad():
                logits = self.model(torch.from_numpy(X_s).float().to(self.device))
            return self.le.inverse_transform(logits.argmax(dim=1).cpu().numpy())

        def predict_proba(self, X):
            X_s = self.scaler_.transform(X)
            with torch.no_grad():
                logits = self.model(torch.from_numpy(X_s).float().to(self.device))
            return torch.softmax(logits, dim=1).cpu().numpy()

    wrapper = MLPWrapper(model, scaler, le, device)
    return wrapper, None  # scaler is inside wrapper


def train_linear_probe(X_train, y_train, X_val, y_val, args):
    """Single linear layer trained with SGD (for very large datasets)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    n_classes = len(le.classes_)

    device = torch.device(args.device)
    dim = X_train_s.shape[1]

    model = nn.Linear(dim, n_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(X_train_s).float(),
        torch.from_numpy(y_train_enc).long(),
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    best_f1, best_state = 0.0, None
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_val_s).float().to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
        from sklearn.metrics import f1_score
        f1 = f1_score(y_val_enc, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    class LinearWrapper:
        def __init__(self, model, scaler, le, device):
            self.model = model
            self.scaler_ = scaler
            self.le = le
            self.device = device

        def predict(self, X):
            X_s = self.scaler_.transform(X)
            with torch.no_grad():
                logits = self.model(torch.from_numpy(X_s).float().to(self.device))
            return self.le.inverse_transform(logits.argmax(dim=1).cpu().numpy())

        def predict_proba(self, X):
            X_s = self.scaler_.transform(X)
            with torch.no_grad():
                logits = self.model(torch.from_numpy(X_s).float().to(self.device))
            return torch.softmax(logits, dim=1).cpu().numpy()

    wrapper = LinearWrapper(model, scaler, le, device)
    return wrapper, None


CLASSIFIERS = {
    "logistic": train_logistic,
    "mlp": train_mlp_classifier,
    "knn": train_knn,
    "linear_probe": train_linear_probe,
    "random_forest": train_random_forest,
}


# ============================================================================
# Evaluation Metrics
# ============================================================================

def evaluate_classifier(clf, scaler, X_test, y_test):
    """Compute classification metrics."""
    from sklearn.metrics import (
        f1_score, balanced_accuracy_score, roc_auc_score,
        average_precision_score, classification_report,
    )

    X_s = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = clf.predict(X_s) if scaler is not None else clf.predict(X_test)

    metrics = {
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
    }

    try:
        y_proba = clf.predict_proba(X_s) if scaler is not None else clf.predict_proba(X_test)
        n_classes = y_proba.shape[1]
        if n_classes == 2:
            metrics["auroc"] = roc_auc_score(y_test, y_proba[:, 1])
            metrics["auprc"] = average_precision_score(y_test, y_proba[:, 1])
        else:
            try:
                metrics["auroc"] = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
            except ValueError:
                metrics["auroc"] = float("nan")
    except AttributeError:
        metrics["auroc"] = float("nan")

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return metrics, report


# ============================================================================
# Virtual Staining (MLP Decoder on Frozen Embeddings)
# ============================================================================

def _load_npz_virtual_stain_data(embeddings_dir, panel_config_path, tokenizer_path, skip_set):
    """Load embeddings from npz and compute per-marker targets from stored patch paths.

    Each image may come from a different panel (different channels).  We map all
    channels to a common tokenizer vocabulary and produce a binary mask indicating
    which markers are present for each sample.

    Returns (embeddings, targets, masks, inv_tokenizer) or (None,)*4 on failure.
    """
    from ruamel.yaml import YAML

    npz_files = sorted(
        f for f in os.listdir(embeddings_dir)
        if f.startswith("embeddings_") and f.endswith(".npz")
    )
    if not npz_files:
        return None, None, None, None

    # Gather all npz data
    emb_parts, path_parts, ds_parts = [], [], []
    for fname in npz_files:
        data = np.load(os.path.join(embeddings_dir, fname), allow_pickle=True)
        emb_parts.append(data["embeddings"])
        path_parts.extend(data["paths"].tolist())
        ds_parts.extend(data["datasets"].tolist())

    embeddings = np.concatenate(emb_parts, axis=0)

    yaml = YAML()
    with open(panel_config_path) as fh:
        panel_config = yaml.load(fh)
    with open(tokenizer_path) as fh:
        tokenizer = yaml.load(fh)
    inv_tokenizer = {v: k for k, v in tokenizer.items()}

    # channel_ids per dataset from panel config
    ch_ids_map = {}
    for ds_name in panel_config.get("datasets", []):
        markers = panel_config.get("markers", {}).get(ds_name, [])
        ch_ids_map[ds_name] = [tokenizer[m] for m in markers if m in tokenizer]

    M = len(tokenizer)
    N = len(embeddings)
    targets = np.zeros((N, M), dtype=np.float32)
    masks = np.zeros((N, M), dtype=bool)
    valid = np.ones(N, dtype=bool)

    for i, (path, ds_name) in enumerate(zip(path_parts, ds_parts)):
        path, ds_name = str(path), str(ds_name)
        if not os.path.exists(path):
            valid[i] = False
            continue
        try:
            patch = np.load(path)
        except Exception:
            valid[i] = False
            continue
        if patch.ndim < 3:
            valid[i] = False
            continue

        means = patch.mean(axis=(1, 2))
        ch_ids = ch_ids_map.get(ds_name, [])
        for c_idx, ch_id in enumerate(ch_ids):
            if c_idx < len(means):
                marker_name = inv_tokenizer.get(ch_id, "")
                if marker_name not in skip_set:
                    targets[i, ch_id] = means[c_idx]
                    masks[i, ch_id] = True

    embeddings = embeddings[valid]
    targets = targets[valid]
    masks = masks[valid]
    return embeddings, targets, masks, inv_tokenizer


def train_virtual_stain_mlp(embeddings_dir, patches_dir, args):
    """Train an MLP decoder head to predict marker intensities from frozen embeddings.

    Supports two embedding formats:
      1) Panel-config NPZ: ``embeddings_*.npz`` with ``paths``/``datasets`` arrays.
         Requires ``--panel-config`` so channel→marker mapping is available.
         Optionally uses ``--train-embeddings-dir`` for a separate train split.
      2) Legacy: ``emb_*.npy`` matched 1-to-1 with patch files in ``--patches-dir``.

    The MLP predicts the spatial-mean intensity of each marker from the embedding.
    Multi-panel data is handled with a masked MSE loss (only present markers).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    skip_set = set(args.skip_markers or [])

    # Load tokenizer for marker names
    if args.tokenizer_config:
        from ruamel.yaml import YAML
        tokenizer = YAML().load(open(args.tokenizer_config))
        inv_tokenizer = {v: k for k, v in tokenizer.items()}
    else:
        tokenizer, inv_tokenizer = None, None

    # ------------------------------------------------------------------
    # Detect format and load data
    # ------------------------------------------------------------------
    npz_files = sorted(
        f for f in os.listdir(embeddings_dir)
        if f.startswith("embeddings_") and f.endswith(".npz")
    )
    use_masked = False

    if npz_files and args.panel_config:
        # ---- NPZ format (panel-config mode) --------------------------
        use_masked = True
        test_emb, test_tgt, test_mask, inv_tokenizer = _load_npz_virtual_stain_data(
            embeddings_dir, args.panel_config, args.tokenizer_config, skip_set,
        )
        if test_emb is None or len(test_emb) == 0:
            print("[ERROR] No valid data loaded from npz format.")
            return {}

        train_embeddings_dir = getattr(args, "train_embeddings_dir", None)
        if train_embeddings_dir:
            train_emb, train_tgt, train_mask, _ = _load_npz_virtual_stain_data(
                train_embeddings_dir, args.panel_config, args.tokenizer_config, skip_set,
            )
            if train_emb is None or len(train_emb) == 0:
                print("[ERROR] No valid training data from --train-embeddings-dir.")
                return {}
            has_separate_test = True
        else:
            has_separate_test = False

        if has_separate_test:
            all_emb = train_emb
            all_targets = train_tgt
            all_masks = train_mask
        else:
            all_emb = test_emb
            all_targets = test_tgt
            all_masks = test_mask

        N, D = all_emb.shape
        C_out = all_targets.shape[1]
        n_present = int(all_masks.any(axis=0).sum())
        print(f"Loaded {N} training samples (npz), embed_dim={D}, "
              f"marker vocab={C_out}, active markers={n_present}")
        if has_separate_test:
            print(f"Loaded {len(test_emb)} test samples from {embeddings_dir}")

    else:
        # ---- Legacy emb_*.npy format ---------------------------------
        emb_files = sorted(
            f for f in os.listdir(embeddings_dir)
            if f.startswith("emb_") and f.endswith(".npy")
        )
        all_emb_l, all_targets_l, all_channel_ids_list, all_names = [], [], [], []
        for ef in emb_files:
            base = ef.replace("emb_", "")
            patch_path = os.path.join(patches_dir, base)
            if not os.path.exists(patch_path):
                npz_path = patch_path.replace(".npy", ".npz")
                if os.path.exists(npz_path):
                    patch_path = npz_path
                else:
                    continue

            emb_raw = np.load(os.path.join(embeddings_dir, ef), allow_pickle=True)
            if isinstance(emb_raw, np.ndarray) and emb_raw.ndim == 0:
                emb_raw = emb_raw.item()
            emb = emb_raw["embeddings"] if isinstance(emb_raw, dict) else emb_raw
            if emb.ndim == 2:
                emb = emb[0]

            patch_raw = np.load(patch_path, allow_pickle=True)
            if isinstance(patch_raw, np.ndarray) and patch_raw.ndim == 0:
                patch_raw = patch_raw.item()
            if isinstance(patch_raw, dict):
                patch = patch_raw.get("patches", patch_raw.get("data"))
                ch_ids = patch_raw.get("channel_ids", None)
            elif isinstance(patch_raw, np.lib.npyio.NpzFile):
                patch = patch_raw["patches"] if "patches" in patch_raw else patch_raw["arr_0"]
                ch_ids = patch_raw.get("channel_ids", None)
            else:
                patch = patch_raw
                ch_ids = None
            if patch.ndim == 4:
                patch = patch[0]
            if patch.ndim == 2:
                continue

            target = patch.mean(axis=(1, 2)) if patch.ndim == 3 else patch.mean(axis=-1)
            all_emb_l.append(emb)
            all_targets_l.append(target)
            all_channel_ids_list.append(ch_ids)
            all_names.append(base)

        if not all_emb_l:
            print("[ERROR] No matched embedding-patch pairs found.")
            return {}

        all_emb = np.stack(all_emb_l, axis=0)
        all_targets = np.stack(all_targets_l, axis=0)
        all_masks = None
        has_separate_test = False
        N, D = all_emb.shape
        C_out = all_targets.shape[1]
        print(f"Loaded {N} samples (legacy), embedding dim={D}, output channels={C_out}")

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------
    if has_separate_test:
        train_idx = np.arange(len(all_emb))
        val_emb_raw = test_emb
        val_tgt_raw = test_tgt
        val_mask_raw = test_mask if use_masked else None
    else:
        np.random.seed(42)
        perm = np.random.permutation(N)
        split_pt = int(0.8 * N)
        train_idx = perm[:split_pt]
        val_idx = perm[split_pt:]
        val_emb_raw = all_emb[val_idx]
        val_tgt_raw = all_targets[val_idx]
        val_mask_raw = all_masks[val_idx] if use_masked else None

    device = torch.device(args.device)

    # Normalize embeddings
    emb_mean = all_emb[train_idx].mean(axis=0)
    emb_std = all_emb[train_idx].std(axis=0) + 1e-8

    train_emb_norm = (all_emb[train_idx] - emb_mean) / emb_std
    val_emb_norm = (val_emb_raw - emb_mean) / emb_std

    # MLP decoder
    hidden = args.mlp_hidden or D
    model = nn.Sequential(
        nn.Linear(D, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(args.mlp_dropout),
        nn.Linear(hidden, hidden // 2),
        nn.BatchNorm1d(hidden // 2),
        nn.ReLU(),
        nn.Dropout(args.mlp_dropout),
        nn.Linear(hidden // 2, C_out),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Build train DataLoader
    if use_masked:
        train_ds = TensorDataset(
            torch.from_numpy(train_emb_norm).float(),
            torch.from_numpy(all_targets[train_idx]).float(),
            torch.from_numpy(all_masks[train_idx]).float(),
        )
    else:
        train_ds = TensorDataset(
            torch.from_numpy(train_emb_norm).float(),
            torch.from_numpy(all_targets[train_idx]).float(),
        )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    val_emb_t = torch.from_numpy(val_emb_norm).float().to(device)
    val_tgt_t = torch.from_numpy(val_tgt_raw).float().to(device)
    val_mask_t = torch.from_numpy(val_mask_raw).float().to(device) if val_mask_raw is not None else None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_loss, best_state = float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        for batch in train_dl:
            if use_masked:
                xb, yb, mb = batch
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = ((pred - yb) ** 2 * mb).sum() / mb.sum().clamp(min=1)
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_emb_t)
            if val_mask_t is not None:
                val_loss = ((val_pred - val_tgt_t) ** 2 * val_mask_t).sum() / val_mask_t.sum().clamp(min=1)
            else:
                val_loss = nn.functional.mse_loss(val_pred, val_tgt_t)
            val_loss = val_loss.item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}  val_MSE={val_loss:.6f}  best={best_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()

    # ------------------------------------------------------------------
    # Evaluate: per-marker Pearson on validation / test set
    # ------------------------------------------------------------------
    with torch.no_grad():
        preds = model(val_emb_t).cpu().numpy()

    results = []
    for c in range(C_out):
        if use_masked:
            marker_name = inv_tokenizer.get(c, f"ch{c}")
            if marker_name in skip_set:
                continue
            col_mask = val_mask_raw[:, c] if val_mask_raw is not None else np.ones(len(preds), dtype=bool)
            if col_mask.sum() < 5:
                continue
            r, _ = pearsonr(preds[col_mask, c], val_tgt_raw[col_mask, c])
        else:
            if inv_tokenizer is not None and all_channel_ids_list[0] is not None:
                ch_id = all_channel_ids_list[0][c] if c < len(all_channel_ids_list[0]) else c
                marker_name = inv_tokenizer.get(int(ch_id), f"ch{c}")
            else:
                marker_name = f"ch{c}"
            if marker_name in skip_set:
                continue
            r, _ = pearsonr(preds[:, c], val_tgt_raw[:, c])

        results.append({"marker": marker_name, "pearson": r})

    df = pd.DataFrame(results)
    print(f"\nVirtual staining (MLP decoder) — Pearson per marker:")
    for _, row in df.iterrows():
        print(f"  {row['marker']:>25s}: {row['pearson']:.4f}")
    print(f"  {'Overall mean':>25s}: {df['pearson'].mean():.4f}")

    return {
        "model": model,
        "emb_mean": emb_mean,
        "emb_std": emb_std,
        "results_df": df,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Downstream evaluation on frozen embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="task", help="Downstream task")

    # --- Cell typing ---
    ct = subparsers.add_parser("cell_typing", help="Cell phenotype classification")
    ct.add_argument("--embeddings-dir", required=True, help="Dir with emb_*.npy files")
    ct.add_argument("--labels-csv", required=True, help="CSV with labels (label/cell_type/phenotype column)")
    ct.add_argument("--output-dir", required=True)
    ct.add_argument("--method", default="logistic", choices=list(CLASSIFIERS.keys()))
    ct.add_argument("--train-split", type=float, default=0.7, help="Fraction for training")
    ct.add_argument("--val-split", type=float, default=0.15, help="Fraction for validation")
    ct.add_argument("--seed", type=int, default=42)
    # Logistic
    ct.add_argument("--n-trials", type=int, default=50, help="Optuna trials (logistic)")
    ct.add_argument("--c-low", type=float, default=1e-6)
    ct.add_argument("--c-high", type=float, default=1e4)
    ct.add_argument("--max-iter", type=int, default=10000)
    # MLP / linear_probe
    ct.add_argument("--epochs", type=int, default=50)
    ct.add_argument("--lr", type=float, default=1e-3)
    ct.add_argument("--batch-size", type=int, default=256)
    ct.add_argument("--weight-decay", type=float, default=1e-4)
    ct.add_argument("--mlp-hidden", type=int, default=None, help="MLP hidden dim (default: embed_dim)")
    ct.add_argument("--mlp-dropout", type=float, default=0.1)
    ct.add_argument("--device", default="cuda")
    # kNN
    ct.add_argument("--knn-k", type=int, default=20)
    # Random Forest
    ct.add_argument("--rf-trees", type=int, default=500)
    ct.add_argument("--rf-depth", type=int, default=None)

    # --- Virtual staining ---
    vs = subparsers.add_parser("virtual_stain", help="Virtual staining via MLP decoder")
    vs.add_argument("--embeddings-dir", required=True,
                    help="Dir with embeddings_*.npz (panel-config) or emb_*.npy (legacy)")
    vs.add_argument("--patches-dir", default=None,
                    help="Dir with original patch .npy files (legacy mode only)")
    vs.add_argument("--train-embeddings-dir", default=None,
                    help="Separate dir with training-split embeddings (npz mode). "
                         "If given, --embeddings-dir is used for test only.")
    vs.add_argument("--panel-config", default=None,
                    help="Path to all_panels_config.yaml (required for npz mode)")
    vs.add_argument("--output-dir", required=True)
    vs.add_argument("--tokenizer-config", default="configs/all_markers_tokenizer.yaml")
    vs.add_argument("--skip-markers", nargs="*", default=["DNA1", "DNA2"])
    vs.add_argument("--method", default="mlp", choices=["mlp"])
    vs.add_argument("--epochs", type=int, default=100)
    vs.add_argument("--lr", type=float, default=1e-3)
    vs.add_argument("--batch-size", type=int, default=256)
    vs.add_argument("--weight-decay", type=float, default=1e-4)
    vs.add_argument("--mlp-hidden", type=int, default=None)
    vs.add_argument("--mlp-dropout", type=float, default=0.1)
    vs.add_argument("--device", default="cuda")

    args = parser.parse_args()

    if args.task is None:
        parser.print_help()
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.task == "cell_typing":
        print(f"=== Cell Typing | method={args.method} ===")

        embeddings, emb_labels, _ = load_embeddings(args.embeddings_dir)
        labels = load_labels_csv(args.labels_csv, n_embeddings=len(embeddings))
        if len(labels) != len(embeddings):
            print(f"[WARN] labels ({len(labels)}) != embeddings ({len(embeddings)}), truncating to min")
            n = min(len(labels), len(embeddings))
            labels, embeddings = labels[:n], embeddings[:n]

        # Stratified split
        np.random.seed(args.seed)
        from sklearn.model_selection import train_test_split
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            embeddings, labels, test_size=1.0 - args.train_split - args.val_split,
            stratify=labels, random_state=args.seed,
        )
        rel_val = args.val_split / (args.train_split + args.val_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=rel_val,
            stratify=y_trainval, random_state=args.seed,
        )
        print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
        print(f"  Classes: {len(np.unique(labels))}  Embedding dim: {embeddings.shape[1]}")

        trainer = CLASSIFIERS[args.method]
        clf, scaler = trainer(X_train, y_train, X_val, y_val, args)

        metrics, report = evaluate_classifier(clf, scaler, X_test, y_test)
        print("\nTest metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Save
        pd.DataFrame([metrics]).to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
        with open(os.path.join(args.output_dir, "classification_report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, default=str)

        print(f"\nResults saved to {args.output_dir}")

    elif args.task == "virtual_stain":
        print(f"=== Virtual Staining | MLP decoder ===")
        result = train_virtual_stain_mlp(args.embeddings_dir, args.patches_dir, args)

        if "results_df" in result:
            result["results_df"].to_csv(
                os.path.join(args.output_dir, "virtual_stain_pearson.csv"), index=False,
            )
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, default=str)

        # Save MLP weights
        import torch
        torch.save({
            "model_state_dict": result["model"].state_dict(),
            "emb_mean": result["emb_mean"],
            "emb_std": result["emb_std"],
        }, os.path.join(args.output_dir, "virtual_stain_mlp.pth"))

        print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
