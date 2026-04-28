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
    """Load all emb_*.npy files from a directory.

    Each file is expected to be saved by generate_embeddings.py as:
        {"embeddings": np.ndarray, "labels": np.ndarray (optional)}

    Returns:
        embeddings: (N, D) array
        labels: (N,) array or None
        file_names: list of source file basenames
    """
    emb_list, label_list, name_list = [], [], []

    files = sorted(f for f in os.listdir(embeddings_dir) if f.startswith("emb_") and f.endswith(".npy"))
    if not files:
        raise FileNotFoundError(f"No emb_*.npy files found in {embeddings_dir}")

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

def load_patch_data(patches_dir, tokenizer_path=None, skip_markers=None):
    """Load patch images for virtual staining evaluation.

    Expects .npy files with shape (C, H, W) or .npz with 'patches' and 'channel_ids'.
    Returns list of (patches_array, channel_ids_array, filename) tuples.
    """
    skip_markers = set(skip_markers or [])
    files = sorted(
        f for f in os.listdir(patches_dir)
        if (f.endswith(".npy") or f.endswith(".npz")) and not f.startswith("emb_")
    )
    return files


def train_virtual_stain_mlp(embeddings_dir, patches_dir, args):
    """Train an MLP decoder head to predict marker intensities from frozen embeddings.

    For each image, we have:
        - embedding: (D,) from generate_embeddings.py
        - patch: (C, H, W) original multiplex image

    The MLP predicts the spatial-mean intensity of each marker from the embedding.
    Evaluation is leave-one-out Pearson correlation per marker.
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

    # Collect matched (embedding, patch) pairs
    emb_files = sorted(f for f in os.listdir(embeddings_dir) if f.startswith("emb_") and f.endswith(".npy"))

    all_emb, all_targets, all_channel_ids_list, all_names = [], [], [], []
    for ef in emb_files:
        base = ef.replace("emb_", "")
        patch_path = os.path.join(patches_dir, base)
        if not os.path.exists(patch_path):
            npz_path = patch_path.replace(".npy", ".npz")
            if os.path.exists(npz_path):
                patch_path = npz_path
            else:
                continue

        # Load embedding
        emb_raw = np.load(os.path.join(embeddings_dir, ef), allow_pickle=True)
        if isinstance(emb_raw, np.ndarray) and emb_raw.ndim == 0:
            emb_raw = emb_raw.item()
        if isinstance(emb_raw, dict):
            emb = emb_raw["embeddings"]
        else:
            emb = emb_raw
        if emb.ndim == 2:
            emb = emb[0]  # take first if batched single

        # Load patch
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

        # Target: spatial-mean intensity per channel → (C,)
        target = patch.mean(axis=(1, 2)) if patch.ndim == 3 else patch.mean(axis=-1)

        all_emb.append(emb)
        all_targets.append(target)
        all_channel_ids_list.append(ch_ids)
        all_names.append(base)

    if not all_emb:
        print("[ERROR] No matched embedding-patch pairs found.")
        return {}

    all_emb = np.stack(all_emb, axis=0)          # (N, D)
    all_targets = np.stack(all_targets, axis=0)   # (N, C)
    N, D = all_emb.shape
    C_out = all_targets.shape[1]
    print(f"Loaded {N} samples, embedding dim={D}, output channels={C_out}")

    # Train/val split (80/20)
    np.random.seed(42)
    perm = np.random.permutation(N)
    split = int(0.8 * N)
    train_idx, val_idx = perm[:split], perm[split:]

    device = torch.device(args.device)

    # Normalize embeddings
    emb_mean = all_emb[train_idx].mean(axis=0)
    emb_std = all_emb[train_idx].std(axis=0) + 1e-8
    all_emb_norm = (all_emb - emb_mean) / emb_std

    # MLP decoder: embedding → channel intensities
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
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(all_emb_norm[train_idx]).float(),
        torch.from_numpy(all_targets[train_idx]).float(),
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    val_emb = torch.from_numpy(all_emb_norm[val_idx]).float().to(device)
    val_tgt = torch.from_numpy(all_targets[val_idx]).float().to(device)

    best_loss, best_state = float("inf"), None
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
            val_pred = model(val_emb)
            val_loss = criterion(val_pred, val_tgt).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}  val_MSE={val_loss:.6f}  best={best_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()

    # Evaluate: per-channel Pearson on validation set
    with torch.no_grad():
        preds = model(val_emb).cpu().numpy()
    targets_val = all_targets[val_idx]

    results = []
    for c in range(C_out):
        if inv_tokenizer is not None and all_channel_ids_list[0] is not None:
            ch_id = all_channel_ids_list[0][c] if c < len(all_channel_ids_list[0]) else c
            marker_name = inv_tokenizer.get(int(ch_id), f"ch{c}")
        else:
            marker_name = f"ch{c}"

        if marker_name in skip_set:
            continue

        r, _ = pearsonr(preds[:, c], targets_val[:, c])
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
    vs.add_argument("--embeddings-dir", required=True, help="Dir with emb_*.npy files")
    vs.add_argument("--patches-dir", required=True, help="Dir with original patch .npy files")
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
