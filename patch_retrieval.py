#!/usr/bin/env python3
# patch_retrieval.py
# -*- coding: utf-8 -*-

"""
Patch retrieval on frozen embeddings produced by generate_embeddings.py.

Two retrieval backends:
  - 'cosine'  — Brute-force cosine similarity (default for DINO / ImmunoKronos)
  - 'kdtree'  — Scipy KD-tree on L2-normalised embeddings (default for ViT Baseline)

Usage:
  # ImmunoKronos (cosine similarity)
  python patch_retrieval.py \
      --query-embeddings embeddings/exp6a/ \
      --index-embeddings embeddings/exp6a_train/ \
      --method cosine \
      --top-k 5 \
      --output-dir results/exp6a_retrieval/ \
      --model-label ImmunoKRONOS

  # ViT Baseline (KD-tree)
  python patch_retrieval.py \
      --query-embeddings embeddings/exp7c/ \
      --index-embeddings embeddings/exp7c_train/ \
      --method kdtree \
      --top-k 5 \
      --output-dir results/exp7c_retrieval/ \
      --model-label VitBaseline

  # With labels for retrieval precision (Precision@K, mAP@K)
  python patch_retrieval.py \
      --query-embeddings embeddings/exp6a/ \
      --index-embeddings embeddings/exp6a_train/ \
      --method cosine \
      --top-k 10 \
      --query-labels data/cell_annotations_test.csv \
      --index-labels data/cell_annotations_train.csv \
      --output-dir results/exp6a_retrieval/
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd


# ============================================================================
# Data Loading
# ============================================================================

def load_embeddings_from_dir(embeddings_dir):
    """Load embeddings from a directory.

    Supports:
      1) Panel-config mode: embeddings_*.npz with 'embeddings', 'paths', 'datasets'
      2) Legacy mode: emb_*.npy files

    Returns:
        embeddings: (N, D) np.ndarray
        file_names: list[str]
        datasets: list[str] or None
    """
    npz_files = sorted(
        f for f in os.listdir(embeddings_dir)
        if f.startswith("embeddings_") and f.endswith(".npz")
    )
    if npz_files:
        emb_list, name_list, ds_list = [], [], []
        for fname in npz_files:
            data = np.load(os.path.join(embeddings_dir, fname), allow_pickle=True)
            emb_list.append(data["embeddings"])
            if "paths" in data:
                name_list.extend(data["paths"].tolist())
            else:
                name_list.extend([fname] * len(data["embeddings"]))
            if "datasets" in data:
                ds_list.extend(data["datasets"].tolist())
        return np.concatenate(emb_list, axis=0), name_list, ds_list or None

    # Legacy mode
    emb_files = sorted(
        f for f in os.listdir(embeddings_dir)
        if f.startswith("emb_") and f.endswith(".npy")
    )
    if not emb_files:
        raise FileNotFoundError(f"No embeddings found in {embeddings_dir}")

    emb_list, name_list = [], []
    for fname in emb_files:
        raw = np.load(os.path.join(embeddings_dir, fname), allow_pickle=True)
        if isinstance(raw, np.ndarray) and raw.ndim == 0:
            raw = raw.item()
        emb = raw["embeddings"] if isinstance(raw, dict) else raw
        if emb.ndim == 1:
            emb = emb[np.newaxis]
        emb_list.append(emb)
        name_list.extend([fname] * len(emb))
    return np.concatenate(emb_list, axis=0), name_list, None


def load_labels(csv_path, n_expected=None):
    """Load labels from CSV. Returns numpy array of string labels."""
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)
    for col in ["label", "cell_type", "phenotype"]:
        if col in df.columns:
            labels = df[col].values.astype(str)
            if n_expected is not None and len(labels) != n_expected:
                print(f"[WARN] Labels ({len(labels)}) != embeddings ({n_expected})")
                labels = labels[:min(len(labels), n_expected)]
            return labels
    raise ValueError(f"Labels CSV must have 'label', 'cell_type', or 'phenotype' column. "
                     f"Found: {list(df.columns)}")


# ============================================================================
# Retrieval Methods
# ============================================================================

def l2_normalize(X):
    """L2-normalize rows of X."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


def retrieve_cosine(query_emb, index_emb, top_k):
    """Brute-force cosine similarity retrieval.

    Args:
        query_emb: (Nq, D)
        index_emb: (Ni, D)
        top_k: int

    Returns:
        indices: (Nq, top_k) — indices into index_emb
        scores: (Nq, top_k) — cosine similarities
    """
    query_norm = l2_normalize(query_emb)
    index_norm = l2_normalize(index_emb)

    # Process in chunks to avoid OOM on large datasets
    chunk_size = 512
    Nq = query_norm.shape[0]
    all_indices = np.zeros((Nq, top_k), dtype=np.int64)
    all_scores = np.zeros((Nq, top_k), dtype=np.float32)

    for start in range(0, Nq, chunk_size):
        end = min(start + chunk_size, Nq)
        sim = query_norm[start:end] @ index_norm.T  # (chunk, Ni)
        top_idx = np.argpartition(-sim, top_k, axis=1)[:, :top_k]
        # Sort top-k by descending similarity
        for i in range(end - start):
            sorted_order = np.argsort(-sim[i, top_idx[i]])
            all_indices[start + i] = top_idx[i][sorted_order]
            all_scores[start + i] = sim[i, top_idx[i][sorted_order]]

    return all_indices, all_scores


def retrieve_kdtree(query_emb, index_emb, top_k):
    """KD-tree retrieval on L2-normalized embeddings (approximate nearest neighbor).

    Args:
        query_emb: (Nq, D)
        index_emb: (Ni, D)
        top_k: int

    Returns:
        indices: (Nq, top_k)
        scores: (Nq, top_k) — cosine similarities (computed post-hoc)
    """
    from scipy.spatial import cKDTree

    query_norm = l2_normalize(query_emb)
    index_norm = l2_normalize(index_emb)

    print(f"Building KD-tree over {index_norm.shape[0]} vectors (dim={index_norm.shape[1]})...")
    tree = cKDTree(index_norm)

    print(f"Querying {query_norm.shape[0]} vectors for top-{top_k} neighbors...")
    distances, indices = tree.query(query_norm, k=top_k, workers=-1)

    # Convert L2 distances to cosine similarity: cos = 1 - d^2/2
    scores = 1.0 - (distances ** 2) / 2.0

    return indices.astype(np.int64), scores.astype(np.float32)


RETRIEVAL_METHODS = {
    "cosine": retrieve_cosine,
    "kdtree": retrieve_kdtree,
}


# ============================================================================
# Evaluation Metrics
# ============================================================================

def precision_at_k(query_labels, index_labels, indices, k=None):
    """Compute Precision@K.

    Args:
        query_labels: (Nq,) string labels
        index_labels: (Ni,) string labels
        indices: (Nq, K) retrieved index positions
        k: if given, truncate to top-k

    Returns:
        per_query_precision: (Nq,) float
    """
    if k is not None:
        indices = indices[:, :k]
    K = indices.shape[1]
    retrieved_labels = index_labels[indices]  # (Nq, K)
    matches = (retrieved_labels == query_labels[:, None])  # (Nq, K)
    return matches.sum(axis=1) / K


def mean_average_precision_at_k(query_labels, index_labels, indices, k=None):
    """Compute mAP@K.

    For each query, AP@K = (1/min(R, K)) * sum_{j=1}^{K} P@j * rel(j)
    where R = number of relevant items in index, rel(j) = 1 if match at rank j.
    """
    if k is not None:
        indices = indices[:, :k]
    K = indices.shape[1]
    Nq = indices.shape[0]

    retrieved_labels = index_labels[indices]
    matches = (retrieved_labels == query_labels[:, None]).astype(np.float32)

    aps = np.zeros(Nq, dtype=np.float32)
    for i in range(Nq):
        cum_matches = np.cumsum(matches[i])
        precisions = cum_matches / np.arange(1, K + 1)
        ap = (precisions * matches[i]).sum()
        # Number of relevant items in the full index
        n_relevant = (index_labels == query_labels[i]).sum()
        if n_relevant > 0:
            aps[i] = ap / min(n_relevant, K)
    return aps


def recall_at_k(query_labels, index_labels, indices, k=None):
    """Compute Recall@K per query."""
    if k is not None:
        indices = indices[:, :k]
    retrieved_labels = index_labels[indices]
    matches = (retrieved_labels == query_labels[:, None])
    n_relevant = np.array([(index_labels == ql).sum() for ql in query_labels], dtype=np.float32)
    n_relevant = np.where(n_relevant == 0, 1, n_relevant)
    return matches.sum(axis=1) / n_relevant


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Patch retrieval on frozen embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query-embeddings", required=True,
                        help="Dir with query (test) embeddings")
    parser.add_argument("--index-embeddings", required=True,
                        help="Dir with index (train/reference) embeddings")
    parser.add_argument("--method", default="cosine",
                        choices=list(RETRIEVAL_METHODS.keys()),
                        help="Retrieval method: 'cosine' (brute-force) or 'kdtree'")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of neighbors to retrieve")
    parser.add_argument("--query-labels", default=None,
                        help="CSV with labels for query (test) set")
    parser.add_argument("--index-labels", default=None,
                        help="CSV with labels for index (train) set")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-label", default=None,
                        help="Label for the model in output files")
    parser.add_argument("--save-retrieval-pairs", action="store_true",
                        help="Save full retrieval indices and scores to NPZ")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load embeddings
    print(f"Loading query embeddings from {args.query_embeddings}")
    query_emb, query_names, query_ds = load_embeddings_from_dir(args.query_embeddings)
    print(f"  -> {query_emb.shape[0]} queries, dim={query_emb.shape[1]}")

    print(f"Loading index embeddings from {args.index_embeddings}")
    index_emb, index_names, index_ds = load_embeddings_from_dir(args.index_embeddings)
    print(f"  -> {index_emb.shape[0]} index items, dim={index_emb.shape[1]}")

    assert query_emb.shape[1] == index_emb.shape[1], \
        f"Dim mismatch: query={query_emb.shape[1]} vs index={index_emb.shape[1]}"

    # Load optional labels
    query_labels = load_labels(args.query_labels, n_expected=len(query_emb))
    index_labels = load_labels(args.index_labels, n_expected=len(index_emb))

    # Run retrieval
    retrieve_fn = RETRIEVAL_METHODS[args.method]
    print(f"\nRunning {args.method} retrieval (top-{args.top_k})...")
    t0 = time.time()
    indices, scores = retrieve_fn(query_emb, index_emb, args.top_k)
    elapsed = time.time() - t0
    print(f"Retrieval done in {elapsed:.2f}s")

    # Report metrics
    metrics = {
        "method": args.method,
        "top_k": args.top_k,
        "n_queries": len(query_emb),
        "n_index": len(index_emb),
        "embed_dim": query_emb.shape[1],
        "retrieval_time_s": round(elapsed, 2),
        "model_label": args.model_label,
    }

    # Score stats
    metrics["mean_top1_score"] = float(scores[:, 0].mean())
    metrics["mean_topk_score"] = float(scores.mean())

    if query_labels is not None and index_labels is not None:
        for k in [1, 3, 5, args.top_k]:
            if k > args.top_k:
                continue
            p_at_k = precision_at_k(query_labels, index_labels, indices, k=k)
            r_at_k = recall_at_k(query_labels, index_labels, indices, k=k)
            map_at_k = mean_average_precision_at_k(query_labels, index_labels, indices, k=k)

            metrics[f"precision@{k}"] = float(p_at_k.mean())
            metrics[f"recall@{k}"] = float(r_at_k.mean())
            metrics[f"mAP@{k}"] = float(map_at_k.mean())

            print(f"  P@{k}={p_at_k.mean():.4f}  R@{k}={r_at_k.mean():.4f}  "
                  f"mAP@{k}={map_at_k.mean():.4f}")

    # Save results
    metrics_path = os.path.join(args.output_dir, "retrieval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    if args.save_retrieval_pairs:
        npz_path = os.path.join(args.output_dir, "retrieval_results.npz")
        save_data = {
            "indices": indices,
            "scores": scores,
            "query_names": np.array(query_names, dtype=object),
            "index_names": np.array(index_names, dtype=object),
        }
        if query_labels is not None:
            save_data["query_labels"] = query_labels
        if index_labels is not None:
            save_data["index_labels"] = index_labels
        np.savez(npz_path, **save_data)
        print(f"Retrieval pairs saved to {npz_path}")

    # Per-query results CSV (top-1 info)
    df_rows = []
    for i in range(len(query_emb)):
        row = {
            "query_idx": i,
            "query_name": query_names[i] if query_names else i,
            "top1_index": int(indices[i, 0]),
            "top1_name": index_names[indices[i, 0]] if index_names else int(indices[i, 0]),
            "top1_score": float(scores[i, 0]),
        }
        if query_labels is not None:
            row["query_label"] = query_labels[i]
        if index_labels is not None:
            row["top1_label"] = index_labels[indices[i, 0]]
            row["top1_match"] = query_labels[i] == index_labels[indices[i, 0]] if query_labels is not None else None
        df_rows.append(row)

    df = pd.DataFrame(df_rows)
    csv_path = os.path.join(args.output_dir, "retrieval_top1.csv")
    df.to_csv(csv_path, index=False)
    print(f"Per-query top-1 results saved to {csv_path}")

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)


if __name__ == "__main__":
    main()
