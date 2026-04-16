#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tissue_search.py


"""
Tissue Search — spatial reverse image search for ImmuVis models.

Retrieves tissue samples with similar phenotypic or spatial patterns
from a support database using the latent representations produced by
any ImmuVis encoder (ConvNeXt, Swin, MambaSwin, ViT, etc.).

Two retrieval modes are supported and can be combined:

1. **Cosine similarity** on global (GAP-pooled) feature vectors, accelerated
   by a `scipy.spatial.cKDTree` in angular space.  Fast O(log N) lookup.
2. **Frobenius distance** on full spatial feature maps (C x H x W), which
   preserves spatial layout information lost by global pooling.  O(N) but
   captures morphological similarities that cosine on pooled vectors misses.

A weighted combination of both scores is used by default
('alpha' controls the cosine vs. Frobenius trade-off).

Usage
-----
>>> from multiplex_model.utils.tissue_search import TissueSearchEngine
>>> engine = TissueSearchEngine()
>>> engine.build_index(support_dir="data/support_embeddings")
>>> results = engine.query(query_dir="data/query_embeddings", topk=5)
>>> results.to_csv("retrieval_results.csv")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_embeddings(folder: str | Path) -> tuple[list[str], list[np.ndarray]]:
    """Load all ``.npy`` embeddings from *folder*.

    Each file can be either:
    * 1-D vector ``(D,)`` — a pre-pooled global descriptor.
    * 3-D tensor ``(C, H, W)`` — a spatial feature map (as produced by
      ``MultiplexAutoencoder.encode()``).

    Returns
    -------
    filenames : list[str]
        Sorted list of file stems used as sample identifiers.
    embeddings : list[np.ndarray]
        Corresponding numpy arrays in the same order.
    """
    folder = Path(folder)
    files = sorted(f for f in folder.iterdir() if f.suffix == ".npy")
    if not files:
        raise FileNotFoundError(f"No .npy files found in {folder}")
    filenames = [f.name for f in files]
    embeddings = [np.load(f).astype(np.float32) for f in files]
    return filenames, embeddings


def _global_pool(emb: np.ndarray) -> np.ndarray:
    """Reduce a spatial feature map ``(C, H, W)`` to a vector ``(C,)`` via GAP."""
    if emb.ndim == 1:
        return emb
    if emb.ndim == 3:
        return emb.mean(axis=(1, 2))  # global average pooling
    raise ValueError(f"Expected 1-D or 3-D array, got shape {emb.shape}")


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalization."""
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norms, eps)


def _frobenius_distance_matrix(
    queries: list[np.ndarray],
    keys: list[np.ndarray],
) -> np.ndarray:
    """Compute pairwise Frobenius distances between spatial feature maps.

    For 1-D vectors this falls back to standard Euclidean distance.

    Parameters
    ----------
    queries, keys : lists of np.ndarray
        Each element is ``(C, H, W)`` or ``(D,)``.

    Returns
    -------
    dist : np.ndarray of shape ``(len(queries), len(keys))``
        Frobenius distance matrix.  Lower = more similar.
    """
    n_q, n_k = len(queries), len(keys)
    dist = np.empty((n_q, n_k), dtype=np.float32)

    for i, q in enumerate(queries):
        for j, k in enumerate(keys):
            # Resize to common spatial dimensions if needed
            q_flat, k_flat = q.ravel(), k.ravel()
            if q_flat.shape[0] != k_flat.shape[0]:
                # If shapes don't match (different spatial sizes), pad the
                # shorter one with zeros — keeps the metric well-defined.
                max_len = max(q_flat.shape[0], k_flat.shape[0])
                q_pad = np.zeros(max_len, dtype=np.float32)
                k_pad = np.zeros(max_len, dtype=np.float32)
                q_pad[: q_flat.shape[0]] = q_flat
                k_pad[: k_flat.shape[0]] = k_flat
                q_flat, k_flat = q_pad, k_pad
            dist[i, j] = np.linalg.norm(q_flat - k_flat)

    return dist


# ------------------------------------------------------------------
# Main engine
# ------------------------------------------------------------------

@dataclass
class TissueSearchEngine:
    """KD-Tree-accelerated tissue retrieval with cosine + Frobenius scoring.

    Parameters
    ----------
    alpha : float
        Blending weight in ``[0, 1]``.
        ``1.0`` = pure cosine similarity, ``0.0`` = pure Frobenius distance.
        Default ``0.7`` favours cosine (fast) but includes spatial structure.
    centering : bool
        Whether to subtract the support-set mean before building the index.
    leafsize : int
        ``cKDTree`` leaf size — tune for your dataset size.
    """

    alpha: float = 0.7
    centering: bool = True
    leafsize: int = 40

    # — internal state (populated by ``build_index``) —
    _tree: cKDTree | None = field(default=None, init=False, repr=False)
    _key_names: list[str] = field(default_factory=list, init=False, repr=False)
    _key_vectors: np.ndarray | None = field(default=None, init=False, repr=False)
    _key_spatial: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _mean: np.ndarray | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(
        self,
        support_dir: str | Path | None = None,
        *,
        filenames: list[str] | None = None,
        embeddings: list[np.ndarray] | None = None,
    ) -> "TissueSearchEngine":

        if support_dir is not None:
            filenames, embeddings = _load_embeddings(support_dir)
        if filenames is None or embeddings is None:
            raise ValueError("Provide support_dir or (filenames, embeddings)")

        self._key_names = list(filenames)
        self._key_spatial = list(embeddings)

        # Global-pooled vectors for the KD-Tree
        vectors = np.stack([_global_pool(e) for e in embeddings])  # (N, D)

        if self.centering:
            self._mean = vectors.mean(axis=0, keepdims=True)
            vectors = vectors - self._mean

        # L2-normalize so that Euclidean distance in the KD-Tree corresponds
        # to angular (cosine) distance: ||a-b||^2 = 2 - 2*cos(a,b) when ||a||=||b||=1
        vectors = _l2_normalize(vectors, axis=1)
        self._key_vectors = vectors

        self._tree = cKDTree(vectors, leafsize=self.leafsize)
        return self

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        query_dir: str | Path | None = None,
        *,
        filenames: list[str] | None = None,
        embeddings: list[np.ndarray] | None = None,
        topk: int = 5,
        results_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Retrieve the top-k most similar support samples for each query.

        Parameters
        ----------
        query_dir : path, optional
            Folder of ``.npy`` query embeddings.
        filenames, embeddings : lists, optional
            In-memory alternative to ``query_dir``.
        topk : int
            Number of neighbours to return per query.
        results_path : path, optional
            If given, save results CSV here.

        Returns
        -------
        pd.DataFrame
            Columns ``top1 … topK`` with support filenames, plus
            ``score1 … scoreK`` with combined similarity scores.
            Index = query filenames.
        """
        if self._tree is None:
            raise RuntimeError("Call build_index() before query()")

        if query_dir is not None:
            filenames, embeddings = _load_embeddings(query_dir)
        if filenames is None or embeddings is None:
            raise ValueError("Provide query_dir or (filenames, embeddings)")

        q_spatial = list(embeddings)
        q_vectors = np.stack([_global_pool(e) for e in embeddings])

        if self.centering and self._mean is not None:
            q_vectors = q_vectors - self._mean
        q_vectors = _l2_normalize(q_vectors, axis=1)

        # Request more candidates than topk so we can re-rank with Frobenius
        n_candidates = min(len(self._key_names), max(topk * 4, 50))
        kd_dists, kd_idxs = self._tree.query(q_vectors, k=n_candidates)
        # Convert Euclidean distance on unit sphere to cosine similarity:
        # cos = 1 - d^2/2
        cosine_sims = 1.0 - kd_dists ** 2 / 2.0  # (n_queries, n_candidates)

        # Frobenius part
        n_queries = len(filenames)
        combined_scores = np.empty((n_queries, n_candidates), dtype=np.float32)

        for i in range(n_queries):
            candidate_idxs = kd_idxs[i]
            candidate_spatial = [self._key_spatial[j] for j in candidate_idxs]
            frob_dists = _frobenius_distance_matrix([q_spatial[i]], candidate_spatial)[0]

            # Normalize Frobenius distances to [0, 1] for blending
            frob_max = frob_dists.max()
            if frob_max > 0:
                frob_norm = frob_dists / frob_max
            else:
                frob_norm = frob_dists
            frob_sim = 1.0 - frob_norm

            # alpha * cosine + (1 - alpha) * frobenius_similarity
            combined_scores[i] = (
                self.alpha * cosine_sims[i] + (1.0 - self.alpha) * frob_sim
            )

        topk_local = np.argsort(-combined_scores, axis=1)[:, :topk]

        key_names_arr = np.array(self._key_names)
        result_names = []
        result_scores = []

        for i in range(n_queries):
            global_idxs = kd_idxs[i][topk_local[i]]
            result_names.append(key_names_arr[global_idxs])
            result_scores.append(combined_scores[i][topk_local[i]])

        result_names = np.array(result_names)   # (n_queries, topk)
        result_scores = np.array(result_scores)  # (n_queries, topk)

        # Build DataFrame
        name_cols = {f"top{k+1}": result_names[:, k] for k in range(topk)}
        score_cols = {f"score{k+1}": result_scores[:, k] for k in range(topk)}
        df = pd.DataFrame({**name_cols, **score_cols}, index=filenames)
        df.index.name = "query"

        if results_path is not None:
            Path(results_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(results_path)

        return df

    # ------------------------------------------------------------------
    # Single-query convenience
    # ------------------------------------------------------------------

    def query_single(
        self,
        embedding: np.ndarray,
        name: str = "query",
        topk: int = 5,
    ) -> pd.DataFrame:
        """Search with a single embedding (1-D or 3-D array)."""
        return self.query(
            filenames=[name],
            embeddings=[embedding],
            topk=topk,
        )

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_embeddings(
        model,
        dataloader,
        device: str = "cuda",
        output_dir: str | Path | None = None,
    ) -> tuple[list[str], list[np.ndarray]]:
        """Extract latent embeddings from an ImmuVis model for all samples.

        Compatible with ``DatasetFromTIFF`` + ``PanelBatchSampler`` which
        yields ``(img, channel_ids, dataset_name, img_path)`` tuples.

        Parameters
        ----------
        model : MultiplexAutoencoder
            A trained ImmuVis autoencoder (frozen, eval mode).
        dataloader : DataLoader
            DataLoader using ``PanelBatchSampler``.  Each batch yields
            ``(img, channel_ids, dataset_name, img_path)``.
        device : str
            Torch device.
        output_dir : path, optional
            If given, save each embedding as ``<stem>.npy``.

        Returns
        -------
        filenames : list[str]
            Sample identifiers (``<stem>.npy``).
        embeddings : list[np.ndarray]
            Encoded feature maps, each ``(D, H', W')``.
        """
        import torch
        from torch.amp import autocast

        model = model.to(device).eval()
        filenames = []
        embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(device, dtype=torch.float32)
                channel_ids = batch[1].to(device, dtype=torch.long)
                img_paths = batch[3]  # list of file paths

                with autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                    out = model.encode(images, channel_ids)
                latent = out["output"]  # (B, D, H', W')

                for b in range(latent.shape[0]):
                    emb = latent[b].float().cpu().numpy()
                    embeddings.append(emb)

                    # Use image path stem as identifier
                    name = img_paths[b] if isinstance(img_paths, (list, tuple)) else img_paths
                    stem = Path(name).stem
                    filenames.append(f"{stem}.npy")

                    if output_dir is not None:
                        out_path = Path(output_dir)
                        out_path.mkdir(parents=True, exist_ok=True)
                        np.save(out_path / f"{stem}.npy", emb)

        return filenames, embeddings
