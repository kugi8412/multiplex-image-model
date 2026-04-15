"""
Patient Stratification pipeline for ImmuKronos (DINOv3).

Uses Multiple Instance Learning (MIL) on patch-level features:
    1. Extract CLS features from all patches
    2. Build h5ad with patient-level metadata
    3. Train MIL model (BagModel) for patient-level classification
    4. Evaluate with AUC via repeated stratified k-fold cross-validation
"""

import os
import warnings
import numpy as np
import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import scanpy as sc

from .inference import load_immukronos_model, ImmuKronosWrapper
from .feature_extraction import PatchDataset, FeatureExtractor, H5ADBuilder, ImmuVisFeatureExtractor


# ==========================================
# MIL Components
# ==========================================

class BagModel(nn.Module):
    """
    Multiple Instance Learning model.

    Processes instances through prepNN, aggregates per bag, then classifies via afterNN.
    """

    def __init__(self, prepNN, afterNN, aggregation_func=None):
        super().__init__()
        self.prepNN = prepNN
        self.afterNN = afterNN
        self.aggregation_func = aggregation_func or torch.mean

    def forward(self, input):
        ids = input[1]
        x = input[0]

        if len(ids.shape) == 1:
            ids = ids.unsqueeze(0)

        inner_ids = ids[-1]
        device = x.device

        nn_out = self.prepNN(x)

        unique, inverse, counts = torch.unique(
            inner_ids, sorted=True, return_inverse=True, return_counts=True
        )
        idx = torch.cat(
            [(inverse == i).nonzero()[0] for i in range(len(unique))]
        ).sort()[1]
        bags = unique[idx]
        counts = counts[idx]

        output = torch.empty((len(bags), nn_out.shape[-1]), device=device)
        for i, bag in enumerate(bags):
            output[i] = self.aggregation_func(nn_out[inner_ids == bag], dim=0)

        output = self.afterNN(output)

        if ids.shape[0] == 1:
            return output
        else:
            ids = ids[:-1]
            mask = torch.empty(0, device=device).long()
            for i in range(len(counts)):
                mask = torch.cat(
                    (mask, torch.sum(counts[:i], dtype=torch.int64).reshape(1))
                )
            return (output, ids[:, mask])


class MilDataset(Dataset):
    """
    MIL dataset: instances grouped by bag IDs with bag-level labels.
    """

    def __init__(self, data, ids, labels, normalize=True):
        self.data = data
        self.labels = labels
        self.ids = ids

        if len(ids.shape) == 1:
            ids = ids.unsqueeze(0)
        self.ids = ids

        self.bags = torch.unique(self.ids[0])

        if normalize:
            data_cpu = data.cpu() if data.is_cuda else data
            mean = data_cpu.mean(dim=0, keepdim=True)
            std = data_cpu.std(dim=0, keepdim=True)
            std[std == 0] = 1
            self.data = (data_cpu - mean) / std
            if data.is_cuda:
                self.data = self.data.cuda()

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        mask = self.ids[0] == self.bags[index]
        data = self.data[mask]
        bagids = self.ids[:, mask]
        labels = self.labels[index]
        return data, bagids, labels

    @property
    def n_features(self):
        return self.data.shape[-1]


class PatientStratificationTrainer:
    """
    Trains and evaluates MIL model for patient stratification.

    Config keys:
        h5ad_path: Path to h5ad file with patch features and patient metadata.
        output_dir: Directory for results.
        label_col: Column in adata.obs for patient-level labels.
        patient_col: Column in adata.obs for patient IDs.
        hidden_dim: MIL hidden dim (default: 128).
        dropout: Dropout rate (default: 0.5).
        lr: Learning rate (default: 1e-3).
        epochs: Training epochs (default: 50).
        n_splits: K-fold splits (default: 5).
        n_repeats: Repeated CV (default: 10).
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))

    def _build_mil_model(self, n_features):
        hidden = self.config.get("hidden_dim", 128)
        dropout = self.config.get("dropout", 0.5)

        prepNN = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        afterNN = nn.Sequential(nn.Linear(hidden, 1))

        return BagModel(prepNN, afterNN, aggregation_func=torch.mean)

    def train_and_evaluate(self):
        """Run repeated stratified k-fold CV and return AUC results."""
        adata = sc.read_h5ad(self.config["h5ad_path"])

        label_col = self.config.get("label_col", "label")
        patient_col = self.config.get("patient_col", "patient_id")

        # Build patient-level data
        patients = adata.obs[patient_col].unique()
        patient_labels = []
        patient_features = []
        patient_ids_list = []

        for pid in patients:
            mask = adata.obs[patient_col] == pid
            features = adata.X[mask]
            label = adata.obs.loc[mask, label_col].iloc[0]

            patient_features.append(torch.tensor(features, dtype=torch.float32))
            patient_labels.append(int(label))
            patient_ids_list.append(pid)

        patient_labels = np.array(patient_labels)

        n_splits = self.config.get("n_splits", 5)
        n_repeats = self.config.get("n_repeats", 10)
        epochs = self.config.get("epochs", 50)
        lr = self.config.get("lr", 1e-3)

        all_aucs = []
        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=42
        )

        output_dir = self.config.get("output_dir", "results")
        os.makedirs(output_dir, exist_ok=True)

        for fold_idx, (train_idx, test_idx) in enumerate(
            tqdm(rskf.split(patients, patient_labels), total=n_splits * n_repeats,
                 desc="MIL CV")
        ):
            # Build datasets
            train_data, train_ids, test_data, test_ids = [], [], [], []
            train_labels = patient_labels[train_idx]
            test_labels = patient_labels[test_idx]

            for i, idx in enumerate(train_idx):
                feats = patient_features[idx]
                train_data.append(feats)
                train_ids.append(torch.full((feats.shape[0],), i, dtype=torch.long))
            for i, idx in enumerate(test_idx):
                feats = patient_features[idx]
                test_data.append(feats)
                test_ids.append(torch.full((feats.shape[0],), i, dtype=torch.long))

            train_data = torch.cat(train_data).to(self.device)
            train_ids = torch.cat(train_ids).to(self.device)
            test_data = torch.cat(test_data).to(self.device)
            test_ids = torch.cat(test_ids).to(self.device)

            train_labels_t = torch.tensor(train_labels, dtype=torch.float32).to(self.device)
            test_labels_t = torch.tensor(test_labels, dtype=torch.float32).to(self.device)

            train_ds = MilDataset(train_data, train_ids, train_labels_t)
            test_ds = MilDataset(test_data, test_ids, test_labels_t, normalize=False)

            n_features = train_ds.n_features
            model = self._build_mil_model(n_features).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            criterion = nn.BCEWithLogitsLoss()

            # Training
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for data, bag_ids, labels in DataLoader(train_ds, batch_size=1, shuffle=True):
                    data = data.squeeze(0).to(self.device)
                    bag_ids = bag_ids.squeeze(0).to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    output = model((data, bag_ids))
                    loss = criterion(output.squeeze(), labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                scheduler.step()

            # Evaluation
            model.eval()
            all_preds, all_true = [], []
            with torch.no_grad():
                for data, bag_ids, labels in DataLoader(test_ds, batch_size=1):
                    data = data.squeeze(0).to(self.device)
                    bag_ids = bag_ids.squeeze(0).to(self.device)

                    output = model((data, bag_ids))
                    prob = torch.sigmoid(output).cpu().numpy().flatten()
                    all_preds.extend(prob)
                    all_true.extend(labels.cpu().numpy().flatten())

            try:
                fpr, tpr, _ = roc_curve(all_true, all_preds)
                fold_auc = auc(fpr, tpr)
            except ValueError:
                fold_auc = float("nan")

            all_aucs.append(fold_auc)

        mean_auc = np.nanmean(all_aucs)
        std_auc = np.nanstd(all_aucs)
        print(f"Patient Stratification AUC: {mean_auc:.4f} ± {std_auc:.4f}")

        results = pd.DataFrame({"fold": range(len(all_aucs)), "auc": all_aucs})
        results.to_csv(os.path.join(output_dir, "mil_cv_results.csv"), index=False)

        return {"mean_auc": mean_auc, "std_auc": std_auc, "all_aucs": all_aucs}


class PatientStratification:
    """
    High-level patient stratification pipeline.

    Config keys:
        checkpoint_path, patch_dir, output_dir,
        marker_list, marker_max_values, marker_info_with_metadata_csv_path,
        metadata_csv_path, metadata_id_col,
        label_col, patient_col,
        batch_size, device, model_config.
    """

    def __init__(self, config):
        self.config = config

    def extract_features(self):
        """Extract CLS features from all patches."""
        extractor = FeatureExtractor(self.config)
        extractor.extract_features()

    def extract_features_immuvis(self):
        """Extract CLS features from ImmuVis .npy data with panel-aware batching."""
        extractor = ImmuVisFeatureExtractor(self.config)
        extractor.extract_features()

    def build_h5ad(self, feature_type="cls_features"):
        """Build h5ad from extracted features."""
        h5ad_config = {
            "feature_dir": self.config["output_dir"],
            "output_path": os.path.join(
                self.config["output_dir"], f"patient_{feature_type}.h5ad"
            ),
            "metadata_csv_path": self.config.get("metadata_csv_path"),
            "metadata_id_col": self.config.get("metadata_id_col", "image_id"),
        }
        builder = H5ADBuilder(h5ad_config)
        return builder.build(feature_type=feature_type)

    def train_and_evaluate(self, h5ad_path=None):
        """Train MIL and run CV evaluation."""
        if h5ad_path is None:
            h5ad_path = os.path.join(
                self.config["output_dir"], "patient_cls_features.h5ad"
            )

        mil_config = {
            **self.config,
            "h5ad_path": h5ad_path,
        }
        trainer = PatientStratificationTrainer(mil_config)
        return trainer.train_and_evaluate()
