"""
Cell Phenotyping pipeline for ImmuKronos (DINOv3).

Mirrors the original KRONOS cell phenotyping tutorial:
    1. Extract cell-centered patches
    2. Extract features using ImmuKronos backbone
    3. Train logistic regression classifier with Optuna hyperparameter search
    4. Evaluate with F1, balanced accuracy, AUROC, AUPRC
"""

import os
import warnings
import numpy as np
import pandas as pd
import pickle as pkl

warnings.filterwarnings("ignore")

import optuna
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
)

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .inference import load_immukronos_model, ImmuKronosWrapper
from .feature_extraction import CellPatchDataset, ImmuVisFeatureExtractor


class CellPhenotypingDataset(CellPatchDataset):
    """Alias for CellPatchDataset used in cell phenotyping pipeline."""
    pass


class CellPhenotypingFeatures(Dataset):
    """Loads pre-extracted features with associated labels for classification."""

    def __init__(self, feature_dir, patch_list, labels):
        self.feature_dir = feature_dir
        self.patch_list = patch_list
        self.labels = labels

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        patch_name = self.patch_list[idx].replace(".h5", ".npy")
        features = np.load(os.path.join(self.feature_dir, patch_name))
        return features, self.labels[idx], patch_name

    def get_all(self):
        feature_list, label_list = [], []
        for i in range(len(self.patch_list)):
            patch_name = self.patch_list[i].replace(".h5", ".npy")
            features = np.load(os.path.join(self.feature_dir, patch_name))
            feature_list.append(features)
            label_list.append(self.labels[i])
        return np.array(feature_list), np.array(label_list)


class CellPhenotypingClassifier:
    """
    Logistic regression classifier with Optuna hyperparameter search.

    Config keys:
        feature_dir: Directory with pre-extracted .npy features.
        output_dir: Directory to save models and results.
        n_trials: Number of Optuna trials (default: 50).
        max_iter: Max iterations for LogisticRegression (default: 5000).
    """

    def __init__(self, config, train_df, valid_df, test_df, output_dir):
        self.config = config
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.output_dir = output_dir
        self.normalizer = StandardScaler()
        self.model = None

        os.makedirs(output_dir, exist_ok=True)

    def _load_features(self, df):
        feature_dir = self.config["feature_dir"]
        ds = CellPhenotypingFeatures(
            feature_dir, df["patch_name"].tolist(), df["label"].tolist()
        )
        return ds.get_all()

    def train(self):
        """Train classifier with Optuna hyperparameter optimization."""
        X_train, y_train = self._load_features(self.train_df)
        X_valid, y_valid = self._load_features(self.valid_df)

        X_train = self.normalizer.fit_transform(X_train)
        X_valid = self.normalizer.transform(X_valid)

        def objective(trial):
            C = trial.suggest_float("C", 1e-4, 100.0, log=True)
            clf = LogisticRegression(
                C=C,
                max_iter=self.config.get("max_iter", 5000),
                solver="lbfgs",
                multi_class="multinomial",
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_valid)
            return f1_score(y_valid, y_pred, average="macro")

        n_trials = self.config.get("n_trials", 50)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_C = study.best_params["C"]
        print(f"Best C: {best_C:.6f} (F1: {study.best_value:.4f})")

        self.model = LogisticRegression(
            C=best_C,
            max_iter=self.config.get("max_iter", 5000),
            solver="lbfgs",
            multi_class="multinomial",
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        # Save model and normalizer
        with open(os.path.join(self.output_dir, "classifier.pkl"), "wb") as f:
            pkl.dump({"model": self.model, "normalizer": self.normalizer}, f)

        return study

    def evaluate(self):
        """Evaluate on test set and return metrics dict."""
        X_test, y_test = self._load_features(self.test_df)
        X_test = self.normalizer.transform(X_test)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        metrics = {
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        }

        # AUROC / AUPRC (handle binary vs multiclass)
        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            metrics["auroc"] = roc_auc_score(y_test, y_proba[:, 1])
            metrics["auprc"] = average_precision_score(y_test, y_proba[:, 1])
        else:
            try:
                metrics["auroc"] = roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="macro"
                )
            except ValueError:
                metrics["auroc"] = float("nan")

        print("Test metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Save metrics
        pd.DataFrame([metrics]).to_csv(
            os.path.join(self.output_dir, "test_metrics.csv"), index=False
        )

        return metrics


class CellPhenotyping:
    """
    High-level pipeline for cell phenotyping.

    Config keys:
        checkpoint_path: Path to ImmuKronos checkpoint.
        patch_dir: Directory with cell-centered .h5 patches.
        output_dir: Base output directory.
        marker_list: List of marker names.
        marker_max_values: Max intensity for normalization.
        marker_info_with_metadata_csv_path: Path to marker metadata CSV.
        annotations_csv_path: Path to cell annotations CSV.
        batch_size: Batch size for feature extraction (default: 32).
        device: 'cuda' or 'cpu'.
        model_config: Dict with model architecture params.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))

    def extract_features(self):
        """Extract marker features from cell patches."""
        model_cfg = self.config.get("model_config", {})
        model, embed_dim = load_immukronos_model(
            checkpoint_path=self.config["checkpoint_path"],
            num_markers=self.config.get("num_markers", 512),
            embed_dim=model_cfg.get("embed_dim", 384),
            depth=model_cfg.get("depth", 12),
            num_heads=model_cfg.get("num_heads", 6),
            patch_size=model_cfg.get("patch_size", 8),
            device=str(self.device),
        )
        wrapper = ImmuKronosWrapper(model, model_cfg.get("patch_size", 8))

        dataset = CellPhenotypingDataset(self.config)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
        )

        feature_dir = os.path.join(self.config["output_dir"], "marker_features")
        os.makedirs(feature_dir, exist_ok=True)

        with torch.no_grad():
            for patches, marker_ids, cell_masks, patch_names in tqdm(
                dataloader, desc="Cell feature extraction"
            ):
                # Apply cell mask before forward pass
                mask_tensor = torch.tensor(cell_masks, dtype=torch.float32).unsqueeze(1)
                patches = patches * mask_tensor

                patches = patches.to(self.device, dtype=torch.float32)
                marker_ids = marker_ids.to(self.device)

                _, marker_features, _ = wrapper(patches, marker_ids=marker_ids)

                marker_features_np = marker_features.cpu().numpy()
                for j, name in enumerate(patch_names):
                    stem = name.replace(".h5", "")
                    np.save(
                        os.path.join(feature_dir, f"{stem}.npy"),
                        marker_features_np[j].flatten(),
                    )

        self.config["feature_dir"] = feature_dir
        print(f"Cell features saved to {feature_dir}")

    def extract_features_immuvis(self):
        """Extract features from ImmuVis .npy data with panel-aware batching.

        Uses ImmuVisFeatureExtractor which handles variable marker counts
        and produces fixed-size features padded to full tokenizer size.

        Requires config keys: panel_config_path, tokenizer_config_path,
        checkpoint_path, output_dir, model_config.
        """
        extractor = ImmuVisFeatureExtractor(self.config)
        extractor.extract_features()
        self.config["feature_dir"] = self.config["output_dir"]
        print("ImmuVis features extracted with panel-aware batching")

    def train_and_evaluate(self, train_df, valid_df, test_df, fold_name="fold_0"):
        """Train classifier and evaluate on a single fold."""
        output_dir = os.path.join(self.config["output_dir"], fold_name)
        classifier = CellPhenotypingClassifier(
            self.config, train_df, valid_df, test_df, output_dir
        )
        study = classifier.train()
        metrics = classifier.evaluate()
        return metrics, study
