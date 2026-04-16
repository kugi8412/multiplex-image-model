"""
Region and Artifact Detection pipeline for ImmuKronos (DINOv3).

Uses full spatial token features (not just marker-averaged) to classify
tissue regions vs artifacts at the spatial-token level.

Pipeline:
    1. Extract patches from tissue images
    2. Extract spatial features: (num_markers, h_p, w_p, embed_dim) per patch
    3. Train logistic regression on spatial token features
    4. Evaluate with F1, balanced accuracy, AUROC
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
    roc_auc_score,
)

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .inference import load_immukronos_model, ImmuKronosWrapper
from .feature_extraction import PatchDataset, ImmuVisFeatureExtractor


class RegionFeatures(Dataset):
    """Loads pre-extracted spatial features with labels."""

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
        return feature_list, np.array(label_list)


class RegionArtifactClassifier:
    """
    Logistic regression classifier for region/artifact detection.

    Features are spatial tokens — each patch produces multiple instances
    (one per spatial position). The classifier operates at the instance level.
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

    def _load_and_flatten_features(self, df):
        """Load spatial features and flatten to instance-level."""
        feature_dir = self.config["feature_dir"]
        all_features, all_labels = [], []

        for _, row in df.iterrows():
            patch_name = row["patch_name"].replace(".h5", ".npy")
            features = np.load(os.path.join(feature_dir, patch_name))
            label = row["label"]

            # features shape: (M, h_p, w_p, D) or (h_p, w_p, D)
            if features.ndim == 4:
                M, h_p, w_p, D = features.shape
                # Reshape to (h_p*w_p, M*D) — each spatial position is an instance
                features = features.transpose(1, 2, 0, 3).reshape(h_p * w_p, M * D)
            elif features.ndim == 3:
                h_p, w_p, D = features.shape
                features = features.reshape(h_p * w_p, D)
            else:
                features = features.reshape(1, -1)

            all_features.append(features)
            all_labels.extend([label] * features.shape[0])

        return np.vstack(all_features), np.array(all_labels)

    def train(self):
        X_train, y_train = self._load_and_flatten_features(self.train_df)
        X_valid, y_valid = self._load_and_flatten_features(self.valid_df)

        X_train = self.normalizer.fit_transform(X_train)
        X_valid = self.normalizer.transform(X_valid)

        def objective(trial):
            C = trial.suggest_float("C", 1e-4, 100.0, log=True)
            clf = LogisticRegression(
                C=C, max_iter=self.config.get("max_iter", 5000),
                solver="lbfgs", n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_valid)
            return f1_score(y_valid, y_pred, average="macro")

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=self.config.get("n_trials", 50),
            show_progress_bar=True,
        )

        best_C = study.best_params["C"]
        self.model = LogisticRegression(
            C=best_C, max_iter=self.config.get("max_iter", 5000),
            solver="lbfgs", n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        with open(os.path.join(self.output_dir, "classifier.pkl"), "wb") as f:
            pkl.dump({"model": self.model, "normalizer": self.normalizer}, f)

        return study

    def evaluate(self):
        X_test, y_test = self._load_and_flatten_features(self.test_df)
        X_test = self.normalizer.transform(X_test)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        metrics = {
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        }

        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            metrics["auroc"] = roc_auc_score(y_test, y_proba[:, 1])
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

        pd.DataFrame([metrics]).to_csv(
            os.path.join(self.output_dir, "test_metrics.csv"), index=False
        )
        return metrics


class RegionArtifactDetection:
    """
    High-level region/artifact detection pipeline.

    Config keys:
        checkpoint_path, patch_dir, output_dir,
        marker_list, marker_max_values, marker_info_with_metadata_csv_path,
        batch_size, device, model_config.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))

    def extract_features(self):
        """Extract full spatial features for region-level classification."""
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

        dataset = PatchDataset(self.config)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 16),
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
        )

        feature_dir = os.path.join(self.config["output_dir"], "spatial_features")
        os.makedirs(feature_dir, exist_ok=True)

        with torch.no_grad():
            for patches, marker_ids, patch_names in tqdm(
                dataloader, desc="Spatial feature extraction"
            ):
                patches = patches.to(self.device, dtype=torch.float32)
                marker_ids = marker_ids.to(self.device)

                _, _, spatial_features = wrapper(patches, marker_ids=marker_ids)

                spatial_np = spatial_features.cpu().numpy()
                for j, name in enumerate(patch_names):
                    stem = name.replace(".h5", "")
                    np.save(
                        os.path.join(feature_dir, f"{stem}.npy"),
                        spatial_np[j],
                    )

        self.config["feature_dir"] = feature_dir
        print(f"Spatial features saved to {feature_dir}")

    def extract_features_immuvis(self):
        """Extract spatial features from ImmuVis .npy data with panel-aware batching."""
        extractor = ImmuVisFeatureExtractor(self.config)
        extractor.extract_features(save_spatial=True)
        self.config["feature_dir"] = os.path.join(self.config["output_dir"], "spatial_features")
        print("ImmuVis spatial features extracted with panel-aware batching")

    def train_and_evaluate(self, train_df, valid_df, test_df, fold_name="fold_0"):
        output_dir = os.path.join(self.config["output_dir"], fold_name)
        classifier = RegionArtifactClassifier(
            self.config, train_df, valid_df, test_df, output_dir
        )
        study = classifier.train()
        metrics = classifier.evaluate()
        return metrics, study
