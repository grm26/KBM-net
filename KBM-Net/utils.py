from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def load_dataset(
    feature_data_path: str | Path,
    feature_dict_path: str | Path,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]], List[str], Dict[object, int]]:
    """Load features, labels, and modality column indices.

    The returned feature and label tensors stay on CPU. Batches are moved to the
    selected device by the training loop, which avoids keeping the full dataset
    on the GPU.
    """

    feature_data_path = Path(feature_data_path)
    feature_dict_path = Path(feature_dict_path)

    feature_dict = np.load(feature_dict_path, allow_pickle=True).item()
    if not isinstance(feature_dict, dict) or not feature_dict:
        raise ValueError(f"Invalid or empty modality dictionary: {feature_dict_path}")

    dataframe = pd.read_csv(feature_data_path, low_memory=False)
    if dataframe.shape[1] < 2:
        raise ValueError("The dataset must contain at least one feature column and one label column.")

    label_column = "LABEL" if "LABEL" in dataframe.columns else dataframe.columns[-1]
    feature_columns = [column for column in dataframe.columns if column != label_column]

    feature_array = dataframe[feature_columns].to_numpy(dtype=np.float32)
    raw_labels = dataframe[label_column].to_numpy()
    unique_labels = sorted(pd.unique(raw_labels).tolist())
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    label_array = np.asarray([label_to_index[label] for label in raw_labels], dtype=np.int64)

    column_to_index = {column: index for index, column in enumerate(feature_columns)}
    modality_names: List[str] = []
    modality_feature_indices: List[List[int]] = []

    for modality_name, modality_columns in feature_dict.items():
        modality_names.append(str(modality_name))
        missing_columns = [
            column for column in modality_columns if column not in column_to_index
        ]
        if missing_columns:
            preview = ", ".join(map(str, missing_columns[:5]))
            raise ValueError(
                f"Modality '{modality_name}' contains missing columns: {preview}"
            )
        modality_feature_indices.append(
            [column_to_index[column] for column in modality_columns]
        )

    features = torch.from_numpy(feature_array)
    labels = torch.from_numpy(label_array)
    return features, labels, modality_feature_indices, modality_names, label_to_index


def make_outer_splits(
    labels: torch.Tensor,
    n_splits: int,
    random_seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create stratified outer cross-validation splits."""

    labels_np = labels.detach().cpu().numpy()
    class_counts = np.bincount(labels_np)
    min_class_count = int(class_counts.min())
    if n_splits > min_class_count:
        raise ValueError(
            f"n_splits={n_splits} is larger than the smallest class count "
            f"({min_class_count}). Reduce n_splits."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )
    dummy_x = np.zeros((labels_np.shape[0], 1), dtype=np.float32)
    return list(splitter.split(dummy_x, labels_np))


def make_train_validation_split(
    train_indices: Sequence[int],
    labels: torch.Tensor,
    validation_ratio: float,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split an outer-training fold into inner-train and validation subsets."""

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")

    train_indices = np.asarray(train_indices, dtype=np.int64)
    labels_np = labels.detach().cpu().numpy()
    train_labels = labels_np[train_indices]

    class_counts = np.bincount(train_labels)
    if class_counts.min() < 2:
        raise ValueError(
            "Every class needs at least two samples in the outer-training fold "
            "for a stratified validation split."
        )

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_ratio,
        random_state=random_seed,
    )
    dummy_x = np.zeros((train_indices.shape[0], 1), dtype=np.float32)
    inner_train_pos, validation_pos = next(splitter.split(dummy_x, train_labels))
    return train_indices[inner_train_pos], train_indices[validation_pos]


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from a training subset only."""

    labels = labels.detach().cpu().long()
    counts = torch.bincount(labels, minlength=num_classes).float()
    if torch.any(counts == 0):
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"Training subset is missing classes: {missing}")

    weights = counts.sum() / (num_classes * counts)
    return weights.float()


def _multiclass_auc(
    y_true: np.ndarray,
    logits: np.ndarray,
    num_classes: int,
) -> float:
    probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()

    if np.unique(y_true).size < 2:
        return float("nan")

    if num_classes == 2:
        return float(roc_auc_score(y_true, probabilities[:, 1]))

    if np.unique(y_true).size < num_classes:
        return float("nan")

    return float(
        roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="macro",
            labels=np.arange(num_classes),
        )
    )


def compute_all_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    """Compute metrics with a fixed class order across all folds."""

    y_true = np.asarray(y_true, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    y_pred = np.argmax(logits, axis=-1)

    labels = np.arange(num_classes)
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    sensitivity_values = []
    specificity_values = []
    total = confusion.sum()

    for class_index in range(num_classes):
        true_positive = confusion[class_index, class_index]
        false_negative = confusion[class_index, :].sum() - true_positive
        false_positive = confusion[:, class_index].sum() - true_positive
        true_negative = total - true_positive - false_negative - false_positive

        sensitivity_denominator = true_positive + false_negative
        specificity_denominator = true_negative + false_positive
        sensitivity_values.append(
            true_positive / sensitivity_denominator
            if sensitivity_denominator > 0
            else 0.0
        )
        specificity_values.append(
            true_negative / specificity_denominator
            if specificity_denominator > 0
            else 0.0
        )

    auc = _multiclass_auc(y_true, logits, num_classes)
    return {
        "acc": float(np.mean(y_true == y_pred)),
        "sen": float(np.mean(sensitivity_values)),
        "spec": float(np.mean(specificity_values)),
        "f1": float(f1),
        "bac": float(balanced_accuracy),
        "auc": auc,
    }
