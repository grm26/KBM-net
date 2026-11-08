from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from Model.models import KBM
from utils import (
    compute_all_metrics,
    compute_class_weights,
    load_dataset,
    make_outer_splits,
    make_train_validation_split,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(gpu_index: int) -> torch.device:
    """Resolve the requested CUDA device, or fall back to CPU when requested."""

    if gpu_index < 0:
        return torch.device("cpu")

    if not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")

    device_count = torch.cuda.device_count()
    if gpu_index >= device_count:
        raise ValueError(
            f"Requested GPU {gpu_index}, but only {device_count} CUDA device(s) "
            "are visible."
        )

    torch.cuda.set_device(gpu_index)
    return torch.device(f"cuda:{gpu_index}")


def resolve_dataset_paths(
    project_root: Path,
    dataset: str,
    task: str,
) -> Tuple[Path, Path]:
    candidates = {
        "TADPOLE": [
            (
                project_root
                / "DATASET"
                / "TADPOLE"
                / f"{task}_ADNI_processed_standard_data.csv",
                project_root
                / "DATASET"
                / "TADPOLE"
                / f"{task}_ADNI_modal_feat_dict.npy",
            ),
        ],
        "ADNI": [
            (
                project_root
                / "DATASET"
                / "ADNI"
                / f"Select_NEW_{task}__ADNI_data.csv",
                project_root
                / "DATASET"
                / "ADNI"
                / f"Select_NEW_{task}__ADNI_modal_feat_dict.npy",
            ),
        ],
        "ABIDE": [
            (
                project_root / "DATASET" / "ABIDE" / "processed_standard_data.csv",
                project_root / "DATASET" / "ABIDE" / "modal_feat_dict.npy",
            ),
            (
                project_root / "DATASET" / "ABIDE" / "ABIDE_processed_data_modal.csv",
                project_root / "DATASET" / "ABIDE" / "ABIDE_modal_feat_dict.npy",
            ),
        ],
        "ABIDE-5": [
            (
                project_root
                / "DATASET"
                / "ABIDE-5"
                / "processed_standard_data.csv",
                project_root / "DATASET" / "ABIDE-5" / "modal_feat_dict.npy",
            ),
            (
                project_root
                / "DATASET"
                / "ABIDE-5"
                / "ABIDE_processed_data_modal.csv",
                project_root / "DATASET" / "ABIDE-5" / "ABIDE_modal_feat_dict.npy",
            ),
        ],
    }

    if dataset not in candidates:
        raise ValueError(f"Unsupported dataset: {dataset}")

    for data_path, feature_dict_path in candidates[dataset]:
        if data_path.exists() and feature_dict_path.exists():
            return data_path, feature_dict_path

    expected = "\n".join(
        f"  - {data_path}\n    {feature_dict_path}"
        for data_path, feature_dict_path in candidates[dataset]
    )
    raise FileNotFoundError(
        f"Could not find files for dataset={dataset}, task={task}.\n{expected}"
    )


def build_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: Iterable[int],
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    indices = torch.as_tensor(list(indices), dtype=torch.long)
    dataset = TensorDataset(features[indices], labels[indices])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def as_index_tensor(indices: Iterable[int]) -> torch.Tensor:
    return torch.as_tensor(list(indices), dtype=torch.long)


def compute_total_loss(
    outputs,
    labels: torch.Tensor,
    criterion: nn.Module,
    fusion_weight: float,
    auxiliary_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fusion_logits = outputs[0]
    modal_logits = outputs[1:]
    fusion_loss = criterion(fusion_logits, labels)

    if modal_logits:
        modal_loss = torch.stack(
            [criterion(logits, labels) for logits in modal_logits]
        ).mean()
    else:
        modal_loss = fusion_loss.new_zeros(())

    total_loss = fusion_weight * fusion_loss + auxiliary_weight * modal_loss
    return total_loss, fusion_loss, modal_loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    fusion_weight: float,
    auxiliary_weight: float,
    max_grad_norm: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_fusion_loss = 0.0
    total_auxiliary_loss = 0.0
    total_samples = 0
    total_correct = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch_x)
        loss, fusion_loss, auxiliary_loss = compute_total_loss(
            outputs,
            batch_y,
            criterion,
            fusion_weight,
            auxiliary_weight,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss detected.")

        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_fusion_loss += fusion_loss.item() * batch_size
        total_auxiliary_loss += auxiliary_loss.item() * batch_size
        total_samples += batch_size
        total_correct += (
            outputs[0].detach().argmax(dim=-1) == batch_y
        ).sum().item()

    return {
        "loss": total_loss / total_samples,
        "fusion_loss": total_fusion_loss / total_samples,
        "auxiliary_loss": total_auxiliary_loss / total_samples,
        "acc": total_correct / total_samples,
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return labels and fusion logits in the loader's original order."""

    model.eval()
    all_logits = []
    all_labels = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        logits = model(batch_x)[0]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits detected.")
        all_logits.append(logits.cpu())
        all_labels.append(batch_y)

    if not all_logits:
        raise RuntimeError("Evaluation loader is empty.")

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return labels, logits


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    labels, logits = predict(model, loader, device)
    return compute_all_metrics(labels, logits, num_classes)


def metric_value(metrics: Dict[str, float], name: str) -> float:
    value = metrics[name]
    return -float("inf") if not np.isfinite(value) else float(value)


def is_better(
    current: Dict[str, float],
    best: Dict[str, float] | None,
    selection_metric: str,
) -> bool:
    if best is None:
        return True

    current_primary = metric_value(current, selection_metric)
    best_primary = metric_value(best, selection_metric)
    if current_primary > best_primary + 1e-12:
        return True
    if abs(current_primary - best_primary) <= 1e-12:
        current_auc = metric_value(current, "auc")
        best_auc = metric_value(best, "auc")
        return current_auc > best_auc
    return False


def search_best_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    train_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    min_learning_rate: float,
    weight_decay: float,
    fusion_weight: float,
    auxiliary_weight: float,
    max_grad_norm: float,
    selection_metric: str,
    description: str,
) -> Tuple[int, Dict[str, float]]:
    class_weights = compute_class_weights(train_labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=min_learning_rate,
    )

    best_epoch = 1
    best_metrics = None
    progress = tqdm(range(1, epochs + 1), desc=description, leave=False)
    for epoch in progress:
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            fusion_weight,
            auxiliary_weight,
            max_grad_norm,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            num_classes,
        )
        scheduler.step()

        if is_better(validation_metrics, best_metrics, selection_metric):
            best_epoch = epoch
            best_metrics = copy.deepcopy(validation_metrics)

        progress.set_postfix(
            train_loss=f"{train_stats['loss']:.4f}",
            val_bac=f"{validation_metrics['bac']:.4f}",
            val_auc=f"{validation_metrics['auc']:.4f}",
        )

    if best_metrics is None:
        raise RuntimeError("Unable to select a validation epoch.")
    return best_epoch, best_metrics


def fit_fixed_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    train_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    min_learning_rate: float,
    weight_decay: float,
    fusion_weight: float,
    auxiliary_weight: float,
    max_grad_norm: float,
    description: str,
) -> None:
    class_weights = compute_class_weights(train_labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=min_learning_rate,
    )

    progress = tqdm(range(1, epochs + 1), desc=description, leave=False)
    for _ in progress:
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            fusion_weight,
            auxiliary_weight,
            max_grad_norm,
        )
        scheduler.step()
        progress.set_postfix(
            loss=f"{train_stats['loss']:.4f}",
            acc=f"{train_stats['acc']:.4f}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-free multimodal KAN + Cross-Modal Mamba training."
    )
    parser.add_argument("--dataset", default="TADPOLE")
    parser.add_argument("--task", default="SMCI_PMCI")
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index, e.g. 0 or 1; use -1 to force CPU.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--min-lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0009)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dim-emb", type=int, default=32)
    parser.add_argument("--d-state", type=int, default=32)
    parser.add_argument("--max-grad-norm", type=float, default=1.5)
    parser.add_argument("--fusion-weight", type=float, default=0.9)
    parser.add_argument("--auxiliary-weight", type=float, default=0.1)
    parser.add_argument(
        "--selection-metric",
        choices=("acc", "bac", "f1", "auc"),
        default="bac",
        help="Validation metric used to select the epoch.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the final model state for every outer fold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.n_splits < 2:
        raise ValueError("epochs must be >= 1 and n_splits must be >= 2.")
    if args.fusion_weight < 0 or args.auxiliary_weight < 0:
        raise ValueError("Loss weights must be non-negative.")

    set_seed(args.seed)
    device = resolve_device(args.gpu)
    project_root = Path(__file__).resolve().parent
    data_path, feature_dict_path = resolve_dataset_paths(
        project_root,
        args.dataset,
        args.task,
    )
    features, labels, feature_list, modality_names, label_map = load_dataset(
        data_path,
        feature_dict_path,
    )
    num_classes = len(label_map)

    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Data: {data_path}")
    print(f"Modalities ({len(modality_names)}): {modality_names}")
    print(f"Samples: {len(labels)}, classes: {num_classes}, device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        "Class counts:",
        torch.bincount(labels, minlength=num_classes).tolist(),
    )

    outer_splits = make_outer_splits(labels, args.n_splits, args.seed)
    metrics_per_fold = []
    oof_sample_indices = []
    oof_fold_ids = []
    oof_labels = []
    oof_probabilities = []
    result_root = project_root / "RESULT" / args.dataset / args.task
    result_root.mkdir(parents=True, exist_ok=True)

    for fold_index, (outer_train_idx, outer_test_idx) in enumerate(
        outer_splits,
        start=1,
    ):
        fold_seed = args.seed + fold_index
        set_seed(fold_seed)
        inner_train_idx, validation_idx = make_train_validation_split(
            outer_train_idx,
            labels,
            args.val_ratio,
            fold_seed,
        )

        search_model = KBM(
            num_classes=num_classes,
            feature_list=feature_list,
            hidden=args.hidden,
            dim_embedding=args.dim_emb,
            d_state=args.d_state,
        ).to(device)
        search_train_loader = build_loader(
            features,
            labels,
            inner_train_idx,
            args.batchsize,
            shuffle=True,
            device=device,
        )
        validation_loader = build_loader(
            features,
            labels,
            validation_idx,
            args.batchsize,
            shuffle=False,
            device=device,
        )

        print(
            f"\nFold {fold_index}/{args.n_splits}: "
            f"train={len(inner_train_idx)}, "
            f"validation={len(validation_idx)}, "
            f"test={len(outer_test_idx)}"
        )
        best_epoch, best_validation_metrics = search_best_epoch(
            search_model,
            search_train_loader,
            validation_loader,
            labels[as_index_tensor(inner_train_idx)],
            num_classes,
            device,
            args.epochs,
            args.lr,
            args.min_lr,
            args.weight_decay,
            args.fusion_weight,
            args.auxiliary_weight,
            args.max_grad_norm,
            args.selection_metric,
            description=f"Fold {fold_index} epoch search",
        )
        del search_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        set_seed(fold_seed + 10000)
        final_model = KBM(
            num_classes=num_classes,
            feature_list=feature_list,
            hidden=args.hidden,
            dim_embedding=args.dim_emb,
            d_state=args.d_state,
        ).to(device)
        outer_train_loader = build_loader(
            features,
            labels,
            outer_train_idx,
            args.batchsize,
            shuffle=True,
            device=device,
        )
        outer_test_loader = build_loader(
            features,
            labels,
            outer_test_idx,
            args.batchsize,
            shuffle=False,
            device=device,
        )

        fit_fixed_epochs(
            final_model,
            outer_train_loader,
            labels[as_index_tensor(outer_train_idx)],
            num_classes,
            device,
            best_epoch,
            args.lr,
            args.min_lr,
            args.weight_decay,
            args.fusion_weight,
            args.auxiliary_weight,
            args.max_grad_norm,
            description=f"Fold {fold_index} final fit",
        )
        test_labels, test_logits = predict(
            final_model,
            outer_test_loader,
            device,
        )
        test_metrics = compute_all_metrics(test_labels, test_logits, num_classes)
        test_probabilities = torch.softmax(
            torch.from_numpy(test_logits),
            dim=-1,
        ).numpy()
        test_sample_indices = np.asarray(outer_test_idx, dtype=np.int64)

        # Keep the held-out predictions needed for ROC/PR curves and later
        # statistical analysis. The outer test fold is never used for training.
        np.savez_compressed(
            result_root / f"fold_{fold_index:02d}_test_predictions.npz",
            sample_indices=test_sample_indices,
            fold_ids=np.full(test_sample_indices.shape, fold_index, dtype=np.int64),
            y_true=test_labels.astype(np.int64),
            y_prob=test_probabilities.astype(np.float32),
        )
        oof_sample_indices.append(test_sample_indices)
        oof_fold_ids.append(
            np.full(test_sample_indices.shape, fold_index, dtype=np.int64)
        )
        oof_labels.append(test_labels.astype(np.int64))
        oof_probabilities.append(test_probabilities.astype(np.float32))

        fold_result = {
            "fold": fold_index,
            "best_epoch": best_epoch,
            "validation": best_validation_metrics,
            "test": test_metrics,
        }
        metrics_per_fold.append(fold_result)
        print(
            f"Fold {fold_index}: best_epoch={best_epoch}, "
            f"ACC={test_metrics['acc']:.4f}, "
            f"BAC={test_metrics['bac']:.4f}, "
            f"F1={test_metrics['f1']:.4f}, "
            f"AUC={test_metrics['auc']:.4f}"
        )

        if args.save_model:
            model_path = result_root / f"fold_{fold_index:02d}_model.pt"
            torch.save(
                {
                    "model_state_dict": final_model.state_dict(),
                    "feature_list": feature_list,
                    "modality_names": modality_names,
                    "label_map": label_map,
                    "args": vars(args),
                    "best_epoch": best_epoch,
                },
                model_path,
            )
        del final_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metric_names = ("acc", "sen", "spec", "f1", "bac", "auc")
    summary = {}
    for metric_name in metric_names:
        values = np.asarray(
            [fold["test"][metric_name] for fold in metrics_per_fold],
            dtype=np.float64,
        )
        summary[metric_name.upper()] = {
            "Mean": float(np.nanmean(values)),
            "Std": float(np.nanstd(values)),
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    class_names = np.asarray(
        [
            str(label)
            for label, _ in sorted(label_map.items(), key=lambda item: item[1])
        ],
        dtype=str,
    )
    oof_data = {
        "sample_indices": np.concatenate(oof_sample_indices, axis=0),
        "fold_ids": np.concatenate(oof_fold_ids, axis=0),
        "y_true": np.concatenate(oof_labels, axis=0),
        "y_prob": np.concatenate(oof_probabilities, axis=0),
        "class_names": class_names,
    }
    timestamped_oof_path = result_root / f"oof_predictions_{timestamp}.npz"
    np.savez_compressed(timestamped_oof_path, **oof_data)
    np.savez_compressed(result_root / "oof_predictions.npz", **oof_data)

    result = {
        "protocol": {
            "outer_cv": args.n_splits,
            "inner_validation_ratio": args.val_ratio,
            "selection_metric": args.selection_metric,
            "test_used_only_for_final_fold_evaluation": True,
        },
        "config": vars(args),
        "dataset": {
            "data_path": str(data_path),
            "feature_dict_path": str(feature_dict_path),
            "modality_names": modality_names,
            "label_map": {str(key): value for key, value in label_map.items()},
        },
        "artifacts": {
            "oof_predictions": str(result_root / "oof_predictions.npz"),
            "timestamped_oof_predictions": str(timestamped_oof_path),
        },
        "folds": metrics_per_fold,
        "summary": summary,
    }
    result_path = result_root / f"run_{timestamp}.json"
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    raw_metrics = np.asarray(
        [
            [
                fold["test"]["acc"],
                fold["test"]["sen"],
                fold["test"]["spec"],
                fold["test"]["f1"],
                fold["test"]["bac"],
                fold["test"]["auc"],
            ]
            for fold in metrics_per_fold
        ],
        dtype=np.float64,
    )
    np.save(result_root / "fold_raw_metrics.npy", raw_metrics)
    with (result_root / "fold_raw_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            [fold["test"] for fold in metrics_per_fold],
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n===== Outer cross-validation summary =====")
    for metric_name, values in summary.items():
        print(
            f"{metric_name}: Mean={values['Mean']:.4f}, "
            f"Std={values['Std']:.4f}"
        )
    print(f"Saved results to: {result_path}")
    print(f"Saved ROC inputs to: {result_root / 'oof_predictions.npz'}")


if __name__ == "__main__":
    main()
