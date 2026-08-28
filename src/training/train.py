"""Train the baseline CNN and track everything in MLflow.

Pipeline position:  data/processed (DVC)  ->  THIS  ->  artifacts/model.pt + mlruns/

What gets logged to MLflow for every run:
  params    - every value from params.yaml (flattened) + dataset sizes + device
  metrics   - train/val loss and accuracy per epoch, plus final test metrics
  artifacts - loss & accuracy curves, confusion matrix, classification report,
              the serialized model, and the dataset split summary

The serialized artifact (artifacts/model.pt) is self-describing: it carries the
weights AND the class order AND the preprocessing config, so the FastAPI service
in M2 can rebuild the exact inference pipeline without guessing.

Run:  python train.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: no GUI needed, safe in CI/Docker

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import PROJECT_ROOT, load_params, resolve  # noqa: E402
from src.data.dataset import build_dataloaders  # noqa: E402
from src.models.cnn import build_model, count_parameters  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def flatten_params(params: dict, prefix: str = "") -> dict:
    """Flatten nested params.yaml into MLflow-friendly 'a.b.c' -> value pairs."""
    flat = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_params(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


def resolve_tracking_uri(mlflow_cfg: dict) -> str:
    """Make a repo-relative sqlite tracking URI absolute.

    'sqlite:///mlflow.db' is relative to the current working directory, which
    would scatter one database per directory you happen to launch from. Anchor
    it to the repo root so every run lands in the same store.
    """
    uri = mlflow_cfg["tracking_uri"]
    prefix = "sqlite:///"
    if uri.startswith(prefix):
        db_path = Path(uri[len(prefix):])
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path
        return prefix + db_path.as_posix()
    return uri


def set_seed(seed: int) -> None:
    """Seed every RNG that affects training, so runs are reproducible."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None) -> dict:
    """One pass over a loader. optimizer=None means evaluation mode."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, seen = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            seen += labels.size(0)

            if not training:
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / seen,
        "accuracy": correct / seen,
        "preds": all_preds,
        "labels": all_labels,
    }


# --------------------------------------------------------------------------- #
# plots (both are explicit M1 deliverables)
# --------------------------------------------------------------------------- #
def plot_curves(history: dict, out_path: Path) -> None:
    """Loss and accuracy curves for train vs validation."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, history["train_loss"], "o-", label="train")
    ax_loss.plot(epochs, history["val_loss"], "s-", label="validation")
    ax_loss.set_title("Loss curve")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_accuracy"], "o-", label="train")
    ax_acc.plot(epochs, history["val_accuracy"], "s-", label="validation")
    ax_acc.set_title("Accuracy curve")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.grid(alpha=0.3)
    ax_acc.legend()

    fig.suptitle("Baseline CNN - Cats vs Dogs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, classes: list, out_path: Path) -> None:
    """Annotated confusion matrix for the held-out test set."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=14,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    params = load_params()
    train_cfg = params["train"]
    model_cfg = params["model"]
    mlflow_cfg = params["mlflow"]

    set_seed(int(train_cfg["seed"]))

    # No GPU on this machine, so use every core: ~4x faster than the default.
    torch.set_num_threads(os.cpu_count() or 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    artifact_dir = resolve(model_cfg["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    loaders, classes = build_dataloaders(params)
    model = build_model(params).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    print(
        f"device={device}  threads={torch.get_num_threads()}  "
        f"trainable_params={count_parameters(model):,}"
    )
    print(
        f"train={len(loaders['train'].dataset)}  "
        f"val={len(loaders['val'].dataset)}  "
        f"test={len(loaders['test'].dataset)}",
        flush=True,
    )

    tracking_uri = resolve_tracking_uri(mlflow_cfg)
    mlflow.set_tracking_uri(tracking_uri)
    print(f"mlflow tracking_uri={tracking_uri}", flush=True)

    # Artifacts live on disk next to the database; create the experiment with an
    # explicit location the first time so re-runs always land in the same place.
    experiment_name = mlflow_cfg["experiment_name"]
    if mlflow.get_experiment_by_name(experiment_name) is None:
        artifact_root = (PROJECT_ROOT / mlflow_cfg["artifact_location"]).resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(experiment_name, artifact_location=artifact_root.as_uri())
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=mlflow_cfg["run_name"]) as run:
        print(f"MLflow run_id={run.info.run_id}", flush=True)

        # ---- parameters ----------------------------------------------------
        mlflow.log_params(flatten_params(params))
        mlflow.log_params(
            {
                "device": str(device),
                "torch_version": torch.__version__,
                "trainable_parameters": count_parameters(model),
                "n_train": len(loaders["train"].dataset),
                "n_val": len(loaders["val"].dataset),
                "n_test": len(loaders["test"].dataset),
            }
        )

        # ---- training loop ---------------------------------------------------
        history = {
            key: []
            for key in ("train_loss", "train_accuracy", "val_loss", "val_accuracy")
        }
        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        patience = int(train_cfg["early_stopping_patience"])
        epochs_without_improvement = 0

        for epoch in range(1, int(train_cfg["epochs"]) + 1):
            started = time.time()
            train_metrics = run_epoch(
                model, loaders["train"], criterion, device, optimizer
            )
            val_metrics = run_epoch(model, loaders["val"], criterion, device)
            duration = time.time() - started

            for prefix, source in (("train", train_metrics), ("val", val_metrics)):
                history[f"{prefix}_loss"].append(source["loss"])
                history[f"{prefix}_accuracy"].append(source["accuracy"])

            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "epoch_seconds": duration,
                },
                step=epoch,
            )

            marker = ""
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                # Keep the best weights in memory; restore them before testing.
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                best_epoch = epoch
                epochs_without_improvement = 0
                marker = "  <- best"
            else:
                epochs_without_improvement += 1

            print(
                f"epoch {epoch:>2}/{train_cfg['epochs']}  "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f}  "
                f"({duration:.0f}s){marker}",
                flush=True,
            )

            if epochs_without_improvement >= patience:
                print(
                    f"early stopping: val_loss did not improve for {patience} epochs",
                    flush=True,
                )
                break

        # ---- restore the best weights before final evaluation -----------------
        if best_state is not None:
            model.load_state_dict(best_state)
        print(f"best epoch: {best_epoch} (val_loss={best_val_loss:.4f})", flush=True)

        # ---- final evaluation on the held-out test split ----------------------
        test_metrics = run_epoch(model, loaders["test"], criterion, device)
        y_true = np.array(test_metrics["labels"])
        y_pred = np.array(test_metrics["preds"])

        final = {
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "test_recall": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "test_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }
        mlflow.log_metrics(final)
        print(
            "\ntest metrics: "
            + json.dumps({k: round(float(v), 4) for k, v in final.items()}, indent=2),
            flush=True,
        )

        # ---- artifacts ---------------------------------------------------------
        curves_path = artifact_dir / "training_curves.png"
        cm_path = artifact_dir / "confusion_matrix.png"
        report_path = artifact_dir / "classification_report.txt"
        metrics_path = artifact_dir / "metrics.json"
        model_path = artifact_dir / model_cfg["artifact_name"]

        plot_curves(history, curves_path)

        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, classes, cm_path)

        report = classification_report(y_true, y_pred, target_names=classes, digits=4)
        report_path.write_text(report, encoding="utf-8")
        print("\n" + report, flush=True)

        # Log confusion-matrix cells as metrics too, so they are queryable in the UI.
        mlflow.log_metrics(
            {
                "cm_true_cat_pred_cat": int(cm[0, 0]),
                "cm_true_cat_pred_dog": int(cm[0, 1]),
                "cm_true_dog_pred_cat": int(cm[1, 0]),
                "cm_true_dog_pred_dog": int(cm[1, 1]),
            }
        )

        metrics_path.write_text(
            json.dumps(
                {
                    "history": history,
                    "final": {k: float(v) for k, v in final.items()},
                    "confusion_matrix": cm.tolist(),
                    "classes": classes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # ---- serialized model artifact ------------------------------------------
        # Self-describing on purpose: weights + class order + preprocessing config,
        # so the API in M2 rebuilds the identical inference pipeline.
        torch.save(
            {
                "state_dict": model.state_dict(),
                "classes": classes,
                "model_config": model_cfg,
                "data_config": {
                    "image_size": params["data"]["image_size"],
                    "channels": params["data"]["channels"],
                    "normalize": params["data"]["normalize"],
                },
                "metrics": {k: float(v) for k, v in final.items()},
                "mlflow_run_id": run.info.run_id,
            },
            model_path,
        )
        print(
            f"\nmodel artifact -> {model_path} "
            f"({model_path.stat().st_size / 1e6:.2f} MB)",
            flush=True,
        )

        for path in (curves_path, cm_path, report_path, metrics_path, model_path):
            mlflow.log_artifact(str(path))
        split_summary = resolve(params["data"]["processed_dir"]) / "split_summary.json"
        if split_summary.exists():
            mlflow.log_artifact(str(split_summary))
        mlflow.log_artifact(str(PROJECT_ROOT / "params.yaml"))

        print(f"\nMLflow run complete: {run.info.run_id}")
        print("View with:  python -m mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    main()
