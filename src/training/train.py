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
from src.models.factory import (  # noqa: E402
    build_from_config,
    count_total,
    count_trainable,
)


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


def parse_args(argv=None):
    """CLI overrides for params.yaml.

    params.yaml stays the canonical config; these flags let a second candidate
    be trained without editing it, and every override is logged to MLflow so
    the run is still fully described by its own parameters.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train a cats-vs-dogs classifier")
    parser.add_argument("--architecture", choices=["baseline_cnn", "resnet18_finetune"],
                        help="Override model.architecture")
    parser.add_argument("--epochs", type=int, help="Override train.epochs")
    parser.add_argument("--learning-rate", type=float, help="Override train.learning_rate")
    parser.add_argument("--run-name", help="Override mlflow.run_name")
    parser.add_argument("--artifact-name", help="Override model.artifact_name")
    parser.add_argument("--no-register", action="store_true",
                        help="Skip MLflow Model Registry registration")
    return parser.parse_args(argv)


def apply_overrides(params: dict, args) -> dict:
    """Fold CLI overrides into the params dict and report what changed."""
    changed = {}
    if args.architecture:
        params["model"]["architecture"] = args.architecture
        params["model"]["name"] = args.architecture
        changed["model.architecture"] = args.architecture
    if args.epochs:
        params["train"]["epochs"] = args.epochs
        changed["train.epochs"] = args.epochs
    if args.learning_rate:
        params["train"]["learning_rate"] = args.learning_rate
        changed["train.learning_rate"] = args.learning_rate
    if args.run_name:
        params["mlflow"]["run_name"] = args.run_name
        changed["mlflow.run_name"] = args.run_name
    if args.artifact_name:
        params["model"]["artifact_name"] = args.artifact_name
        changed["model.artifact_name"] = args.artifact_name

    if changed:
        print("CLI overrides applied:", json.dumps(changed), flush=True)
    return params


def register_model_version(
    model, run_id: str, mlflow_cfg: dict, model_cfg: dict, final: dict, classes: list
):
    """Log the model as an MLflow flavour and register a new registry version.

    The new version is tagged with everything needed to compare candidates
    later and given the 'challenger' alias. Promotion to 'champion' is a
    separate, gated decision made by scripts/promote_model.py -- a training run
    never promotes itself.
    """
    from mlflow.tracking import MlflowClient

    name = mlflow_cfg["registered_model_name"]

    # MLflow 3.x defaults to the traced 'pt2' format, which requires an
    # input_example so it can trace model.forward. Supplying one is the better
    # fix anyway: it also gives the registered version a model signature.
    example = torch.randn(1, 3, 224, 224).numpy()
    try:
        info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            registered_model_name=name,
            input_example=example,
        )
    except Exception as exc:
        # Some architectures are not cleanly traceable; fall back to the
        # pickle-based flavour rather than losing the registration.
        print(f"  pt2 logging failed ({exc}); retrying as pickle", flush=True)
        info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            registered_model_name=name,
            serialization_format="pickle",
        )
    print(f"logged model flavour -> {info.model_uri}", flush=True)

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{name}'")
    version = max(versions, key=lambda v: int(v.version))

    for key, value in {
        "architecture": model_cfg.get("architecture", "baseline_cnn"),
        "test_accuracy": f"{final['test_accuracy']:.4f}",
        "test_f1": f"{final['test_f1']:.4f}",
        "test_loss": f"{final['test_loss']:.4f}",
        "best_epoch": str(int(final["best_epoch"])),
        "classes": ",".join(classes),
        "source_run_id": run_id,
    }.items():
        client.set_model_version_tag(name, version.version, key, value)

    client.update_model_version(
        name=name,
        version=version.version,
        description=(
            f"{model_cfg.get('architecture')} | test_accuracy="
            f"{final['test_accuracy']:.4f} | run={run_id}"
        ),
    )

    # Every new candidate becomes the challenger; promotion is gated elsewhere.
    client.set_registered_model_alias(
        name, mlflow_cfg["challenger_alias"], version.version
    )

    print(
        f"registered {name} version {version.version} "
        f"(alias '{mlflow_cfg['challenger_alias']}')",
        flush=True,
    )
    return name, version.version


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
def main(argv=None) -> None:
    args = parse_args(argv)
    params = apply_overrides(load_params(), args)
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
    model = build_from_config(model_cfg, params["data"]).to(device)
    criterion = nn.CrossEntropyLoss()
    # Only optimise parameters that actually require grad: for a frozen
    # backbone that is just the new head.
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    print(
        f"architecture={model_cfg.get('architecture')}  device={device}  "
        f"threads={torch.get_num_threads()}"
    )
    print(
        f"parameters: {count_trainable(model):,} trainable "
        f"/ {count_total(model):,} total"
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
                "trainable_parameters": count_trainable(model),
                "total_parameters": count_total(model),
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
        # Artefact names are namespaced per candidate so training a second
        # model never overwrites the first one's evidence. The baseline keeps
        # the original unprefixed filenames.
        stem = Path(model_cfg["artifact_name"]).stem
        prefix = "" if stem == "model" else f"{stem}_"

        curves_path = artifact_dir / f"{prefix}training_curves.png"
        cm_path = artifact_dir / f"{prefix}confusion_matrix.png"
        report_path = artifact_dir / f"{prefix}classification_report.txt"
        metrics_path = artifact_dir / f"{prefix}metrics.json"
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
                # architecture is what lets api/model_loader.py rebuild the
                # right network for whichever version was promoted.
                "architecture": model_cfg.get("architecture", "baseline_cnn"),
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

        # ---- MLflow Model Registry -------------------------------------------
        if not args.no_register:
            try:
                registered_name, version = register_model_version(
                    model, run.info.run_id, mlflow_cfg, model_cfg, final, classes
                )
                mlflow.set_tags(
                    {
                        "registered_model": registered_name,
                        "registered_version": version,
                    }
                )
            except Exception as exc:  # never lose a training run over this
                print(f"WARNING: model registration failed: {exc}", flush=True)

        print(f"\nMLflow run complete: {run.info.run_id}")
        print(
            "View with:  python -m mlflow ui --backend-store-uri sqlite:///mlflow.db"
        )
        print("Promote with:  python scripts/promote_model.py")


if __name__ == "__main__":
    main()
