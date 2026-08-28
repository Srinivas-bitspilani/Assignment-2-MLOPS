"""Register an existing checkpoint into the MLflow Model Registry.

Training runs register themselves, but this script covers two real cases:

  * a run that finished before registry support existed (backfill)
  * a run whose registration step failed while the training itself was fine --
    re-registering costs seconds, retraining costs hours

It resumes the original MLflow run so the registered version stays attached to
the parameters, metrics and artifacts that produced it, rather than creating a
detached copy.

Usage
    python scripts/register_model.py --run-id d604ad0f... --checkpoint artifacts/model.pt
    python scripts/register_model.py --all      # register every .pt in artifacts/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import load_params  # noqa: E402
from src.models.factory import build_from_config  # noqa: E402
from src.training.train import (  # noqa: E402
    register_model_version,
    resolve_tracking_uri,
)


def register_one(checkpoint_path: Path, run_id: str | None, mlflow_cfg: dict) -> bool:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    run_id = run_id or checkpoint.get("mlflow_run_id")
    if not run_id:
        print(f"  {checkpoint_path.name}: no run id available, skipping")
        return False

    model_cfg = dict(checkpoint["model_config"])
    model_cfg["architecture"] = checkpoint.get(
        "architecture", model_cfg.get("architecture", "baseline_cnn")
    )
    model_cfg["pretrained"] = False        # weights come from the checkpoint

    model = build_from_config(model_cfg, checkpoint["data_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    metrics = {k: float(v) for k, v in checkpoint.get("metrics", {}).items()}
    classes = list(checkpoint["classes"])

    print(f"  {checkpoint_path.name}: architecture={model_cfg['architecture']} "
          f"accuracy={metrics.get('test_accuracy')} run={run_id[:8]}")

    # Resuming the run keeps the version tied to its own params and metrics.
    with mlflow.start_run(run_id=run_id):
        name, version = register_model_version(
            model, run_id, mlflow_cfg, model_cfg, metrics, classes
        )
        mlflow.set_tags(
            {"registered_model": name, "registered_version": version}
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Register a checkpoint")
    parser.add_argument("--checkpoint", help="Path to a .pt checkpoint")
    parser.add_argument("--run-id", help="Override the run id in the checkpoint")
    parser.add_argument("--all", action="store_true",
                        help="Register every .pt file in the artifacts directory")
    args = parser.parse_args()

    params = load_params()
    mlflow_cfg = params["mlflow"]
    mlflow.set_tracking_uri(resolve_tracking_uri(mlflow_cfg))

    artifact_dir = PROJECT_ROOT / params["model"]["artifact_dir"]

    if args.all:
        targets = sorted(artifact_dir.glob("*.pt"))
    elif args.checkpoint:
        targets = [Path(args.checkpoint)]
        if not targets[0].is_absolute():
            targets = [PROJECT_ROOT / args.checkpoint]
    else:
        parser.error("pass --checkpoint or --all")

    if not targets:
        print(f"No checkpoints found in {artifact_dir}")
        return 1

    print(f"Registering {len(targets)} checkpoint(s) into "
          f"'{mlflow_cfg['registered_model_name']}'")
    ok = 0
    for path in targets:
        try:
            ok += register_one(path, args.run_id, mlflow_cfg)
        except Exception as exc:
            print(f"  {path.name}: FAILED -> {exc}")

    print(f"\nregistered {ok}/{len(targets)}")
    print("Next:  python scripts/promote_model.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
