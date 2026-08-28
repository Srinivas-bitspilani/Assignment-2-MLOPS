"""Gated promotion in the MLflow Model Registry.

A training run never promotes itself. Every run registers a new version and
tags it `challenger`. This script is the gate that decides whether that
challenger replaces the `champion`:

    promote if   challenger.test_accuracy  >  champion.test_accuracy + min_delta

If it wins, the script:
  1. moves the `champion` alias to the winning version,
  2. exports that version's checkpoint to artifacts/model.pt -- the file the
     Docker image bakes in, so **only a promoted model can ever be served**,
  3. writes artifacts/model_registry.json as an auditable record.

If it loses, nothing changes and the exit code is still 0 (a challenger losing
is a normal outcome, not a pipeline failure). Use --fail-on-reject in CI if you
want the opposite.

Usage
    python scripts/promote_model.py                  # evaluate + promote if better
    python scripts/promote_model.py --dry-run        # report only
    python scripts/promote_model.py --version 2      # force a specific version
    python scripts/promote_model.py --show           # print the registry table
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import load_params  # noqa: E402
from src.training.train import resolve_tracking_uri  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def as_float(value, default: float = -1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def version_row(version, alias_map: dict[int, list[str]] | None = None) -> dict:
    tags = version.tags or {}
    number = int(version.version)
    # search_model_versions() does not populate .aliases, so the authoritative
    # alias -> version mapping is read from the registered model itself.
    aliases = (alias_map or {}).get(number) or list(version.aliases or [])
    return {
        "version": number,
        "architecture": tags.get("architecture", "?"),
        "test_accuracy": as_float(tags.get("test_accuracy")),
        "test_f1": as_float(tags.get("test_f1")),
        "test_loss": as_float(tags.get("test_loss")),
        "best_epoch": tags.get("best_epoch", "?"),
        "source_run_id": tags.get("source_run_id", ""),
        "aliases": sorted(aliases),
        "created": version.creation_timestamp,
    }


def decide(candidate: dict | None, champion: dict | None, min_delta: float):
    """The promotion gate, isolated so it can be unit-tested.

    Returns (promote: bool, reason: str).

    Rules:
      * no candidate            -> nothing to do
      * no champion yet         -> promote the candidate
      * candidate beats champion by more than min_delta -> promote
      * otherwise               -> reject (a normal outcome, not an error)
    """
    if candidate is None:
        return False, "no candidate version available"

    if champion is None:
        return True, "no champion yet: promoting the best available version"

    margin = candidate["test_accuracy"] - champion["test_accuracy"]
    promote = margin > min_delta
    reason = (
        f"challenger v{candidate['version']} accuracy "
        f"{candidate['test_accuracy']:.4f} vs champion "
        f"v{champion['version']} {champion['test_accuracy']:.4f} "
        f"(margin {margin:+.4f}, required > {min_delta})"
    )
    return promote, reason


def alias_map(client: MlflowClient, name: str) -> dict[int, list[str]]:
    """version number -> aliases pointing at it, read from the registered model."""
    mapping: dict[int, list[str]] = {}
    try:
        model = client.get_registered_model(name)
    except Exception:
        return mapping
    for alias in getattr(model, "aliases", None) or []:
        # Depending on the MLflow version this is either a list of objects with
        # .alias/.version or a plain {alias: version} dict.
        if isinstance(alias, str):
            number = int((model.aliases or {})[alias])
            mapping.setdefault(number, []).append(alias)
        else:
            mapping.setdefault(int(alias.version), []).append(alias.alias)
    return mapping


def load_registry(client: MlflowClient, name: str) -> list[dict]:
    versions = client.search_model_versions(f"name='{name}'")
    aliases = alias_map(client, name)
    return sorted(
        (version_row(v, aliases) for v in versions), key=lambda r: r["version"]
    )


def print_table(rows: list[dict], champion_alias: str) -> None:
    print(f"{'ver':>4}  {'architecture':<18} {'test_acc':>9} {'test_f1':>8} "
          f"{'loss':>7}  {'aliases':<22} run")
    print("-" * 96)
    for row in rows:
        aliases = ",".join(row["aliases"]) or "-"
        star = " <<" if champion_alias in row["aliases"] else ""
        print(f"{row['version']:>4}  {row['architecture']:<18} "
              f"{row['test_accuracy']:>9.4f} {row['test_f1']:>8.4f} "
              f"{row['test_loss']:>7.4f}  {aliases:<22} "
              f"{row['source_run_id'][:8]}{star}")


def export_checkpoint(client: MlflowClient, row: dict, target: Path) -> bool:
    """Copy the champion version's raw checkpoint to `target`.

    The .pt checkpoint is logged as a plain run artifact by train.py, and it is
    what api/model_loader.py consumes. Exporting it here is what ties
    "promoted in the registry" to "actually served".
    """
    run_id = row["source_run_id"]
    if not run_id:
        print("  cannot export: version has no source_run_id tag")
        return False

    candidates = [
        a.path for a in client.list_artifacts(run_id) if a.path.endswith(".pt")
    ]
    if not candidates:
        print(f"  cannot export: no .pt artifact on run {run_id[:8]}")
        return False

    local = client.download_artifacts(run_id, candidates[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local, target)
    size_mb = target.stat().st_size / 1e6
    print(f"  exported {candidates[0]} -> {target.name} ({size_mb:.2f} MB)")
    return True


def write_record(
    rows: list[dict], name: str, champion: dict | None, decision: dict, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "registered_model": name,
                "total_versions": len(rows),
                "champion": champion,
                "decision": decision,
                "versions": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  registry record -> {path}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a registry version")
    parser.add_argument("--min-delta", type=float, default=0.005,
                        help="Accuracy a challenger must beat the champion by")
    parser.add_argument("--version", type=int,
                        help="Force-promote this version, skipping the gate")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the decision without changing anything")
    parser.add_argument("--show", action="store_true",
                        help="Print the registry table and exit")
    parser.add_argument("--fail-on-reject", action="store_true",
                        help="Exit 1 when the challenger is rejected")
    args = parser.parse_args()

    params = load_params()
    mlflow_cfg = params["mlflow"]
    name = mlflow_cfg["registered_model_name"]
    champion_alias = mlflow_cfg["champion_alias"]

    mlflow.set_tracking_uri(resolve_tracking_uri(mlflow_cfg))
    client = MlflowClient()

    rows = load_registry(client, name)
    if not rows:
        print(f"No versions registered under '{name}'. Run training first.")
        return 1

    print("=" * 96)
    print(f"MLFLOW MODEL REGISTRY  '{name}'")
    print("=" * 96)
    print_table(rows, champion_alias)
    print()

    if args.show:
        return 0

    champion = next((r for r in rows if champion_alias in r["aliases"]), None)

    # ---- pick the candidate ------------------------------------------------
    if args.version:
        candidate = next((r for r in rows if r["version"] == args.version), None)
        if candidate is None:
            print(f"Version {args.version} not found.")
            return 1
        reason = f"forced by --version {args.version}"
        promote = True
    else:
        # Best non-champion version by accuracy.
        others = [r for r in rows if r is not champion]
        candidate = max(others, key=lambda r: r["test_accuracy"], default=None)
        if candidate is None:
            print("Only one version exists and it is already champion.")
            return 0

        promote, reason = decide(candidate, champion, args.min_delta)

    print("DECISION")
    print(f"  candidate : v{candidate['version']} ({candidate['architecture']}) "
          f"acc={candidate['test_accuracy']:.4f}")
    print(f"  champion  : "
          + (f"v{champion['version']} ({champion['architecture']}) "
             f"acc={champion['test_accuracy']:.4f}" if champion else "none"))
    print(f"  reason    : {reason}")
    print(f"  outcome   : {'PROMOTE' if promote else 'REJECT'}"
          + ("  (dry run, nothing changed)" if args.dry_run else ""))
    print()

    decision = {
        "candidate_version": candidate["version"],
        "candidate_accuracy": candidate["test_accuracy"],
        "champion_version_before": champion["version"] if champion else None,
        "champion_accuracy_before": champion["test_accuracy"] if champion else None,
        "min_delta": args.min_delta,
        "promoted": bool(promote and not args.dry_run),
        "reason": reason,
    }

    record_path = PROJECT_ROOT / params["model"]["artifact_dir"] / "model_registry.json"

    if promote and not args.dry_run:
        client.set_registered_model_alias(name, champion_alias, candidate["version"])
        print(f"  alias '{champion_alias}' -> v{candidate['version']}")

        # Once a version is champion it is no longer a challenger: leaving both
        # aliases on one version makes the registry ambiguous to read.
        challenger_alias = mlflow_cfg.get("challenger_alias")
        if challenger_alias:
            try:
                client.delete_registered_model_alias(name, challenger_alias)
                print(f"  alias '{challenger_alias}' cleared (it is now champion)")
            except Exception:
                pass

        target = (
            PROJECT_ROOT
            / params["model"]["artifact_dir"]
            / "model.pt"          # the filename the Docker image bakes in
        )
        export_checkpoint(client, candidate, target)

        rows = load_registry(client, name)          # refresh aliases
        champion = next((r for r in rows if champion_alias in r["aliases"]), None)
        print()
        print_table(rows, champion_alias)
        print()

    write_record(rows, name, champion, decision, record_path)
    print("=" * 96)

    if not promote and args.fail_on_reject:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
