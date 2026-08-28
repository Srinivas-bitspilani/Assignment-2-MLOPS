"""Central config loader.

Every stage of the pipeline (download, preprocess, train, evaluate, serve)
reads its settings from params.yaml through this one helper, so there is a
single source of truth and nothing is hardcoded.
"""

from pathlib import Path

import yaml

# Repository root = two levels up from this file (src/config.py -> src -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params(path: Path | str | None = None) -> dict:
    """Load params.yaml and return it as a plain dict."""
    path = Path(path) if path else PARAMS_PATH
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(relative_path: str | Path) -> Path:
    """Turn a params.yaml path (always repo-relative) into an absolute path."""
    return (PROJECT_ROOT / Path(relative_path)).resolve()


if __name__ == "__main__":
    import json

    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(json.dumps(load_params(), indent=2))
