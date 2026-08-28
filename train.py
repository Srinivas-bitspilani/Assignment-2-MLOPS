"""Project entrypoint for training.

Thin wrapper so the whole training pipeline runs with `python train.py` from
the repository root (and later from a DVC stage or a CI job).
"""

from src.training.train import main

if __name__ == "__main__":
    main()
