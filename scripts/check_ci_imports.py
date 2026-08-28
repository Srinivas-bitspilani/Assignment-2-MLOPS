"""Run the test suite as CI sees it: training-only dependencies unavailable.

Usage:  python scripts/check_ci_imports.py

Run this before pushing. It catches the class of failure where a test imports a
training-time module (mlflow, scikit-learn, matplotlib, dvc) that CI does not
install, which passes locally and fails in CI.

CI installs requirements-api.txt + requirements-dev.txt, which deliberately
exclude mlflow, scikit-learn, matplotlib and dvc. Locally those ARE installed,
so a test that imports one passes here and fails in CI. This makes them
unimportable to reproduce the CI environment before pushing.

The finder returns a spec whose loader raises, rather than raising from
find_spec itself: code that merely *probes* for an optional module (torch's
dynamo does this for pandas) then behaves normally, while a real import fails
exactly as it would in CI.
"""

import importlib.abc
import importlib.machinery
import sys

BLOCKED = {"mlflow", "sklearn", "matplotlib", "dvc"}


class FailingLoader(importlib.abc.Loader):
    def create_module(self, spec):
        raise ModuleNotFoundError(
            f"No module named {spec.name!r} (not installed in CI serving env)"
        )

    def exec_module(self, module):
        raise ModuleNotFoundError(f"No module named {module.__name__!r}")


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            return importlib.machinery.ModuleSpec(name, FailingLoader())
        return None


for module in list(sys.modules):
    if module.split(".")[0] in BLOCKED:
        del sys.modules[module]

sys.meta_path.insert(0, Blocker())

import pytest  # noqa: E402

print("Hiding training-only imports:", sorted(BLOCKED))
print()
sys.exit(pytest.main(["-q", "--no-header", "-p", "no:cacheprovider"]))
