"""The model-promotion gate, as pure logic.

Deliberately free of any MLflow import. The gate is the rule that decides what
reaches production, so it must be unit-testable in the serving environment,
where MLflow is not installed (it is a training-time dependency, excluded from
requirements-api.txt to keep the image small).

scripts/promote_model.py wires this to the real registry.
"""

from __future__ import annotations


def as_float(value, default: float = -1.0) -> float:
    """Parse a registry tag into a float.

    Registry tags are strings, sometimes written by other tools or by hand, so
    a malformed one must degrade to `default` rather than crash the gate.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def decide(
    candidate: dict | None, champion: dict | None, min_delta: float
) -> tuple[bool, str]:
    """Decide whether `candidate` should replace `champion`.

    Returns (promote, reason).

    Rules:
      * no candidate      -> nothing to do
      * no champion yet   -> promote the candidate
      * candidate beats the champion by more than min_delta -> promote
      * otherwise         -> reject, which is a normal outcome, not an error

    min_delta exists so a statistically meaningless gain (say 0.2 pp) does not
    churn production.
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
