"""Tests for the model factory and the registry promotion gate.

These cover the two pieces that decide *which model reaches production*:

  * src/models/factory.py    - rebuilds the right architecture from a checkpoint
  * scripts/promote_model.py - the accuracy gate a challenger must pass

A bug in either would ship the wrong model, which no amount of API testing
would catch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.promote_model import as_float, decide  # noqa: E402
from src.models.cnn import BaselineCNN  # noqa: E402
from src.models.factory import (  # noqa: E402
    BASELINE,
    RESNET18,
    SUPPORTED,
    build_from_config,
    count_total,
    count_trainable,
)


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def test_factory_builds_baseline(params):
    cfg = dict(params["model"])
    cfg["architecture"] = BASELINE
    model = build_from_config(cfg, params["data"])
    assert isinstance(model, BaselineCNN)
    assert count_trainable(model) == count_total(model)  # nothing frozen


def test_factory_defaults_to_baseline_for_old_checkpoints(params):
    """A checkpoint written before `architecture` existed must still load."""
    cfg = {k: v for k, v in params["model"].items() if k != "architecture"}
    assert isinstance(build_from_config(cfg, params["data"]), BaselineCNN)


def test_factory_rejects_unknown_architecture(params):
    cfg = dict(params["model"])
    cfg["architecture"] = "vit_giant_totally_real"
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_from_config(cfg, params["data"])


def test_supported_list_is_accurate():
    assert BASELINE in SUPPORTED and RESNET18 in SUPPORTED


@pytest.mark.parametrize("architecture", [BASELINE, RESNET18])
def test_every_architecture_returns_two_logits(params, architecture):
    cfg = dict(params["model"])
    cfg["architecture"] = architecture
    cfg["pretrained"] = False           # no network access in tests
    model = build_from_config(cfg, params["data"]).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 2)
    assert torch.allclose(
        torch.softmax(logits, 1).sum(1), torch.ones(2), atol=1e-5
    )


def test_resnet_freezing_leaves_only_the_head_trainable(params):
    """The whole point of freeze_backbone: backward touches ~1k params, not 11M."""
    cfg = dict(params["model"])
    cfg.update({"architecture": RESNET18, "pretrained": False, "freeze_backbone": True})
    model = build_from_config(cfg, params["data"])

    trainable, total = count_trainable(model), count_total(model)
    assert total > 10_000_000            # ResNet18 is ~11.2M parameters
    assert trainable < 5_000             # only the 2-class head
    assert trainable < total / 1000


def test_resnet_unfrozen_trains_everything(params):
    cfg = dict(params["model"])
    cfg.update({"architecture": RESNET18, "pretrained": False, "freeze_backbone": False})
    model = build_from_config(cfg, params["data"])
    assert count_trainable(model) == count_total(model)


def test_checkpoint_roundtrip_rebuilds_identical_model(params):
    """Save -> load must reproduce identical outputs for every architecture.

    This is the guarantee the serving path depends on.
    """
    for architecture in (BASELINE, RESNET18):
        cfg = dict(params["model"])
        cfg.update({"architecture": architecture, "pretrained": False})

        original = build_from_config(cfg, params["data"]).eval()
        state = {k: v.clone() for k, v in original.state_dict().items()}

        rebuilt = build_from_config(cfg, params["data"])
        rebuilt.load_state_dict(state)
        rebuilt.eval()

        batch = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            assert torch.equal(original(batch), rebuilt(batch)), architecture


# --------------------------------------------------------------------------- #
# promotion gate
# --------------------------------------------------------------------------- #
def version(number: int, accuracy: float) -> dict:
    return {"version": number, "test_accuracy": accuracy, "architecture": "x"}


def test_first_version_is_promoted_when_there_is_no_champion():
    promote, reason = decide(version(1, 0.72), None, min_delta=0.005)
    assert promote is True
    assert "no champion" in reason


def test_clearly_better_challenger_is_promoted():
    promote, _ = decide(version(2, 0.97), version(1, 0.72), min_delta=0.005)
    assert promote is True


def test_worse_challenger_is_rejected():
    promote, reason = decide(version(2, 0.65), version(1, 0.72), min_delta=0.005)
    assert promote is False
    assert "margin -0.0700" in reason


def test_marginally_better_challenger_is_rejected():
    """A 0.2 pp gain must not churn production when min_delta is 0.5 pp."""
    promote, _ = decide(version(2, 0.7220), version(1, 0.7200), min_delta=0.005)
    assert promote is False


def test_gate_boundary_behaviour():
    """Just under min_delta is rejected; comfortably over is promoted.

    Deliberately avoids asserting on an exactly-equal margin: 0.7050 - 0.7000
    is 0.005000000000000004 in IEEE-754, so an "exactly at the threshold" test
    would be asserting on float representation rather than on the gate.
    """
    champion = version(1, 0.7000)
    assert decide(version(2, 0.7040), champion, 0.005)[0] is False  # 0.4 pp < 0.5 pp
    assert decide(version(2, 0.7100), champion, 0.005)[0] is True   # 1.0 pp > 0.5 pp


def test_no_candidate_is_not_a_promotion():
    promote, reason = decide(None, version(1, 0.9), min_delta=0.005)
    assert promote is False
    assert "no candidate" in reason


def test_min_delta_zero_promotes_any_improvement():
    assert decide(version(2, 0.7201), version(1, 0.7200), 0.0)[0] is True


def test_reason_string_reports_both_accuracies():
    _, reason = decide(version(3, 0.9725), version(1, 0.7250), min_delta=0.005)
    assert "0.9725" in reason and "0.7250" in reason


# --------------------------------------------------------------------------- #
# tag parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [("0.9725", 0.9725), ("0", 0.0), (None, -1.0), ("", -1.0), ("not-a-number", -1.0)],
)
def test_as_float_survives_bad_tags(raw, expected):
    """Registry tags are strings written by hand or by other tools; a malformed
    one must not crash the promotion gate."""
    assert as_float(raw) == expected
