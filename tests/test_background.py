"""Tests for smokesight.background."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from smokesight.background import VALID_METHODS, background
from smokesight.calibrate import calibrate


def _cal(synthetic_video: Path, minimal_config: Dict[str, Any]):
    return calibrate(synthetic_video, minimal_config, progress=False)


def test_default_returns_correct_shapes(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    bg = background(_cal(synthetic_video, minimal_config), n_frames=20)
    assert bg.L0.shape == (64, 64, 1)
    assert bg.sigma_L0.shape == (64, 64, 1)
    assert bg.confidence.shape == (64, 64)
    assert bg.method == "temporal_median"
    assert bg.n_frames_used == 20


def test_confidence_in_unit_range(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    bg = background(_cal(synthetic_video, minimal_config), n_frames=20)
    assert np.all(bg.confidence >= 0.0)
    assert np.all(bg.confidence <= 1.0)


@pytest.mark.parametrize(
    "method", ["temporal_median", "temporal_mean", "percentile_10"]
)
def test_methods_dont_raise(
    method: str, synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    bg = background(_cal(synthetic_video, minimal_config), n_frames=20, method=method)
    assert bg.method == method
    assert np.all(np.isfinite(bg.L0))
    assert np.all(bg.sigma_L0 >= 0)


def test_mask_zeroes_confidence_where_set(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    cal = _cal(synthetic_video, minimal_config)
    mask = np.zeros((64, 64), dtype=bool)
    mask[0:10, 0:10] = True  # mark a corner as known-static
    bg = background(cal, n_frames=20, mask=mask)
    assert np.all(bg.confidence[0:10, 0:10] == 0.0)
    # outside the mask, confidence should still be a normal value somewhere
    assert bg.confidence[30:40, 30:40].max() > 0.0


def test_n_frames_too_large_raises(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    cal = _cal(synthetic_video, minimal_config)
    with pytest.raises(ValueError, match="exceeds total frames"):
        background(cal, n_frames=cal.L.shape[0] + 1)


def test_unknown_method_raises(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        background(_cal(synthetic_video, minimal_config), method="bogus")


def test_valid_methods_constant_lists_all_branches() -> None:
    """The dispatcher and the public list must agree."""
    assert set(VALID_METHODS) == {
        "temporal_median",
        "temporal_mean",
        "gmm",
        "percentile_10",
    }


def test_gmm_method_runs_when_sklearn_available(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    """GMM method exists as a code path but is slow; run it on a small
    crop so the test stays fast. Skips cleanly if sklearn isn't installed."""
    pytest.importorskip("sklearn")
    cal = _cal(synthetic_video, minimal_config)
    # crop down to keep the per-pixel GMM affordable in the test suite
    cal.L = cal.L[:20, :8, :8, :]
    bg = background(cal, n_frames=20, method="gmm")
    assert bg.method == "gmm"
    assert bg.L0.shape == (8, 8, 1)
    assert np.all((bg.confidence >= 0.0) & (bg.confidence <= 1.0))


def test_gmm_method_errors_without_sklearn(monkeypatch: pytest.MonkeyPatch) -> None:
    """If sklearn isn't importable, the gmm method raises a friendly
    ImportError pointing at the install command rather than the bare
    ModuleNotFoundError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("sklearn"):
            raise ImportError("no sklearn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from smokesight._atmos import IdentityAtmos
    from smokesight._results import CalibrationResult
    from smokesight._sensor import SensorModel

    L = np.ones((5, 8, 8, 1), dtype=np.float32)
    cal = CalibrationResult(
        L=L,
        sigma_L=np.full_like(L, 0.01),
        metadata={},
        sensor=SensorModel.from_config(
            {"sensor": {"gain": 0.012, "bit_depth": 14, "ner": 0.002}}
        ),
        atmos=IdentityAtmos(),
    )
    with pytest.raises(ImportError, match="scikit-learn"):
        background(cal, n_frames=5, method="gmm")
