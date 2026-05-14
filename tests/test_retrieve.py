"""Tests for smokesight.retrieve.

The headline test is `test_tau_recovers_ground_truth` -- the synthetic
plume in conftest has known peak tau=0.5 and the pipeline must come
back within the spec's 10% tolerance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from smokesight.background import background
from smokesight.calibrate import calibrate
from smokesight.retrieve import retrieve


def _pipeline(synthetic_video: Path, minimal_config: Dict[str, Any]):
    cal = calibrate(synthetic_video, minimal_config, progress=False)
    bg = background(cal, n_frames=20, method="temporal_median")
    return cal, bg


def test_tau_recovers_ground_truth(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    """Peak tau should land within ±10% of the injected 0.5.

    The fixture's first 20 frames are plume-free (used to build L0); the
    remaining 30 frames carry the plume, and that's where we check tau.
    """
    cal, bg = _pipeline(synthetic_video, minimal_config)
    res = retrieve(cal, bg, min_confidence=0.0)
    plume_frames = res.tau[20:, 32, 32]  # plume present from frame 20 onward
    plume_frames = plume_frames[~np.isnan(plume_frames)]
    assert plume_frames.size > 0, "no valid tau pixels at the plume centre"
    assert float(plume_frames.mean()) == pytest.approx(0.5, rel=0.10)


def test_sigma_tau_finite_where_valid_nan_where_masked(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    cal, bg = _pipeline(synthetic_video, minimal_config)
    res = retrieve(cal, bg, min_confidence=0.0)
    valid = res.mask.astype(bool)
    assert np.all(np.isfinite(res.sigma_tau[valid]))
    assert np.all(np.isnan(res.sigma_tau[~valid]))


def test_low_confidence_pixels_get_masked(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    cal, bg = _pipeline(synthetic_video, minimal_config)
    bg.confidence[10:20, 10:20] = 0.0  # force a corner to fail the gate
    res = retrieve(cal, bg, min_confidence=0.5)
    assert np.all(np.isnan(res.tau[:, 10:20, 10:20]))


def test_tau_max_caps_output(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    cal, bg = _pipeline(synthetic_video, minimal_config)
    res = retrieve(cal, bg, tau_max=0.1, min_confidence=0.0)
    valid = res.mask.astype(bool)
    if valid.any():
        assert float(np.nanmax(res.tau[valid])) <= 0.1


def test_invalid_min_confidence_rejected(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    cal, bg = _pipeline(synthetic_video, minimal_config)
    with pytest.raises(ValueError, match="min_confidence"):
        retrieve(cal, bg, min_confidence=-0.1)
    with pytest.raises(ValueError, match="min_confidence"):
        retrieve(cal, bg, min_confidence=2.0)


def test_shape_mismatch_rejected(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    cal, bg = _pipeline(synthetic_video, minimal_config)
    bg.L0 = bg.L0[:32, :32, :]  # corrupt the shape
    bg.sigma_L0 = bg.sigma_L0[:32, :32, :]
    bg.confidence = bg.confidence[:32, :32]
    with pytest.raises(ValueError, match="does not match"):
        retrieve(cal, bg, min_confidence=0.5)


def test_metadata_inherits_and_extends(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    cal, bg = _pipeline(synthetic_video, minimal_config)
    res = retrieve(cal, bg, tau_max=1.5, min_confidence=0.5)
    # cal metadata fields preserved
    assert res.metadata["height"] == cal.metadata["height"]
    # retrieval params added
    assert res.metadata["tau_max"] == 1.5
    assert res.metadata["background_method"] == "temporal_median"
