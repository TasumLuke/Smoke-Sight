"""Tests for smokesight.dynamics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from smokesight._results import RetrievalResult
from smokesight.background import background
from smokesight.calibrate import calibrate
from smokesight.dynamics import dynamics
from smokesight.retrieve import retrieve


def _retrieval(synthetic_video: Path, minimal_config: Dict[str, Any]):
    cal = calibrate(synthetic_video, minimal_config, progress=False)
    bg = background(cal, n_frames=20)
    return retrieve(cal, bg, min_confidence=0.0)


def _drifting_plume_result(t: int = 30, h: int = 64, w: int = 64) -> RetrievalResult:
    """Build a RetrievalResult with a Gaussian that drifts upward over time
    so we have a controllable rise velocity ground truth."""
    yy, xx = np.mgrid[0:h, 0:w]
    tau = np.zeros((t, h, w), dtype=np.float32)
    sigma_tau = np.full_like(tau, 0.01)
    mask = np.ones_like(tau)
    drift_per_frame = 0.5  # pixels/frame
    sigma = 5.0
    cy0 = 50.0
    for i in range(t):
        cy = cy0 - drift_per_frame * i  # moving "up" = decreasing row index
        cx = 32.0
        gauss = 0.5 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
        tau[i] = gauss
    return RetrievalResult(
        tau=tau,
        sigma_tau=sigma_tau,
        mask=mask,
        metadata={"fps": 25.0, "pixel_scale": 0.25},
    )


def test_centroid_track_has_one_row_per_frame() -> None:
    res = _drifting_plume_result()
    dyn = dynamics(res)
    assert dyn.centroid_track.shape == (res.tau.shape[0], 2)


def test_rise_velocity_sign_matches_drift_direction() -> None:
    """Plume centroid moving to smaller y -> negative slope in y(t) ->
    negative rise_velocity (in the image frame). The sign just has to be
    consistent with the drift."""
    res = _drifting_plume_result()
    dyn = dynamics(res)
    assert np.isfinite(dyn.rise_velocity)
    assert dyn.rise_velocity < 0  # y decreases over time in our setup


def test_rise_velocity_uncertainty_finite() -> None:
    res = _drifting_plume_result()
    dyn = dynamics(res)
    assert np.isfinite(dyn.sigma_rise_velocity)
    assert dyn.sigma_rise_velocity >= 0


def test_sigma_y_coeffs_shape_is_two() -> None:
    res = _drifting_plume_result()
    dyn = dynamics(res)
    assert dyn.sigma_y_coeffs.shape == (2,)
    assert dyn.sigma_z_coeffs.shape == (2,)
    assert dyn.sigma_y_cov.shape == (2, 2)
    assert dyn.sigma_z_cov.shape == (2, 2)


def test_fully_masked_input_returns_nan_rise() -> None:
    """If every frame is masked, rise velocity is NaN, not 0."""
    res = RetrievalResult(
        tau=np.full((10, 32, 32), np.nan, dtype=np.float32),
        sigma_tau=np.full((10, 32, 32), np.nan, dtype=np.float32),
        mask=np.zeros((10, 32, 32), dtype=np.float32),
        metadata={"fps": 25.0, "pixel_scale": 0.25},
    )
    dyn = dynamics(res)
    assert np.isnan(dyn.rise_velocity)
    assert np.isnan(dyn.sigma_rise_velocity)


def test_fps_must_be_known() -> None:
    res = RetrievalResult(
        tau=np.zeros((5, 8, 8), dtype=np.float32),
        sigma_tau=np.zeros((5, 8, 8), dtype=np.float32),
        mask=np.ones((5, 8, 8), dtype=np.float32),
        metadata={},  # no fps anywhere
    )
    with pytest.raises(ValueError, match="fps"):
        dynamics(res)


def test_explicit_fps_overrides_metadata() -> None:
    res = _drifting_plume_result()
    dyn_default = dynamics(res)
    dyn_doubled = dynamics(res, fps=50.0)  # 2x metadata fps
    # rise velocity scales linearly with fps
    np.testing.assert_allclose(
        dyn_doubled.rise_velocity, dyn_default.rise_velocity * 2.0, rtol=1e-6
    )


def test_runs_end_to_end_on_synthetic_pipeline(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    """Smoke test: dynamics() shouldn't crash on the real retrieval output
    even though the fixture has a static (non-drifting) plume."""
    res = _retrieval(synthetic_video, minimal_config)
    # metadata may be missing fps for the synthetic TIFF; pass it explicitly
    dyn = dynamics(res, fps=25.0, pixel_scale=0.25)
    assert dyn.centroid_track.shape[0] == res.tau.shape[0]
