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


def test_pg_fit_recovers_known_sigma() -> None:
    """For a synthetic plume whose sigma follows a known power law,
    the fitted Pasquill-Gifford `a` should land within 20% of ground truth.

    Construct frames where the centroid moves 1 px/frame downwind from the
    source and the cross-wind sigma follows sigma_y(x) = A_TRUE * x^B_TRUE
    in metres. With pixel_scale = 0.25 m/px, downwind distance at frame t
    is 0.25 * t metres, and sigma in pixels at that frame is
    (A_TRUE * (0.25 * t) ** B_TRUE) / 0.25.
    """
    a_true = 1.0  # sigma at 1 m downwind
    b_true = 0.7  # PG stability-class-D style exponent
    pixel_scale = 0.25

    t = 30
    h, w = 96, 96
    yy, xx = np.mgrid[0:h, 0:w]
    src_x = w // 2
    src_y = h - 1  # source at the bottom edge
    tau = np.zeros((t, h, w), dtype=np.float32)
    for i in range(1, t + 1):  # skip i=0 so dist > 0
        cy = src_y - i  # 1 px/frame upward
        dist_m = pixel_scale * i
        sigma_m = a_true * (dist_m**b_true)
        sigma_px = max(sigma_m / pixel_scale, 1.0)
        tau[i - 1] = 0.5 * np.exp(
            -((xx - src_x) ** 2 + (yy - cy) ** 2) / (2 * sigma_px**2)
        ).astype(np.float32)

    res = RetrievalResult(
        tau=tau,
        sigma_tau=np.full_like(tau, 0.01),
        mask=np.ones_like(tau),
        metadata={"fps": 10.0, "pixel_scale": pixel_scale},
    )
    dyn = dynamics(res, source_location=(int(src_x), int(src_y)))
    a_fit = float(dyn.sigma_y_coeffs[0])
    assert np.isfinite(a_fit), f"fit failed: a={a_fit}"
    # The Gaussian-width estimator on a discretised image isn't perfect;
    # 20% tolerance (spec sec 8.5) gives the fit room to breathe.
    assert (
        abs(a_fit - a_true) / a_true < 0.20
    ), f"fitted a={a_fit:.3f} not within 20% of ground truth {a_true}"
