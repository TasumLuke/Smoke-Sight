"""Plume rise velocity and Pasquill-Gifford dispersion fits.

Rise velocity comes from tracking the tau-weighted vertical centroid
over time and fitting a line to centroid_y(t). The slope, in
m/frame, becomes m/s once we multiply by fps.

Dispersion coefficients come from fitting Gaussians to the cross-wind
(x) and vertical (y) profiles of integrated tau, frame by frame. Each
sigma_y(t) and sigma_z(t) is then fitted to a power law
``sigma = a * x^b`` against downwind distance.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit

from smokesight._results import DynamicsResult, RetrievalResult
from smokesight._types import FloatArray
from smokesight._uncertainty import gaussian_fit_uncertainty

# Coefficients are stored as (a, b) in sigma = a * x^b for both axes.
_PowerLawCoeffs = Tuple[float, float]


def dynamics(
    result: RetrievalResult,
    *,
    fps: Optional[float] = None,
    pixel_scale: Optional[float] = None,
    source_location: Optional[Tuple[int, int]] = None,
    stability_class: Optional[str] = None,
) -> DynamicsResult:
    """Estimate rise velocity, dispersion coefficients, and centroid track."""
    fps_val = _resolve_fps(fps, result)
    pixel_scale_val = _resolve_pixel_scale(pixel_scale, result)

    centroid_track = _track_centroid(result.tau)  # (T, 2) in pixels, NaN if frame empty
    rise_v, sigma_rise_v = _fit_rise_velocity(
        centroid_track, fps=fps_val, pixel_scale=pixel_scale_val
    )

    if source_location is None:
        source_location = _estimate_source_from_first_valid(centroid_track)

    sigma_y_coeffs, sigma_y_cov = _fit_dispersion_axis(
        result.tau,
        axis="x",
        source_location=source_location,
        pixel_scale=pixel_scale_val,
    )
    sigma_z_coeffs, sigma_z_cov = _fit_dispersion_axis(
        result.tau,
        axis="y",
        source_location=source_location,
        pixel_scale=pixel_scale_val,
    )

    metadata = {
        "fps": fps_val,
        "pixel_scale": pixel_scale_val,
        "source_location": source_location,
    }

    return DynamicsResult(
        rise_velocity=rise_v,
        sigma_rise_velocity=sigma_rise_v,
        sigma_y_coeffs=np.asarray(sigma_y_coeffs, dtype=np.float32),
        sigma_z_coeffs=np.asarray(sigma_z_coeffs, dtype=np.float32),
        sigma_y_cov=sigma_y_cov.astype(np.float32),
        sigma_z_cov=sigma_z_cov.astype(np.float32),
        centroid_track=centroid_track.astype(np.float32),
        stability_class=stability_class,
        metadata=metadata,
    )


def _resolve_fps(fps: Optional[float], result: RetrievalResult) -> float:
    if fps is not None:
        return float(fps)
    meta_fps = result.metadata.get("fps")
    if meta_fps is None or float(meta_fps) <= 0:
        raise ValueError(
            "fps unknown -- pass fps= explicitly or include it in the "
            "CalibrationResult metadata"
        )
    return float(meta_fps)


def _resolve_pixel_scale(
    pixel_scale: Optional[float], result: RetrievalResult
) -> float:
    if pixel_scale is not None:
        return float(pixel_scale)
    if "pixel_scale" in result.metadata:
        return float(result.metadata["pixel_scale"])
    # Fall back to 1.0 m/pixel -- rise_velocity will then be in pixels/s,
    # but the result still has a meaningful shape.
    return 1.0


def _track_centroid(tau: FloatArray) -> FloatArray:
    """Tau-weighted centroid (x, y) per frame; NaN if the frame is fully masked."""
    t = tau.shape[0]
    track = np.full((t, 2), np.nan, dtype=np.float64)
    for i in range(t):
        frame = tau[i]
        valid = np.isfinite(frame) & (frame > 0)
        if not valid.any():
            continue
        # ndimage uses (row, col) = (y, x); invert to (x, y) on the way out.
        cy, cx = ndimage.center_of_mass(np.where(valid, frame, 0.0))
        track[i, 0] = cx
        track[i, 1] = cy
    return track


def _fit_rise_velocity(
    centroid_track: FloatArray, *, fps: float, pixel_scale: float
) -> Tuple[float, float]:
    """Linear fit to centroid_y(t). Returns (rise_velocity, sigma) in m/s."""
    y = centroid_track[:, 1]
    valid = np.isfinite(y)
    if valid.sum() < 3:
        return float("nan"), float("nan")

    t = np.arange(len(y))[valid].astype(np.float64)
    y_valid = y[valid].astype(np.float64)
    n = len(t)

    # Closed-form simple linear regression (avoids dragging in scipy.stats).
    t_mean = t.mean()
    y_mean = y_valid.mean()
    sxx = float(((t - t_mean) ** 2).sum())
    sxy = float(((t - t_mean) * (y_valid - y_mean)).sum())
    if sxx <= 0:
        return float("nan"), float("nan")
    slope_px_per_frame = sxy / sxx
    intercept = y_mean - slope_px_per_frame * t_mean

    # Slope standard error.
    residual = y_valid - (slope_px_per_frame * t + intercept)
    if n <= 2:
        slope_se = float("nan")
    else:
        mse = float((residual**2).sum() / (n - 2))
        slope_se = float(np.sqrt(mse / sxx)) if mse >= 0 else float("nan")

    rise_v = float(slope_px_per_frame * pixel_scale * fps)
    sigma_rise_v = float(slope_se * pixel_scale * fps)
    return rise_v, sigma_rise_v


def _estimate_source_from_first_valid(centroid_track: FloatArray) -> Tuple[int, int]:
    """Use the earliest non-NaN centroid as the source pixel."""
    for row in centroid_track:
        if np.isfinite(row).all():
            return int(row[0]), int(row[1])
    return 0, 0  # nothing to anchor to; downstream fits will return NaN


def _fit_dispersion_axis(
    tau: FloatArray,
    *,
    axis: str,
    source_location: Tuple[int, int],
    pixel_scale: float,
) -> Tuple[_PowerLawCoeffs, FloatArray]:
    """Fit sigma = a * x^b to the per-frame Gaussian widths along an axis.

    axis='x' integrates over y and fits the cross-wind profile.
    axis='y' integrates over x and fits the vertical profile.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    widths = []
    distances = []
    sx, sy = source_location

    for frame in tau:
        valid_frame = np.where(np.isfinite(frame), frame, 0.0)
        if axis == "x":
            profile = valid_frame.sum(axis=0)  # integrate over y
            anchor = sx
        else:
            profile = valid_frame.sum(axis=1)
            anchor = sy

        sigma = _gaussian_width_or_nan(profile)
        if not np.isfinite(sigma):
            continue

        # Use the frame's centroid distance from source as "downwind distance".
        # ndimage.center_of_mass returns (row, col) -- (y, x) in image-space.
        cy_frame, cx_frame = ndimage.center_of_mass(valid_frame)
        if np.isnan(cx_frame) or np.isnan(cy_frame):
            continue
        dx = cx_frame - sx
        dy = cy_frame - sy
        dist = float(np.hypot(dx, dy) * pixel_scale)
        if dist <= 0:
            continue

        widths.append(sigma * pixel_scale)
        distances.append(dist)

    if len(widths) < 3:
        return (float("nan"), float("nan")), np.full((2, 2), np.nan)

    return _fit_power_law(np.asarray(distances), np.asarray(widths))


def _gaussian_width_or_nan(profile: FloatArray) -> float:
    """Fit a 1-D Gaussian to a profile; return sigma in pixels, or NaN on failure."""
    profile = np.asarray(profile, dtype=np.float64)
    if profile.size < 5 or profile.sum() <= 0:
        return float("nan")
    x = np.arange(profile.size, dtype=np.float64)

    def gaussian(x: FloatArray, a: float, mu: float, sigma: float) -> FloatArray:
        return np.asarray(a * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2)))

    # Reasonable starting guesses: amp = max, mean = argmax, sigma = profile_std/2.
    a0 = float(profile.max())
    mu0 = float(np.argmax(profile))
    sigma0 = max(float(np.std(profile)) / 2.0, 1.0)
    try:
        popt, _ = curve_fit(gaussian, x, profile, p0=[a0, mu0, sigma0], maxfev=2000)
    except (RuntimeError, ValueError):
        return float("nan")
    return float(abs(popt[2]))


def _fit_power_law(x: FloatArray, y: FloatArray) -> Tuple[_PowerLawCoeffs, FloatArray]:
    """Fit y = a * x^b via curve_fit, returning ((a, b), 2x2 covariance)."""

    def model(x: FloatArray, a: float, b: float) -> FloatArray:
        return a * np.power(x, b)

    try:
        popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.5], maxfev=2000)
    except (RuntimeError, ValueError):
        return (float("nan"), float("nan")), np.full((2, 2), np.nan)

    # Use _uncertainty.gaussian_fit_uncertainty to sanitise the covariance; we
    # store the full matrix on the result, but the caller often wants
    # sqrt(diag(pcov)) and that path will get NaNs if pcov is degenerate.
    _ = gaussian_fit_uncertainty(pcov)
    return (float(popt[0]), float(popt[1])), np.asarray(pcov, dtype=np.float64)
