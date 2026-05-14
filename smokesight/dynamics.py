import numpy as np

from smokesight.results import DynamicsResult, RetrievalResult


def dynamics(
    result: RetrievalResult,
    *,
    fps=None,
    pixel_scale=None,
    source_location=None,
    stability_class=None,
) -> DynamicsResult:
    """Estimate plume rise velocity and simple dispersion widths from tau frames."""
    fps = float(fps or result.metadata.get("fps", 1.0))
    pixel_scale = float(pixel_scale or result.metadata.get("pixel_scale", 1.0))
    centroids = []
    widths_x = []
    widths_y = []

    for frame in result.tau:
        valid = np.isfinite(frame) & (frame > 0)
        if not np.any(valid):
            centroids.append([np.nan, np.nan])
            widths_x.append(np.nan)
            widths_y.append(np.nan)
            continue
        weights = np.where(valid, frame, 0.0)
        y, x = np.indices(frame.shape)
        total = float(np.sum(weights))
        cx = float(np.sum(x * weights) / total)
        cy = float(np.sum(y * weights) / total)
        centroids.append([cx, cy])
        widths_x.append(float(np.sqrt(np.sum(weights * (x - cx) ** 2) / total)))
        widths_y.append(float(np.sqrt(np.sum(weights * (y - cy) ** 2) / total)))

    centroid_track = np.asarray(centroids, dtype=np.float32)
    ok = np.isfinite(centroid_track[:, 1])
    if np.count_nonzero(ok) >= 2:
        tt = np.arange(len(centroid_track), dtype=np.float64)[ok]
        yy = centroid_track[ok, 1].astype(np.float64)
        slope, intercept = np.polyfit(tt, yy, 1)
        rise_velocity = float(-slope * pixel_scale * fps)
        residuals = yy - (slope * tt + intercept)
        sxx = float(np.sum((tt - np.mean(tt)) ** 2))
        mse = float(np.mean(residuals**2))
        sigma = float(np.sqrt(max(mse, 1e-12) / sxx)) if sxx > 0 else np.nan
    else:
        rise_velocity = np.nan
        sigma = np.nan

    wy = np.asarray(widths_x, dtype=np.float64)
    wz = np.asarray(widths_y, dtype=np.float64)
    finite_y = wy[np.isfinite(wy) & (wy > 0)]
    finite_z = wz[np.isfinite(wz) & (wz > 0)]
    sigma_y_coeffs = np.array(
        [float(np.nanmean(finite_y)) if finite_y.size else np.nan, 1.0],
        dtype=np.float32,
    )
    sigma_z_coeffs = np.array(
        [float(np.nanmean(finite_z)) if finite_z.size else np.nan, 1.0],
        dtype=np.float32,
    )
    sigma_y_cov = np.diag(np.nan_to_num(sigma_y_coeffs, nan=0.0)).astype(np.float32)
    sigma_z_cov = np.diag(np.nan_to_num(sigma_z_coeffs, nan=0.0)).astype(np.float32)

    return DynamicsResult(
        rise_velocity=rise_velocity,
        sigma_rise_velocity=sigma,
        sigma_y_coeffs=sigma_y_coeffs,
        sigma_z_coeffs=sigma_z_coeffs,
        sigma_y_cov=sigma_y_cov,
        sigma_z_cov=sigma_z_cov,
        stability_class=stability_class,
        centroid_track=centroid_track,
        metadata=dict(result.metadata),
    )
