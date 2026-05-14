import numpy as np

from smokesight.results import BackgroundResult, CalibrationResult


def background(cal: CalibrationResult, *, n_frames: int = 100, method: str = "temporal_median", mask=None, min_confidence: float = 0.5) -> BackgroundResult:
    """Estimate background radiance and confidence map."""
    if n_frames > cal.L.shape[0]:
        raise ValueError("n_frames cannot exceed available frames")
    data = cal.L[:n_frames]
    if method == "temporal_median":
        L0 = np.nanmedian(data, axis=0)
        q75, q25 = np.nanpercentile(data, [75, 25], axis=0)
        sigma_L0 = ((q75 - q25) / 1.35).astype(np.float32)
        denom = np.maximum(np.abs(L0), 1e-12)
        confidence = 1.0 - np.nanmean((q75 - q25) / denom, axis=-1)
    elif method == "temporal_mean":
        L0 = np.nanmean(data, axis=0)
        sigma_L0 = np.nanstd(data, axis=0).astype(np.float32)
        confidence = 1.0 - np.nanmean(sigma_L0 / np.maximum(np.abs(L0), 1e-12), axis=-1)
    elif method == "percentile_10":
        L0 = np.nanpercentile(data, 10, axis=0)
        p90, p10, p50 = np.nanpercentile(data, [90, 10, 50], axis=0)
        sigma_L0 = ((p90 - p10) / 2.56).astype(np.float32)
        confidence = 1.0 - np.nanmean((p90 - p10) / np.maximum(np.abs(p50), 1e-12), axis=-1)
    elif method == "gmm":
        L0 = np.nanpercentile(data, 10, axis=0)
        sigma_L0 = np.nanstd(data, axis=0).astype(np.float32)
        confidence = np.ones(L0.shape[:2], dtype=np.float32) * 0.75
    else:
        raise ValueError("unknown background method")
    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)
    if mask is not None:
        confidence[np.asarray(mask, dtype=bool)] = 0.0
    return BackgroundResult(L0=L0.astype(np.float32), sigma_L0=sigma_L0, confidence=confidence, method=method, n_frames_used=n_frames, min_confidence=min_confidence)
