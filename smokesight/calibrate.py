from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import imageio.v3 as iio
import numpy as np
import yaml

from smokesight._atmos import make_atmos
from smokesight._sensor import SensorModel
from smokesight._uncertainty import radiance_uncertainty
from smokesight.results import CalibrationResult


def calibrate(
    video_path: Union[str, Path],
    config: Union[str, Path, Dict[str, Any]],
    *,
    frame_range: Optional[Tuple[int, int]] = None,
    n_workers: int = 1,
    progress: bool = True,
) -> CalibrationResult:
    """Convert raw DN image/video data to calibrated radiance."""
    cfg = _load_config(config)
    frames = _read_frames(video_path)
    if frame_range is not None:
        start, stop = frame_range
        frames = frames[start : stop + 1]
    if frames.ndim == 3:
        frames = frames[..., None]
    if frames.ndim != 4:
        raise ValueError("video must be T,H,W or T,H,W,C")
    t, h, w, _ = frames.shape
    sensor = SensorModel.from_config(cfg, (h, w))
    atmos = make_atmos(cfg)
    dn = frames.astype(np.float32)
    l_raw = (dn - sensor.dark[None, :, :, None]) / np.maximum(sensor.flat[None, :, :, None], 1e-12)
    l_abs = l_raw * sensor.gain
    response = sensor.spectral_response.reshape(1, 1, 1, -1)
    if l_abs.shape[-1] == 1:
        L = l_abs * response
    else:
        L = l_abs
    L = atmos.correct(L).astype(np.float32)
    sigma_L = radiance_uncertainty(L, sensor, atmos)
    metadata = {
        "fps": 1.0,
        "n_frames": int(t),
        "height": int(h),
        "width": int(w),
        "bit_depth": int(sensor.bit_depth),
        "video_path": str(video_path),
        "config_path": str(config) if not isinstance(config, dict) else "<dict>",
        "calibration_timestamp": datetime.now(timezone.utc).isoformat(),
        "institution": cfg.get("institution", ""),
        "wavelengths": sensor.wavelengths.tolist(),
    }
    return CalibrationResult(L=L, sigma_L=sigma_L, metadata=metadata, sensor=sensor, atmos=atmos)


def _load_config(config):
    if isinstance(config, dict):
        return config
    with open(config, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_frames(video_path):
    path = Path(video_path)
    arr = np.asarray(iio.imread(path))
    if arr.ndim == 2:
        arr = arr[None, :, :]
    return arr
