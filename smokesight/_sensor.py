from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import imageio.v3 as iio
import numpy as np


@dataclass
class SensorModel:
    dark: np.ndarray
    flat: np.ndarray
    gain: float
    spectral_response: np.ndarray
    wavelengths: np.ndarray
    bit_depth: int
    noise_equivalent_radiance: float

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], frame_shape: Optional[Tuple[int, int]] = None
    ) -> "SensorModel":
        sensor = config.get("sensor", {})
        for key in ("gain", "bit_depth", "ner"):
            if key not in sensor:
                raise ValueError(f"Missing required sensor.{key}")
        if frame_shape is None:
            frame_shape = (1, 1)
        h, w = frame_shape
        dark = _load_or_default(sensor.get("dark_current"), (h, w), 0.0)
        flat = _load_or_default(sensor.get("flat_field"), (h, w), 1.0)
        if flat.shape != dark.shape or flat.shape != (h, w):
            raise ValueError("flat_field and dark_current must match video HxW")
        mean = float(np.mean(flat))
        if mean != 0.0:
            flat = flat / mean
        sr = sensor.get("spectral_response") or {}
        response = np.asarray(sr.get("response", [1.0]), dtype=np.float32)
        wavelengths = np.asarray(sr.get("wavelengths", [1.0]), dtype=np.float32)
        if response.shape != wavelengths.shape:
            raise ValueError("spectral_response.response and wavelengths must match")
        return cls(
            dark=dark.astype(np.float32),
            flat=flat.astype(np.float32),
            gain=float(sensor["gain"]),
            spectral_response=response.astype(np.float32),
            wavelengths=wavelengths.astype(np.float32),
            bit_depth=int(sensor["bit_depth"]),
            noise_equivalent_radiance=float(sensor["ner"]),
        )


def _load_or_default(path: Optional[str], shape: Tuple[int, int], value: float) -> np.ndarray:
    if not path:
        return np.full(shape, value, dtype=np.float32)
    arr = np.asarray(iio.imread(Path(path)), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr
