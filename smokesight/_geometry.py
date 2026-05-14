from dataclasses import dataclass

import numpy as np


@dataclass
class CameraGeometry:
    focal_length: float = 50.0
    sensor_width: float = 12.8
    image_width: int = 128
    altitude: float = 300.0
    tilt_angle: float = 0.0
    pixel_scale: float = 1.0

    @classmethod
    def from_config(cls, config):
        geom = config.get("geometry", {})
        return cls(pixel_scale=float(geom.get("pixel_scale", 1.0)))


def project_to_ground(pixels: np.ndarray, geom: CameraGeometry) -> np.ndarray:
    return np.asarray(pixels, dtype=np.float32) * geom.pixel_scale
