from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class CalibrationResult:
    L: np.ndarray
    sigma_L: np.ndarray
    metadata: Dict[str, Any]
    sensor: Any
    atmos: Any

    def to_netcdf(self, path: str) -> None:
        from smokesight.io import to_netcdf
        to_netcdf(self, path)

    def __repr__(self) -> str:
        return f"CalibrationResult(shape={self.L.shape})"


@dataclass
class BackgroundResult:
    L0: np.ndarray
    sigma_L0: np.ndarray
    confidence: np.ndarray
    method: str
    n_frames_used: int
    min_confidence: float = 0.5

    def to_netcdf(self, path: str) -> None:
        from smokesight.io import to_netcdf
        to_netcdf(self, path)

    def __repr__(self) -> str:
        return f"BackgroundResult(shape={self.L0.shape}, method={self.method})"


@dataclass
class RetrievalResult:
    tau: np.ndarray
    sigma_tau: np.ndarray
    T_lambda: Optional[np.ndarray]
    N: Optional[np.ndarray]
    sigma_N: Optional[np.ndarray]
    mask: np.ndarray
    metadata: Dict[str, Any]

    def to_netcdf(self, path: str) -> None:
        from smokesight.io import to_netcdf
        to_netcdf(self, path)

    def __repr__(self) -> str:
        valid = int(np.count_nonzero(self.mask))
        return f"RetrievalResult(shape={self.tau.shape}, valid={valid})"


@dataclass
class DynamicsResult:
    rise_velocity: float
    sigma_rise_velocity: float
    sigma_y_coeffs: np.ndarray
    sigma_z_coeffs: np.ndarray
    sigma_y_cov: np.ndarray
    sigma_z_cov: np.ndarray
    stability_class: Optional[str]
    centroid_track: np.ndarray
    metadata: Dict[str, Any]

    def to_netcdf(self, path: str) -> None:
        from smokesight.io import to_netcdf
        to_netcdf(self, path)

    def __repr__(self) -> str:
        return f"DynamicsResult(rise_velocity={self.rise_velocity:.3g})"
