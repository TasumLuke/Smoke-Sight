import warnings
from typing import Any, Dict

import numpy as np


class IdentityAtmos:
    T_atm: float = 1.0
    L_path: float = 0.0

    def correct(self, L: np.ndarray) -> np.ndarray:
        return L

    def uncertainty(self, L: np.ndarray) -> np.ndarray:
        return np.zeros_like(L, dtype=np.float32)


class AtmosModel:
    def __init__(self, config: Dict[str, Any]):
        self.T_atm = float(config.get("T_atm", 1.0))
        self.L_path = float(config.get("L_path", 0.0))
        if self.T_atm == 0:
            raise ValueError("atmosphere T_atm cannot be zero")

    def correct(self, L: np.ndarray) -> np.ndarray:
        return (L - self.L_path) / self.T_atm

    def uncertainty(self, L: np.ndarray) -> np.ndarray:
        return np.asarray(0.05 * np.abs(L), dtype=np.float32)


def make_atmos(config: Dict[str, Any]):
    atmos = config.get("atmosphere")
    if not atmos or atmos.get("model", "identity") == "identity":
        return IdentityAtmos()
    warnings.warn("External atmospheric model disabled in MVP; using simple AtmosModel")
    return AtmosModel(atmos)
