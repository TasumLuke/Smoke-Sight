from typing import Callable, Sequence, Tuple

import numpy as np


def radiance_uncertainty(L, sensor, atmos):
    L_abs = np.abs(np.asarray(L, dtype=np.float32))
    sigma_shot = np.sqrt(np.maximum(L_abs * max(sensor.gain, 1e-12), 0.0))
    sigma_read = sensor.noise_equivalent_radiance
    sigma_flat = 0.01 * L_abs
    sigma_atmos = atmos.uncertainty(L)
    return np.sqrt(sigma_shot**2 + sigma_read**2 + sigma_flat**2 + sigma_atmos**2).astype(np.float32)


def tau_uncertainty(L, sigma_L, L0, sigma_L0):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.sqrt((sigma_L / L) ** 2 + (sigma_L0 / L0) ** 2)
    return out.astype(np.float32)


def centroid_uncertainty(tau, sigma_tau):
    valid = np.isfinite(tau) & np.isfinite(sigma_tau)
    if not np.any(valid):
        return np.array([np.nan, np.nan], dtype=np.float32)
    return np.array([np.nanmean(sigma_tau[valid]), np.nanmean(sigma_tau[valid])], dtype=np.float32)


def gaussian_fit_uncertainty(profile, sigma_profile):
    return np.asarray(sigma_profile, dtype=np.float32)


def monte_carlo(func: Callable, inputs: Sequence[np.ndarray], sigmas: Sequence[np.ndarray], n: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(max(n, 1000)):
        noisy = [rng.normal(inp, sig) for inp, sig in zip(inputs, sigmas)]
        samples.append(func(*noisy))
    arr = np.asarray(samples)
    lo, hi = np.percentile(arr, [16, 84], axis=0)
    return np.mean(arr, axis=0), ((hi - lo) / 2.0)
