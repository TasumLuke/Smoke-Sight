import warnings

import numpy as np

from smokesight._uncertainty import tau_uncertainty
from smokesight.results import BackgroundResult, CalibrationResult, RetrievalResult


def retrieve(cal: CalibrationResult, bg: BackgroundResult, *, tau_max: float = 2.0, wavelengths=None, species_xsec=None) -> RetrievalResult:
    """Compute optical depth from radiance and background."""
    L = cal.L
    L0 = bg.L0[None, :, :, :]
    bad_l0 = (~np.isfinite(L0)) | (L0 == 0)
    if np.any(bad_l0):
        warnings.warn(f"L0 is zero or NaN at {int(np.count_nonzero(bad_l0))} pixels")
    with np.errstate(divide="ignore", invalid="ignore"):
        R = L / L0
    invalid = (bg.confidence[None, :, :] < bg.min_confidence) | (np.nanmean(R, axis=-1) > 1.05) | (np.nanmean(R, axis=-1) < 0.01) | np.any(bad_l0, axis=-1)
    R_clip = np.clip(R, 0.01, 1.05)
    tau_lambda = -np.log(R_clip)
    tau = np.nanmean(tau_lambda, axis=-1).astype(np.float32)
    tau[invalid] = np.nan
    sigma_l0 = bg.sigma_L0[None, :, :, :]
    sigma_tau_lambda = tau_uncertainty(L, cal.sigma_L, L0, sigma_l0)
    sigma_tau = np.nanmean(sigma_tau_lambda, axis=-1).astype(np.float32)
    sigma_tau[invalid] = np.nan
    too_high = tau > tau_max
    tau[too_high] = np.nan
    sigma_tau[too_high] = np.nan
    valid = np.isfinite(tau)
    T_lambda = np.exp(-tau_lambda).astype(np.float32) if tau_lambda.shape[-1] > 1 else None
    N = None
    sigma_N = None
    if species_xsec:
        first = next(iter(species_xsec.values()))
        xsec = float(np.ravel(first)[0])
        if xsec != 0:
            N = (tau / xsec).astype(np.float32)
            sigma_N = (sigma_tau / abs(xsec)).astype(np.float32)
    meta = dict(cal.metadata)
    meta.update({"tau_max": tau_max})
    return RetrievalResult(tau=tau, sigma_tau=sigma_tau, T_lambda=T_lambda, N=N, sigma_N=sigma_N, mask=valid, metadata=meta)
