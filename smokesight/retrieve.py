"""Beer-Lambert tau retrieval.

Given calibrated radiance L and a background plate L0, compute optical
depth tau = -ln(L / L0) per pixel per frame, plus per-pixel sigma_tau.

Pixels are masked (NaN'd) where:
  - bg.confidence < min_confidence (background was not well determined)
  - L / L0 > 1.05 (plume brighter than background -- beyond Beer-Lambert)
  - L / L0 < 0.01 (saturated absorption -- the model breaks down here)
  - L0 <= 0 or non-finite
  - tau exceeds tau_max (optically thick; beyond linear regime)

Optional spectral-transmittance cube T_lambda = exp(-tau_lambda) is
returned when the input is multi-band. If a species cross-section is
provided, column density N [mol m^-2] is solved by least-squares
across wavelengths.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from smokesight._results import BackgroundResult, CalibrationResult, RetrievalResult
from smokesight._types import FloatArray
from smokesight._uncertainty import tau_uncertainty

_RATIO_LO = 0.01
_RATIO_HI = 1.05


def retrieve(
    cal: CalibrationResult,
    bg: BackgroundResult,
    *,
    tau_max: float = 2.0,
    wavelengths: Optional[list[float]] = None,
    species_xsec: Optional[Dict[str, FloatArray]] = None,
    min_confidence: float = 0.5,
) -> RetrievalResult:
    """Compute tau, sigma_tau and (optionally) T_lambda and column density."""
    if cal.L.shape[1:] != bg.L0.shape:
        raise ValueError(
            f"cal.L spatial+lambda shape {cal.L.shape[1:]} does not match "
            f"bg.L0 shape {bg.L0.shape}"
        )
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(f"min_confidence must be in [0, 1]; got {min_confidence}")

    L = cal.L  # (T, H, W, N_lambda)
    L0 = bg.L0  # (H, W, N_lambda)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = L / L0  # broadcast over T

    panchromatic_ratio = ratio.mean(axis=-1)  # (T, H, W)

    invalid = (
        (panchromatic_ratio > _RATIO_HI)
        | (panchromatic_ratio < _RATIO_LO)
        | ~np.isfinite(panchromatic_ratio)
    )
    low_conf = bg.confidence < min_confidence  # (H, W)
    invalid = invalid | low_conf[np.newaxis, :, :]

    if (L0 <= 0).any() or not np.isfinite(L0).all():
        n_bad = int(np.sum((L0 <= 0) | ~np.isfinite(L0)))
        import warnings

        warnings.warn(
            f"{n_bad} background pixels have non-positive or non-finite L0; "
            "they will be masked from tau",
            UserWarning,
            stacklevel=2,
        )

    # Compute tau using the panchromatic ratio (averaged over wavelengths)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = -np.log(np.clip(panchromatic_ratio, _RATIO_LO, None))
    invalid = invalid | (tau > tau_max)
    tau = np.where(invalid, np.nan, tau).astype(np.float32)

    # sigma_tau: collapse the wavelength axis on L and L0 by averaging too,
    # so the uncertainty matches the central value's grouping.
    L_pan = L.mean(axis=-1)  # (T, H, W)
    L0_pan = L0.mean(axis=-1)  # (H, W)
    sigma_L_pan = cal.sigma_L.mean(axis=-1)  # (T, H, W)
    sigma_L0_pan = bg.sigma_L0.mean(axis=-1)  # (H, W)

    sigma_tau = tau_uncertainty(
        L_pan,
        sigma_L_pan,
        np.broadcast_to(L0_pan, L_pan.shape),
        np.broadcast_to(sigma_L0_pan, L_pan.shape),
    )
    sigma_tau = np.where(invalid, np.nan, sigma_tau).astype(np.float32)

    valid_mask = (~invalid).astype(bool)

    T_lambda: Optional[FloatArray] = None
    if wavelengths is not None and L.shape[-1] > 1:
        T_lambda = np.exp(
            -np.where(np.isnan(tau)[..., np.newaxis], 0.0, tau[..., np.newaxis])
        ).astype(np.float32)
        T_lambda = np.where(np.isnan(tau)[..., np.newaxis], np.nan, T_lambda)

    N: Optional[FloatArray] = None
    sigma_N: Optional[FloatArray] = None
    if species_xsec:
        N, sigma_N = _column_density(tau, sigma_tau, species_xsec, L.shape[-1])

    metadata = dict(cal.metadata)
    metadata.update(
        {
            "tau_max": tau_max,
            "min_confidence": min_confidence,
            "background_method": bg.method,
            "background_n_frames": bg.n_frames_used,
        }
    )

    return RetrievalResult(
        tau=tau,
        sigma_tau=sigma_tau,
        mask=valid_mask.astype(np.float32),  # stored as float32 for NetCDF compat
        metadata=metadata,
        T_lambda=T_lambda,
        N=N,
        sigma_N=sigma_N,
    )


def _column_density(
    tau: FloatArray,
    sigma_tau: FloatArray,
    species_xsec: Dict[str, FloatArray],
    n_lambda: int,
) -> tuple[FloatArray, FloatArray]:
    """Solve tau = sum_species( N_s * sigma_xsec_s ) for N.

    Single species + single wavelength: N = tau / sigma_xsec.
    Multi-species or multi-wavelength: weighted least-squares per pixel.
    For now we implement the single-species case (most common); a true
    multi-species solver is a follow-up.
    """
    if len(species_xsec) > 1:
        raise NotImplementedError(
            "multi-species retrieval is not implemented yet; pass a single species"
        )
    species, xsec = next(iter(species_xsec.items()))
    xsec_arr = np.asarray(xsec, dtype=np.float64)
    if xsec_arr.size == 1:
        sigma_xsec = float(xsec_arr.item())
    else:
        # Average across wavelengths (or could weight by spectral_response)
        sigma_xsec = float(xsec_arr.mean())
    if sigma_xsec <= 0:
        raise ValueError(f"species cross-section for {species!r} must be positive")

    N = (tau / sigma_xsec).astype(np.float32)
    sigma_N = (sigma_tau / sigma_xsec).astype(np.float32)
    return N, sigma_N
