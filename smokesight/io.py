"""CF-1.9 NetCDF writer plus an xarray accessor.

Every result dataclass routes through ``to_netcdf`` here. The output
file always carries:

    Conventions   = 'CF-1.9'
    title         = 'SmokeSight retrieval output'
    source        = 'SmokeSight v<version>'
    history       = ISO-8601 timestamp + how it was produced
    references    = repo URL
    institution   = from config if present, else ''

Each variable gets CF-compliant ``standard_name`` and ``units``;
sigma_* variables carry an ``ancillary_variables`` attribute pointing
back at the quantity they are the uncertainty of. JOSS reviewers look
at exactly this, so don't go renaming the attributes casually.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import xarray as xr

from smokesight import __version__
from smokesight._results import (
    BackgroundResult,
    CalibrationResult,
    DynamicsResult,
    RetrievalResult,
)

PathLike = Union[str, "os.PathLike[str]"]
_AnyResult = Union[CalibrationResult, BackgroundResult, RetrievalResult, DynamicsResult]


def to_netcdf(
    result: _AnyResult,
    path: PathLike,
    *,
    complevel: int = 4,
    mode: Literal["w", "a"] = "w",
) -> None:
    """Write any SmokeSight result to a CF-1.9 NetCDF4 file."""
    ds = _to_dataset(result)
    encoding = _build_encoding(ds, complevel=complevel)
    ds.to_netcdf(str(path), mode=mode, format="NETCDF4", encoding=encoding)


def _to_dataset(result: _AnyResult) -> xr.Dataset:
    if isinstance(result, CalibrationResult):
        ds = _calibration_to_dataset(result)
    elif isinstance(result, BackgroundResult):
        ds = _background_to_dataset(result)
    elif isinstance(result, RetrievalResult):
        ds = _retrieval_to_dataset(result)
    elif isinstance(result, DynamicsResult):
        ds = _dynamics_to_dataset(result)
    else:
        raise TypeError(f"unknown result type: {type(result).__name__}")
    return _apply_global_attrs(ds, result)


def _calibration_to_dataset(r: CalibrationResult) -> xr.Dataset:
    t, h, w, n_lambda = r.L.shape
    coords = _spatial_coords(h, w, n_lambda, t=t)
    data_vars: Dict[str, Any] = {
        "L": (
            ("time", "y", "x", "wavelength"),
            r.L,
            {
                "standard_name": "toa_outgoing_radiance_per_unit_wavelength",
                "units": "W m-2 sr-1 um-1",
                "long_name": "calibrated radiance",
            },
        ),
        "sigma_L": (
            ("time", "y", "x", "wavelength"),
            r.sigma_L,
            {
                "units": "W m-2 sr-1 um-1",
                "long_name": "1-sigma uncertainty on L",
                "ancillary_variables": "L",
            },
        ),
    }
    return xr.Dataset(data_vars=data_vars, coords=coords)


def _background_to_dataset(r: BackgroundResult) -> xr.Dataset:
    h, w, n_lambda = r.L0.shape
    coords = _spatial_coords(h, w, n_lambda)
    data_vars: Dict[str, Any] = {
        "L0": (
            ("y", "x", "wavelength"),
            r.L0,
            {
                "long_name": "background radiance",
                "standard_name": "toa_outgoing_radiance_per_unit_wavelength",
                "units": "W m-2 sr-1 um-1",
            },
        ),
        "sigma_L0": (
            ("y", "x", "wavelength"),
            r.sigma_L0,
            {
                "units": "W m-2 sr-1 um-1",
                "long_name": "1-sigma uncertainty on L0",
                "ancillary_variables": "L0",
            },
        ),
        "confidence": (
            ("y", "x"),
            r.confidence,
            {
                "long_name": "background-estimation confidence",
                "units": "1",
                "valid_range": np.array([0.0, 1.0], dtype=np.float32),
            },
        ),
    }
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs["background_method"] = r.method
    ds.attrs["background_n_frames_used"] = int(r.n_frames_used)
    return ds


def _retrieval_to_dataset(r: RetrievalResult) -> xr.Dataset:
    t, h, w = r.tau.shape
    n_lambda = r.T_lambda.shape[-1] if r.T_lambda is not None else 1
    coords = _spatial_coords(h, w, n_lambda, t=t)
    data_vars: Dict[str, Any] = {
        "tau": (
            ("time", "y", "x"),
            r.tau,
            {
                "standard_name": "atmosphere_optical_thickness_due_to_aerosol",
                "units": "1",
                "long_name": "optical depth",
            },
        ),
        "sigma_tau": (
            ("time", "y", "x"),
            r.sigma_tau,
            {
                "units": "1",
                "long_name": "1-sigma uncertainty on tau",
                "ancillary_variables": "tau",
            },
        ),
        "mask": (
            ("time", "y", "x"),
            r.mask.astype(np.int8),
            {"long_name": "valid-pixel mask (1=valid, 0=masked)", "units": "1"},
        ),
    }
    if r.T_lambda is not None:
        data_vars["T_lambda"] = (
            ("time", "y", "x", "wavelength"),
            r.T_lambda,
            {
                "standard_name": "transmittance",
                "units": "1",
                "long_name": "spectral transmittance",
            },
        )
    if r.N is not None:
        data_vars["N"] = (
            ("time", "y", "x"),
            r.N,
            {
                "standard_name": "column_number_density",
                "units": "mol m-2",
                "long_name": "column density",
            },
        )
    if r.sigma_N is not None:
        data_vars["sigma_N"] = (
            ("time", "y", "x"),
            r.sigma_N,
            {
                "units": "mol m-2",
                "long_name": "1-sigma uncertainty on N",
                "ancillary_variables": "N",
            },
        )
    return xr.Dataset(data_vars=data_vars, coords=coords)


def _dynamics_to_dataset(r: DynamicsResult) -> xr.Dataset:
    t = r.centroid_track.shape[0]
    coords: Dict[str, Any] = {
        "time": ("time", np.arange(t, dtype=np.float64)),
        "axis": ("axis", ["x", "y"]),
        "coeff": ("coeff", ["a", "b"]),
    }
    data_vars: Dict[str, Any] = {
        "rise_velocity": (
            (),
            r.rise_velocity,
            {"units": "m s-1", "long_name": "plume rise velocity"},
        ),
        "sigma_rise_velocity": (
            (),
            r.sigma_rise_velocity,
            {
                "units": "m s-1",
                "long_name": "1-sigma uncertainty on rise_velocity",
                "ancillary_variables": "rise_velocity",
            },
        ),
        "sigma_y_coeffs": (
            ("coeff",),
            r.sigma_y_coeffs,
            {
                "long_name": (
                    "Pasquill-Gifford power-law coefficients (a, b) for sigma_y"
                ),
            },
        ),
        "sigma_z_coeffs": (
            ("coeff",),
            r.sigma_z_coeffs,
            {
                "long_name": (
                    "Pasquill-Gifford power-law coefficients (c, d) for sigma_z"
                ),
            },
        ),
        "sigma_y_cov": (
            ("coeff", "coeff"),
            r.sigma_y_cov,
            {"long_name": "covariance matrix for sigma_y_coeffs"},
        ),
        "sigma_z_cov": (
            ("coeff", "coeff"),
            r.sigma_z_cov,
            {"long_name": "covariance matrix for sigma_z_coeffs"},
        ),
        "centroid_track": (
            ("time", "axis"),
            r.centroid_track,
            {"long_name": "tau-weighted centroid per frame", "units": "pixel"},
        ),
    }
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    if r.stability_class is not None:
        ds.attrs["pasquill_gifford_stability_class"] = r.stability_class
    return ds


def _spatial_coords(
    h: int, w: int, n_lambda: int, *, t: Optional[int] = None
) -> Dict[str, Any]:
    coords: Dict[str, Any] = {
        "x": (
            "x",
            np.arange(w, dtype=np.float64),
            {"long_name": "x", "units": "pixel"},
        ),
        "y": (
            "y",
            np.arange(h, dtype=np.float64),
            {"long_name": "y", "units": "pixel"},
        ),
        "wavelength": (
            "wavelength",
            np.arange(n_lambda, dtype=np.float64),
            {"standard_name": "radiation_wavelength", "units": "um"},
        ),
    }
    if t is not None:
        coords["time"] = (
            "time",
            np.arange(t, dtype=np.float64),
            {"standard_name": "time", "units": "frame"},
        )
    return coords


def _apply_global_attrs(ds: xr.Dataset, result: _AnyResult) -> xr.Dataset:
    meta = getattr(result, "metadata", {}) or {}
    institution = str(meta.get("institution", ""))
    history = (
        f"{datetime.now(timezone.utc).isoformat()} "
        f"SmokeSight {__version__} Python API"
    )
    ds.attrs.update(
        {
            "Conventions": "CF-1.9",
            "title": "SmokeSight retrieval output",
            "institution": institution,
            "source": f"SmokeSight v{__version__}",
            "history": history,
            "references": "https://github.com/TasumLuke/smokesight",
        }
    )
    return ds


def _build_encoding(ds: xr.Dataset, *, complevel: int) -> Dict[str, Dict[str, Any]]:
    """zlib-compress every numeric data variable at the requested level."""
    encoding: Dict[str, Dict[str, Any]] = {}
    for name, var in ds.data_vars.items():
        if np.issubdtype(var.dtype, np.number) and var.ndim > 0:
            encoding[str(name)] = {"zlib": True, "complevel": int(complevel)}
    return encoding


# ---------------------------------------------------------------------------
# xarray accessor: ds.smokesight.tau, ds.smokesight.plot_frame(t), ...
# ---------------------------------------------------------------------------


@xr.register_dataset_accessor("smokesight")  # type: ignore[no-untyped-call]
class SmokeSightAccessor:
    """``Dataset.smokesight`` -- handy access for SmokeSight-produced datasets."""

    def __init__(self, ds: xr.Dataset):
        self._ds = ds

    @property
    def tau(self) -> Optional[xr.DataArray]:
        return self._ds.get("tau")

    @property
    def sigma_tau(self) -> Optional[xr.DataArray]:
        return self._ds.get("sigma_tau")

    def plot_frame(self, t: int, **kwargs: Any) -> Any:
        """Plot tau at frame t with a diverging colormap, masked pixels grey."""
        if "tau" not in self._ds:
            raise KeyError("dataset has no 'tau' variable")
        import matplotlib.pyplot as plt

        frame = self._ds["tau"].isel(time=t).values
        cmap = kwargs.pop("cmap", "RdBu_r")
        fig, ax = plt.subplots()
        im = ax.imshow(frame, cmap=cmap, **kwargs)
        ax.set_facecolor("0.5")  # grey background shows through NaN cells
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Optical Depth")
        ax.set_title(f"tau, frame {t}")
        return fig

    def animate(self, output_path: PathLike, *, fps: int = 10, **kwargs: Any) -> None:
        """Write an MP4 of every tau frame using matplotlib's FFMpegWriter."""
        if "tau" not in self._ds:
            raise KeyError("dataset has no 'tau' variable")
        import matplotlib.animation as manim
        import matplotlib.pyplot as plt

        tau = self._ds["tau"].values  # (T, H, W)
        cmap = kwargs.pop("cmap", "RdBu_r")
        fig, ax = plt.subplots()
        im = ax.imshow(tau[0], cmap=cmap, **kwargs)
        ax.set_facecolor("0.5")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Optical Depth")

        def update(i: int) -> Any:
            im.set_data(tau[i])
            ax.set_title(f"tau, frame {i}")
            return (im,)

        anim = manim.FuncAnimation(fig, update, frames=tau.shape[0], blit=False)
        writer = manim.FFMpegWriter(fps=fps)
        anim.save(str(output_path), writer=writer)
        plt.close(fig)
