from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from smokesight.results import CalibrationResult, DynamicsResult, RetrievalResult


def to_netcdf(result, path, *, complevel: int = 4, mode: str = "w") -> None:
    """Write a CF-1.9-ish NetCDF file."""
    attrs = {
        "Conventions": "CF-1.9",
        "title": "SmokeSight retrieval output",
        "institution": getattr(result, "metadata", {}).get("institution", ""),
        "source": "SmokeSight v0.1.0",
        "history": datetime.now(timezone.utc).isoformat() + " SmokeSight Python API",
        "references": "https://github.com/TasumLuke/smokesight",
    }
    if isinstance(result, RetrievalResult):
        t, h, w = result.tau.shape
        ds = xr.Dataset(
            data_vars={
                "tau": (("time", "y", "x"), result.tau, {"standard_name": "atmosphere_optical_thickness_due_to_aerosol", "units": "1"}),
                "sigma_tau": (("time", "y", "x"), result.sigma_tau, {"units": "1", "ancillary_variables": "tau"}),
                "mask": (("time", "y", "x"), result.mask),
            },
            coords={"time": np.arange(t), "y": np.arange(h), "x": np.arange(w)},
            attrs=attrs,
        )
    elif isinstance(result, CalibrationResult):
        t, h, w, n = result.L.shape
        ds = xr.Dataset(
            data_vars={"L": (("time", "y", "x", "wavelength"), result.L), "sigma_L": (("time", "y", "x", "wavelength"), result.sigma_L)},
            coords={"time": np.arange(t), "y": np.arange(h), "x": np.arange(w), "wavelength": np.arange(n)},
            attrs=attrs,
        )
    elif isinstance(result, DynamicsResult):
        ds = xr.Dataset(
            data_vars={"centroid_track": (("time", "xy"), result.centroid_track)},
            attrs={**attrs, "rise_velocity": result.rise_velocity, "sigma_rise_velocity": result.sigma_rise_velocity},
        )
    else:
        raise TypeError("unsupported result type")
    try:
        encoding = {name: {"zlib": True, "complevel": complevel} for name in ds.data_vars if ds[name].dtype.kind in "fiu"}
        ds.to_netcdf(Path(path), mode=mode, encoding=encoding)
    except ValueError:
        # scipy backend does not support compression; retry without encoding.
        ds.to_netcdf(Path(path), mode=mode)


@xr.register_dataset_accessor("smokesight")
class SmokeSightAccessor:
    def __init__(self, ds):
        self._ds = ds

    @property
    def tau(self):
        return self._ds["tau"]

    @property
    def sigma_tau(self):
        return self._ds["sigma_tau"]

    def plot_frame(self, t):
        return self._ds["tau"].isel(time=t).plot()

    def animate(self, output_path):
        raise NotImplementedError("Animation is not implemented in the MVP")

