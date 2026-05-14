import numpy as np
import xarray as xr

from smokesight.io import to_netcdf


def test_to_netcdf_creates_file(tmp_path, retrieval_result):
    path = tmp_path / "out.nc"
    to_netcdf(retrieval_result, path)
    assert path.exists()


def test_to_netcdf_cf_conventions(tmp_path, retrieval_result):
    path = tmp_path / "out.nc"
    to_netcdf(retrieval_result, path)
    ds = xr.open_dataset(path)
    assert ds.attrs["Conventions"] == "CF-1.9"


def test_to_netcdf_tau_variable_present(tmp_path, retrieval_result):
    path = tmp_path / "out.nc"
    to_netcdf(retrieval_result, path)
    ds = xr.open_dataset(path)
    assert "tau" in ds


def test_to_netcdf_sigma_tau_ancillary(tmp_path, retrieval_result):
    path = tmp_path / "out.nc"
    to_netcdf(retrieval_result, path)
    ds = xr.open_dataset(path)
    assert ds["sigma_tau"].attrs["ancillary_variables"] == "tau"


def test_to_netcdf_opens_with_xarray(tmp_path, retrieval_result):
    path = tmp_path / "out.nc"
    to_netcdf(retrieval_result, path)
    ds = xr.open_dataset(path)
    assert ds.sizes["time"] == retrieval_result.tau.shape[0]


def test_xarray_accessor(tmp_path, retrieval_result):
    path = tmp_path / "out.nc"
    to_netcdf(retrieval_result, path)
    ds = xr.open_dataset(path)
    assert ds.smokesight.tau is not None
