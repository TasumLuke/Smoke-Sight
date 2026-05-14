"""Tests for smokesight.io."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import xarray as xr

from smokesight.background import background
from smokesight.calibrate import calibrate
from smokesight.io import to_netcdf
from smokesight.retrieve import retrieve


def _pipeline(synthetic_video: Path, minimal_config: Dict[str, Any]):
    cal = calibrate(synthetic_video, minimal_config, progress=False)
    bg = background(cal, n_frames=20)
    res = retrieve(cal, bg, min_confidence=0.0)
    return cal, bg, res


def test_calibration_to_netcdf_writes_file(
    synthetic_video: Path, minimal_config: Dict[str, Any], tmp_path: Path
) -> None:
    cal, _, _ = _pipeline(synthetic_video, minimal_config)
    out = tmp_path / "cal.nc"
    to_netcdf(cal, out)
    assert out.exists()


def test_retrieval_netcdf_has_cf_attributes(
    synthetic_video: Path, minimal_config: Dict[str, Any], tmp_path: Path
) -> None:
    _, _, res = _pipeline(synthetic_video, minimal_config)
    out = tmp_path / "ret.nc"
    to_netcdf(res, out)
    ds = xr.open_dataset(out)
    try:
        assert ds.attrs["Conventions"] == "CF-1.9"
        assert ds.attrs["title"].startswith("SmokeSight")
        assert ds.attrs["source"].startswith("SmokeSight v")
        assert "history" in ds.attrs
    finally:
        ds.close()


def test_retrieval_netcdf_has_tau_and_sigma_tau(
    synthetic_video: Path, minimal_config: Dict[str, Any], tmp_path: Path
) -> None:
    _, _, res = _pipeline(synthetic_video, minimal_config)
    out = tmp_path / "ret.nc"
    to_netcdf(res, out)
    ds = xr.open_dataset(out)
    try:
        assert "tau" in ds
        assert "sigma_tau" in ds
        # sigma_tau must point back at tau via the CF ancillary_variables attr
        assert ds["sigma_tau"].attrs.get("ancillary_variables") == "tau"
        assert (
            ds["tau"].attrs.get("standard_name")
            == "atmosphere_optical_thickness_due_to_aerosol"
        )
    finally:
        ds.close()


def test_xarray_accessor_exposes_tau(
    synthetic_video: Path, minimal_config: Dict[str, Any], tmp_path: Path
) -> None:
    _, _, res = _pipeline(synthetic_video, minimal_config)
    out = tmp_path / "ret.nc"
    to_netcdf(res, out)
    ds = xr.open_dataset(out)
    try:
        assert ds.smokesight.tau is not None
        assert ds.smokesight.sigma_tau is not None
    finally:
        ds.close()


def test_background_netcdf_has_confidence(
    synthetic_video: Path, minimal_config: Dict[str, Any], tmp_path: Path
) -> None:
    _, bg, _ = _pipeline(synthetic_video, minimal_config)
    out = tmp_path / "bg.nc"
    to_netcdf(bg, out)
    ds = xr.open_dataset(out)
    try:
        assert "confidence" in ds
        assert ds["confidence"].attrs.get("units") == "1"
        assert ds.attrs.get("background_method") == "temporal_median"
    finally:
        ds.close()


def test_unknown_result_type_raises(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="unknown result type"):
        to_netcdf("not a result", tmp_path / "junk.nc")  # type: ignore[arg-type]


def test_result_to_netcdf_method_now_works(
    synthetic_video: Path, minimal_config: Dict[str, Any], tmp_path: Path
) -> None:
    """The lazy import in _results.to_netcdf should pick up io.to_netcdf
    now that it exists -- no NotImplementedError."""
    cal, _, _ = _pipeline(synthetic_video, minimal_config)
    out = tmp_path / "cal.nc"
    cal.to_netcdf(out)  # would have raised in Phase 3
    assert out.exists()
