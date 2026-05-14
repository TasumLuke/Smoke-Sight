"""Click CLI: ``smokesight run`` / ``calibrate`` / ``inspect``.

Errors go to stderr via click.echo(err=True). Exit code 1 on any
SmokeSight* exception; exit 0 on success. Unhandled exceptions
intentionally propagate so users see the traceback (real bugs should
crash loudly rather than being swallowed and turned into a generic 1).
"""

from __future__ import annotations

import sys
from typing import Optional

import click
import numpy as np
import xarray as xr

from smokesight import __version__
from smokesight._types import FloatArray
from smokesight.background import background
from smokesight.calibrate import SmokeSightCalibrationError, calibrate
from smokesight.dynamics import dynamics
from smokesight.io import to_netcdf
from smokesight.retrieve import retrieve


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="smokesight")
def main() -> None:
    """SmokeSight: radiometric plume measurement from EO/IR video."""


@main.command("run")
@click.argument("video", type=click.Path(exists=True, dir_okay=False))
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output",
    "output",
    required=True,
    type=click.Path(dir_okay=False),
    help="Path to write the CF-NetCDF output.",
)
@click.option(
    "--n-frames",
    type=int,
    default=None,
    help="Limit processing to the first N frames.",
)
@click.option(
    "--no-dynamics",
    is_flag=True,
    default=False,
    help="Skip the rise-velocity / dispersion-fit step.",
)
@click.option(
    "--bg-method",
    type=click.Choice(["temporal_median", "temporal_mean", "gmm", "percentile_10"]),
    default="temporal_median",
    show_default=True,
)
@click.option(
    "--bg-frames",
    type=int,
    default=100,
    show_default=True,
    help="How many frames to use for the background estimate.",
)
@click.option(
    "--fps",
    type=float,
    default=None,
    help=(
        "Override the video frame rate (Hz). TIFF stacks rarely carry "
        "an fps in their metadata; pass this if you want dynamics to run."
    ),
)
def run_cmd(
    video: str,
    config: str,
    output: str,
    n_frames: Optional[int],
    no_dynamics: bool,
    bg_method: str,
    bg_frames: int,
    fps: Optional[float],
) -> None:
    """Run the full pipeline: calibrate -> background -> retrieve -> dynamics."""
    try:
        frame_range = (0, n_frames - 1) if n_frames is not None else None
        cal = calibrate(video, config, frame_range=frame_range, progress=True)
        bg = background(
            cal,
            n_frames=min(bg_frames, cal.L.shape[0]),
            method=bg_method,
        )
        res = retrieve(cal, bg, min_confidence=0.5)

        if not no_dynamics:
            effective_fps = fps or cal.metadata.get("fps") or None
            if not effective_fps:
                # No fps anywhere. Skip dynamics rather than crash. TIFF
                # stacks don't carry one in their headers; users can pass
                # --fps to opt in, or --no-dynamics to silence this warning.
                click.echo(
                    "warning: skipping dynamics (no fps available; "
                    "pass --fps to override or --no-dynamics to silence)",
                    err=True,
                )
            else:
                dyn = dynamics(res, fps=effective_fps)
                # to_netcdf doesn't yet accept a NetCDF group, so dynamics
                # goes to <output>.dynamics.nc next to the retrieval output.
                to_netcdf(dyn, output + ".dynamics.nc")
                click.echo(f"wrote {output}.dynamics.nc")

        to_netcdf(res, output)
        click.echo(f"wrote {output}")
    except SmokeSightCalibrationError as exc:
        _fail(f"calibration failed: {exc}")
    except ValueError as exc:
        _fail(f"configuration / input error: {exc}")


@main.command("calibrate")
@click.argument("video", type=click.Path(exists=True, dir_okay=False))
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output",
    "output",
    required=True,
    type=click.Path(dir_okay=False),
    help="Path to write the calibrated radiance cube.",
)
def calibrate_cmd(video: str, config: str, output: str) -> None:
    """Run calibrate only and save the radiance cube."""
    try:
        cal = calibrate(video, config, progress=True)
        to_netcdf(cal, output)
        click.echo(f"wrote {output}")
    except SmokeSightCalibrationError as exc:
        _fail(f"calibration failed: {exc}")
    except ValueError as exc:
        _fail(f"configuration error: {exc}")


@main.command("inspect")
@click.argument("netcdf_path", type=click.Path(exists=True, dir_okay=False))
def inspect_cmd(netcdf_path: str) -> None:
    """Print a human-readable summary of a SmokeSight NetCDF file."""
    ds = xr.open_dataset(netcdf_path)
    try:
        click.echo(f"file:        {netcdf_path}")
        click.echo(f"Conventions: {ds.attrs.get('Conventions', '?')}")
        click.echo(f"source:      {ds.attrs.get('source', '?')}")
        click.echo("variables:")
        for name, var in ds.data_vars.items():
            click.echo(
                f"  {name:<14} shape={tuple(var.shape)}  "
                f"units={var.attrs.get('units', '?')}"
            )
        for var_name in ("tau", "sigma_tau"):
            if var_name in ds:
                _print_stats(var_name, ds[var_name].values)
    finally:
        ds.close()


def _print_stats(name: str, arr: FloatArray) -> None:
    finite = np.isfinite(arr)
    pct_masked = 100.0 * float((~finite).sum()) / float(arr.size)
    if finite.any():
        click.echo(
            f"  {name} stats: min={arr[finite].min():.4g} "
            f"max={arr[finite].max():.4g} mean={arr[finite].mean():.4g} "
            f"masked={pct_masked:.1f}%"
        )
    else:
        click.echo(f"  {name} stats: all masked")


def _fail(message: str) -> None:
    click.echo(message, err=True)
    sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
