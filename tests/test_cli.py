"""Tests for smokesight.cli using Click's CliRunner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from click.testing import CliRunner, Result

from smokesight.cli import main


def _write_config(tmp_path: Path, cfg: Dict[str, Any]) -> Path:
    out = tmp_path / "cal.yaml"
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return out


def _all_output(res: Result) -> str:
    """stdout+stderr regardless of Click version.

    Click 8.1 merges both streams into ``res.output``; 8.2+ keeps them
    separate. The merged view is what tests actually want when they're
    looking for a message regardless of where it landed.
    """
    text = res.output or ""
    try:
        text += res.stderr or ""
    except ValueError:
        # Click 8.1: stderr not captured separately, already in output.
        pass
    return text


def test_version_flag() -> None:
    runner = CliRunner()
    res = runner.invoke(main, ["--version"])
    assert res.exit_code == 0
    assert "smokesight" in res.output.lower()


def test_help_lists_subcommands() -> None:
    runner = CliRunner()
    res = runner.invoke(main, ["--help"])
    assert res.exit_code == 0
    for cmd in ("run", "calibrate", "inspect"):
        assert cmd in res.output


def test_calibrate_subcommand_writes_netcdf(
    synthetic_video: Path,
    minimal_config: Dict[str, Any],
    tmp_path: Path,
) -> None:
    cfg_path = _write_config(tmp_path, minimal_config)
    output = tmp_path / "cal.nc"
    runner = CliRunner()
    res = runner.invoke(
        main, ["calibrate", str(synthetic_video), str(cfg_path), "-o", str(output)]
    )
    assert res.exit_code == 0, res.output
    assert output.exists()


def test_run_subcommand_full_pipeline(
    synthetic_video: Path,
    minimal_config: Dict[str, Any],
    tmp_path: Path,
) -> None:
    cfg_path = _write_config(tmp_path, minimal_config)
    output = tmp_path / "ret.nc"
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "run",
            str(synthetic_video),
            str(cfg_path),
            "-o",
            str(output),
            "--bg-frames",
            "20",
            "--no-dynamics",
        ],
    )
    assert res.exit_code == 0, res.output
    assert output.exists()


def test_inspect_subcommand_prints_summary(
    synthetic_video: Path,
    minimal_config: Dict[str, Any],
    tmp_path: Path,
) -> None:
    # First produce a NetCDF file
    cfg_path = _write_config(tmp_path, minimal_config)
    output = tmp_path / "cal.nc"
    runner = CliRunner()
    runner.invoke(
        main, ["calibrate", str(synthetic_video), str(cfg_path), "-o", str(output)]
    )

    res = runner.invoke(main, ["inspect", str(output)])
    assert res.exit_code == 0, res.output
    assert "Conventions" in res.output
    assert "variables:" in res.output


def test_calibrate_with_bad_config_exits_nonzero(
    synthetic_video: Path, tmp_path: Path
) -> None:
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("sensor: {bit_depth: 14, ner: 0.002}\n")  # missing gain
    runner = CliRunner()
    res = runner.invoke(
        main, ["calibrate", str(synthetic_video), str(bad_cfg), "-o", "/dev/null"]
    )
    assert res.exit_code == 1
    # res.output combines stdout + stderr on older Click; both work here
    combined = _all_output(res)
    assert "gain" in combined


def test_run_skips_dynamics_when_fps_unknown(
    synthetic_video: Path,
    minimal_config: Dict[str, Any],
    tmp_path: Path,
) -> None:
    """TIFF stacks don't carry fps metadata. The CLI should warn and skip
    dynamics rather than fail the whole run."""
    cfg_path = _write_config(tmp_path, minimal_config)
    output = tmp_path / "ret.nc"
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "run",
            str(synthetic_video),
            str(cfg_path),
            "-o",
            str(output),
            "--bg-frames",
            "20",
            "--bg-method",
            "temporal_mean",
        ],
    )
    assert res.exit_code == 0, res.output
    assert output.exists()
    # Click 8.1 mixes stderr into output; 8.2+ keeps them separate.
    combined = _all_output(res)
    assert "skipping dynamics" in combined
    # no dynamics file produced
    assert not (tmp_path / "ret.nc.dynamics.nc").exists()


def test_run_with_explicit_fps_writes_dynamics(
    synthetic_video: Path,
    minimal_config: Dict[str, Any],
    tmp_path: Path,
) -> None:
    """Passing --fps lets the run command drive dynamics through to its
    own NetCDF output."""
    cfg_path = _write_config(tmp_path, minimal_config)
    output = tmp_path / "ret.nc"
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "run",
            str(synthetic_video),
            str(cfg_path),
            "-o",
            str(output),
            "--bg-frames",
            "20",
            "--fps",
            "25.0",
        ],
    )
    assert res.exit_code == 0, res.output
    assert output.exists()
    dyn_out = tmp_path / "ret.nc.dynamics.nc"
    assert dyn_out.exists()


def test_inspect_prints_tau_stats_when_present(
    synthetic_video: Path,
    minimal_config: Dict[str, Any],
    tmp_path: Path,
) -> None:
    """The inspect subcommand should report tau min/max/mean for a retrieval file."""
    cfg_path = _write_config(tmp_path, minimal_config)
    output = tmp_path / "ret.nc"
    runner = CliRunner()
    runner.invoke(
        main,
        [
            "run",
            str(synthetic_video),
            str(cfg_path),
            "-o",
            str(output),
            "--bg-frames",
            "20",
            "--no-dynamics",
        ],
    )
    res = runner.invoke(main, ["inspect", str(output)])
    assert res.exit_code == 0, res.output
    assert "tau stats" in res.output or "all masked" in res.output
