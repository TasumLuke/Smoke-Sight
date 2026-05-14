"""Tests for smokesight.cli using Click's CliRunner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from click.testing import CliRunner

from smokesight.cli import main


def _write_config(tmp_path: Path, cfg: Dict[str, Any]) -> Path:
    out = tmp_path / "cal.yaml"
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return out


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


def test_calibrate_with_bad_config_exits_nonzero_to_stderr(
    synthetic_video: Path, tmp_path: Path
) -> None:
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("sensor: {bit_depth: 14, ner: 0.002}\n")  # missing gain
    runner = CliRunner(mix_stderr=False)
    res = runner.invoke(
        main, ["calibrate", str(synthetic_video), str(bad_cfg), "-o", "/dev/null"]
    )
    assert res.exit_code == 1
    assert "gain" in (res.stderr or "")
