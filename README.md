# SmokeSight

[![CI](https://github.com/TasumLuke/SmokeSight/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/TasumLuke/SmokeSight/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TasumLuke/SmokeSight/branch/main/graph/badge.svg)](https://codecov.io/gh/TasumLuke/SmokeSight)
[![PyPI](https://img.shields.io/pypi/v/smokesight.svg)](https://pypi.org/project/smokesight/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![JOSS](https://img.shields.io/badge/JOSS-submitted-orange.svg)](https://joss.theoj.org/)

Radiometrically-calibrated plume measurement from EO/IR surveillance video.

In defence and atmospheric research, detection is a solved problem. SmokeSight
does the part after that: per-pixel optical depth `tau(x, y, t)`,
wavelength-dependent transmittance, line-of-sight column density, and
Pasquill–Gifford dispersion coefficients — every measurement with a documented
1-sigma uncertainty.

---

## Why this exists

Three capabilities that do not coexist in any open package today:

- **Imagery-to-radiometry.** DN-to-radiance with sensor model, atmospheric
  path correction, and per-pixel noise propagation. Closed military codes do
  this. Open tools assume the radiance is already calibrated.
- **Uncertainty-propagated inversion.** Per-pixel `sigma_tau` from the
  Beer-Lambert step makes the output a measurement, not a visualisation.
  Every numeric quantity that ships ships with a 1-sigma error bar; if it
  can't, it isn't returned.
- **Atmospheric-science output formats.** CF-1.9 NetCDF4 plus an xarray
  accessor means dispersion modellers (HYSPLIT, FLEXPART, STILT), STE
  solvers, and sensor-evaluation researchers can consume the data directly,
  no conversion step.

---

## Install

```bash
pip install smokesight                  # core
pip install "smokesight[calibrate]"     # adds py6s / pymodtran for atmos correction
pip install "smokesight[dev]"           # tests, mypy, black, isort, sphinx, ...
```

Supports Python 3.8 – 3.11.

---

## Quick start

```python
import smokesight as ss

cal    = ss.calibrate("plume.tif", config="cal.yaml")
bg     = ss.background(cal, n_frames=100)
result = ss.retrieve(cal, bg)

# result.tau        -- optical depth tau(x, y, t)
# result.sigma_tau  -- per-pixel 1-sigma uncertainty
# result.T_lambda   -- transmittance cube (multi-band input)
# result.mask       -- valid-pixel mask

dyn = ss.dynamics(result)
print(dyn.rise_velocity)       # m/s
print(dyn.sigma_y_coeffs)      # Pasquill-Gifford (a, b) for sigma_y = a * x^b

result.to_netcdf("output.nc")
```

Output is a CF-1.9 NetCDF4 file and opens directly in xarray:

```python
import xarray as xr
ds = xr.open_dataset("output.nc")
ds.smokesight.plot_frame(t=30)   # registered accessor
```

See `docs/tutorials/01_quickstart.ipynb` for an end-to-end runnable example.

---

## Pipeline

| Module        | Input                          | Output                              |
| ------------- | ------------------------------ | ----------------------------------- |
| `calibrate`   | raw video + cal metadata       | radiance cube `L(x, y, t, lambda)`  |
| `background`  | radiance cube                  | background plate `L0` + confidence  |
| `retrieve`    | `L`, `L0`                      | `tau(x, y, t)` + `sigma_tau`        |
| `dynamics`    | retrieval result               | rise velocity, `sigma_y`, `sigma_z` |
| `io`          | any result                     | CF-NetCDF4 + xarray accessor        |

---

## Outputs

Every result carries the central value, a 1-sigma uncertainty, and metadata
sufficient to reproduce it.

| Symbol          | Quantity                              | Units                       |
| --------------- | ------------------------------------- | --------------------------- |
| `L`             | calibrated radiance                   | W m<sup>-2</sup> sr<sup>-1</sup> µm<sup>-1</sup> |
| `sigma_L`       | 1-sigma uncertainty on `L`            | same as `L`                 |
| `L0`            | background radiance                   | same as `L`                 |
| `tau`           | optical depth                         | dimensionless               |
| `sigma_tau`     | 1-sigma uncertainty on `tau`          | dimensionless               |
| `T_lambda`      | spectral transmittance                | dimensionless               |
| `N`             | column number density (optional)      | mol m<sup>-2</sup>          |
| `rise_velocity` | buoyant plume rise velocity           | m s<sup>-1</sup>            |
| `sigma_y, sigma_z` | Pasquill-Gifford dispersion fits   | m                           |

---

## Uncertainty

`sigma_L` combines shot noise, read noise, flat-field calibration uncertainty,
and atmospheric uncertainty in quadrature. `sigma_tau` propagates analytically
through Beer-Lambert. Centroid and dispersion-fit uncertainties use the
delta method and `scipy.optimize.curve_fit` covariances respectively. Monte
Carlo propagation is available as a fallback for non-linear paths; its RNG is
internally seeded so two runs on identical inputs produce identical output.

Masked pixels (low background confidence, ratio out of physical bounds, tau
beyond `tau_max`) come back as `NaN` in both `tau` and `sigma_tau` — a masked
measurement never carries a numeric uncertainty.

---

## Contributing

Open an issue before any non-trivial change. The one rule that's
non-negotiable: every measurement output ships with a propagated
uncertainty; if you can't propagate it, you can't ship it. See
[CONTRIBUTING.md](CONTRIBUTING.md) for setup and conventions.

```bash
git clone https://github.com/TasumLuke/SmokeSight
cd SmokeSight
pip install -e ".[dev]"
pre-commit install
pytest                  # 110+ tests, ~25 seconds
```

CI runs black, isort, flake8, mypy --strict, and pytest with a 90% coverage
gate on Python 3.8 / 3.10 / 3.11.

---

## Citation

If you use SmokeSight in published work, please cite via the
[CITATION.cff](CITATION.cff) metadata or:

```bibtex
@software{smokesight,
  title  = {SmokeSight: Radiometric plume measurement from EO/IR video},
  author = {SmokeSight Contributors},
  year   = {2026},
  url    = {https://github.com/TasumLuke/SmokeSight},
  license = {Apache-2.0}
}
```

A JOSS paper is forthcoming.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
