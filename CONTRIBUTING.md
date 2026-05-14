# Contributing to SmokeSight

Thanks for considering a contribution. This project is small and the
rules below exist to keep it that way.

## Before you start work

Open an issue first if your change is non-trivial. "Non-trivial" means
anything more than:

- a typo or docstring fix,
- a bug fix with an obvious test, or
- adding test coverage for an existing code path.

Anything bigger gets a quick chat first. New dispersion models,
alternative retrieval algorithms, breaking API changes, that kind of
thing. Saves both of us time when scope turns out to be different than
either side assumed.

## The one rule that's non-negotiable

**Every measurement output must carry a propagated uncertainty.**

If a function in `smokesight/` returns a number that came from a sensor
reading, that number ships with a matching `sigma_*` field. If the
uncertainty can't be propagated for some reason, the quantity is not
returned at all. We don't ship measurements without error bars.

This is enforced by convention, not by the type system, so reviewers
look for it explicitly. The architecture in `smokesight/_uncertainty.py`
gives you the building blocks (analytic propagation, delta method,
Monte Carlo); use them.

## Development setup

```bash
git clone https://github.com/TasumLuke/SmokeSight.git
cd SmokeSight
pip install -e ".[dev]"
pre-commit install     # optional but recommended; runs black/isort/flake8/mypy
pytest                 # 110+ tests, ~25 seconds
```

The `[calibrate]` extra (`pip install -e ".[dev,calibrate]"`) pulls in
`py6s` and `pymodtran` for atmospheric correction. Those are optional;
the test suite skips two cases when they're missing.

## What gets checked in CI

Every push runs, on Python 3.8 / 3.10 / 3.11:

- `black --check` + `isort --check-only` (formatting)
- `flake8` (lint, 88-char line length)
- `mypy --strict` (3.9+ analysis baseline)
- `pytest --cov-fail-under=90` (coverage gate)

CI will block your PR if any of these fail. The easiest way to find out
locally is to run them yourself:

```bash
black smokesight tests
isort smokesight tests
flake8 smokesight tests
mypy smokesight
pytest
```

## Conventions

- **Public API** is exactly `calibrate`, `background`, `retrieve`,
  `dynamics`, `io` plus `__version__`. Anything else is private
  (`_sensor`, `_atmos`, `_geometry`, `_uncertainty`, `_results`) and
  may change without a deprecation cycle.
- **Type-annotate everything public.** mypy `--strict` is enforced; new
  code that can't type-check needs a brief comment explaining why and
  a targeted `# type: ignore[code]` rather than a blanket ignore.
- **Tests live in `tests/`.** Name them after what they assert, not
  after the function under test. `test_tau_recovers_ground_truth` is
  more useful in a failure report than `test_retrieve_3`.
- **No new runtime dependencies without justification.** Each entry in
  `install_requires` ought to earn its keep. If a feature you want
  needs a new dep, mention it in the issue.

## Reporting bugs

Open an issue with:

- the SmokeSight version (`python -c "import smokesight; print(smokesight.__version__)"`),
- Python version and OS,
- a minimal reproducing snippet (a synthetic video and config dict is
  fine, anything that runs against the test fixtures works), and
- what you expected vs. what happened.

## Security

If you find a vulnerability, please open a private security advisory
on GitHub rather than a public issue.

## Code of conduct

Be respectful. We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
