SmokeSight
==========

Radiometrically-calibrated plume measurement from EO/IR surveillance video.

SmokeSight converts raw digital-number (DN) frames from electro-optical and
infrared cameras into physically calibrated optical depth :math:`\tau(x, y, t)`,
spectral transmittance, column density, and dispersion coefficients --
every measurement with a documented uncertainty.

Quick start
-----------

.. code-block:: python

   import smokesight as ss

   cal    = ss.calibrate("plume.tif", config="cal.yaml")
   bg     = ss.background(cal, n_frames=100)
   result = ss.retrieve(cal, bg)
   dyn    = ss.dynamics(result)

   result.to_netcdf("output.nc")

See :doc:`tutorials/01_quickstart` for a full walk-through.

Pipeline
--------

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Module
     - Input
     - Output
   * - :mod:`smokesight.calibrate`
     - raw video + cal metadata
     - radiance cube ``L(x, y, t, lambda)``
   * - :mod:`smokesight.background`
     - radiance cube
     - background plate ``L0`` + confidence map
   * - :mod:`smokesight.retrieve`
     - ``L``, ``L0``
     - ``tau(x, y, t)`` + ``sigma_tau``
   * - :mod:`smokesight.dynamics`
     - retrieval result
     - rise velocity, Pasquill-Gifford coefficients
   * - :mod:`smokesight.io`
     - any result
     - CF-NetCDF4 + xarray accessor

Uncertainty
-----------

Every measurement output carries a propagated 1-sigma uncertainty. The
sensor noise model combines shot noise, read noise, flat-field
uncertainty and (optional) atmospheric uncertainty in quadrature.
Beer-Lambert uncertainty propagates analytically through the
:math:`\tau = -\ln(L/L_0)` inversion; centroid and dispersion-fit
uncertainties come from the delta method and ``curve_fit`` covariances.

If a quantity's uncertainty cannot be propagated, the quantity is not
returned. This rule is enforced by reviewers; see
`CONTRIBUTING.md <https://github.com/TasumLuke/SmokeSight/blob/main/CONTRIBUTING.md>`_.

Contents
--------

.. toctree::
   :maxdepth: 2

   tutorials/01_quickstart
   tutorials/02_calibration
   tutorials/03_uncertainty
   api/index

Citation
--------

If you use SmokeSight, please cite via ``CITATION.cff`` in the repository.
A JOSS paper is forthcoming.

Index
-----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
