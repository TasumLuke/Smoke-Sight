API reference
=============

Public API
----------

The five user-facing functions plus the result dataclasses.

.. autofunction:: smokesight.calibrate
.. autofunction:: smokesight.background
.. autofunction:: smokesight.retrieve
.. autofunction:: smokesight.dynamics

I/O
---

.. autofunction:: smokesight.io.to_netcdf
.. autoclass:: smokesight.io.SmokeSightAccessor
   :members:

Result dataclasses
------------------

.. autoclass:: smokesight._results.CalibrationResult
   :members:
.. autoclass:: smokesight._results.BackgroundResult
   :members:
.. autoclass:: smokesight._results.RetrievalResult
   :members:
.. autoclass:: smokesight._results.DynamicsResult
   :members:

Sensor and atmosphere models
----------------------------

.. autoclass:: smokesight._sensor.SensorModel
   :members:
.. autoclass:: smokesight._atmos.IdentityAtmos
   :members:
.. autoclass:: smokesight._atmos.AtmosModel
   :members:
.. autofunction:: smokesight._atmos.make_atmos

Geometry
--------

.. autoclass:: smokesight._geometry.CameraGeometry
   :members:
.. autofunction:: smokesight._geometry.compute_pixel_scale
.. autofunction:: smokesight._geometry.project_to_ground

Uncertainty propagation
-----------------------

.. autofunction:: smokesight._uncertainty.radiance_uncertainty
.. autofunction:: smokesight._uncertainty.tau_uncertainty
.. autofunction:: smokesight._uncertainty.centroid_uncertainty
.. autofunction:: smokesight._uncertainty.gaussian_fit_uncertainty
.. autofunction:: smokesight._uncertainty.monte_carlo
