"""SmokeSight: radiometric plume measurement from EO/IR video.

The public API is exactly five names plus ``__version__``::

    import smokesight as ss

    cal    = ss.calibrate("plume.tif", config="cal.yaml")
    bg     = ss.background(cal)
    result = ss.retrieve(cal, bg)
    dyn    = ss.dynamics(result)
    ss.io.to_netcdf(result, "out.nc")

Anything starting with ``_`` is private and may change without notice.
"""

from __future__ import annotations

# __version__ has to come BEFORE the submodule imports below: smokesight.io
# does `from smokesight import __version__` and we're still mid-import here,
# so the name has to exist before we trigger that.
__version__: str = "0.1.0"

from smokesight import io  # noqa: E402
from smokesight.background import background  # noqa: E402
from smokesight.calibrate import calibrate  # noqa: E402
from smokesight.dynamics import dynamics  # noqa: E402
from smokesight.retrieve import retrieve  # noqa: E402

__all__ = [
    "calibrate",
    "background",
    "retrieve",
    "dynamics",
    "io",
    "__version__",
]
