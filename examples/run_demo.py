import numpy as np

import smokesight as ss
from smokesight.results import CalibrationResult


# Fake calibrated video: 20 frames, 64x64 pixels, 1 wavelength band
L = np.ones((20, 64, 64, 1), dtype=np.float32)

# Add a fake plume by darkening a square region
L[5:, 25:40, 25:40, 0] *= 0.7

sigma_L = np.full_like(L, 0.01)

cal = CalibrationResult(
    L=L,
    sigma_L=sigma_L,
    metadata={
        "fps": 25.0,
        "n_frames": 20,
        "height": 64,
        "width": 64,
        "bit_depth": 16,
    },
    sensor=None,
    atmos=None,
)

bg = ss.background(cal, n_frames=5)
result = ss.retrieve(cal, bg)
dyn = ss.dynamics(result)

print("Tau shape:", result.tau.shape)
print("Mean tau:", np.nanmean(result.tau))
print("Max tau:", np.nanmax(result.tau))
print("Rise velocity:", dyn.rise_velocity)

result.to_netcdf("demo_output.nc")
print("Saved demo_output.nc")