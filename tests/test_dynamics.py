import numpy as np

from smokesight import background, calibrate, dynamics, retrieve


def _moving_retrieval(moving_video, minimal_config):
    cal = calibrate(moving_video, minimal_config)
    bg = background(cal, n_frames=1)
    return retrieve(cal, bg)


def test_dynamics_rise_velocity_sign(moving_video, minimal_config):
    ret = _moving_retrieval(moving_video, minimal_config)
    dyn = dynamics(ret, fps=10.0, pixel_scale=0.5)
    assert dyn.rise_velocity > 0


def test_dynamics_rise_velocity_uncertainty_finite(moving_video, minimal_config):
    ret = _moving_retrieval(moving_video, minimal_config)
    dyn = dynamics(ret, fps=10.0, pixel_scale=0.5)
    assert np.isfinite(dyn.sigma_rise_velocity)
    assert dyn.sigma_rise_velocity > 0


def test_dynamics_sigma_y_coeffs_shape(retrieval_result):
    dyn = dynamics(retrieval_result)
    assert dyn.sigma_y_coeffs.shape == (2,)


def test_dynamics_pg_fit_synthetic(retrieval_result):
    dyn = dynamics(retrieval_result)
    assert np.isfinite(dyn.sigma_y_coeffs[0])
    assert dyn.sigma_y_coeffs[0] > 0
