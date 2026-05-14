import numpy as np

from smokesight import calibrate, retrieve
from smokesight.results import CalibrationResult


def test_retrieve_tau_ground_truth(retrieval_result):
    peak = np.nanmax(retrieval_result.tau[30])
    assert np.isclose(peak, 0.5, rtol=0.10)


def test_retrieve_sigma_tau_finite_where_valid(retrieval_result):
    assert np.all(np.isfinite(retrieval_result.sigma_tau[retrieval_result.mask]))


def test_retrieve_sigma_tau_nan_where_masked(retrieval_result):
    assert np.all(np.isnan(retrieval_result.sigma_tau[~retrieval_result.mask]))


def test_retrieve_low_confidence_masked(cal_result, bg_result):
    bg_result.confidence[0, 0] = 0.0
    ret = retrieve(cal_result, bg_result)
    assert not np.any(ret.mask[:, 0, 0])


def test_retrieve_tau_max_respected(cal_result, bg_result):
    ret = retrieve(cal_result, bg_result, tau_max=0.2)
    assert np.nanmax(ret.tau) <= 0.2


def test_retrieve_multi_band(cal_result, bg_result):
    multi = CalibrationResult(
        L=np.repeat(cal_result.L, 4, axis=-1),
        sigma_L=np.repeat(cal_result.sigma_L, 4, axis=-1),
        metadata=dict(cal_result.metadata),
        sensor=cal_result.sensor,
        atmos=cal_result.atmos,
    )
    bg_result.L0 = np.repeat(bg_result.L0, 4, axis=-1)
    bg_result.sigma_L0 = np.repeat(bg_result.sigma_L0, 4, axis=-1)
    ret = retrieve(multi, bg_result, wavelengths=[3.5, 4.0, 4.5, 5.0])
    assert ret.T_lambda.shape[-1] == 4
