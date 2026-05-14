import numpy as np
import pytest

from smokesight import background


def test_background_median_shape(cal_result):
    bg = background(cal_result, n_frames=20)
    assert bg.L0.shape == (64, 64, 1)


def test_background_confidence_range(cal_result):
    bg = background(cal_result, n_frames=20)
    assert np.all(bg.confidence >= 0.0)
    assert np.all(bg.confidence <= 1.0)


def test_background_all_methods(cal_result):
    for method in ["temporal_median", "temporal_mean", "gmm", "percentile_10"]:
        bg = background(cal_result, n_frames=20, method=method)
        assert bg.L0.shape == (64, 64, 1)


def test_background_mask_respected(cal_result):
    mask = np.zeros((64, 64), dtype=bool)
    mask[3, 4] = True
    bg = background(cal_result, n_frames=20, mask=mask)
    assert bg.confidence[3, 4] == 0.0


def test_background_n_frames_gt_available_raises(cal_result):
    with pytest.raises(ValueError):
        background(cal_result, n_frames=999)
