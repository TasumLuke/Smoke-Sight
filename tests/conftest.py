import numpy as np
import pytest
import imageio.v3 as iio

from smokesight import background, calibrate, retrieve


@pytest.fixture
def minimal_config():
    return {"sensor": {"gain": 0.012, "bit_depth": 16, "ner": 0.002}}


@pytest.fixture
def full_config(tmp_path):
    h, w = 64, 64
    flat = np.ones((h, w), dtype=np.float32)
    flat[10, 10] = 2.0
    dark = np.full((h, w), 10.0, dtype=np.float32)
    flat_path = tmp_path / "flat.tif"
    dark_path = tmp_path / "dark.tif"
    iio.imwrite(flat_path, flat)
    iio.imwrite(dark_path, dark)
    return {
        "sensor": {
            "gain": 0.012,
            "bit_depth": 16,
            "ner": 0.002,
            "flat_field": str(flat_path),
            "dark_current": str(dark_path),
            "spectral_response": {
                "wavelengths": [3.5],
                "response": [1.0],
            },
        },
        "institution": "SmokeSight Test Lab",
    }


@pytest.fixture
def synthetic_video(tmp_path):
    """50-frame 64x64 TIFF stack with peak tau ~= 0.5 after frame 25."""
    t, h, w = 50, 64, 64
    base_dn = 1000.0
    frames = np.full((t, h, w), base_dn, dtype=np.float32)
    y, x = np.indices((h, w))
    gaussian = np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / (2.0 * 5.0**2))
    tau = 0.5 * gaussian
    frames[25:] = base_dn * np.exp(-tau)
    rng = np.random.default_rng(42)
    frames += rng.normal(0.0, 0.25, size=frames.shape).astype(np.float32)
    path = tmp_path / "synthetic_video.tif"
    iio.imwrite(path, np.clip(frames, 0, 65535).astype(np.uint16))
    return path


@pytest.fixture
def moving_video(tmp_path):
    t, h, w = 30, 64, 64
    base_dn = 1000.0
    frames = np.full((t, h, w), base_dn, dtype=np.float32)
    y, x = np.indices((h, w))
    for k in range(t):
        cy = 52 - 0.35 * k
        gaussian = np.exp(-((x - 32) ** 2 + (y - cy) ** 2) / (2.0 * 5.0**2))
        tau = 0.45 * gaussian
        frames[k] = base_dn * np.exp(-tau)
    path = tmp_path / "moving_video.tif"
    iio.imwrite(path, np.clip(frames, 0, 65535).astype(np.uint16))
    return path


@pytest.fixture
def cal_result(synthetic_video, minimal_config):
    return calibrate(synthetic_video, minimal_config)


@pytest.fixture
def bg_result(cal_result):
    return background(cal_result, n_frames=20)


@pytest.fixture
def retrieval_result(cal_result, bg_result):
    return retrieve(cal_result, bg_result)
