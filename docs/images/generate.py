"""Generate the README example images.

Re-run this when the test pipeline numerics change or you want to refresh
the marketing figures. Outputs go to this directory:

    docs/images/pipeline.png
    docs/images/tau_recovery.png
    docs/images/uncertainty_components.png
    docs/images/analytic_vs_mc.png
    docs/images/dispersion_fit.png

Run from the repo root:

    python docs/images/generate.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from smokesight._atmos import IdentityAtmos
from smokesight._results import RetrievalResult
from smokesight._uncertainty import (
    monte_carlo,
    radiance_uncertainty,
    tau_uncertainty,
)
from smokesight.background import background
from smokesight.calibrate import calibrate
from smokesight.dynamics import dynamics
from smokesight.retrieve import retrieve

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Match the test fixture so the recovered numbers match what the test suite
# reports. If the fixture changes, regenerate.
PEAK_TAU = 0.5
SIGMA_PX = 5.0
CENTER = (32, 32)
BACKGROUND_DN = 5000
N_FRAMES = 50
N_BG_FRAMES = 20
H, W = 64, 64
BIT_DEPTH = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_synthetic_video(path: Path) -> tuple[Path, np.ndarray]:
    """Same recipe as tests/conftest.py; returns the file plus the ground truth tau."""
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = CENTER
    tau_truth = PEAK_TAU * np.exp(
        -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * SIGMA_PX**2)
    )
    transmittance = np.exp(-tau_truth)

    rng = np.random.default_rng(0)
    L_with_plume = (BACKGROUND_DN * transmittance).astype(np.float64)
    L_no_plume = np.full_like(L_with_plume, float(BACKGROUND_DN))
    max_dn = 2**BIT_DEPTH - 1

    frames = []
    for i in range(N_FRAMES):
        clean = L_no_plume if i < N_BG_FRAMES else L_with_plume
        noisy = (
            clean
            + rng.normal(0.0, np.sqrt(clean), size=clean.shape)
            + rng.normal(0.0, 5.0, size=clean.shape)
        )
        frames.append(np.clip(noisy, 0, max_dn).astype(np.uint16))

    imageio.mimwrite(path, frames)
    return path, tau_truth


def run_pipeline(video_path: Path):
    cfg = {"sensor": {"gain": 0.012, "bit_depth": BIT_DEPTH, "ner": 0.002}}
    cal = calibrate(video_path, cfg, progress=False)
    bg = background(cal, n_frames=N_BG_FRAMES)
    res = retrieve(cal, bg, min_confidence=0.0)
    return cal, bg, res


# ---------------------------------------------------------------------------
# Figure 1 -- pipeline.png (the headline)
# ---------------------------------------------------------------------------


def fig_pipeline(cal, bg, res, tau_truth: np.ndarray) -> None:
    frame = imageio.mimread(VIDEO_PATH, memtest=False)[30]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    im0 = axes[0].imshow(frame, cmap="gray")
    axes[0].set_title("(1) Raw DN frame\n(uint16)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, label="DN")

    im1 = axes[1].imshow(cal.L[30, ..., 0], cmap="viridis")
    axes[1].set_title("(2) Calibrated radiance L\n(W m$^{-2}$ sr$^{-1}$ µm$^{-1}$)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(res.tau[30], cmap="RdBu_r", vmin=-0.1, vmax=0.6)
    axes[2].set_title("(3) Retrieved τ\n(optical depth)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(res.sigma_tau[30], cmap="magma", vmin=0)
    axes[3].set_title("(4) σ$_τ$\n(1-sigma uncertainty)")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("SmokeSight pipeline: DN → radiance → τ → σ$_τ$", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pipeline.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'pipeline.png'}")


# ---------------------------------------------------------------------------
# Figure 2 -- tau_recovery.png
# ---------------------------------------------------------------------------


def fig_tau_recovery(res, tau_truth: np.ndarray) -> None:
    # Average tau across the plume-present frames (20..49) for the cleanest
    # comparison against ground truth.
    tau_recovered = np.nanmean(res.tau[N_BG_FRAMES:], axis=0)
    residual = tau_recovered - tau_truth

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    im0 = axes[0].imshow(tau_truth, cmap="RdBu_r", vmin=0, vmax=0.6)
    axes[0].set_title(f"Ground truth\npeak τ = {tau_truth.max():.3f}")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(tau_recovered, cmap="RdBu_r", vmin=0, vmax=0.6)
    peak_recovered = float(tau_recovered[CENTER[1], CENTER[0]])
    axes[1].set_title(f"Recovered\npeak τ = {peak_recovered:.3f}")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(residual, cmap="seismic", vmin=-0.05, vmax=0.05)
    axes[2].set_title(
        f"Residual (recovered − truth)\nRMS = {float(np.sqrt(np.nanmean(residual**2))):.4f}"
    )
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter([CENTER[0]], [CENTER[1]], marker="x", color="black", s=60)

    fig.suptitle(
        "Plume centre recovery (±10% tolerance is the spec gate)", y=1.02
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tau_recovery.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'tau_recovery.png'}")


# ---------------------------------------------------------------------------
# Figure 3 -- uncertainty_components.png
# ---------------------------------------------------------------------------


@dataclass
class _StubSensor:
    gain: float = 0.012
    noise_equivalent_radiance: float = 0.002
    flat_field_relative_uncertainty: float = 0.01


def fig_uncertainty_components() -> None:
    sensor = _StubSensor()
    L = np.logspace(-3, 3, 200)
    total = radiance_uncertainty(L, sensor, IdentityAtmos())
    shot = np.sqrt(L * sensor.gain)
    read = np.full_like(L, sensor.noise_equivalent_radiance)
    flat = L * sensor.flat_field_relative_uncertainty

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.loglog(L, total, "k-", lw=2.5, label="total σ$_L$")
    ax.loglog(L, shot, "--", color="tab:blue", label="shot noise (∝√L)")
    ax.loglog(L, flat, "--", color="tab:orange", label="flat-field (1% of L)")
    ax.loglog(L, read, ":", color="tab:green", label="read noise floor (NER)")

    # Annotate the regime boundaries
    L_crossover = (sensor.noise_equivalent_radiance**2) / sensor.gain
    ax.axvline(L_crossover, color="grey", alpha=0.3, ls=":")
    ax.text(
        L_crossover * 1.3,
        2e-4,
        "shot = read",
        rotation=90,
        color="grey",
        fontsize=9,
        va="bottom",
    )

    ax.set_xlabel("L  [W m$^{-2}$ sr$^{-1}$ µm$^{-1}$]")
    ax.set_ylabel("σ$_L$  [same units]")
    ax.set_title("Radiance-uncertainty components (combined in quadrature)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "uncertainty_components.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'uncertainty_components.png'}")


# ---------------------------------------------------------------------------
# Figure 4 -- analytic_vs_mc.png
# ---------------------------------------------------------------------------


def fig_analytic_vs_mc() -> None:
    L = np.linspace(0.1, 1.0, 50)
    L0 = np.ones_like(L)
    sigma_L = np.full_like(L, 0.05)
    sigma_L0 = np.full_like(L, 0.05)

    analytic = tau_uncertainty(L, sigma_L, L0, sigma_L0)

    def beer_lambert(L_in, L0_in):
        return -np.log(L_in / L0_in)

    _, sigma_mc = monte_carlo(
        beer_lambert, [L, L0], [sigma_L, sigma_L0], n=2000
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(L, analytic, "k-", lw=2, label="analytic propagation")
    ax.plot(L, sigma_mc, "o", markersize=4, color="tab:red", label="Monte Carlo (n=2000)")
    ax.set_xlabel("L / L$_0$")
    ax.set_ylabel("σ$_τ$")
    ax.set_title(
        "Validation: analytic σ$_τ$ matches Monte Carlo where the linearisation holds"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "analytic_vs_mc.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'analytic_vs_mc.png'}")


# ---------------------------------------------------------------------------
# Figure 5 -- dispersion_fit.png
# ---------------------------------------------------------------------------


def fig_dispersion_fit() -> None:
    """Same synthetic as tests/test_dynamics.py::test_pg_fit_recovers_known_sigma."""
    a_true, b_true = 1.0, 0.7
    pixel_scale = 0.25
    t = 30
    h, w = 96, 96
    yy, xx = np.mgrid[0:h, 0:w]
    src_x, src_y = w // 2, h - 1

    tau = np.zeros((t, h, w), dtype=np.float32)
    distances = []
    sigmas_truth = []
    for i in range(1, t + 1):
        cy = src_y - i
        dist_m = pixel_scale * i
        sigma_m = a_true * (dist_m**b_true)
        sigma_px = max(sigma_m / pixel_scale, 1.0)
        tau[i - 1] = 0.5 * np.exp(
            -((xx - src_x) ** 2 + (yy - cy) ** 2) / (2 * sigma_px**2)
        ).astype(np.float32)
        distances.append(dist_m)
        sigmas_truth.append(sigma_m)

    res = RetrievalResult(
        tau=tau,
        sigma_tau=np.full_like(tau, 0.01),
        mask=np.ones_like(tau),
        metadata={"fps": 10.0, "pixel_scale": pixel_scale},
    )
    dyn = dynamics(res, source_location=(src_x, src_y))
    a_fit, b_fit = float(dyn.sigma_y_coeffs[0]), float(dyn.sigma_y_coeffs[1])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_smooth = np.linspace(min(distances), max(distances), 200)
    ax.plot(
        x_smooth, a_true * x_smooth**b_true, "k-", lw=2, label=f"truth: σ = {a_true:.2f} x^{b_true:.2f}"
    )
    ax.plot(distances, sigmas_truth, "o", markersize=4, color="tab:blue", label="per-frame truth")
    ax.plot(
        x_smooth,
        a_fit * x_smooth**b_fit,
        "--",
        color="tab:red",
        lw=2,
        label=f"fit:   σ = {a_fit:.2f} x^{b_fit:.2f}",
    )
    err_pct = abs(a_fit - a_true) / a_true * 100
    ax.text(
        0.95,
        0.05,
        f"recovery error on a: {err_pct:.1f}% (spec bar: ±20%)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax.set_xlabel("downwind distance [m]")
    ax.set_ylabel("σ$_y$  [m]")
    ax.set_title("Pasquill-Gifford power-law fit on a synthetic dispersing plume")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "dispersion_fit.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'dispersion_fit.png'}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


VIDEO_PATH = OUT_DIR / "_synthetic.tif"


def main() -> None:
    _, tau_truth = build_synthetic_video(VIDEO_PATH)
    cal, bg, res = run_pipeline(VIDEO_PATH)

    fig_pipeline(cal, bg, res, tau_truth)
    fig_tau_recovery(res, tau_truth)
    fig_uncertainty_components()
    fig_analytic_vs_mc()
    fig_dispersion_fit()

    # clean up the intermediate video so it doesn't get committed
    VIDEO_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
