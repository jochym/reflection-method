"""Plotting utilities for reflection-method (optional [plot] extra).

Uses matplotlib. All functions accept an optional `ax` parameter for subplot embedding.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .core import LSQUnivariateSpline


def plot_original_spline(
    x: np.ndarray,
    y: np.ndarray,
    spl1: LSQUnivariateSpline,
    x0_opt: float,
    x0_std: float,
    xlabel: str = "Time",
    ylabel: str = "relative magnitude",
    x_unit: str = "",
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot original data, spline, and x0 vertical line."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    xs = np.linspace(x.min(), x.max(), 600)

    ax.scatter(x, y, s=8, alpha=0.6, label="data", color="C0", edgecolors="none")
    ax.plot(xs, spl1(xs), label="original spline", color="royalblue", linewidth=1.5)

    y_min, y_max = float(y.min()), float(y.max())
    x0_label = f"x₀ = {x0_opt:.2f} ± {x0_std:.2f}"
    if x_unit:
        x0_label += f" {x_unit}"
    ax.axvline(x0_opt, color="red", linewidth=1.5, label=x0_label)

    ax.set_xlabel(f"{xlabel} [{x_unit}]" if x_unit else xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Light curve and original spline with x₀")
    ax.legend(loc="lower left", framealpha=0.7)
    ax.grid(True, alpha=0.3)

    return ax


def plot_scan(
    x0_grid: np.ndarray,
    sigma1: np.ndarray,
    sigma2: np.ndarray,
    spl_sigma: LSQUnivariateSpline,
    x0_opt: float,
    x0_grid_idx_min: int,
    xlabel: str = "x₀",
    ylabel: str = "residual σ",
    x_unit: str = "",
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot σ₁, σ₂, and spline fit to σ₂ with vertical lines at minima."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    xs = np.linspace(x0_grid[0], x0_grid[-1], 200)

    ax.plot(x0_grid, sigma1, label="σ w.r.t. original spline", color="royalblue", linewidth=1)
    ax.plot(x0_grid, sigma2, label="σ w.r.t. refitted spline", color="darkgreen", linewidth=1)
    ax.plot(xs, spl_sigma(xs), label="spline for σ₂", color="darkgreen", linestyle=":", linewidth=1.5)

    ax.axvline(x0_grid[x0_grid_idx_min], color="royalblue", linestyle="--", linewidth=1, alpha=0.7)
    x0_label = f"x₀ = {x0_opt:.2f}"
    if x_unit:
        x0_label += f" {x_unit}"
    ax.axvline(x0_opt, color="red", linewidth=1.5, label=x0_label)

    ax.set_xlabel(f"{xlabel} [{x_unit}]" if x_unit else xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("σ (standard deviation of residual) as a function of the reflection point x₀")
    ax.legend(loc="upper left", framealpha=0.7)
    ax.grid(True, alpha=0.3)

    return ax


def plot_all(
    x: np.ndarray,
    y: np.ndarray,
    spl1: LSQUnivariateSpline,
    xr: np.ndarray,
    spl2: LSQUnivariateSpline,
    x0_opt: float,
    x0_std: float,
    x0_grid: np.ndarray,
    sigma2: np.ndarray,
    spl_sigma: LSQUnivariateSpline,
    x0_boot: np.ndarray,
    xlabel: str = "Time",
    ylabel: str = "relative magnitude",
    x_unit: str = "",
    **kwargs,
) -> Figure:
    """Create the full 4-panel figure.

    Returns the Figure object for saving.
    """
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.62, 0.38], height_ratios=[0.5, 0.5])
    ax_main = fig.add_subplot(gs[:, 0])
    ax_sigma = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, 1])

    # Determine x0 range for right panels (±3σ)
    x0_range = 3 * x0_std
    x0_xlim = (x0_opt - x0_range, x0_opt + x0_range)

    # --- Panel 1: Main ---
    xs = np.linspace(x.min(), x.max(), 600)
    xs2 = np.linspace(min(x.min(), xr.min()), max(x.max(), xr.max()), 600)

    ax_main.scatter(x, y, s=8, alpha=0.6, label="original data", color="C0", edgecolors="none")
    ax_main.scatter(xr, y, s=8, alpha=0.5, label="reflected points", color="orange", edgecolors="none")
    ax_main.plot(xs, spl1(xs), label="original spline", color="royalblue", linewidth=1.5)
    ax_main.plot(xs2, spl2(xs2), label="refitted spline (data + reflection)", color="darkgreen", linestyle="--", linewidth=1.5)

    y_min, y_max = float(y.min()), float(y.max())
    x0_label = f"x₀ = {x0_opt:.2f} ± {x0_std:.2f}"
    if x_unit:
        x0_label += f" {x_unit}"
    ax_main.axvline(x0_opt, color="red", linewidth=1.5, label=x0_label)

    ax_main.set_xlabel(f"{xlabel} [{x_unit}]" if x_unit else xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.set_title(f"Reflected light curve; x₀ = {x0_opt:.2f} ± {x0_std:.2f} {x_unit}".strip())
    ax_main.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax_main.grid(True, alpha=0.3)

    # --- Panel 2: σ₂ ---
    # Dense sampling for smooth spline curve
    xs_sig = np.linspace(x0_grid[0], x0_grid[-1], 500)
    ax_sigma.plot(x0_grid, sigma2, 'o', label="σ₂(x₀)", color="darkgreen", markersize=3, alpha=0.7, linewidth=0)
    ax_sigma.plot(xs_sig, spl_sigma(xs_sig), label="σ₂ spline", color="darkgreen", linestyle="-", linewidth=1.5)
    ax_sigma.axvline(x0_opt, color="red", linewidth=1.5)

    # Set x-limits to ±3σ around x0_opt for consistency with histogram
    ax_sigma.set_xlim(x0_xlim)
    # Set y-limits to show the minimum region nicely
    sigma_min = float(np.min(sigma2))
    sigma_max_in_range = float(np.max(sigma2[(x0_grid >= x0_xlim[0]) & (x0_grid <= x0_xlim[1])]))
    y_pad = 0.1 * (sigma_max_in_range - sigma_min) if sigma_max_in_range > sigma_min else 0.01
    ax_sigma.set_ylim(sigma_min - y_pad, sigma_max_in_range + y_pad)

    ax_sigma.set_ylabel("σ₂")
    ax_sigma.set_title("σ₂ scan")
    ax_sigma.legend(fontsize=8, framealpha=0.7)
    ax_sigma.grid(True, alpha=0.3)
    ax_sigma.tick_params(labelbottom=False)

    # --- Panel 3: Histogram ---
    hist_counts, hist_bins, _ = ax_hist.hist(x0_boot, bins=15, color=(120/255, 120/255, 170/255, 0.6), edgecolor="none", alpha=0.7, label="bootstrap distribution")
    hist_max = float(hist_counts.max())
    ax_hist.axvline(x0_opt, color="red", linewidth=1.5)

    # Same x-limits as σ₂ panel
    ax_hist.set_xlim(x0_xlim)

    x0_label = f"x₀ [{x_unit}]" if x_unit else "x₀"
    ax_hist.set_xlabel(x0_label)
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Bootstrap distribution of x₀")
    ax_hist.legend(fontsize=8, framealpha=0.7)
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig