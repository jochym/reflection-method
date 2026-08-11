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
    xlabel: str = "x",
    ylabel: str = "y",
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot original data, spline, and x0 vertical line."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    xs = np.linspace(x.min(), x.max(), 600)

    ax.scatter(x, y, s=8, alpha=0.6, label="data", color="C0", edgecolors="none")
    ax.plot(xs, spl1(xs), label="original spline", color="royalblue", linewidth=1.5)

    y_min, y_max = float(y.min()), float(y.max())
    ax.axvline(x0_opt, color="red", linewidth=1.5, label=f"x₀ = {x0_opt:.2f} ± {x0_std:.2f}")

    ax.set_xlabel(xlabel)
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
    ax.axvline(x0_opt, color="red", linewidth=1.5, label=f"x₀ = {x0_opt:.2f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("σ (standard deviation of residual) as a function of the reflection point x₀")
    ax.legend(loc="upper left", framealpha=0.7)
    ax.grid(True, alpha=0.3)

    return ax


def plot_composite(
    x: np.ndarray,
    y: np.ndarray,
    xr: np.ndarray,
    spl1: LSQUnivariateSpline,
    spl2: LSQUnivariateSpline,
    x0_opt: float,
    x0_std: float,
    xlabel: str = "x",
    ylabel: str = "y",
    x0_grid: Optional[np.ndarray] = None,
    sigma2: Optional[np.ndarray] = None,
    spl_sigma: Optional[LSQUnivariateSpline] = None,
    x0_boot: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """4-panel composite: main light curve, σ₂ scan, bootstrap histogram."""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[0.62, 0.38], height_ratios=[0.5, 0.5])
        ax_main = fig.add_subplot(gs[:, 0])
        ax_sigma = fig.add_subplot(gs[0, 1])
        ax_hist = fig.add_subplot(gs[1, 1])
    else:
        # Assume ax is the main axis of a pre-created grid
        fig = ax.figure
        # Create subplots sharing the figure
        gs = fig.add_gridspec(2, 2, width_ratios=[0.62, 0.38], height_ratios=[0.5, 0.5])
        ax_main = fig.add_subplot(gs[:, 0])
        ax_sigma = fig.add_subplot(gs[0, 1])
        ax_hist = fig.add_subplot(gs[1, 1])

    # --- Panel 1: Main light curve with reflection ---
    xs = np.linspace(x.min(), x.max(), 600)
    xs2 = np.linspace(min(x.min(), xr.min()), max(x.max(), xr.max()), 600)

    ax_main.scatter(x, y, s=8, alpha=0.6, label="original data", color="C0", edgecolors="none")
    ax_main.scatter(xr, y, s=8, alpha=0.5, label="reflected points", color="orange", edgecolors="none")
    ax_main.plot(xs, spl1(xs), label="original spline", color="royalblue", linewidth=1.5)
    ax_main.plot(xs2, spl2(xs2), label="refitted spline (data + reflection)", color="darkgreen", linestyle="--", linewidth=1.5)

    y_min, y_max = float(y.min()), float(y.max())
    ax_main.axvline(x0_opt, color="red", linewidth=1.5, label=f"x₀ = {x0_opt:.2f} ± {x0_std:.2f}")

    # CI rectangle
    from reflection_method.core import find_x0
    # We'll compute CI from x0_grid if provided, otherwise approximate
    if x0_grid is not None and sigma2 is not None:
        from scipy.interpolate import UnivariateSpline
        spl_sig = UnivariateSpline(x0_grid, sigma2, k=3, s=0)
        # Find 16th/84th percentiles via interpolation
        # For now just use simple approx
        pass

    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.set_title(f"Reflected light curve; x₀ = {x0_opt:.2f} ± {x0_std:.2f}")
    ax_main.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax_main.grid(True, alpha=0.3)

    # --- Panel 2: σ₂ scan ---
    if x0_grid is not None and sigma2 is not None and spl_sigma is not None:
        xs_sig = np.linspace(x0_grid[0], x0_grid[-1], 200)
        ax_sigma.plot(x0_grid, sigma2, label="σ₂(x₀)", color="darkgreen", linewidth=1)
        ax_sigma.plot(xs_sig, spl_sigma(xs_sig), label="σ₂ spline", color="darkgreen", linestyle=":", linewidth=1.5)
        ax_sigma.axvline(x0_opt, color="red", linewidth=1.5)
        ax_sigma.set_ylabel("σ₂")
        ax_sigma.set_title("σ₂ scan")
        ax_sigma.legend(fontsize=8, framealpha=0.7)
        ax_sigma.grid(True, alpha=0.3)
        # Hide x labels on top panel
        ax_sigma.tick_params(labelbottom=False)
    else:
        ax_sigma.text(0.5, 0.5, "σ₂ data not provided", ha="center", va="center", transform=ax_sigma.transAxes)

    # --- Panel 3: Bootstrap histogram ---
    if x0_boot is not None and len(x0_boot) > 0:
        hist_counts, hist_bins, _ = ax_hist.hist(x0_boot, bins=15, color=(120/255, 120/255, 170/255, 0.6), edgecolor="none", alpha=0.7, label="bootstrap distribution")
        hist_max = float(hist_counts.max())
        ax_hist.axvline(x0_opt, color="red", linewidth=1.5)
        ax_hist.set_xlabel("x₀")
        ax_hist.set_ylabel("count")
        ax_hist.set_title("Bootstrap distribution of x₀")
        ax_hist.legend(fontsize=8, framealpha=0.7)
        ax_hist.grid(True, alpha=0.3)
    else:
        ax_hist.text(0.5, 0.5, "Bootstrap data not provided", ha="center", va="center", transform=ax_hist.transAxes)

    plt.tight_layout()
    return ax_main


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
    xlabel: str = "x",
    ylabel: str = "y",
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

    # --- Panel 1: Main ---
    xs = np.linspace(x.min(), x.max(), 600)
    xs2 = np.linspace(min(x.min(), xr.min()), max(x.max(), xr.max()), 600)

    ax_main.scatter(x, y, s=8, alpha=0.6, label="original data", color="C0", edgecolors="none")
    ax_main.scatter(xr, y, s=8, alpha=0.5, label="reflected points", color="orange", edgecolors="none")
    ax_main.plot(xs, spl1(xs), label="original spline", color="royalblue", linewidth=1.5)
    ax_main.plot(xs2, spl2(xs2), label="refitted spline (data + reflection)", color="darkgreen", linestyle="--", linewidth=1.5)

    y_min, y_max = float(y.min()), float(y.max())
    ax_main.axvline(x0_opt, color="red", linewidth=1.5, label=f"x₀ = {x0_opt:.2f} ± {x0_std:.2f}")

    # CI rectangle (approximate from x0_grid/sigma2)
    if len(x0_grid) > 1:
        # Find region where sigma2 <= min + 1*std (rough CI)
        pass

    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.set_title(f"Reflected light curve; x₀ = {x0_opt:.2f} ± {x0_std:.2f}")
    ax_main.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax_main.grid(True, alpha=0.3)

    # --- Panel 2: σ₂ ---
    xs_sig = np.linspace(x0_grid[0], x0_grid[-1], 200)
    ax_sigma.plot(x0_grid, sigma2, label="σ₂(x₀)", color="darkgreen", linewidth=1)
    ax_sigma.plot(xs_sig, spl_sigma(xs_sig), label="σ₂ spline", color="darkgreen", linestyle=":", linewidth=1.5)
    ax_sigma.axvline(x0_opt, color="red", linewidth=1.5)
    ax_sigma.set_ylabel("σ₂")
    ax_sigma.set_title("σ₂ scan")
    ax_sigma.legend(fontsize=8, framealpha=0.7)
    ax_sigma.grid(True, alpha=0.3)
    ax_sigma.tick_params(labelbottom=False)

    # --- Panel 3: Histogram ---
    hist_counts, hist_bins, _ = ax_hist.hist(x0_boot, bins=15, color=(120/255, 120/255, 170/255, 0.6), edgecolor="none", alpha=0.7, label="bootstrap distribution")
    hist_max = float(hist_counts.max())
    ax_hist.axvline(x0_opt, color="red", linewidth=1.5)
    ax_hist.set_xlabel("x₀")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Bootstrap distribution of x₀")
    ax_hist.legend(fontsize=8, framealpha=0.7)
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig