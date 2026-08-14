"""Plotting utilities for reflection-method (optional `[plot]` extra).

Uses matplotlib. All functions accept an optional `ax` parameter for subplot
embedding. The main function `plot_all` creates the standard 4-panel composite
figure matching the notebook visualization.

Dependencies
------------
Requires the `[plot]` extra: ``pip install "reflection-method[plot]"``.
This installs matplotlib as a dependency.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.dates import AutoDateLocator, DateFormatter

from .core import LSQUnivariateSpline


def _minutes_to_datetime64(t0: np.datetime64, minutes: np.ndarray) -> np.ndarray:
    """Convert array of minutes-from-t0 to matplotlib-compatible datetime64."""
    return t0 + (minutes * 60_000).astype("timedelta64[ms]")


def _utc_label(t0: np.datetime64, x0_minutes: float) -> str:
    """ISO-8601 UTC string for an ``x0`` expressed in minutes from ``t0``."""
    ts = t0 + np.timedelta64(int(round(x0_minutes * 60)), "s")
    return np.datetime_as_string(ts, unit="s", timezone="UTC")


def plot_original_spline(
    x: np.ndarray,
    y: np.ndarray,
    spl1: LSQUnivariateSpline,
    x0_opt: float,
    x0_std: float,
    xlabel: str = "Time",
    ylabel: str = "magnitude",
    x_unit: str = "",
    invert_y: bool = False,
    utc0: Optional[np.datetime64] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot original data, initial spline fit, and the optimal ``x₀`` line.

    Parameters
    ----------
    x : np.ndarray
        Original abscissae (time, phase, etc.).
    y : np.ndarray
        Original ordinates (flux, magnitude, etc.).
    spl1 : LSQUnivariateSpline
        Initial spline fit to the original data (without reflection).
    x0_opt : float
        Optimal reflection point found by the reflection method.
    x0_std : float
        Bootstrap standard error of ``x0_opt``.
    xlabel : str, optional
        Label for the x-axis (without unit). Default "Time".
    ylabel : str, optional
        Label for the y-axis. Default "magnitude".
    x_unit : str, optional
        Unit string for the x-axis (e.g., "min", "JD", "phase"). Appended to
        the xlabel in brackets. Default "" (no unit).
    invert_y : bool, optional
        If True, invert the y-axis so that smaller values are on top — the
        standard convention when plotting magnitudes (brighter stars up).
        Default False.
    utc0 : np.datetime64 or None, optional
        Origin timestamp for the abscissae, which are then interpreted as
        minutes from ``utc0`` and the x-axis is labelled with UTC clock times.
        Default None (numeric x-axis, ``x_unit`` used for the label).
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.

    Examples
    --------
    >>> from reflection_method import fit_spline, find_x0
    >>> from reflection_method.plot import plot_original_spline
    >>> import matplotlib.pyplot as plt
    >>> spl1 = fit_spline(x, y, pts_per_knot=10)
    >>> x0_opt, _, _ = find_x0(x, y)
    >>> ax = plot_original_spline(x, y, spl1, x0_opt, 0.01, xlabel="Phase", x_unit="")
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    if utc0 is not None:
        x_plot = _minutes_to_datetime64(utc0, x)
        xs = np.linspace(x.min(), x.max(), 600)
        xs_plot = _minutes_to_datetime64(utc0, xs)
        x0_opt_plot = _minutes_to_datetime64(utc0, np.array([x0_opt]))[0]
    else:
        x_plot = x
        xs = np.linspace(x.min(), x.max(), 600)
        xs_plot = xs
        x0_opt_plot = x0_opt

    ax.scatter(x_plot, y, s=8, alpha=0.6, label="data", color="C0", edgecolors="none")
    ax.plot(xs_plot, spl1(xs), label="original spline", color="royalblue", linewidth=1.5)

    y_min, y_max = float(y.min()), float(y.max())
    if utc0 is not None:
        x0_label = f"x₀ = {_utc_label(utc0, x0_opt)} ± {x0_std * 60:.0f} s"
    else:
        x0_label = f"x₀ = {x0_opt:.2f} ± {x0_std:.2f}"
        if x_unit:
            x0_label += f" {x_unit}"
    ax.axvline(x0_opt_plot, color="red", linewidth=1.5, label=x0_label)

    if utc0 is not None:
        ax.set_xlabel(f"{xlabel} [UTC]")
        ax.xaxis.set_major_locator(AutoDateLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    else:
        ax.set_xlabel(f"{xlabel} [{x_unit}]" if x_unit else xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Light curve and original spline with x₀")
    ax.legend(loc="lower left", framealpha=0.7)
    ax.grid(True, alpha=0.3)

    if invert_y:
        ax.invert_yaxis()

    return ax


def plot_scan(
    x0_grid: np.ndarray,
    sigma1: np.ndarray,
    sigma2: np.ndarray,
    spl_sigma: np.poly1d,
    x0_opt: float,
    x0_grid_idx_min: int,
    xlabel: str = "x₀",
    ylabel: str = "residual σ",
    x_unit: str = "",
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot σ₁, σ₂, and the polynomial fit to σ₂ with vertical lines at minima.

    Parameters
    ----------
    x0_grid : np.ndarray
        Grid of trial reflection points.
    sigma1 : np.ndarray
        σ values with respect to the original spline (σ₁).
    sigma2 : np.ndarray
        σ values with respect to the refitted spline (σ₂).
    spl_sigma : np.poly1d
        Polynomial fit (np.poly1d) to the σ₂ curve.
    x0_opt : float
        Optimal reflection point from the refined minimum.
    x0_grid_idx_min : int
        Index of the minimum ``sigma2`` on the grid (before refinement).
    xlabel : str, optional
        Label for the x-axis. Default "x₀".
    ylabel : str, optional
        Label for the y-axis. Default "residual σ".
    x_unit : str, optional
        Unit string for the x-axis. Default "".
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    xs = np.linspace(x0_grid[0], x0_grid[-1], 200)

    ax.plot(x0_grid, sigma1, label="σ w.r.t. original spline", color="royalblue", linewidth=1)
    ax.plot(x0_grid, sigma2, label="σ w.r.t. refitted spline", color="mediumseagreen", linewidth=1)
    ax.plot(xs, spl_sigma(xs), label="polynomial fit to σ₂", color="darkgreen", linestyle=":", linewidth=1.5)

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
    spl_sigma: np.poly1d,
    x0_boot: np.ndarray,
    xlabel: str = "Time",
    ylabel: str = "magnitude",
    x_unit: str = "",
    invert_y: bool = False,
    utc0: Optional[np.datetime64] = None,
    fit_window_lo: Optional[float] = None,
    fit_window_hi: Optional[float] = None,
    **kwargs,
) -> Figure:
    """Create the standard 4-panel reflection-method diagnostic figure.

    Produces a 2×2 figure with the following panels:

    1. **Main (left, full height)**: Light curve with original data, reflected
       points, original spline, refitted spline (data + reflection), and
       the optimal ``x₀`` vertical line with uncertainty.
    2. **σ₂ scan (top right)**: The ``σ₂(x₀)`` curve as data points with the
       polynomial fit (drawn only over the window used for the fit), and the
       optimal ``x₀`` line. X-axis limited to ``±3σ`` around the optimum for
       consistency with the histogram.
    3. **Bootstrap histogram (bottom right)**: Distribution of bootstrap
       ``x₀`` estimates with the optimal ``x₀`` line. Same x-axis limits
       as the σ₂ scan.

    Parameters
    ----------
    x : np.ndarray
        Original abscissae.
    y : np.ndarray
        Original ordinates.
    spl1 : LSQUnivariateSpline
        Initial spline fit to original data.
    xr : np.ndarray
        Reflected abscissae (``2 * x0_opt - x``).
    spl2 : LSQUnivariateSpline
        Spline fit to combined (original + reflected) data.
    x0_opt : float
        Optimal reflection point.
    x0_std : float
        Bootstrap standard error of ``x0_opt``.
    x0_grid : np.ndarray
        Scan grid of trial ``x₀`` values.
    sigma2 : np.ndarray
        ``σ₂`` values at each grid point.
    spl_sigma : np.poly1d
        Polynomial fit (``np.poly1d``) to ``(x0_grid, sigma2)``, as returned
        by :func:`reflection_method.core.refine_x0_minimum`.
    x0_boot : np.ndarray
        Array of bootstrap ``x₀`` estimates (length = ``n_bootstrap``).
    xlabel : str, optional
        Label for the main x-axis (e.g., "Time", "Phase"). Default "Time".
    ylabel : str, optional
        Label for the y-axis (e.g., "magnitude", "Flux"). Default
        "magnitude".
    x_unit : str, optional
        Unit for the x-axis (e.g., "min", "JD", "HJD", "phase"). Appended
        to axis labels and ``x₀`` labels. Default "".
    invert_y : bool, optional
        If True, invert the y-axis of the main panel so that smaller values
        are on top — the standard convention when plotting magnitudes
        (brighter stars up). Default False.
    utc0 : np.datetime64 or None, optional
        Origin timestamp for the abscissae, which are then interpreted as
        minutes from ``utc0``. The main panel is then labelled with UTC
        clock times instead of the raw numeric axis, and the ``x₀`` label on
        the main panel shows the UTC time of the minimum. The right-hand
        panels (σ₂ scan and histogram) always use relative minutes from x₀
        to show precision. Default None (numeric x-axis, ``x_unit`` used
        for the labels).
    fit_window_lo : float or None, optional
        Lower bound of the window used for the polynomial fit of ``σ₂``,
        in the same units as ``x0_grid``. The fit curve is drawn only over
        this window (the points actually used for the fit). Default None
        (draw over the whole scan range).
    fit_window_hi : float or None, optional
        Upper bound of the fit window, same convention as
        ``fit_window_lo``. Default None.
    **kwargs : dict
        Additional keyword arguments (currently unused, for forward
        compatibility).

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object. Call ``fig.savefig("output.png")`` to save.

    Notes
    -----
    - The right-hand panels (σ₂ scan and histogram) share the same x-axis
      limits: ``x₀ ± 3 * x0_std``. This ensures consistent visual comparison.
    - The σ₂ scan y-limits are automatically set to the range of the points
      visible within the ``±3σ`` window, with a 10% padding.
    - The σ₂ scan uses a polynomial fit to the curve (quartic by default);
      the fit curve is drawn only over the window that was actually used for
      the fit (``fit_window_lo`` … ``fit_window_hi``), so the fit can be
      judged against the points it was fitted to.
    - The figure size is 10×8 inches with width ratios 0.62:0.38 and
      height ratios 0.5:0.5.
    - The layout uses ``plt.tight_layout()`` for automatic spacing.

    Examples
    --------
    >>> from reflection_method import find_minimum, find_x0, fit_spline, combine
    >>> from reflection_method.plot import plot_all
    >>> from reflection_method.core import refine_x0_minimum
    >>> import matplotlib.pyplot as plt
    >>>
    >>> result = find_minimum(x, y)
    >>> x0_opt, x0_grid, sigma2 = find_x0(x, y)
    >>> _, spl_sigma, fit_lo, fit_hi = refine_x0_minimum(x0_grid, sigma2)
    >>> spl1 = fit_spline(x, y)
    >>> xr = 2 * x0_opt - x
    >>> spl2 = fit_spline(*combine(x, y, x0_opt))
    >>> # Bootstrap samples needed for the histogram:
    >>> from reflection_method import bootstrap_x0
    >>> *_, x0_boot = bootstrap_x0(x, y, x0_opt, spl1, 10, 3, None, 60, 80, 0.1,
    ...                            return_samples=True)
    >>>
    >>> fig = plot_all(x, y, spl1, xr, spl2, x0_opt, result.x0_std,
    ...                x0_grid, sigma2, spl_sigma, x0_boot,
    ...                xlabel="Time", ylabel="Flux", x_unit="min",
    ...                fit_window_lo=fit_lo, fit_window_hi=fit_hi)
    >>> fig.savefig("output.png", dpi=150, bbox_inches="tight")
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

    if utc0 is not None:
        x_plot = _minutes_to_datetime64(utc0, x)
        xr_plot = _minutes_to_datetime64(utc0, xr)
        xs_plot = _minutes_to_datetime64(utc0, xs)
        xs2_plot = _minutes_to_datetime64(utc0, xs2)
        x0_opt_plot = _minutes_to_datetime64(utc0, np.array([x0_opt]))[0]
    else:
        x_plot = x
        xr_plot = xr
        xs_plot = xs
        xs2_plot = xs2
        x0_opt_plot = x0_opt

    ax_main.scatter(x_plot, y, s=8, alpha=0.6, label="original data", color="C0", edgecolors="none")
    ax_main.scatter(xr_plot, y, s=8, alpha=0.5, label="reflected points", color="orange", edgecolors="none")
    ax_main.plot(xs_plot, spl1(xs), label="original spline", color="royalblue", linewidth=1.5)
    ax_main.plot(xs2_plot, spl2(xs2), label="refitted spline (data + reflection)", color="darkgreen", linestyle="--", linewidth=1.5)

    y_min, y_max = float(y.min()), float(y.max())
    if utc0 is not None:
        x0_label = f"x₀ = {_utc_label(utc0, x0_opt)} ± {x0_std * 60:.0f} s"
    else:
        x0_label = f"x₀ = {x0_opt:.2f} ± {x0_std:.2f}"
        if x_unit:
            x0_label += f" {x_unit}"
    ax_main.axvline(x0_opt_plot, color="red", linewidth=1.5, label=x0_label)

    if utc0 is not None:
        ax_main.set_xlabel(f"{xlabel} [UTC]")
        ax_main.xaxis.set_major_locator(AutoDateLocator())
        ax_main.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    else:
        ax_main.set_xlabel(f"{xlabel} [{x_unit}]" if x_unit else xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.set_title(f"Reflected light curve; {x0_label}")
    ax_main.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax_main.grid(True, alpha=0.3)

    if invert_y:
        ax_main.invert_yaxis()

    # --- Panel 2: σ₂ ---
    # Relative to x0_opt to show precision, same x-limits as the histogram
    x0_rel = x0_grid - x0_opt
    ax_sigma.plot(x0_rel, sigma2, 'o', label="σ₂(x₀)", color="mediumseagreen", markersize=3, alpha=0.7, linewidth=0)
    ax_sigma.axvline(0, color="red", linewidth=1.5)

    # Fit curve drawn only over the window used for the polynomial fit
    fit_lo = x0_grid[0] if fit_window_lo is None else fit_window_lo
    fit_hi = x0_grid[-1] if fit_window_hi is None else fit_window_hi
    xs_fit_rel = np.linspace(fit_lo - x0_opt, fit_hi - x0_opt, 200)
    ax_sigma.plot(xs_fit_rel, spl_sigma(xs_fit_rel + x0_opt), label="polynomial fit to σ₂", color="darkgreen", linestyle="-", linewidth=1.5)

    # Same x-limits as the histogram (±3σ)
    ax_sigma.set_xlim(-3 * x0_std, 3 * x0_std)
    # Y-limits from the points visible within the ±3σ window
    in_view = np.abs(x0_rel) <= 3 * x0_std
    sigma_min = float(sigma2[in_view].min())
    sigma_max = float(sigma2[in_view].max())
    y_pad = 0.1 * (sigma_max - sigma_min) if sigma_max > sigma_min else 0.01
    ax_sigma.set_ylim(sigma_min - y_pad, sigma_max + y_pad)

    ax_sigma.set_ylabel("σ₂")
    ax_sigma.set_title("σ₂(x₀) scan")
    ax_sigma.legend(fontsize=8, framealpha=0.7)
    ax_sigma.grid(True, alpha=0.3)
    ax_sigma.tick_params(labelbottom=False)

    # --- Panel 3: Histogram ---
    # Plot relative to x0_opt (in minutes) to show precision
    x0_boot_rel = x0_boot - x0_opt
    hist_counts, hist_bins, _ = ax_hist.hist(x0_boot_rel, bins=15, color=(120/255, 120/255, 170/255, 0.6), edgecolor="none", alpha=0.7, label="bootstrap distribution")
    hist_max = float(hist_counts.max())
    ax_hist.axvline(0, color="red", linewidth=1.5)

    # Same x-limits as σ₂ panel (±3σ)
    x0_rel_xlim = (-3 * x0_std, 3 * x0_std)
    ax_hist.set_xlim(x0_rel_xlim)

    ax_hist.set_xlabel(f"Δx₀ [{x_unit}]" if x_unit else "Δx₀")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Bootstrap distribution of x₀")
    ax_hist.legend(fontsize=8, framealpha=0.7)
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig