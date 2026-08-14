#!/usr/bin/env python3
"""Reflection method on real AAVSO light curves — full worked example.

Loads two real AAVSO 'extended format' photometry files, finds the primary
eclipse minimum of each star with the reflection method, and produces the
full 4-panel diagnostic figures plus an aligned comparison plot.

Run with::

    python examples/plot_aavso.py

Requires the plotting extra:

    pip install "reflection-method[plot]"

The input files are AAVSO 'extended format' CSV: the column header lives in
a ``#``-prefixed comment line. Only the standard library is used for parsing
(the core library itself deliberately does no I/O).
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reflection_method import combine, find_minimum, find_x0, fit_spline
from reflection_method.core import refine_x0_minimum
from reflection_method.plot import plot_all

# Location of the AAVSO fixtures (also used by the test suite)
FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
STARS = [
    FIXTURES / "V500_Peg_2461261_2026-08-08_G.aavso.txt",
    FIXTURES / "V456_Cyg_2461251_2026-07-29_G.aavso.txt",
]
OUT_DIR = Path(__file__).resolve().parent / "plots"

SEED = 42
N_BOOTSTRAP = 200


def load_aavso(
    path: Path,
    x_col: str = "DATE-OBS",
    y_col: str = "MAG",
    err_col: str = "MAG_ERR",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.datetime64, str, np.ndarray]:
    """Read an AAVSO extended-format file into arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.datetime64, str, np.ndarray]
        ``(x_minutes, y_mag, w, t0_utc, star_name, mag_err)`` where ``x`` is
        the time in minutes from the first observation, ``y`` the magnitude
        as reported (logarithmic scale), ``w`` the weights ``1 / MAG_ERR``,
        ``t0_utc`` the first timestamp, ``star_name`` the value of the
        ``NAME`` column and ``mag_err`` the photometric error of each point
        in magnitudes (the ``MAG_ERR`` column as-is).
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    # AAVSO stores the header in a comment line: #NAME,DATE-OBS,MAG,...
    header_line = next(
        line[1:].strip()
        for line in lines
        if line.startswith("#") and "," in line and "=" not in line[1:]
    )
    fieldnames = [col.strip() for col in header_line.split(",")]
    data_lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    rows = list(csv.DictReader(data_lines, fieldnames=fieldnames))
    if not rows:
        raise ValueError(f"No data rows in {path}")

    star_name = rows[0]["NAME"]
    ts = np.array(
        [np.datetime64(datetime.datetime.fromisoformat(r[x_col].strip().replace(" ", "T")), "ms")
         for r in rows]
    )
    t0 = ts.min()
    x = ((ts - t0) / np.timedelta64(1, "m")).astype(float)

    # Magnitudes are used as-is, on the logarithmic scale
    y = np.array([float(r[y_col]) for r in rows])

    err = np.array([float(r[err_col]) for r in rows])
    w = 1.0 / np.maximum(err, 1e-6)
    # MAG_ERR is already the photometric error in magnitudes — used directly

    return x, y, w, t0, star_name, err


def analyze_star(path: Path, out_png: Path) -> tuple[str, float, float]:
    """Run the full pipeline on one star and save the diagnostic figure."""
    x, y, w, t0, star_name, _ = load_aavso(path)
    rng = np.random.default_rng(SEED)

    # High-level pipeline (returns the bootstrap samples for the histogram).
    # An eclipse is a *maximum* of the magnitude light curve -> find_peak=True.
    result, x0_boot = find_minimum(
        x, y,
        pts_per_knot=10,
        degree=3,
        w=w,
        n_scan=200,
        x0_window=0.1,
        n_bootstrap=N_BOOTSTRAP,
        n_scan_boot=80,
        find_peak=True,
        rng=rng,
        return_samples=True,
    )

    # Reconstruct intermediates needed by the diagnostic figure
    x0_opt, x0_grid, sigma2 = find_x0(
        x, y, pts_per_knot=10, degree=3, w=w, n_scan=200, x0_window=0.1,
        find_peak=True,
    )
    spl1 = fit_spline(x, y, 10, 3, w)

    # Polynomial fit for the σ₂ curve, matching the algorithm's refinement
    _, spl_sigma, fit_window_lo, fit_window_hi = refine_x0_minimum(x0_grid, sigma2)

    xr = 2 * x0_opt - x
    x_all, y_all, w_all = combine(x, y, x0_opt, w)
    spl2 = fit_spline(x_all, y_all, 20, 3, w_all)

    fig = plot_all(
        x, y, spl1, xr, spl2, x0_opt, result.x0_std,
        x0_grid, sigma2, spl_sigma, x0_boot,
        xlabel="Time", ylabel="magnitude", x_unit="min",
        invert_y=True, utc0=t0,
        fit_window_lo=fit_window_lo, fit_window_hi=fit_window_hi,
    )
    fig.suptitle(
        f"{star_name} — primary minimum {np.datetime_as_string(t0, unit='s')}Z\n"
        f"x0 = {result.x0:.2f} ± {result.x0_std:.2f} min  |  68% CI "
        f"[{result.x0_lo:.2f}, {result.x0_hi:.2f}] min",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # UTC time of the minimum (origin + x0 minutes)
    t_min = np.datetime_as_string(
        t0 + np.timedelta64(int(result.x0 * 60), "s"), unit="s", timezone="UTC"
    )
    print(f"{star_name:<12} minimum {t_min}  ± {result.x0_std * 60:.0f} s  "
          f"->  {out_png.name}")
    return star_name, result.x0, result.x0_std


def comparison_figure(results: list[tuple[str, float, float]], out_png: Path) -> None:
    """Overlay both light curves, minima aligned at zero, as scatter with
    magnitude errors (yerr = MAG_ERR directly).

    The magnitude axis is inverted (brighter = up). The data is plotted
    exactly as reported by the observer — no zoom or other adjustment.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for path, (name, x0, x0_std) in zip(STARS, results):
        x, y, w, t0, _, mag_err = load_aavso(path)
        color = "C0" if "V500" in str(path) else "C1"
        ax.errorbar(
            x - x0, y, yerr=mag_err,
            fmt="o", ms=3.5, linewidth=0, alpha=0.9,
            color=color, ecolor=color, elinewidth=1.0, capsize=2.5,
            label=f"{name} (x0 = {x0:.2f} ± {x0_std:.2f} min)",
        )
    ax.axvline(0, color="red", linewidth=1.5)
    ax.set_xlabel("Time from minimum [min]")
    ax.set_ylabel("magnitude")
    ax.set_title("Both eclipses aligned to their detected minima")
    ax.invert_yaxis()  # brighter (smaller magnitude) at the top
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{'comparison':<12} saved                           ->  {out_png.name}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for path in STARS:
        results.append(analyze_star(path, OUT_DIR / f"{path.stem.split('_')[0]}_result.png"))
    comparison_figure(results, OUT_DIR / "comparison.png")
    print(f"\nPlots written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
