"""CLI for reflection-method: handles I/O, time conversion, and calls core library."""

import json
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np

try:
    from reflection_method import find_minimum, MinimumResult
except ImportError:
    # Allow running from source without install
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from reflection_method import find_minimum, MinimumResult


def parse_datetime_column(values: list[str]) -> tuple[np.ndarray, Optional[np.datetime64]]:
    """Parse ISO datetime strings to minutes from first timestamp.

    Returns (x_minutes, t0_utc) where t0_utc is the first timestamp as datetime64[ms] (UTC).
    """
    import datetime

    # Try parsing as ISO format (AAVSO DATE-OBS)
    timestamps = []
    for v in values:
        try:
            # Handle formats like "2026-08-08 20:43:16.427" or "2026-08-08T20:43:16.427"
            v_clean = v.strip().replace(" ", "T")
            dt = datetime.datetime.fromisoformat(v_clean)
            timestamps.append(dt)
        except ValueError:
            raise ValueError(f"Cannot parse datetime: {v}")

    # Convert to numpy datetime64[ms] (naive = UTC per AAVSO convention)
    ts = np.array([np.datetime64(dt, "ms") for dt in timestamps])
    t0 = ts.min()
    x_minutes = ((ts - t0) / np.timedelta64(1, "m")).astype(float)
    return x_minutes, t0


def parse_jd_column(values: list[str]) -> np.ndarray:
    """Parse Julian Date column (float)."""
    return np.array([float(v) for v in values], dtype=float)


def parse_phase_column(values: list[str]) -> np.ndarray:
    """Parse phase column (float 0-1)."""
    return np.array([float(v) for v in values], dtype=float)


def mag_to_flux(mag: np.ndarray) -> np.ndarray:
    """Convert magnitudes to relative flux (median-normalized)."""
    median_mag = float(np.median(mag))
    return 10.0 ** (-0.4 * (mag - median_mag))


def read_csv_file(
    path: str,
    x_col: str,
    y_col: str,
    w_col: Optional[str],
    time_format: str,
    invert_mag: bool,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.datetime64]]:
    """Read CSV file and return x, y, w, t0_utc.

    Handles AAVSO extended format where header is in a comment line like:
    #NAME,DATE-OBS,MAG,MAG_ERR,...

    Args:
        path: Path to CSV file
        x_col: X column name
        y_col: Y column name
        w_col: Weight column name (or None)
        time_format: "jd" | "hjd" | "mjd" | "iso" | "minutes"
        invert_mag: Convert MAG to flux

    Returns:
        x, y, w, t0_utc (None if time_format != "iso")
    """
    import csv

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find header line: first line starting with # that contains commas
    header_line = None
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") and "," in stripped and "=" not in stripped[1:]:
            header_line = stripped[1:].strip()  # Remove leading #
            data_start = i + 1
            break

    if header_line is None:
        # Fallback: assume first non-comment line is header
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith("#"):
                header_line = line.strip()
                data_start = i + 1
                break

    if header_line is None:
        raise ValueError("No header found in file")

    fieldnames = [c.strip() for c in header_line.split(",")]

    # Parse data rows
    data_lines = [line for line in lines[data_start:] if line.strip() and not line.strip().startswith("#")]
    if not data_lines:
        raise ValueError("No data rows found")

    reader = csv.DictReader(data_lines, fieldnames=fieldnames)
    rows = list(reader)
    if not rows:
        raise ValueError("No data rows found")

    # Check columns exist
    for col in [x_col, y_col] + ([w_col] if w_col else []):
        if col not in fieldnames:
            raise ValueError(f"Column '{col}' not found. Available: {fieldnames}")

    # Extract columns
    x_raw = [row[x_col] for row in rows]
    y_raw = np.array([float(row[y_col]) for row in rows])
    w_raw = np.array([float(row[w_col]) for row in rows]) if w_col else None

    # ... rest of function unchanged
    t0_utc = None
    if time_format == "iso":
        x, t0_utc = parse_datetime_column(x_raw)
    elif time_format in ("jd", "hjd", "mjd"):
        x = parse_jd_column(x_raw)
    elif time_format == "phase":
        x = parse_phase_column(x_raw)
    elif time_format == "minutes":
        x = np.array([float(v) for v in x_raw], dtype=float)
    else:
        raise ValueError(f"Unknown time_format: {time_format}")

    # Invert magnitudes to flux if requested
    if invert_mag:
        y = mag_to_flux(y_raw)
    else:
        y = y_raw

    # Weights: 1/sigma if provided, else None
    if w_raw is not None:
        w = 1.0 / np.maximum(w_raw, 1e-6)
    else:
        w = None

    return x, y, w, t0_utc


def result_to_dict(result: MinimumResult, t0_utc: Optional[np.datetime64] = None) -> dict:
    """Convert MinimumResult to JSON-serializable dict."""
    out = {
        "x0": float(result.x0),
        "x0_std": float(result.x0_std),
        "x0_lo": float(result.x0_lo),
        "x0_hi": float(result.x0_hi),
        "sigma_min": float(result.sigma_min),
        "n_points": int(result.n_points),
        "n_bootstrap": int(result.n_bootstrap),
    }
    if t0_utc is not None:
        # Convert x0 to UTC time
        ts = t0_utc + np.timedelta64(int(result.x0 * 60), "s")
        # Proper ISO 8601 UTC format: 2026-08-08T22:14:04Z
        utc_str = np.datetime_as_string(ts, unit="s", timezone="UTC")
        if utc_str.endswith("Z"):
            pass  # already has Z
        elif "T" in utc_str:
            utc_str += "Z"
        else:
            utc_str = utc_str.replace(" ", "T") + "Z"
        utc_uncertainty_s = float(result.x0_std * 60)
        out["utc_time"] = utc_str
        out["utc_uncertainty_s"] = utc_uncertainty_s
    return out


@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("--x-col", default="DATE-OBS", help="X column name (time)")
@click.option("--y-col", default="MAG", help="Y column name (magnitude/flux)")
@click.option("--w-col", default=None, help="Weight/error column name (optional)")
@click.option("--time-format", type=click.Choice(["jd", "hjd", "mjd", "iso", "minutes"]), default="iso", help="Time format of X column")
@click.option("--invert-mag/--no-invert-mag", default=True, help="Convert MAG to relative flux")
@click.option("--pts-per-knot", default=10, type=int, help="Points per spline knot")
@click.option("--degree", default=3, type=int, help="Spline degree (1-5)")
@click.option("--n-scan", default=200, type=int, help="Scan resolution for x0 search")
@click.option("--x0-window", default=0.1, type=float, help="Scan window as fraction of x-range")
@click.option("--n-bootstrap", default=60, type=int, help="Bootstrap iterations")
@click.option("--n-scan-boot", default=80, type=int, help="Scan resolution per bootstrap iteration")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output JSON file (default: stdout)")
@click.option("--plot", type=click.Path(path_type=Path), default=None, help="Save plot to file (requires [plot] extra)")
@click.option("--plot-format", type=click.Choice(["png", "pdf", "svg"]), default="png", help="Plot format")
def find(
    input_file: Path,
    x_col: str,
    y_col: str,
    w_col: Optional[str],
    time_format: str,
    invert_mag: bool,
    pts_per_knot: int,
    degree: int,
    n_scan: int,
    x0_window: float,
    n_bootstrap: int,
    n_scan_boot: int,
    seed: Optional[int],
    output: Optional[Path],
    plot: Optional[Path],
    plot_format: str,
):
    """Find eclipse minimum in a light curve using the reflection method.

    Reads CSV (AAVSO format with # comments supported), converts time/magnitude,
    runs the reflection method, outputs JSON with x0 and uncertainties.
    """
    # Read data
    try:
        x, y, w, t0_utc = read_csv_file(
            str(input_file), x_col, y_col, w_col, time_format, invert_mag
        )
    except Exception as e:
        click.echo(f"Error reading data: {e}", err=True)
        sys.exit(1)

    click.echo(f"Loaded {len(x)} points", err=True)
    if t0_utc is not None:
        click.echo(f"Time origin (UTC): {np.datetime_as_string(t0_utc, unit='s')}Z", err=True)

    # RNG
    rng = np.random.default_rng(seed) if seed is not None else None

    # Run core algorithm
    try:
        result = find_minimum(
            x, y,
            pts_per_knot=pts_per_knot,
            degree=degree,
            w=w,
            n_scan=n_scan,
            x0_window=x0_window,
            n_bootstrap=n_bootstrap,
            n_scan_boot=n_scan_boot,
            rng=rng,
        )
    except Exception as e:
        click.echo(f"Error in algorithm: {e}", err=True)
        sys.exit(1)

    # Prepare output
    out_dict = result_to_dict(result, t0_utc)
    out_dict["input_file"] = str(input_file)
    out_dict["time_format"] = time_format
    out_dict["parameters"] = {
        "pts_per_knot": pts_per_knot,
        "degree": degree,
        "n_scan": n_scan,
        "x0_window": x0_window,
        "n_bootstrap": n_bootstrap,
        "n_scan_boot": n_scan_boot,
    }

    # Write JSON
    json_str = json.dumps(out_dict, indent=2)
    if output:
        output.write_text(json_str)
        click.echo(f"Result written to {output}", err=True)
    else:
        click.echo(json_str)

    # Optional plot
    if plot:
        try:
            from reflection_method.plot import plot_all
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from reflection_method import find_x0, fit_spline, combine
            from reflection_method.core import spline_variance, bootstrap_x0
            from scipy.interpolate import UnivariateSpline
            from scipy.optimize import minimize_scalar

            # Reconstruct intermediate data
            x0_opt, x0_grid, sigma2 = find_x0(x, y, pts_per_knot, degree, w, n_scan, x0_window)
            spl1 = fit_spline(x, y, pts_per_knot, degree, w)
            spl_sigma = UnivariateSpline(x0_grid, sigma2, k=3, s=0)
            xr = 2 * x0_opt - x
            x_all, y_all, w_all = combine(x, y, x0_opt, w)
            spl2 = fit_spline(x_all, y_all, 2 * pts_per_knot, degree, w_all)

            # Generate bootstrap samples for histogram
            rng_boot = np.random.default_rng(seed) if seed is not None else None
            # Use the same bootstrap function logic but capture x0_boot array
            residuals = y - spl1(x)
            n_pts = len(x)
            x0_boot = np.empty(n_bootstrap)
            xmin, xmax = float(x.min()), float(x.max())
            window_width = x0_window * (xmax - xmin)

            for k in range(n_bootstrap):
                y_boot = spl1(x) + residuals[rng_boot.integers(0, n_pts, n_pts)]
                spl_b = fit_spline(x, y_boot, pts_per_knot, degree, w)

                x_fine = np.linspace(xmin, xmax, 1001)
                center = float(x_fine[int(np.argmin(spl_b(x_fine)))])
                lo = max(xmin, center - window_width / 2)
                hi = min(xmax, center + window_width / 2)

                grid = np.linspace(lo, hi, n_scan_boot)
                sig = np.empty(len(grid))

                for i, x0 in enumerate(grid):
                    xs, ys, ws = combine(x, y_boot, x0, w)
                    sp = fit_spline(xs, ys, 2 * pts_per_knot, degree, ws)
                    sig[i] = np.sqrt(spline_variance(sp, xs, ys))

                sig = np.maximum(sig, 1e-6)
                spl_sig = UnivariateSpline(grid, sig, k=3, s=0)
                x0_boot[k] = minimize_scalar(spl_sig, bounds=(grid[0], grid[-1]), method="bounded").x

            fig = plot_all(
                x, y, spl1, xr, spl2, x0_opt, result.x0_std,
                x0_grid, sigma2, spl_sigma, x0_boot,
                xlabel=x_col, ylabel=y_col
            )
            fig.savefig(plot, format=plot_format, dpi=150, bbox_inches="tight")
            plt.close(fig)
            click.echo(f"Plot saved to {plot}", err=True)
        except ImportError:
            click.echo("Warning: [plot] extra not installed, skipping plot", err=True)
        except Exception as e:
            click.echo(f"Warning: Could not generate plot: {e}", err=True)


if __name__ == "__main__":
    find()