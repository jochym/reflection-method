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
    """Parse ISO datetime strings to minutes elapsed from the first timestamp.

    Accepts ISO-8601 timestamps with either a space or ``T`` separator
    (e.g. AAVSO ``DATE-OBS`` values like ``2026-08-08 20:43:16.427``).
    Timestamps are treated as naive UTC, following the AAVSO convention.

    Parameters
    ----------
    values : list[str]
        ISO datetime strings to parse.

    Returns
    -------
    tuple[np.ndarray, np.datetime64 or None]
        ``(x_minutes, t0_utc)`` where ``x_minutes`` is the time of each point
        in minutes relative to the earliest timestamp, and ``t0_utc`` is that
        earliest timestamp as ``datetime64[ms]`` (UTC).

    Raises
    ------
    ValueError
        If any value cannot be parsed as an ISO datetime.
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
    """Parse a Julian Date (or HJD/MJD) column to floats.

    Parameters
    ----------
    values : list[str]
        String representations of Julian Dates.

    Returns
    -------
    np.ndarray
        Float array of Julian Dates.
    """
    return np.array([float(v) for v in values], dtype=float)


def parse_phase_column(values: list[str]) -> np.ndarray:
    """Parse a phase column (values in 0-1) to floats.

    Parameters
    ----------
    values : list[str]
        String representations of phases.

    Returns
    -------
    np.ndarray
        Float array of phases.
    """
    return np.array([float(v) for v in values], dtype=float)


def mag_to_flux(mag: np.ndarray) -> np.ndarray:
    """Convert magnitudes to median-normalized relative flux.

    Applies the standard photometric relation ``F = 10 ** (-0.4 * (mag - m0))``
    where ``m0`` is the median magnitude. The result is dimensionless and
    centered near unity.

    Parameters
    ----------
    mag : np.ndarray
        Magnitude values.

    Returns
    -------
    np.ndarray
        Relative flux values (median-normalized).
    """
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
    """Read a CSV file and extract ``x``, ``y`` and optional weights.

    Handles AAVSO-style extended format where the column header is in a
    comment line like ``#NAME,DATE-OBS,MAG,MAG_ERR,...``. Falls back to
    treating the first non-comment line as the header.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    x_col : str
        Name of the column holding the time / abscissa values.
    y_col : str
        Name of the column holding the magnitude / flux values.
    w_col : str or None
        Name of the column holding measurement uncertainties, or None.
    time_format : str
        One of ``"jd"``, ``"hjd"``, ``"mjd"``, ``"iso"``, ``"phase"``,
        ``"minutes"``.
    invert_mag : bool
        If True, convert magnitudes to relative flux (``mag_to_flux``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray or None, np.datetime64 or None]
        ``(x, y, w, t0_utc)`` where:
        - ``x`` : abscissae in the units implied by ``time_format``
          (for ``"iso"`` these are minutes from the first timestamp)
        - ``y`` : ordinates (flux or magnitude)
        - ``w`` : weights ``1 / sigma``, or None if no ``w_col``
        - ``t0_utc`` : earliest timestamp as ``datetime64[ms]``, or None
          unless ``time_format == "iso"``

    Raises
    ------
    ValueError
        If no header or no data rows are found, a requested column is
        missing, or a value cannot be converted.
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
    """Convert a ``MinimumResult`` into a JSON-serializable dictionary.

    When ``t0_utc`` is given, ``x0`` (minutes from origin) is converted to
    an ISO-8601 UTC timestamp and the uncertainty is expressed in seconds.

    Parameters
    ----------
    result : MinimumResult
        Result of ``find_minimum``.
    t0_utc : np.datetime64 or None, optional
        Time origin (earliest timestamp of the light curve). If None, only
        the raw x0 values are included in the output.

    Returns
    -------
    dict
        Dictionary with keys ``x0``, ``x0_std``, ``x0_lo``, ``x0_hi``,
        ``sigma_min``, ``n_points``, ``n_bootstrap`` and, when ``t0_utc`` is
        given, ``utc_time`` (ISO-8601 with trailing ``Z``) and
        ``utc_uncertainty_s``.
    """
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
    """Find an eclipse minimum in a light curve using the reflection method.

    Reads a CSV file (AAVSO-style extended format with ``#`` comment header
    is supported), converts time and magnitude columns, runs the reflection
    method via :func:`reflection_method.find_minimum`, and prints a JSON
    report with the minimum location ``x0`` and its uncertainties. An
    optional diagnostic plot can be written with ``--plot`` (requires the
    ``[plot]`` extra).

    Parameters
    ----------
    input_file : Path
        Path to the input CSV file.
    x_col : str
        Name of the time column. Default ``DATE-OBS``.
    y_col : str
        Name of the magnitude/flux column. Default ``MAG``.
    w_col : str or None
        Name of the uncertainty column, or None for equal weights.
    time_format : str
        One of ``jd``, ``hjd``, ``mjd``, ``iso``, ``minutes``.
    invert_mag : bool
        Convert magnitudes to relative flux. Default True.
    pts_per_knot : int
        Points per spline knot. Default 10.
    degree : int
        Spline degree (1-5). Default 3.
    n_scan : int
        Scan resolution for the main x0 search. Default 200.
    x0_window : float
        Scan window as a fraction of the x range. Default 0.1.
    n_bootstrap : int
        Bootstrap iterations for uncertainty estimation. Default 60.
    n_scan_boot : int
        Scan resolution per bootstrap iteration. Default 80.
    seed : int or None
        Random seed for reproducibility.
    output : Path or None
        Write JSON to this file instead of stdout.
    plot : Path or None
        Save a diagnostic plot to this file.
    plot_format : str
        Plot format: ``png``, ``pdf`` or ``svg``.
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
        result, x0_boot = find_minimum(
            x, y,
            pts_per_knot=pts_per_knot,
            degree=degree,
            w=w,
            n_scan=n_scan,
            x0_window=x0_window,
            n_bootstrap=n_bootstrap,
            n_scan_boot=n_scan_boot,
            rng=rng,
            return_samples=True,
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
            from scipy.interpolate import UnivariateSpline

            # Reconstruct intermediate data
            x0_opt, x0_grid, sigma2 = find_x0(x, y, pts_per_knot, degree, w, n_scan, x0_window)
            spl1 = fit_spline(x, y, pts_per_knot, degree, w)
            spl_sigma = UnivariateSpline(x0_grid, sigma2, k=3, s=0)
            xr = 2 * x0_opt - x
            x_all, y_all, w_all = combine(x, y, x0_opt, w)
            spl2 = fit_spline(x_all, y_all, 2 * pts_per_knot, degree, w_all)

            # Determine axis labels and units
            if time_format == "iso":
                xlabel = "Time"
                ylabel = "relative magnitude"
                x_unit = "min"
            elif time_format in ("jd", "hjd", "mjd"):
                xlabel = "Time"
                ylabel = "relative magnitude"
                x_unit = time_format.upper()
            elif time_format == "phase":
                xlabel = "Phase"
                ylabel = "relative magnitude"
                x_unit = ""
            elif time_format == "minutes":
                xlabel = "Time"
                ylabel = "relative magnitude"
                x_unit = "min"
            else:
                xlabel = "Time"
                ylabel = "relative magnitude"
                x_unit = ""

            fig = plot_all(
                x, y, spl1, xr, spl2, x0_opt, result.x0_std,
                x0_grid, sigma2, spl_sigma, x0_boot,
                xlabel=xlabel, ylabel=ylabel, x_unit=x_unit
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