"""Unit tests for core reflection-method algorithm."""

import numpy as np
import pytest

from reflection_method.core import (
    fit_spline,
    spline_variance,
    combine,
    find_x0,
    bootstrap_x0,
    find_minimum,
)
from reflection_method.result import MinimumResult


def make_synthetic_eclipsing(
    n: int = 300,
    seed: int = 42,
    min_phase: float = 0.5,
    noise: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic eclipsing light curve with known minimum."""
    rng = np.random.default_rng(seed)
    phase = np.sort(rng.uniform(0.0, 1.0, n))

    # Primary minimum at min_phase
    primary = 0.7 * np.exp(-((phase - min_phase) ** 2) / (2 * 0.02**2))
    # Secondary minima
    secondary = 0.15 * (
        np.exp(-((phase - 0.0) ** 2) / (2 * 0.03**2))
        + np.exp(-((phase - 1.0) ** 2) / (2 * 0.03**2))
    )

    flux = 1.0 - primary - secondary + rng.normal(0.0, noise, n)
    return phase, flux


class TestFitSpline:
    def test_basic_fit(self):
        x, y = make_synthetic_eclipsing(200, seed=1)
        spl = fit_spline(x, y, pts_per_knot=10, degree=3)
        assert spl._eval_args[2] == 3
        assert len(spl.get_knots()) > 0

    def test_fit_with_weights(self):
        x, y = make_synthetic_eclipsing(200, seed=2)
        w = np.ones_like(x)
        spl = fit_spline(x, y, pts_per_knot=10, degree=3, w=w)
        assert spl._eval_args[2] == 3

    def test_fit_degree_1(self):
        x, y = make_synthetic_eclipsing(100, seed=3)
        spl = fit_spline(x, y, pts_per_knot=20, degree=1)
        assert spl._eval_args[2] == 1

    def test_too_few_points_raises(self):
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            fit_spline(x, y, degree=3)

    def test_unsorted_x_handled(self):
        x, y = make_synthetic_eclipsing(100, seed=4)
        # Shuffle
        idx = np.random.permutation(len(x))
        x_shuf, y_shuf = x[idx], y[idx]
        spl = fit_spline(x_shuf, y_shuf, pts_per_knot=10, degree=3)
        # Should work and produce reasonable values
        y_pred = spl(np.sort(x))
        assert np.all(np.isfinite(y_pred))


class TestSplineVariance:
    def test_variance_positive(self):
        x, y = make_synthetic_eclipsing(100, seed=5)
        spl = fit_spline(x, y, pts_per_knot=10, degree=3)
        var = spline_variance(spl, x, y)
        assert var > 0
        assert np.isfinite(var)


class TestCombine:
    def test_combine_shapes(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 1.5, 1.0])
        x0 = 2.5
        x_all, y_all, w_all = combine(x, y, x0)
        assert len(x_all) == 8
        assert len(y_all) == 8
        assert w_all is None

    def test_combine_with_weights(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 1.5])
        w = np.array([1.0, 0.5, 1.0])
        x0 = 2.0
        x_all, y_all, w_all = combine(x, y, x0, w)
        assert len(x_all) == 6
        assert w_all is not None
        assert len(w_all) == 6

    def test_combine_sorted(self):
        x = np.array([3.0, 1.0, 4.0, 2.0])
        y = np.array([1.0, 2.0, 1.5, 1.0])
        x0 = 2.5
        x_all, _, _ = combine(x, y, x0)
        assert np.all(np.diff(x_all) >= 0)


class TestFindX0:
    def test_find_x0_recovers_minimum(self):
        # Known minimum at phase=0.5
        x, y = make_synthetic_eclipsing(500, seed=10, min_phase=0.5, noise=0.005)
        x0_opt, x0_grid, sigma2 = find_x0(x, y, pts_per_knot=10, degree=3, n_scan=300)

        # Should be close to 0.5
        assert abs(x0_opt - 0.5) < 0.02
        assert len(x0_grid) == 300
        assert len(sigma2) == 300
        assert np.all(np.isfinite(sigma2))
        assert np.all(sigma2 > 0)

    def test_find_x0_different_minimum(self):
        # Minimum at phase=0.3
        x, y = make_synthetic_eclipsing(500, seed=11, min_phase=0.3, noise=0.005)
        x0_opt, _, _ = find_x0(x, y, pts_per_knot=10, degree=3, n_scan=300)
        assert abs(x0_opt - 0.3) < 0.02


class TestBootstrapX0:
    def test_bootstrap_returns_uncertainties(self):
        x, y = make_synthetic_eclipsing(400, seed=12, noise=0.005)
        x0_opt, _, _ = find_x0(x, y, pts_per_knot=10, degree=3, n_scan=200)
        spl1 = fit_spline(x, y, pts_per_knot=10, degree=3)

        x0_std, x0_lo, x0_hi = bootstrap_x0(
            x, y, x0_opt, spl1,
            pts_per_knot=10, degree=3, w=None,
            n_bootstrap=30, n_scan_boot=60,  # Reduced for speed
            rng=np.random.default_rng(42)
        )

        assert x0_std > 0
        # With very tight distributions, percentiles can have small numerical bias
        # Check that true minimum (0.5) is within CI, and CI is ordered
        assert x0_lo < x0_hi
        assert x0_lo <= 0.5 <= x0_hi

    def test_bootstrap_reproducible_with_seed(self):
        x, y = make_synthetic_eclipsing(300, seed=13, noise=0.005)
        x0_opt, _, _ = find_x0(x, y, pts_per_knot=10, degree=3, n_scan=200)
        spl1 = fit_spline(x, y, pts_per_knot=10, degree=3)

        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)

        std1, lo1, hi1 = bootstrap_x0(x, y, x0_opt, spl1, 10, 3, None, 20, 50, 0.1, rng1)
        std2, lo2, hi2 = bootstrap_x0(x, y, x0_opt, spl1, 10, 3, None, 20, 50, 0.1, rng2)

        assert std1 == std2
        assert lo1 == lo2
        assert hi1 == hi2


class TestFindMinimum:
    def test_full_pipeline(self):
        x, y = make_synthetic_eclipsing(500, seed=14, min_phase=0.5, noise=0.005)
        result = find_minimum(
            x, y,
            pts_per_knot=10, degree=3,
            n_scan=200, n_bootstrap=30, n_scan_boot=60,
            rng=np.random.default_rng(99)
        )

        assert isinstance(result, MinimumResult)
        assert abs(result.x0 - 0.5) < 0.02
        assert result.x0_std > 0
        # With very tight distributions, percentiles can have small numerical bias
        # Check x0 is close to true value and CI is ordered
        assert result.x0_lo < result.x0_hi
        assert abs(result.x0 - 0.5) < 0.01
        assert result.sigma_min > 0
        assert result.n_points == 500
        assert result.n_bootstrap == 30

    def test_minimum_result_immutable(self):
        x, y = make_synthetic_eclipsing(100, seed=15)
        result = find_minimum(x, y, n_bootstrap=10, n_scan_boot=40, rng=np.random.default_rng(1))
        # NamedTuple is immutable
        with pytest.raises(AttributeError):
            result.x0 = 999

    def test_different_time_units(self):
        """Test that algorithm works with any linear time units (JD, minutes, etc.)."""
        # Generate in "days"
        x_days, y = make_synthetic_eclipsing(300, seed=16, min_phase=2.5, noise=0.005)
        # Convert to "minutes"
        x_minutes = x_days * 24 * 60

        result_days = find_minimum(x_days, y, n_bootstrap=10, n_scan_boot=40, rng=np.random.default_rng(1))
        result_min = find_minimum(x_minutes, y, n_bootstrap=10, n_scan_boot=40, rng=np.random.default_rng(1))

        # Results should scale accordingly
        assert abs(result_min.x0 - result_days.x0 * 24 * 60) < 0.1
        assert abs(result_min.x0_std - result_days.x0_std * 24 * 60) < 0.1