# reflection-method: Technical Documentation

**Version:** 0.2.0 (Draft)  
**Author:** Jochym  
**Date:** 2024

---

## 1. Introduction

The **reflection-method** library implements a technique for determining the moment of minimum light in eclipsing binary stars by exploiting the symmetry of the light curve about the primary minimum. The method was originally proposed as an alternative to classical techniques like the Kwee & van Woerden (KvW) method and polynomial fitting.

This document describes the algorithm, its theoretical basis, implementation details, and provides a comparison with established methods in the literature.

---

## 2. The Reflection Method: Algorithm

### 2.1 Principle

If a light curve is perfectly symmetric about the eclipse minimum at phase `x₀`, then reflecting all observed points about the vertical axis through `x₀` reconstructs the original curve. For a set of observations `(x_i, y_i)`, the reflected points are `(2x₀ - x_i, y_i)`. 

When the trial reflection point `x₀` coincides with the true minimum, the original and reflected points together form a symmetric set that is well-described by a single smooth curve (spline). The residual variance of a spline fitted to this combined dataset is minimized at the true minimum.

### 2.2 Mathematical Formulation

Given data points `{(x_i, y_i)}` for `i = 1...N` with optional weights `w_i`:

1. **Reflection:** For a trial `x₀`, define reflected abscissae `x'_i = 2x₀ - x_i`. The combined dataset is:
   ```
   X = {x_i} ∪ {x'_i}
   Y = {y_i} ∪ {y_i}
   W = {w_i} ∪ {w_i}  (if weights provided)
   ```

2. **Spline Fit:** Fit a least-squares spline `S(x; x₀)` to the combined `(X, Y, W)` using a fixed number of points per knot (`pts_per_knot`) and spline degree `k`.

3. **Objective Function:** Compute the residual standard deviation:
   ```
   σ₂(x₀) = √[ Σ W_j (Y_j - S(X_j; x₀))² / (N_eff - dof) ]
   ```
   where `N_eff` is the number of combined points and `dof` accounts for spline degrees of freedom.

4. **Optimization:** Find `x₀* = argmin σ₂(x₀)`. The function `σ₂(x₀)` is smooth and unimodal near the true minimum. We scan over a window around an initial guess, then refine the grid minimum with a local polynomial fit to the `σ₂` curve (Section 3.2).

### 2.3 Uncertainty Estimation: Residual Bootstrap

To estimate the uncertainty of `x₀*`:

1. Fit initial spline `S₁(x)` to original data (without reflection).
2. Compute residuals `r_i = y_i - S₁(x_i)`.
3. For `b = 1...B` bootstrap iterations:
   - Generate bootstrap sample: `y_i^(b) = S₁(x_i) + r_{π(i)}` where `π` is a random permutation.
   - Run full reflection-method scan on `{(x_i, y_i^(b))}` to obtain `x₀^(b)`.
4. The bootstrap distribution `{x₀^(b)}` provides:
   - Standard error: `σ(x₀) = std({x₀^(b)})`
   - 68% confidence interval: `[P₁₆, P₈₄]` percentiles

This residual bootstrap preserves the noise structure while reflecting the uncertainty in the minimum location.

---

## 3. Implementation Details

### 3.1 Spline Fitting

- **Method:** `LSQUnivariateSpline` (SciPy) — least-squares B-spline with specified knots.
- **Knot placement:** Uniform in `x`, `n_inner = max(1, N / pts_per_knot)` interior knots.
- **Degree:** Configurable (default `k=3`, cubic).
- **Extrapolation:** `ext=3` — raises error outside knot range (prevents silent extrapolation).
- **Weights:** Supported via `w` parameter (inverse variance weights).

### 3.2 Scanning Procedure

1. **Initial guess:** Minimum of coarse spline fit to original data (2001-point grid).
2. **Window:** `± x0_window * (xmax - xmin)` around guess (default 10% of range).
3. **Grid:** `n_scan` points (default 200) linearly spaced in window.
4. **Refinement:** A local polynomial fit to the `σ₂(x₀)` grid.

### 3.3 Refinement of the σ₂ Minimum

The grid scan locates `σ₂(x₀)` to within the grid spacing. The true minimum
is refined by fitting a polynomial to the `σ₂` curve *inside a narrow window
around the grid minimum*:

- The fit window is `± parabola_window * (x_grid_max - x_grid_min)` around
  the grid minimum (default `parabola_window = 0.05`, i.e. 5% of the scan
  range).
- When at least **five** grid points fall in the window, a **quartic**
  polynomial is fitted (`numpy.polyfit`, degree 4). The σ₂ curve is not
  exactly parabolic near the minimum, so the higher order reduces bias.
- With fewer than five but at least three points, a **parabola** is fitted.
  With fewer than three, the grid minimum itself is used.
- The minimum of the polynomial is located **analytically**: the roots of
  the derivative (a cubic for the quartic) are filtered to those inside the
  fit window, and the candidate (plus the window endpoints) with the lowest
  polynomial value is chosen. For a parabola the vertex formula is used.
- The result is clamped to the scan bounds.

Rationale: interpolating the `σ₂` curve with a spline and minimizing it
numerically (an earlier implementation) is unstable when the curve is
nearly flat near the minimum — tiny noise then creates spurious local
minima. A low-order polynomial fitted to a few local points is robust to
this, because fitting averages over the noise instead of interpolating it.
The same refinement is applied inside every bootstrap iteration, so the
uncertainty estimate reflects the actual production algorithm.

### 3.4 Bootstrap Details

- **Residuals:** From initial spline fit (not the reflected fit).
- **Resampling:** With replacement from residuals.
- **Per-iteration scan:** Coarser grid (`n_scan_boot`, default 80) for speed.
- **Refinement:** Same local polynomial fit as the main scan
  (`parabola_window`).
- **RNG:** NumPy PCG64 (`np.random.default_rng()`), seedable for reproducibility.

### 3.5 Time/Unit Handling

The core library is **unit-agnostic**: `x` can be JD, HJD, MJD, phase, minutes, etc. All outputs (`x₀`, `σ`, percentiles) are in the same units as input `x`. Time conversion (e.g., DATE-OBS → minutes → UTC) is handled by the CLI layer.

---

## 4. Comparison with Established Methods

### 4.1 Kwee & van Woerden (1956) — The "KvW" Method

**Reference:** Kwee, K. K., & van Woerden, H. (1956). *A method for computing accurately the time of minimum of an eclipsing variable*. Bulletin of the Astronomical Institutes of the Netherlands, 12, 327.

**Principle:** Assume the light curve near minimum is parabolic: `y = a(x - x₀)² + b(x - x₀) + c`. Fit a parabola to a selected symmetric interval around the apparent minimum using weighted least squares. The vertex gives `x₀`.

**Procedure:**
1. Select symmetric points around the apparent minimum.
2. Fit parabola `y = Ax² + Bx + C` (or `y = a(x - x₀)² + c`).
3. Vertex at `x₀ = -B/(2A)`.
4. Uncertainty via propagation of errors or Jackknife.

**Comparison:**

| Aspect | KvW | Reflection Method |
|--------|-----|-------------------|
| **Model assumption** | Parabolic (quadratic) near min | Non-parametric (spline) |
| **Data usage** | Subset near minimum | All available data (reflected) |
| **Symmetry handling** | Assumes perfect symmetry | Tests symmetry via residual variance |
| **Asymmetric curves** | Biased if curve non-parabolic/asymmetric | Robust: σ₂ increases if asymmetry present |
| **Outliers** | Sensitive (least squares on subset) | Robust (spline smoothing, all points) |
| **Parameter selection** | Interval width | `pts_per_knot`, `x0_window`, `degree` |
| **Uncertainty** | Analytical (parabola) | Bootstrap (non-parametric) |

**Key advantage of reflection method:** Does not assume a specific functional form (parabola). The spline adapts to the actual shape of the light curve. This is critical for:
- Contact/overcontact binaries with flat minima
- Stars with significant limb darkening distortions
- Cases where the minimum is broad or asymmetric

**Key disadvantage:** More computationally intensive; requires tuning of spline parameters.

### 4.2 Polynomial Fitting (Low-Order)

**Principle:** Fit a polynomial (typically 2nd–4th order) to a window around the minimum using weighted least squares. The minimum of the polynomial is found analytically or numerically.

**Comparison:**

| Aspect | Polynomial Fit | Reflection Method |
|--------|----------------|-------------------|
| **Model** | Fixed global polynomial | Local adaptive spline |
| **Window** | Must be chosen carefully | Uses all data via reflection |
| **Runge phenomenon** | Risk with high degree near edges | None (B-splines local support) |
| **Asymmetry** | Bias if curve not polynomial | Naturally handled |
| **Derivatives** | Analytical | Numerical (spline) |

**Literature:** Andronov (1990, 2003) discusses polynomial and trigonometric fits for minima. The reflection method can be seen as a non-parametric generalization that avoids the "which polynomial degree?" and "which window?" dilemmas.

### 4.3 Other Methods

| Method | Reference | Notes |
|--------|-----------|-------|
| **Trigonometric fit** | Mikulášek et al. (2008) | Good for periodic data, overkill for single minimum |
| **Gaussian/sigmoid fit** | Common in exoplanet transits | Model-dependent; reflection is model-free |
| **Bisector/Chord method** | Historical | Low precision, obsolete |
| **MCMC light-curve modeling** | Modern (e.g., `jktebop`, `PHOEBE`) | Full physical modeling; different goal (system parameters) |

---

## 5. Theoretical Justification

### 5.1 Symmetry as Optimality Criterion

The reflection method minimizes the **lack of symmetry** measured by residual variance. If the true light curve is symmetric about `x₀*`, then for any `x₀ ≠ x₀*` the reflected points introduce artificial distortions that increase the residual variance of any smooth interpolant. This is equivalent to maximizing the symmetry of the combined dataset.

### 5.2 Connection to Cross-Validation

The combined dataset (original + reflected) can be viewed as a form of self-consistency check. The spline fit to the combined data is essentially a symmetry-constrained smoothing problem. Minimizing `σ₂(x₀)` finds the axis of symmetry that makes the data most "self-consistent" under reflection.

### 5.3 Statistical Properties

- **Consistency:** As `N → ∞` and `pts_per_knot → ∞` appropriately, `x₀* → x₀_true` under mild smoothness assumptions.
- **Efficiency:** The method uses all data points twice (original + reflected), achieving near-optimal information extraction for symmetric curves.
- **Robustness:** Spline smoothing provides automatic outlier resistance compared to raw least-squares parabola fits.

---

## 6. Practical Recommendations

### 6.1 Parameter Choices

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `pts_per_knot` | 10 | 5–20; smaller = more flexible, larger = smoother. For sparse data, use larger values. |
| `degree` | 3 | 3 (cubic) standard. Degree 1 for very noisy data, 4–5 only for very dense, smooth curves. |
| `n_scan` | 200 | Sufficient for smooth `σ₂` curve. Increase for very broad minima. |
| `x0_window` | 0.1 | 5–20% of range. Must contain true minimum. Check scan plot. |
| `parabola_window` | 0.05 | 2–10% of scan range. Window for the polynomial refinement; wider = more smoothing, narrower = more local. |
| `n_bootstrap` | 60 | 50–200. More = better CI precision, slower. |
| `n_scan_boot` | 80 | Coarser than main scan for speed. |

### 6.2 Diagnostics

Always inspect:
1. **Scan plot (`σ₂(x₀)`):** Should show clear, smooth minimum. Multiple minima → check `x0_window` or data quality.
2. **Main plot (light curve + splines):** Original vs. refitted spline should overlap closely at minimum. Systematic deviations → asymmetry or model mismatch.
3. **Bootstrap histogram:** Should be roughly unimodal and symmetric. Skewed → non-Gaussian uncertainty; report percentiles.

### 6.3 When to Use Reflection Method

- **Yes:** Eclipsing binaries with well-sampled minima; contact/overcontact systems; cases where parabolic fit is questionable; need for robust uncertainty.
- **No:** Very sparse data (< 20 points near minimum); extremely asymmetric curves (e.g., heartbeat stars); real-time processing (too slow).

---

## 7. References

### Core Method
- **Kwee, K. K., & van Woerden, H. (1956).** *A method for computing accurately the time of minimum of an eclipsing variable*. Bull. Astron. Inst. Netherlands, 12, 327. [Classic parabolic method]

### Polynomial/Trigonometric Fits
- **Andronov, I. L. (1990).** *On the determination of moments of minimum light of eclipsing variables*. In *IAU Colloq. 121: Photometric and Spectroscopic Variations of Stars*.
- **Andronov, I. L. (2003).** *Asymptotic behaviour of the period of eclipsing variables*. Odessa Astron. Publ., 16, 55.
- **Mikulášek, Z., et al. (2008).** *New methods for determination of moments of minima of eclipsing binaries*. Astron. Nachr., 329, 118.

### Spline-Based Methods
- **De Boor, C. (2001).** *A Practical Guide to Splines*. Springer. [B-spline theory]
- **Eilers, P. H. C., & Marx, B. D. (1996).** *Flexible smoothing with B-splines and penalties*. Stat. Sci., 11, 89. [P-splines, related concept]

### Bootstrap
- **Efron, B., & Tibshirani, R. J. (1993).** *An Introduction to the Bootstrap*. Chapman & Hall.
- **Davison, A. C., & Hinkley, D. V. (1997).** *Bootstrap Methods and their Application*. Cambridge Univ. Press.

### Light Curve Analysis / Eclipsing Binaries
- **Kallrath, J., & Milone, E. F. (2009).** *Eclipsing Binary Stars: Modeling and Analysis*. Springer.
- **Southworth, J. (2011).** *Eclipsing binary systems: the key to understanding stellar physics*. In *Eclipsing Binaries as Astrophysical Tools*.
- **Příbulla, T., et al. (2012).** *Minima of eclipsing binaries — new methods and results*. Contrib. Astron. Obs. Skalnaté Pleso, 42, 217.

### Related Software
- **Kwee-van Woerden implementations:** `VARTOOLS` (Hartman & Bakos 2016), `PyAstronomy` (Czesla et al. 2019).
- **Full light-curve modeling:** `PHOEBE` (Prša et al. 2016), `jktebop` (Southworth 2013), `ellc` (Maxted 2016).

---

## 8. Limitations & Future Work

### Known Limitations
1. **Assumes near-symmetry:** Strongly asymmetric minima (e.g., ellipsoidal variables, spots) will bias `x₀`.
2. **Spline tuning:** Requires user judgment on `pts_per_knot`; no fully automatic selection yet.
3. **Computational cost:** Bootstrap with spline refitting is O(B × N_scan × N_logN); slower than analytical KvW.
4. **No physical parameters:** Outputs `x₀` only; no inclination, mass ratio, etc.

### Planned Improvements
- **Automatic knot selection** via generalized cross-validation (GCV) or AIC.
- **Robust spline fitting** (Huber loss) for outlier-heavy data.
- **Asymmetry diagnostics** (quantify skewness of `σ₂(x₀)` curve).
- **MCMC-based uncertainty** as alternative to bootstrap.
- **Parallel bootstrap** for speed.

---

## 9. License

MIT License — see LICENSE file.

---

*Document version 0.2.0 — accompanying reflection-method library v0.2.0*