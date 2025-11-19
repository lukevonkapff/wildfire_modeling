import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import lognorm, norm
from scipy.special import gammaincc, gamma
import warnings
from scipy.optimize import differential_evolution
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import rasterio
import rasterio.mask
from scipy import stats
from collections import Counter
import re
import statsmodels.api as sm
import pickle
import wildfire_powerlaw as wfpl


def ccdf_exponential(x: np.ndarray, lam: float, xmin: int = 12) -> np.ndarray:
    """Compute the complementary CDF (CCDF) of an exponential distribution.

    Args:
        x: Input array of fire sizes.
        lam: Exponential rate parameter λ.
        xmin: Lower cutoff value.

    Returns:
        CCDF values normalized to 1 at xmin.
    """
    # Restrict support to values at or above xmin so the CCDF is well-defined on the tail
    mask = x >= xmin

    # Compute exponential CCDF shifted to start at xmin
    ccdf = np.exp(-lam * (x - xmin))

    # Normalize so that CCDF(xmin) = 1 (on the valid mask)
    ccdf /= max(ccdf[mask][0], 1e-12)

    # Clip to avoid exact zeros or ones, which helps with log-scale plotting
    return np.clip(ccdf, 1e-12, 1.0)


def ccdf_power_law(x: np.ndarray, alpha: float, xmin: int = 12) -> np.ndarray:
    """Compute the CCDF of a pure power-law distribution.

    Args:
        x: Fire sizes or other continuous data.
        alpha: Power-law exponent α (>1 typical).
        xmin: Minimum cutoff for valid values.

    Returns:
        CCDF normalized at xmin.
    """
    # Restrict to the domain x >= xmin where the power-law model is assumed to hold
    mask = x >= xmin

    # Compute CCDF in closed form relative to xmin
    ccdf = (x / xmin) ** (1 - alpha)

    # Normalize to 1 at xmin for comparability across distributions
    ccdf /= max(ccdf[mask][0], 1e-12)

    # Clip to guard against numerical underflow/overflow
    return np.clip(ccdf, 1e-12, 1.0)


def ccdf_truncated_power_law(
    x: np.ndarray, alpha: float, lambd: float, xmin: int = 4
) -> np.ndarray:
    """Stable truncated power-law CCDF.

    Automatically reduces to a pure power law when λ ≈ 0 or α is large.

    Args:
        x: Input array of data (e.g., fire sizes).
        alpha: Power-law exponent.
        lambd: Truncation rate λ.
        xmin: Minimum cutoff for validity.

    Returns:
        CCDF array (same shape as x), numerically stable.
    """
    # Ensure floating-point array for downstream operations
    x = np.asarray(x, dtype=float)

    # Identify values above the lower cutoff and precompute λx
    mask = x >= xmin
    z = lambd * x[mask]

    # If λ is very small or α is large, fall back to pure power law (truncation negligible)
    if lambd < 1e-3 or np.all(z < 1e-2) or alpha > 5:
        return ccdf_power_law(x, alpha, xmin)

    try:
        # Compute normalized incomplete gamma expression for the CCDF
        val = gammaincc(1 - alpha, z) / gamma(1 - alpha)

        # If the incomplete gamma evaluation is numerically unstable, fall back to power law
        if not np.any(np.isfinite(val)) or np.all(val < 1e-12):
            # fallback to power law if gamma underflows
            return ccdf_power_law(x, alpha, xmin)

        # Initialize CCDF and assign values only where x >= xmin
        ccdf = np.zeros_like(x)
        ccdf[mask] = np.clip(val, 1e-12, 1.0)

        # Normalize at xmin for consistency across distributions
        ccdf /= max(ccdf[mask][0], 1e-12)
    except Exception:
        # On any unexpected numerical error, revert to a pure power-law CCDF
        return ccdf_power_law(x, alpha, xmin)

    # Final clipping to ensure numerical robustness on log plots
    return np.clip(ccdf, 1e-12, 1.0)


def ccdf_stretched_exponential(
    x: np.ndarray, lam: float, beta: float, xmin: int = 12
) -> np.ndarray:
    """Compute the CCDF of a stretched exponential (Weibull form).

    Args:
        x: Input array of values ≥ xmin.
        lam: Rate parameter λ.
        beta: Stretching exponent β.
        xmin: Lower cutoff for normalization.

    Returns:
        Normalized CCDF array.
    """
    # Restrict CCDF definition to tail values x >= xmin
    mask = x >= xmin

    # Compute stretched exponential CCDF shifted to normalize at xmin
    ccdf = np.exp(-(lam * x) ** beta + (lam * xmin) ** beta)

    # Normalize to CCDF(xmin) = 1
    ccdf /= max(ccdf[mask][0], 1e-12)

    # Clip extremes for numerical stability
    return np.clip(ccdf, 1e-12, 1.0)


def ccdf_weibull(x: np.ndarray, k: float, lam: float, xmin: int = 12) -> np.ndarray:
    """Compute the CCDF of a Weibull distribution.

    Args:
        x: Input data array.
        k: Shape parameter (β in some notation).
        lam: Scale parameter λ.
        xmin: Lower truncation cutoff.

    Returns:
        CCDF normalized to 1 at xmin.
    """
    # Restrict CCDF to the domain x >= xmin
    mask = x >= xmin

    # Compute Weibull CCDF with a shift to normalize at xmin
    ccdf = np.exp(-((x / lam) ** k - (xmin / lam) ** k))

    # Normalize to CCDF(xmin) = 1
    ccdf /= max(ccdf[mask][0], 1e-12)

    # Clip to avoid exactly 0 or 1 on log-scales
    return np.clip(ccdf, 1e-12, 1.0)


def ccdf_lognormal(x: np.ndarray, mu: float, sigma: float, xmin: int = 12) -> np.ndarray:
    """Compute the CCDF of a lognormal distribution.

    Args:
        x: Input values.
        mu: Log-mean of the distribution.
        sigma: Log-standard deviation.
        xmin: Minimum value for normalization.

    Returns:
        Normalized CCDF array.
    """
    # Restrict CCDF to x >= xmin for the tail behavior
    mask = x >= xmin

    # Use SciPy's lognormal survival function to compute CCDF
    ccdf = lognorm.sf(x, sigma, scale=np.exp(mu))

    # Normalize at xmin so CCDF(xmin) = 1
    ccdf /= max(ccdf[mask][0], 1e-12)

    # Clip for numerical safety
    return np.clip(ccdf, 1e-12, 1.0)


def ccdf_genpareto(x: np.ndarray, xi: float, sigma: float, xmin: int = 12) -> np.ndarray:
    """Compute CCDF for the Generalized Pareto Distribution (GPD).

    Args:
        x: Input data (e.g., fire sizes).
        xi: Shape parameter ξ.
        sigma: Scale parameter σ.
        xmin: Lower truncation bound.

    Returns:
        CCDF array normalized at xmin.
    """
    # Restrict to tail values and shift domain to start at xmin
    mask = x >= xmin
    y = x[mask] - xmin

    # Use exponential limit when shape parameter is very close to zero
    if abs(xi) < 1e-6:
        ccdf = np.exp(-y / sigma)
    else:
        # Standard GPD CCDF expression
        ccdf = (1 + xi * y / sigma) ** (-1 / xi)

    # Initialize full output and assign computed CCDF only to valid entries
    full_ccdf = np.zeros_like(x)
    full_ccdf[mask] = ccdf

    # Normalize to CCDF(xmin) = 1
    full_ccdf /= max(full_ccdf[mask][0], 1e-12)

    # Final clipping for numerical stability
    return np.clip(full_ccdf, 1e-12, 1.0)


def plot_distribution_evolution_ccdf(df: pd.DataFrame, xmin: int = 12) -> None:
    """Plot CCDF evolution over relative time for each fitted distribution.

    Degenerate truncated power laws (tiny λ) are drawn as pure power laws.

    Args:
        df: DataFrame with columns ["distribution", "biome", "p1", "p2", "p1'", "p2'", "p1_slope_sig", "p2_slope_sig"].
        xmin: Minimum fire size cutoff.
    """
    # Define x-axis range (fire sizes) on a log grid and relative time steps
    x = np.logspace(np.log10(xmin), 3, 500)
    time_steps = np.linspace(-1, 1, 5)

    # Set up colormap and normalization to encode relative time with color
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=time_steps.min(), vmax=time_steps.max())

    # Iterate over rows, each representing a biome/distribution combination
    for _, row in df.iterrows():
        dist = row["distribution"]
        biome = row["biome"]

        # Helper to compute time-varying first parameter (p1) with slope if significant
        def p1(t: float) -> float:
            if row.get("p1_slope_sig", 0) == 0:
                return row["p1"]
            return row["p1"] + row["p1'"] * t

        # Helper to compute time-varying second parameter (p2) with slope if significant
        def p2(t: float) -> float | None:
            if np.isnan(row.get("p2", np.nan)):
                return None
            if row.get("p2_slope_sig", 0) == 0:
                return row["p2"]
            return row["p2"] + row["p2'"] * t

        # Create a new figure for this biome/distribution
        plt.figure(figsize=(6, 4))
        success = False

        # Loop over relative time points and compute corresponding CCDF curves
        for t in time_steps:
            y = None
            try:
                # Dispatch to appropriate CCDF function depending on distribution name
                if dist == "exponential":
                    y = ccdf_exponential(x, p1(t), xmin)
                elif dist == "power_law":
                    y = ccdf_power_law(x, p1(t), xmin)
                elif dist == "truncated_power_law":
                    alpha, lambd = p1(t), max(p2(t), 1e-8)
                    y = ccdf_truncated_power_law(x, alpha, lambd, xmin)
                elif dist == "stretched_exponential":
                    y = ccdf_stretched_exponential(x, p1(t), p2(t), xmin)
                elif dist == "weibull":
                    y = ccdf_weibull(x, p1(t), p2(t), xmin)
                elif dist == "lognormal":
                    y = ccdf_lognormal(x, p1(t), p2(t), xmin)
                elif dist == "generalized_pareto":
                    y = ccdf_genpareto(x, p1(t), p2(t), xmin)
            except Exception:
                # If something fails for this time step, skip this curve
                y = None

            # Skip invalid or fully NaN curves
            if y is None or np.all(np.isnan(y)):
                continue

            # Plot this CCDF curve, colored by relative time
            plt.plot(x, y, color=cmap(norm(t)), label=f"t={t:+.1f}")
            success = True

        # If no valid CCDFs were produced, skip plotting and move on
        if not success:
            plt.close()
            print(f"Skipped {biome} ({dist}) — all NaN CCDFs.")
            continue

        # Add titles, labels, and log scales
        plt.title(f"{biome}\n{dist.replace('_', ' ').title()} (CCDF)")
        plt.xlabel("Fire size (km²)")
        plt.ylabel("CCDF")
        plt.legend(title="Relative time")
        plt.xscale("log")
        plt.yscale("log")
        plt.tight_layout()
        plt.show()


def summarize_timevary_results_mode(
    timevary_results: dict, mode: str = "both"
) -> pd.DataFrame:
    """Convert one mode ('both', 'p1_only', or 'p2_only') from nested results into a DataFrame.

    The resulting DataFrame contains per-biome regression coefficients and
    slope significance flags for time-varying parameters.

    Args:
        timevary_results: Nested dictionary of results by biome → distribution → mode.
        mode: Which fit mode to extract ("both", "p1_only", or "p2_only").

    Returns:
        DataFrame with columns:
        biome, distribution, n, p1, p1_se, p1', p1'_se, p2, p2_se, p2', p2'_se,
        p1_slope_sig, p2_slope_sig.
    """
    # Initialize list to hold tidy records for each biome×distribution combination
    rows: list[dict] = []

    # Iterate through biomes and distributions stored in the nested results dict
    for biome, dists in timevary_results.items():
        for dist, modes in dists.items():
            # Skip distributions without the requested mode
            if mode not in modes:
                continue

            # Extract coefficient and SE arrays plus sample size
            res = modes[mode]
            coeffs = np.array(res.get("coeffs", []), dtype=float)
            ses = np.array(res.get("ses", []), dtype=float)
            n = res.get("n", np.nan)

            # Initialize parameters and significance flags with NaN/zero defaults
            p1 = p1_se = p1s = p1s_se = np.nan
            p2 = p2_se = p2s = p2s_se = np.nan
            p1_sig = p2_sig = 0

            # For each mode, interpret the shape of the coeff vector and derive significance
            # based on 95% confidence intervals around the slope terms.
            # interpret coefficients by mode length
            if mode == "both":
                if len(coeffs) == 2:
                    # Only p1 and its slope vary; no second parameter
                    p1, p1s = coeffs
                    p1_se, p1s_se = ses
                    ci_low, ci_high = p1s - 1.96 * p1s_se, p1s + 1.96 * p1s_se
                    p1_sig = 1 if (ci_low > 0 or ci_high < 0) else 0
                elif len(coeffs) == 4:
                    # Both p1 and p2 and their slopes are present
                    p1, p1s, p2, p2s = coeffs
                    p1_se, p1s_se, p2_se, p2s_se = ses
                    ci_low1, ci_high1 = p1s - 1.96 * p1s_se, p1s + 1.96 * p1s_se
                    ci_low2, ci_high2 = p2s - 1.96 * p2s_se, p2s + 1.96 * p2s_se
                    p1_sig = 1 if (ci_low1 > 0 or ci_high1 < 0) else 0
                    p2_sig = 1 if (ci_low2 > 0 or ci_high2 < 0) else 0

            elif mode == "p1_only":
                if len(coeffs) == 2:
                    # Only p1 varies over time; no p2 parameter
                    p1, p1s = coeffs
                    p1_se, p1s_se = ses
                    ci_low, ci_high = p1s - 1.96 * p1s_se, p1s + 1.96 * p1s_se
                    p1_sig = 1 if (ci_low > 0 or ci_high < 0) else 0
                elif len(coeffs) == 3:
                    # p1 varies with time and p2 is static
                    p1, p1s, p2 = coeffs
                    p1_se, p1s_se, p2_se = ses
                    ci_low, ci_high = p1s - 1.96 * p1s_se, p1s + 1.96 * p1s_se
                    p1_sig = 1 if (ci_low > 0 or ci_high < 0) else 0

            elif mode == "p2_only":
                if len(coeffs) == 2:
                    # p1 is static; p2 varies with time
                    p1, p2s = coeffs
                    p1_se, p2s_se = ses
                    ci_low, ci_high = p2s - 1.96 * p2s_se, p2s + 1.96 * p2s_se
                    p2_sig = 1 if (ci_low > 0 or ci_high < 0) else 0
                elif len(coeffs) == 3:
                    # p1 and p2 are both present, but only p2 has a slope
                    p1, p2, p2s = coeffs
                    p1_se, p2_se, p2s_se = ses
                    ci_low, ci_high = p2s - 1.96 * p2s_se, p2s + 1.96 * p2s_se
                    p2_sig = 1 if (ci_low > 0 or ci_high < 0) else 0

            # Append a tidy record summarizing coefficients and slope flags
            rows.append(
                {
                    "biome": biome,
                    "distribution": dist,
                    "n": n,
                    "p1": p1,
                    "p1_se": p1_se,
                    "p1'": p1s,
                    "p1'_se": p1s_se,
                    "p2": p2,
                    "p2_se": p2_se,
                    "p2'": p2s,
                    "p2'_se": p2s_se,
                    "p1_slope_sig": p1_sig,
                    "p2_slope_sig": p2_sig,
                }
            )

    # Convert accumulated records into a DataFrame and enforce column order
    df = pd.DataFrame(rows)
    df = df[
        [
            "biome",
            "distribution",
            "n",
            "p1",
            "p1_se",
            "p1'",
            "p1'_se",
            "p2",
            "p2_se",
            "p2'",
            "p2'_se",
            "p1_slope_sig",
            "p2_slope_sig",
        ]
    ]
    return df


def analyze_time_varying_mle(
    mtbs_classified: pd.DataFrame | gpd.GeoDataFrame,
    overall_results: dict,
    year_col: str = "year",
    xmin: float = 4,
    llhr_cutoff: float = 2.0,
    R_boot: int = 20,
    relerr_cutoff: float = 1.0,  # kept for API compatibility (not used)
    min_total: int = 400,
    verbose: bool = True,
    prior_weight: float = 1e-3,
) -> dict:
    """Time-varying MLE using global Differential Evolution optimization.

    This fits linear-in-time parameterizations (on stable transforms) for each
    candidate distribution per biome. For the truncated power law (TPL), it uses:
        α(t) = 1 + exp(a1 + b1 * t)
        λ(t) = exp(a2 + b2 * t)
    which guarantees α>1 and λ>0 while keeping slopes interpretable.

    A weak prior nudges parameters toward the static fit (from `overall_results`)
    to prevent drifting, especially for short series.

    Args:
        mtbs_classified: Fire records with columns `[year_col, area_km2, modis_class_static]`.
        overall_results: Nested dict from your static fitting pass:
            overall_results[biome]['params'] (DataFrame)
            overall_results[biome]['likelihood_matrix'] (DataFrame)
        year_col: Name of year column.
        xmin: Minimum size threshold for tail fitting.
        llhr_cutoff: Δlog-likelihood threshold used to filter candidate dists (smaller is better).
        R_boot: Bootstrap reps for SEs (via DE on resampled data).
        relerr_cutoff: Reserved; retained for compatibility (not used in this routine).
        min_total: Minimum number of tail observations per biome to attempt fitting.
        verbose: Print progress and filtering info.
        prior_weight: Strength of weak prior toward static parameters.

    Returns:
        Nested dict: results[biome][dist_name][mode] with keys:
          - "coeffs": transformed coefficients (p1, p1', p2, p2') or subset by mode
          - "ses": bootstrap SEs for the same
          - "loglik": best value of penalized log-likelihood (negated in optimizer)
          - "n": number of tail observations used
    """
    # Suppress runtime warnings inside the optimization loop to keep logs cleaner
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # --------- Log-PDF helpers (vectorized, stable ranges) ----------
    # Each helper returns log-PDF values for a given paramization, with clipping to avoid
    # numerical issues. These are used by the time-varying negative log-likelihood.
    def logpdf_lognormal(x: np.ndarray, mu: float, sigma: float, xmin: float = 0) -> np.ndarray:
        x = np.asarray(x)
        valid = x > xmin
        pdf = -np.inf * np.ones_like(x, dtype=float)

        # Avoid degenerate sigma by clipping to a reasonable range
        sigma = np.clip(sigma, 1e-6, 50)

        # Standard lognormal log-density for valid support
        pdf[valid] = (
            -np.log(x[valid]) - np.log(sigma)
            - 0.5 * ((np.log(x[valid]) - mu) / sigma) ** 2
            - np.log(np.sqrt(2 * np.pi))
        )
        return pdf

    def logpdf_powerlaw(x: np.ndarray, alpha: float, xmin: float = 1) -> np.ndarray:
        x = np.asarray(x)
        valid = x >= xmin
        pdf = -np.inf * np.ones_like(x, dtype=float)

        # Clip exponent to a plausible range for numerical stability
        alpha = np.clip(alpha, 0.0, 5)

        # Normalization constant for power-law density (with special-case α=1)
        C = (alpha - 1) / xmin if alpha != 1 else 1 / xmin

        # Log-density for valid x
        pdf[valid] = np.log(np.abs(C)) - alpha * np.log(x[valid] / xmin)
        return pdf

    def logpdf_trunc_powerlaw(x: np.ndarray, alpha: float, lambd: float, xmin: float = 1) -> np.ndarray:
        """Properly normalized truncated power law; stable for small/large α, λ."""
        x = np.asarray(x)
        valid = x >= xmin
        pdf = -np.inf * np.ones_like(x, dtype=float)

        # Clip α and λ to avoid excessively extreme values
        alpha = np.clip(alpha, 0, 5)
        lambd = np.clip(lambd, 0, 5)

        try:
            # Compute normalization constant using incomplete gamma for truncation at xmin
            Z = (
                (lambd ** (1 - alpha))
                * np.exp(lambd * xmin)
                * gammaincc(1 - alpha, lambd * xmin)
                * gamma(1 - alpha)
            )
            # Clip Z to avoid log underflow/overflow
            Z = np.clip(Z, 1e-300, 1e300)
            logZ = np.log(Z)
        except Exception:
            # On failure, fall back to unnormalized log-density (still usable for optimization)
            logZ = 0.0

        # Log-density over valid range
        pdf[valid] = -alpha * np.log(x[valid]) - lambd * (x[valid] - xmin) - logZ
        return pdf

    def logpdf_genpareto(x: np.ndarray, xi: float, sigma: float, xmin: float = 0) -> np.ndarray:
        x = np.asarray(x)
        y = x - xmin
        pdf = -np.inf * np.ones_like(x, dtype=float)

        # Clip shape and scale into a stable, valid range
        xi = np.clip(xi, -1, 2)
        sigma = np.clip(sigma, 1e-6, 10)

        # Guard against invalid combinations that would yield negative support
        valid = (sigma > 0) & (1 + xi * y / sigma > 0)

        # Standard GPD log-density
        pdf[valid] = -np.log(sigma) - (1 / xi + 1) * np.log(1 + xi * y[valid] / sigma)
        return pdf

    def logpdf_weibull(x: np.ndarray, k: float, lam: float, xmin: float = 0) -> np.ndarray:
        x = np.asarray(x)
        valid = x >= xmin
        pdf = -np.inf * np.ones_like(x, dtype=float)

        # Clip shape and scale to avoid degeneracy
        k = np.clip(k, 1e-6, 50)
        lam = np.clip(lam, 1e-6, 50)

        # Compute dimensionless ratio and ensure it is bounded
        z = np.clip(x[valid] / lam, 1e-12, 1e6)

        # Weibull log-density expression
        pdf[valid] = np.log(k) - np.log(lam) + (k - 1) * np.log(z) - z ** k
        return pdf

    def logpdf_stretched_exp(x: np.ndarray, lam: float, beta: float, xmin: float = 0) -> np.ndarray:
        x = np.asarray(x)
        valid = x >= xmin
        pdf = -np.inf * np.ones_like(x, dtype=float)

        # Constrain parameters to plausible ranges
        lam = np.clip(lam, 1e-6, 50)
        beta = np.clip(beta, 0.1, 5)

        # Stretched exponential (Weibull-form) log-density with shift to xmin
        pdf[valid] = (
            np.log(beta)
            + np.log(lam)
            + (beta - 1) * (np.log(x[valid]) + np.log(lam))
            + (lam * xmin) ** beta
            - (lam * x[valid]) ** beta
        )
        return pdf

    # Map distribution name to its corresponding log-PDF function
    dist_logpdfs: dict[str, callable] = {
        "lognormal": logpdf_lognormal,
        "power_law": logpdf_powerlaw,
        "truncated_power_law": logpdf_trunc_powerlaw,
        "generalized_pareto": logpdf_genpareto,
        "weibull": logpdf_weibull,
        "stretched_exponential": logpdf_stretched_exp,
    }

    # Initialize final results container
    timevary_results: dict = {}

    # Work on a copy of the MTBS input to avoid mutating caller's object
    mtbs_classified = mtbs_classified.copy()

    # Center years by mean and scale to decades to make slopes interpretable and well-scaled
    mtbs_classified["year_c"] = (mtbs_classified[year_col] - mtbs_classified[year_col].mean()) / 10.0

    # Global random generator used for bootstrapping indices
    rng_global = np.random.default_rng(42)

    # Loop over biomes (static MODIS classes) and fit time-varying models per biome
    for biome, subset in mtbs_classified.groupby("modis_class_static"):
        # Extract fire sizes and enforce tail threshold xmin
        data = subset["area_km2"].values
        data = data[data >= xmin]

        # If not enough fires above threshold, skip this biome
        if len(data) < min_total:
            if verbose:
                print(f"\n=== {biome} skipped: only {len(data)} fires above xmin ({xmin}) ===")
            continue

        # Extract centered years corresponding to tail data
        years = subset.loc[subset["area_km2"] >= xmin, "year_c"].values

        if verbose:
            print(f"\n=== {biome} (n={len(data)} fires ≥ {xmin}) ===")

        # Pull static fit results and likelihood matrix for this biome
        res = overall_results.get(biome, {})
        if not res:
            if verbose:
                print(f"No overall_results found for {biome}")
            continue

        params_df: pd.DataFrame = res["params"]
        llhr: pd.DataFrame = res["likelihood_matrix"]

        # Filter candidate distributions by availability, reductions, and Δloglik
        candidates: list[str] = []
        for dist, row in params_df.iterrows():
            # Skip distributions that are not implemented in the time-varying log-PDF map
            if dist not in dist_logpdfs:
                continue

            # Skip distributions that are recorded as reductions of other distributions
            if isinstance(row.get("reduces_to"), str):
                continue

            # Drop distributions that are everywhere dominated by alternatives in Δloglik
            if dist in llhr.index and llhr.loc[dist].min() > llhr_cutoff:
                continue

            candidates.append(dist)

        # If no suitable candidate distributions remain, skip this biome
        if not candidates:
            if verbose:
                print(f"No viable candidates for {biome}")
            continue

        # Container for per-distribution fit results within this biome
        biome_res: dict = {}

        # Loop over candidate distributions and fit multiple "modes" (which params can vary over time)
        for dist_name in candidates:
            fit_modes: dict = {}

            # Which parameters may vary with time in each "mode"
            fit_configs = {"both": [True, True], "p1_only": [True, False], "p2_only": [False, True]}

            # Extract static parameter estimates to define a weak prior
            static_row = params_df.loc[dist_name]
            p1_static = float(static_row.get("p1", 1.0))
            p2_static = float(static_row.get("p2", 1.0)) if not pd.isna(static_row.get("p2", np.nan)) else 1.0

            # Iterate across modes that allow different parameters to vary with time
            for mode, (fit_p1, fit_p2) in fit_configs.items():
                # -------- Negative log-likelihood with weak prior ----------
                # This function is passed to the differential evolution optimizer. It unpacks
                # the linear-in-time parameterization, computes log-PDF values for each
                # observation/time, and adds a quadratic penalty to keep parameters near
                # the static fit on a transformed scale.
                def neg_loglik(params: np.ndarray, data: np.ndarray = data) -> float:
                    try:
                        # unpack by distribution + mode
                        # For the TPL, interpret parameters as a1,b1,a2,b2 (or subsets)
                        if dist_name == "truncated_power_law":
                            if mode == "both":
                                a1, b1, a2, b2 = params
                            elif mode == "p1_only":
                                a1, b1, a2 = params
                                b2 = 0.0
                            elif mode == "p2_only":
                                a1, a2, b2 = params
                                b1 = 0.0
                        else:
                            # For other distributions, we still treat (a1,b1,a2,b2) as generic
                            if mode == "both":
                                a1, b1, a2, b2 = params
                            elif mode == "p1_only":
                                a1, b1, a2 = params
                                b2 = 0.0
                            elif mode == "p2_only":
                                a1, a2, b2 = params
                                b1 = 0.0

                        # time-varying parameterization (per distribution)
                        # For each distribution, map (a1,b1,a2,b2) to interpretable parameters (e.g. α, λ)
                        if dist_name == "truncated_power_law":
                            alpha = 1.0 + np.exp(a1 + b1 * years)
                            lambd = np.exp(a2 + b2 * years)
                            ll = logpdf_trunc_powerlaw(data, alpha, lambd, xmin)
                        elif dist_name == "lognormal":
                            mu = a1 + b1 * years
                            sigma = np.clip(a2 + b2 * years, 1e-6, 50)
                            ll = logpdf_lognormal(data, mu, sigma, xmin)
                        elif dist_name == "generalized_pareto":
                            xi = np.clip(a1 + b1 * years, -1, 2)
                            sigma = np.clip(a2 + b2 * years, 1e-6, 10)
                            ll = logpdf_genpareto(data, xi, sigma, xmin)
                        elif dist_name == "power_law":
                            alpha = np.clip(a1 + b1 * years, 0.0, 5)
                            ll = logpdf_powerlaw(data, alpha, xmin)
                        elif dist_name == "weibull":
                            k = np.clip(a1 + b1 * years, 1e-3, 10)
                            lam = np.clip(a2 + b2 * years, 1e-3, 50)
                            ll = logpdf_weibull(data, k, lam, xmin)
                        elif dist_name == "stretched_exponential":
                            lam = np.clip(a1 + b1 * years, 1e-4, 50)
                            beta = np.clip(a2 + b2 * years, 0.1, 5)
                            ll = logpdf_stretched_exp(data, lam, beta, xmin)
                        else:
                            # Return infinite cost for unknown distributions
                            return np.inf

                        # Drop any non-finite log-likelihood contributions
                        valid_ll = ll[np.isfinite(ll)]

                        # weak prior → center around static p1/p2 (on stable transforms)
                        # Encode prior on transformed scale to discourage extreme deviations
                        if dist_name == "truncated_power_law":
                            prior_center_a1 = np.log(max(p1_static - 1, 1e-3))  # α → log(α-1)
                        else:
                            prior_center_a1 = np.log(max(p1_static, 1e-6))      # p1 → log(p1)
                        prior_center_a2 = np.log(max(p2_static, 1e-6))          # p2 → log(p2)

                        # Quadratic penalty around prior centers
                        prior_penalty = prior_weight * ((a1 - prior_center_a1) ** 2 + (a2 - prior_center_a2) ** 2)

                        # Return penalized negative log-likelihood
                        return -np.sum(valid_ll) + prior_penalty
                    except Exception:
                        # Any unexpected failure yields infinite cost to be rejected by the optimizer
                        return np.inf

                # --------- Bounds by distribution and mode ----------
                # Define parameter bounds for each distribution on the (a1,b1,a2,b2) scale.
                # These are chosen to keep transformed parameters in reasonable ranges.
                if dist_name == "truncated_power_law":
                    bounds = [(-2, 2), (-1, 1), (-9, 0), (-1, 1)]  # (a1,b1,a2,b2)
                elif dist_name == "generalized_pareto":
                    bounds = [(-1, 2), (-1, 1), (1e-6, 10), (-1, 1)]
                elif dist_name == "power_law":
                    bounds = [(0.0, 5), (-1, 1)]  # (a1,b1) for α(t)=a1 + b1*t (clipped later)
                elif dist_name == "weibull":
                    bounds = [(1e-3, 10), (-1, 1), (1e-3, 50), (-1, 10)]
                elif dist_name == "stretched_exponential":
                    bounds = [(1e-4, 50), (-1, 1), (0.1, 5), (-1, 1)]
                else:
                    bounds = [(-5, 5), (-1, 1), (-5, 5), (-1, 1)]

                # Adjust bounds to drop parameters that are fixed in this mode
                if mode == "both":
                    bnds = bounds
                elif mode == "p1_only":
                    bnds = [bounds[0], bounds[1], bounds[2]]
                elif mode == "p2_only":
                    bnds = [bounds[0], bounds[2], bounds[3]]

                # -------- Differential Evolution global search ----------
                # Use SciPy's differential_evolution to globally optimize the penalized
                # negative log-likelihood for this biome/distribution/mode.
                opt = differential_evolution(
                    neg_loglik,
                    bounds=bnds,
                    strategy="best1bin",
                    maxiter=600,
                    popsize=15,
                    polish=True,
                    seed=42,
                    updating="deferred",
                    workers=1,
                    init="latinhypercube",
                )
                coeffs = opt.x
                ll_max = -opt.fun

                # -------- Bootstrap via DE on resampled data ----------
                # Approximate standard errors by re-fitting on bootstrap resamples of the data.
                boot_params: list[np.ndarray] = []
                for _ in range(R_boot):
                    # Sample indices with replacement
                    idx = rng_global.choice(len(data), size=len(data), replace=True)
                    boot_data = data[idx]
                    try:
                        # Optimize on bootstrap sample (less exhaustive to save compute)
                        opt_b = differential_evolution(
                            lambda p: neg_loglik(p, data=boot_data),
                            bounds=bnds,
                            maxiter=300,
                            polish=False,
                            seed=None,
                            updating="deferred",
                            workers=1,
                        )
                        boot_params.append(opt_b.x)
                    except Exception:
                        # allow some failures (ill-conditioned resamples)
                        continue

                # -------- Transform coefficients to (p1, p1', p2, p2') --------
                # Convert the (a1,b1,a2,b2) representation into the more interpretable
                # (p1, p1', p2, p2') summary, with analogous transforms applied to bootstraps.
                if dist_name == "truncated_power_law":
                    # Unpack coefficients according to mode
                    if mode == "both":
                        a1, b1, a2, b2 = coeffs
                    elif mode == "p1_only":
                        a1, b1, a2 = coeffs
                        b2 = 0.0
                    elif mode == "p2_only":
                        a1, a2, b2 = coeffs
                        b1 = 0.0

                    # Transform back to α(t) = 1 + exp(a1 + b1*t), λ(t) = exp(a2 + b2*t)
                    p1 = 1 + np.exp(a1)
                    p1_prime = b1 * np.exp(a1)
                    p2 = np.exp(a2)
                    p2_prime = b2 * np.exp(a2)

                    # If bootstraps succeeded, transform and compute standard deviations
                    if boot_params:
                        boot_p1, boot_p1p, boot_p2, boot_p2p = [], [], [], []
                        for bp in boot_params:
                            if mode == "both":
                                ba1, bb1, ba2, bb2 = bp
                            elif mode == "p1_only":
                                ba1, bb1, ba2 = bp
                                bb2 = 0.0
                            elif mode == "p2_only":
                                ba1, ba2, bb2 = bp
                                bb1 = 0.0
                            boot_p1.append(1 + np.exp(ba1))
                            boot_p1p.append(bb1 * np.exp(ba1))
                            boot_p2.append(np.exp(ba2))
                            boot_p2p.append(bb2 * np.exp(ba2))
                        p1_se = np.std(boot_p1)
                        p1p_se = np.std(boot_p1p)
                        p2_se = np.std(boot_p2)
                        p2p_se = np.std(boot_p2p)
                    else:
                        # If no successful bootstraps, fall back to zero SEs
                        p1_se = p1p_se = p2_se = p2p_se = 0.0

                    coeffs_trans = [p1, p1_prime, p2, p2_prime]
                    ses_trans = [p1_se, p1p_se, p2_se, p2p_se]
                else:
                    # For non-TPL distributions we keep coefficients on the original scale
                    ses = np.std(boot_params, axis=0) if boot_params else np.zeros_like(coeffs)
                    coeffs_trans = coeffs
                    ses_trans = ses

                # Store mode-specific results for this distribution
                fit_modes[mode] = {
                    "coeffs": coeffs_trans,
                    "ses": ses_trans,
                    "loglik": ll_max,
                    "n": len(data),
                }

            # If at least one mode was fitted, attach results for this distribution
            if fit_modes:
                biome_res[dist_name] = fit_modes

        # Attach biome-level results if any distributions succeeded
        if biome_res:
            timevary_results[biome] = biome_res

    # Return nested dictionary of all biome/distribution/mode fits
    return timevary_results


def plot_savanna_fires(mtbs_classified: pd.DataFrame | gpd.GeoDataFrame, biome: str = "both") -> None:
    """Plot Savanna / Woody Savanna fires on a Cartopy basemap (includes Alaska fallback).

    Args:
        mtbs_classified: Dataset with columns LATITUDE, LONGITUDE, modis_class_static, geometry (if GeoDataFrame).
        biome: One of {"Savannas", "Woody savannas", "both"}.
    """
    # Suppress runtime warnings from Cartopy/geopandas operations
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Validate requested biome option
    valid_biomes = ["Savannas", "Woody savannas", "both"]
    if biome not in valid_biomes:
        raise ValueError(f"biome must be one of {valid_biomes}")

    # Decide which biomes to retain based on user input
    target_biomes = ["Savannas", "Woody savannas"] if biome == "both" else [biome]

    # Filter to the selected biomes and require finite coordinates
    subset = mtbs_classified[mtbs_classified["modis_class_static"].isin(target_biomes)].copy()
    subset = subset[subset["LATITUDE"].apply(np.isfinite) & subset["LONGITUDE"].apply(np.isfinite)]

    # If no valid fires, report and return early
    if subset.empty:
        print(f"No valid fires found for {', '.join(target_biomes)}.")
        return

    # ensure geospatial types
    # If the subset is not already a GeoDataFrame, construct geometry from lat/lon
    if not isinstance(subset, gpd.GeoDataFrame):
        subset = gpd.GeoDataFrame(
            subset,
            geometry=gpd.points_from_xy(subset.LONGITUDE, subset.LATITUDE),
            crs="EPSG:4326",
        )

    # Drop invalid or empty geometries to avoid projection/plotting issues
    subset = subset[subset.geometry.notna() & (~subset.geometry.is_empty) & subset.geometry.is_valid].copy()
    if subset.empty:
        print(f"No valid geometries found for {', '.join(target_biomes)}.")
        return

    # Set up Cartopy map with PlateCarree projection and basic features
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.add_feature(cfeature.LAND, facecolor="lightgrey")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    # Color mapping by biome class for the scatter points
    colors = {"Savannas": "tab:orange", "Woody savannas": "tab:green"}
    for biome_name, group in subset.groupby("modis_class_static"):
        # Plot points for each biome group
        ax.scatter(
            group["LONGITUDE"],
            group["LATITUDE"],
            color=colors.get(biome_name, "red"),
            label=biome_name,
            s=20,
            alpha=0.6,
            transform=ccrs.PlateCarree(),
        )

    # extent with padding; fallback to CONUS/Alaska view if bounds broken
    try:
        # Try to compute bounds of the fire points and pad a bit for nicer view
        minx, miny, maxx, maxy = subset.total_bounds
        if np.all(np.isfinite([minx, miny, maxx, maxy])) and (minx < maxx and miny < maxy):
            pad_x = max((maxx - minx) * 0.05, 5)
            pad_y = max((maxy - miny) * 0.05, 5)
            ax.set_extent([minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y], crs=ccrs.PlateCarree())
        else:
            # If bounds are degenerate, fall through to global/Alaska default
            raise ValueError("Invalid bounds")
    except Exception:
        # Default view covering CONUS and Alaska
        ax.set_extent([-170, -50, 15, 75], crs=ccrs.PlateCarree())

    # Build title depending on whether we combined both savanna classes or not
    title_str = "Savanna & Woody Savanna Fires (MTBS)" if biome == "both" else f"{biome} Fires (MTBS)"
    ax.set_title(title_str, fontsize=14, pad=10)

    # Add a legend if there are any labeled handles
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="Biome", loc="upper right", frameon=True)

    # Overlay gridlines with labels on left/bottom axes only
    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.3)
    gl.top_labels = gl.right_labels = False

    plt.tight_layout()
    plt.show()


def plot_fire_counts_faceted(
    mtbs_classified: pd.DataFrame | gpd.GeoDataFrame,
    year_col: str = "year",
    static_col: str = "modis_class_static",
    ncols: int = 3,
    panel_width: float = 6,
    panel_height: float = 4,
) -> None:
    """Faceted barplots of fire counts per year by MODIS class + overall.

    Args:
        mtbs_classified: DataFrame/GeoDataFrame with `[year_col, static_col]`.
        year_col: Column containing the year.
        static_col: Column with static MODIS class labels.
        ncols: Number of columns in the facet grid.
        panel_width: Width of each panel (inches).
        panel_height: Height of each panel (inches).
    """
    # Aggregate total yearly fire counts across all MODIS classes
    yearly_counts = mtbs_classified.groupby(year_col).size()

    # Aggregate yearly counts by MODIS class and pivot to wide format
    category_counts = mtbs_classified.groupby([year_col, static_col]).size().unstack(fill_value=0)

    # Determine categories and layout of facet grid
    categories = category_counts.columns.tolist()
    n_categories = len(categories)
    n_plots = n_categories + 1  # one panel per category + overall panel
    nrows = int(np.ceil(n_plots / ncols))

    # Set up figure with shared axes across facets
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()

    # per-category panels
    # For each MODIS class, draw a bar plot of yearly fire counts
    for i, cat in enumerate(categories):
        ax = axes[i]
        category_counts[cat].plot(kind="bar", ax=ax, color="tab:blue", alpha=0.7)
        ax.set_title(cat)
        ax.set_xlabel("")
        ax.set_ylabel("Fires")

    # overall panel
    # In the next panel, plot the total yearly fire count across all classes
    ax = axes[n_categories]
    yearly_counts.plot(kind="bar", ax=ax, color="black", alpha=0.8)
    ax.set_title("Overall Fires")
    ax.set_xlabel("")
    ax.set_ylabel("Fires")

    # remove any extra subplot axes
    # If grid has more panels than needed, remove the unused axes
    for j in range(n_categories + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a common title and tidy layout
    fig.suptitle("MTBS Fires by Year and Static MODIS Classification", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()


def fire_threshold_analysis(
    mtbs_classified: pd.DataFrame | gpd.GeoDataFrame,
    year_col: str = "year",
    area_col: str = "area_km2",
    thresholds: list[float] = [0, 4, 10, 20, 50],
) -> None:
    """Plot yearly fire counts at multiple minimum-size thresholds.

    Args:
        mtbs_classified: Input with `[year_col, area_col]`.
        year_col: Name of year column.
        area_col: Name of size column in km².
        thresholds: List of minimum size thresholds to consider.
    """
    # For each minimum threshold, compute yearly fire counts of fires above that size
    yearly_counts_by_thresh: dict[float, pd.Series] = {}
    for t in thresholds:
        filtered = mtbs_classified[mtbs_classified[area_col] >= t]
        yearly_counts = filtered.groupby(year_col).size()
        yearly_counts_by_thresh[t] = yearly_counts

    # Combine threshold-specific time series into a single DataFrame
    counts_df = pd.DataFrame(yearly_counts_by_thresh).fillna(0).astype(int)

    # Plot multiseries line chart comparing thresholds
    fig, ax = plt.subplots(figsize=(12, 6))
    for t in thresholds:
        ax.plot(counts_df.index, counts_df[t], marker="o", label=f"≥ {t} km²")

    # Add informative labels and legend
    ax.set_title("Yearly Fire Counts by Threshold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Fires")
    ax.legend(title="Min fire size")
    plt.tight_layout()
    plt.show()


def build_static_modis_tiles(modis_by_year: dict[int, list[str]], out_dir: str = "../data/static_modis_tiles") -> dict[str, str]:
    """Build static MODIS (per tile) rasters by pixelwise mode across years.

    Args:
        modis_by_year: Mapping from year → list of MODIS raster file paths.
        out_dir: Output directory for static rasters.

    Returns:
        Dict mapping tile_id → written GeoTIFF path.
    """
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Regex to extract MODIS tile IDs (e.g., h10v04) from filenames
    tile_pattern = re.compile(r"h\d{2}v\d{2}")

    # collect rasters by MODIS tile id
    # Build a mapping from tile_id to all rasters (across years) covering that tile
    tile_files: dict[str, list[str]] = {}
    for year, files in modis_by_year.items():
        for f in files:
            m = tile_pattern.search(os.path.basename(f))
            if not m:
                continue
            tile_id = m.group()
            tile_files.setdefault(tile_id, []).append(f)

    # Dict to store output GeoTIFF paths per tile
    out_paths: dict[str, str] = {}

    # For each tile, compute pixelwise mode across its raster stack to form a static map
    for tile_id, files in tile_files.items():
        arrays: list[np.ndarray] = []
        ref_profile: dict | None = None
        nodata = 255

        # Read raster bands for this tile and store them in a stack
        for f in files:
            with rasterio.open(f) as src:
                # Set the reference profile from the first raster
                if ref_profile is None:
                    ref_profile = src.profile.copy()
                    nodata = ref_profile.get("nodata", 255)
                arr = src.read(1).astype(float)
                arrays.append(arr)

        # stack and mask nodata + class 17 as NaN (per your convention)
        # Stack arrays along a time axis and mask out nodata and a specific class (17)
        stack = np.stack(arrays, axis=0)
        stack = np.where((stack == nodata) | (stack == 17), np.nan, stack)

        # pixelwise mode across time
        # Use SciPy's mode to get the most frequent land cover class at each pixel
        mode_map, _ = stats.mode(stack, axis=0, nan_policy="omit")
        static_map = np.squeeze(mode_map).astype(np.uint8)

        # Construct output filename for the static tile map
        out_path = os.path.join(out_dir, f"static_modis_mode_{tile_id}.tif")

        # Write the static map as a GeoTIFF with the reference metadata
        assert ref_profile is not None
        ref_profile.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=nodata)
        with rasterio.open(out_path, "w", **ref_profile) as dst:
            dst.write(static_map, 1)
        out_paths[tile_id] = out_path

    # Return mapping from tile_id to static GeoTIFF path
    return out_paths


def classify_with_static(
    fires_gdf: gpd.GeoDataFrame,
    static_tile_dict: dict[str, str],
    modis_to_gfa: dict[int, str],
) -> gpd.GeoDataFrame:
    """Classify fire polygons using static (per-tile) MODIS rasters.

    Args:
        fires_gdf: Fire perimeters (geometry must be present).
        static_tile_dict: Mapping of tile_id → path to static raster (from `build_static_modis_tiles`).
        modis_to_gfa: Map from MODIS class code → GFA category string.

    Returns:
        Copy of `fires_gdf` with a new column 'modis_class_static'.
    """
    # Accumulate (fire_index, assigned_class) tuples here
    results: list[tuple[int, str]] = []

    # Loop over each static MODIS tile raster
    for tile_id, raster_path in static_tile_dict.items():
        with rasterio.open(raster_path) as src:
            # Reproject fire perimeters into the same CRS as the raster
            fires_proj = fires_gdf.to_crs(src.crs)

            # For each fire polygon in this CRS, sample the underlying raster
            for idx, fire in fires_proj.iterrows():
                try:
                    # Clip raster to fire geometry to extract pixel values within the fire
                    out_image, _ = rasterio.mask.mask(src, [fire.geometry], crop=True)
                except ValueError:
                    # Skip fires that fail clipping (e.g., geometry outside raster)
                    continue

                # Filter out nodata pixels and sentinel value 255
                data = out_image[0]
                data = data[(data != src.nodata) & (data != 255)]
                if data.size > 0:
                    # Map MODIS numeric codes to GFA categories
                    mapped = [modis_to_gfa.get(int(val), "Other") for val in data]

                    # Take the most common class within the polygon as its assignment
                    majority = Counter(mapped).most_common(1)[0][0]
                    results.append((idx, majority))

    # Create output GeoDataFrame and initialize classification column
    fires_out = fires_gdf.copy()
    fires_out["modis_class_static"] = "Unknown"

    # Assign classifications where available
    for idx, cls in results:
        fires_out.at[idx, "modis_class_static"] = cls
    return fires_out


def classify_with_static_majority_threshold(
    fires_gdf: gpd.GeoDataFrame,
    static_tile_dict: dict[str, str],
    modis_to_gfa: dict[int, str],
    min_majority_frac: float = 0.8,
) -> gpd.GeoDataFrame:
    """Classify fire polygons using static (per-tile) MODIS rasters,
    but only if one land cover class occupies ≥ `min_majority_frac` of the area.

    Otherwise, classify as "Unknown".

    Args:
        fires_gdf: Fire perimeters (geometry must be present).
        static_tile_dict: Mapping of tile_id → path to static raster (from `build_static_modis_tiles`).
        modis_to_gfa: Map from MODIS class code → GFA category string.
        min_majority_frac: Minimum fraction of pixels needed for confident classification (default 0.8).

    Returns:
        Copy of `fires_gdf` with new column 'modis_class_static_majority'.
    """
    # Base directory containing static MODIS tiles; used to rebuild full paths defensively
    base_dir = "../data/static_modis_tiles"

    # Collect (fire_index, assigned_class) for all fires with a strong majority
    results: list[tuple[int, str]] = []

    # Loop through each static tile and attempt to classify overlapping fires
    for tile_id, raster_path in static_tile_dict.items():
        # Construct absolute path inside base_dir
        raster_path_full = os.path.join(base_dir, os.path.basename(raster_path))

        try:
            with rasterio.open(raster_path_full) as src:
                # Reproject fire polygons into raster CRS for sampling
                fires_proj = fires_gdf.to_crs(src.crs)

                for idx, fire in fires_proj.iterrows():
                    try:
                        # Mask raster by fire polygon to get land cover pixels inside
                        out_image, _ = rasterio.mask.mask(src, [fire.geometry], crop=True)
                    except ValueError:
                        # Skip if polygon is outside this raster or clipping fails
                        continue

                    # Filter out nodata and sentinel 255 values
                    data = out_image[0]
                    data = data[(data != src.nodata) & (data != 255)]
                    if data.size == 0:
                        continue

                    # Convert MODIS numeric codes to GFA categories
                    mapped = [modis_to_gfa.get(int(val), "Other") for val in data]

                    # Count frequencies of each GFA category within the fire polygon
                    counts = Counter(mapped)
                    total = sum(counts.values())
                    if total == 0:
                        continue

                    # Get most common class and its fractional coverage
                    top_class, top_count = counts.most_common(1)[0]
                    top_frac = top_count / total

                    # Apply majority threshold
                    if top_frac >= min_majority_frac:
                        results.append((idx, top_class))
                    else:
                        # Explicitly record insufficient-majority fires as "Unknown"
                        results.append((idx, "Unknown"))

        except rasterio.errors.RasterioIOError:
            # If the tile raster is missing, log and move on to the next tile
            print(f"Skipping missing tile {tile_id}: {raster_path_full}")
            continue

    # Apply classifications
    # Start with all fires classified as "Unknown"
    fires_out = fires_gdf.copy()
    fires_out["modis_class_static_majority"] = "Unknown"

    # Overwrite with majority-based classes where they were computed
    for idx, cls in results:
        fires_out.at[idx, "modis_class_static_majority"] = cls

    return fires_out


def classify_with_modis(
    fires_gdf: gpd.GeoDataFrame,
    year_col: str,
    modis_by_year: dict[int, list[str]],
    modis_to_gfa: dict[int, str],
) -> gpd.GeoDataFrame:
    """Classify fires with year-specific MODIS rasters.

    Args:
        fires_gdf: Fire polygons with a year column.
        year_col: Name of the year column in `fires_gdf`.
        modis_by_year: Mapping from year → list of MODIS rasters for that year.
        modis_to_gfa: Map from MODIS class code → GFA category.

    Returns:
        Copy of `fires_gdf` with new column 'modis_class_timevary'.
    """
    # Collect (fire_index, class) for time-varying MODIS-based classification
    results: list[tuple[int, str]] = []

    # Group fires by year so that each group can be matched to that year's MODIS rasters
    for year, fires in fires_gdf.groupby(fires_gdf[year_col]):
        # Skip years for which we have no MODIS rasters
        if year not in modis_by_year:
            continue

        # Loop through each MODIS file for this year
        for modis_file in modis_by_year[year]:
            with rasterio.open(modis_file) as src:
                # Reproject fire polygons into the raster CRS for sampling
                fires_proj = fires.to_crs(src.crs)
                for idx, fire in fires_proj.iterrows():
                    try:
                        # Clip raster to fire geometry
                        out_image, _ = rasterio.mask.mask(src, [fire.geometry], crop=True)
                    except ValueError:
                        # Skip fires outside this particular MODIS tile
                        continue

                    # Filter out nodata and sentinel value 255
                    data = out_image[0]
                    data = data[(data != src.nodata) & (data != 255)]
                    if data.size > 0:
                        # Map MODIS numeric codes to GFA categories
                        mapped = [modis_to_gfa.get(int(val), "Other") for val in data]

                        # Use majority class within the polygon as time-varying classification
                        majority = Counter(mapped).most_common(1)[0][0]
                        results.append((idx, majority))

    # Initialize output GeoDataFrame and classification column
    fires_out = fires_gdf.copy()
    fires_out["modis_class_timevary"] = "Unknown"

    # Assign year-specific MODIS classifications where available
    for idx, cls in results:
        fires_out.at[idx, "modis_class_timevary"] = cls
    return fires_out


def load_shapefile(shp_path: str, projection: str, area_col: str | None = None) -> gpd.GeoDataFrame:
    """Load a shapefile, set input CRS, reproject to equal-area (EPSG:6933), compute area if requested.

    Args:
        shp_path: Path to shapefile or GeoPackage layer.
        projection: CRS string/EPSG to assign before reprojection.
        area_col: If provided and missing, compute polygon area (km²) into this column.

    Returns:
        GeoDataFrame in EPSG:6933; includes area column if requested.
    """
    # Read the vector file (shapefile/GeoPackage layer) into a GeoDataFrame
    gdf = gpd.read_file(shp_path)

    # Assign the original CRS to the GeoDataFrame
    gdf = gdf.set_crs(projection)

    # Reproject to an equal-area projection (EPSG:6933) for area calculations
    gdf = gdf.to_crs("EPSG:6933")

    # Optionally compute area if an area column is requested and not already present
    if area_col is not None and area_col not in gdf.columns:
        gdf[area_col] = gdf.geometry.area / 1e6  # m² → km²
    return gdf


def summarize_ecoregion_fits(
    df_both: pd.DataFrame,
    df_p1: pd.DataFrame,
    df_p2: pd.DataFrame,
    overall_results: dict,
) -> pd.DataFrame:
    """Summarize main results per ecoregion (primary distribution per biome).

    Adds Δloglik (from `overall_results`) for quick comparison.

    Args:
        df_both: Summary with joint (p1 & p2) mode results.
        df_p1: Summary for p1-only mode.
        df_p2: Summary for p2-only mode.
        overall_results: Static fit results to look up Δloglik.

    Returns:
        DataFrame summarizing key parameters, trends, and Δloglik.
    """
    # Map from biome to the primary distribution considered best/representative
    dist_map = {
        "Deciduous Broadleaf forest": "generalized_pareto",
        "Evergreen Broadleaf forest": "generalized_pareto",
        "Mixed forest": "generalized_pareto",
        "Savannas": "truncated_power_law",
        "Woody savannas": "truncated_power_law",
        "Evergreen Needleleaf forest": "lognormal",
        "Grasslands": "lognormal",
        "Open shrublands": "lognormal",
    }

    # Store one summary record per biome in this list
    records: list[dict] = []

    # Iterate over the set of primary biomes and distributions
    for biome, best_dist in dist_map.items():
        # Extract rows for this biome/distribution combination from each summary table
        row_both = df_both[(df_both["biome"] == biome) & (df_both["distribution"] == best_dist)]
        row_p1 = df_p1[(df_p1["biome"] == biome) & (df_p1["distribution"] == best_dist)]
        row_p2 = df_p2[(df_p2["biome"] == biome) & (df_p2["distribution"] == best_dist)]

        # If no 'both' fit exists, log and move on to the next biome
        if row_both.empty:
            print(f"No match for {biome} ({best_dist}) in df_both.")
            continue

        # Grab the first (and presumably only) row for this biome/distribution
        r_b = row_both.iloc[0]
        n = int(r_b["n"])

        # Extract static p1/p2 and their SEs for formatted reporting
        p1, p1_se = r_b["p1"], r_b["p1_se"]
        p2, p2_se = r_b["p2"], r_b["p2_se"]

        # Look up Δloglik from the static fit results, if available
        delta_ll = np.nan
        if biome in overall_results:
            ll_df: pd.DataFrame = overall_results[biome]["likelihood_matrix"]
            if best_dist in ll_df.index:
                delta_ll = ll_df.loc[best_dist].iloc[0]

        # Initialize time trend summaries as NaN; they may or may not be filled below
        p1_trend = p1_trend_se = p2_trend = p2_trend_se = np.nan

        # Only record a p1 trend if both the joint and p1-only fits indicate a significant slope
        if not row_p1.empty:
            if (r_b["p1_slope_sig"] == 1) and (row_p1.iloc[0]["p1_slope_sig"] == 1):
                p1_trend = r_b["p1'"]
                p1_trend_se = r_b["p1'_se"]

        # Likewise for p2 trends, requiring consistency and significance across summaries
        if not row_p2.empty:
            if (r_b["p2_slope_sig"] == 1) and (row_p2.iloc[0]["p2_slope_sig"] == 1):
                p2_trend = r_b["p2'"]
                p2_trend_se = r_b["p2'_se"]

        # Compose a compact record summarizing parameters and time trends for this biome
        records.append(
            {
                "biome": biome,
                "distribution": best_dist,
                "n": n,
                "Δloglik": delta_ll,
                "p1 ± se": f"{p1:.3f} ± {p1_se:.3f}" if pd.notna(p1) else np.nan,
                "p2 ± se": f"{p2:.3f} ± {p2_se:.3f}" if pd.notna(p2) else np.nan,
                "Δp1 ± se": f"{p1_trend:.3f} ± {p1_trend_se:.3f}" if pd.notna(p1_trend) else "",
                "Δp2 ± se": f"{p2_trend:.3f} ± {p2_trend_se:.3f}" if pd.notna(p2_trend) else "",
            }
        )

    # Return the final summary table as a DataFrame
    return pd.DataFrame(records)


def summarize_other_fits(
    df_both: pd.DataFrame,
    df_p1: pd.DataFrame,
    df_p2: pd.DataFrame,
    overall_results: dict,
) -> pd.DataFrame:
    """Complementary summary table of *non-primary* fits, with Δloglik.

    Args:
        df_both: Joint-mode summary results.
        df_p1: p1-only mode summary.
        df_p2: p2-only mode summary.
        overall_results: Static fit results for Δloglik lookup.

    Returns:
        DataFrame with remaining biome×distribution combos not in the primary set.
    """
    # Define which biome/distribution combinations are considered "primary"
    primary_map = {
        "Deciduous Broadleaf forest": "generalized_pareto",
        "Evergreen Broadleaf forest": "generalized_pareto",
        "Mixed forest": "generalized_pareto",
        "Savannas": "truncated_power_law",
        "Woody savannas": "truncated_power_law",
        "Evergreen Needleleaf forest": "lognormal",
        "Grasslands": "lognormal",
        "Open shrublands": "lognormal",
    }

    # Records here will cover all fits not already captured as primary
    records: list[dict] = []

    # Iterate across rows in df_both and skip entries belonging to the primary set
    for _, r_b in df_both.iterrows():
        biome = r_b["biome"]
        dist = r_b["distribution"]

        # Skip primaries: they are summarized elsewhere
        if biome in primary_map and dist == primary_map[biome]:
            continue  # skip primaries

        # Retrieve corresponding rows from p1-only and p2-only summaries
        row_p1 = df_p1[(df_p1["biome"] == biome) & (df_p1["distribution"] == dist)]
        row_p2 = df_p2[(df_p2["biome"] == biome) & (df_p2["distribution"] == dist)]

        # Extract sample size and static parameter estimates
        n = int(r_b["n"])
        p1, p1_se = r_b["p1"], r_b["p1_se"]
        p2, p2_se = r_b["p2"], r_b["p2_se"]

        # Look up Δloglik in overall_results likelihood matrix if present
        delta_ll = np.nan
        if biome in overall_results:
            ll_df: pd.DataFrame = overall_results[biome]["likelihood_matrix"]
            if dist in ll_df.index:
                delta_ll = ll_df.loc[dist].iloc[0]

        # Initialize trend terms for p1/p2
        p1_trend = p1_trend_se = p2_trend = p2_trend_se = np.nan

        # Require consistency and significance for p1 trends
        if not row_p1.empty:
            if (r_b["p1_slope_sig"] == 1) and (row_p1.iloc[0]["p1_slope_sig"] == 1):
                p1_trend = r_b["p1'"]
                p1_trend_se = r_b["p1'_se"]

        # Similarly require consistency and significance for p2 trends
        if not row_p2.empty:
            if (r_b["p2_slope_sig"] == 1) and (row_p2.iloc[0]["p2_slope_sig"] == 1):
                p2_trend = r_b["p2'"]
                p2_trend_se = r_b["p2'_se"]

        # Append summary record for this non-primary biome×distribution combination
        records.append(
            {
                "biome": biome,
                "distribution": dist,
                "n": n,
                "Δloglik": delta_ll,
                "p1 ± se": f"{p1:.3f} ± {p1_se:.3f}" if pd.notna(p1) else np.nan,
                "p2 ± se": f"{p2:.3f} ± {p2_se:.3f}" if pd.notna(p2) else np.nan,
                "Δp1 ± se": f"{p1_trend:.3f} ± {p1_trend_se:.3f}" if pd.notna(p1_trend) else "",
                "Δp2 ± se": f"{p2_trend:.3f} ± {p2_trend_se:.3f}" if pd.notna(p2_trend) else "",
            }
        )

    # Return complementary summary of all non-primary fits
    return pd.DataFrame(records)


def plot_biome_facets(mtbs_classified: pd.DataFrame | gpd.GeoDataFrame) -> None:
    """Create a 3×3 faceted map of MTBS fire locations by individual biome.

    One panel per biome (up to 9 panels); here we use 8 MODIS classes:
      - Deciduous Broadleaf forest
      - Evergreen Broadleaf forest
      - Mixed forest
      - Savannas
      - Woody savannas
      - Evergreen Needleleaf forest
      - Grasslands
      - Open shrublands
    """
    # Silence runtime warnings from Cartopy/geopandas
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Order of individual biomes to plot (one per panel)
    biomes = [
        "Deciduous Broadleaf forest",
        "Evergreen Broadleaf forest",
        "Mixed forest",
        "Savannas",
        "Woody savannas",
        "Evergreen Needleleaf forest",
        "Grasslands",
        "Open shrublands",
    ]

    # Assign a distinct color to each biome
    biome_colors = {
        "Deciduous Broadleaf forest": "tab:orange",
        "Evergreen Broadleaf forest": "tab:green",
        "Mixed forest": "tab:brown",
        "Savannas": "tab:olive",
        "Woody savannas": "tab:gray",
        "Evergreen Needleleaf forest": "tab:blue",
        "Grasslands": "tab:red",
        "Open shrublands": "tab:purple",
    }

    # Copy input to avoid mutating the caller's object
    df = mtbs_classified.copy()

    # Drop rows with missing MODIS classification or non-finite coordinates
    df = df[df["modis_class_static"].notna()].copy()
    df = df[df["LATITUDE"].apply(np.isfinite) & df["LONGITUDE"].apply(np.isfinite)]

    # If not already a GeoDataFrame, construct geometry points from lat/lon
    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE),
            crs="EPSG:4326",
        )

    # Remove invalid or empty geometries
    df = df[df.geometry.notna() & (~df.geometry.is_empty) & df.geometry.is_valid].copy()
    if df.empty:
        print("No valid fire locations found for plotting.")
        return

    # 3×3 grid of panels
    # Create a faceted 3×3 grid of Cartopy subplots
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18, 12),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes_flat = axes.ravel()

    # Plot each biome in its own panel
    for ax, biome in zip(axes_flat, biomes):
        subset = df[df["modis_class_static"] == biome].copy()

        # Add geographic context (land, coastlines, borders)
        ax.add_feature(cfeature.LAND, facecolor="lightgrey")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)

        if subset.empty:
            # If no fires for this biome, display "no data" and default extent
            ax.set_title(f"{biome}\n(no data)", fontsize=11, pad=8)
            ax.set_extent([-170, -50, 15, 75], crs=ccrs.PlateCarree())
        else:
            # Scatter plot all fire locations for this biome
            ax.scatter(
                subset["LONGITUDE"],
                subset["LATITUDE"],
                color=biome_colors.get(biome, "black"),
                s=15,
                alpha=0.6,
                transform=ccrs.PlateCarree(),
                label=biome,
            )

            # Zoom to data with a little padding
            try:
                minx, miny, maxx, maxy = subset.total_bounds
                pad_x = max((maxx - minx) * 0.05, 5)
                pad_y = max((maxy - miny) * 0.05, 5)
                ax.set_extent(
                    [minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y],
                    crs=ccrs.PlateCarree(),
                )
            except Exception:
                # If bounds are invalid, use a default North America view
                ax.set_extent([-170, -50, 15, 75], crs=ccrs.PlateCarree())

            ax.set_title(biome, fontsize=11, pad=8)

        # Gridlines (labels only on left/bottom to avoid clutter)
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.3)
        gl.top_labels = False
        gl.right_labels = False

    # Turn off any unused axes (9th panel, since we only have 8 biomes)
    if len(biomes) < len(axes_flat):
        for ax in axes_flat[len(biomes) :]:
            ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def fires_time_series_by_ecoregion(
    df: pd.DataFrame | gpd.GeoDataFrame,
    year_col: str = "year",
    class_col: str = "modis_class_static",
    period_years: int = 5,
    valid_classes: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    include_total: bool = False,
    plot: bool = True,
    figsize: tuple[int, int] = (10, 5),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate fire counts per ecoregion over fixed-length periods (drops final partial bin).

    Args:
        df: Input with `year_col` and `class_col`.
        year_col: Column with integer years.
        class_col: Column with MODIS class labels.
        period_years: Width of each time bin.
        valid_classes: Optional list of classes to include (defaults to 8-core classes).
        start_year: Optional start bound for years.
        end_year: Optional end bound for years.
        include_total: Add a "Total" series across classes.
        plot: If True, draw a multi-series line chart.
        figsize: Figure size.

    Returns:
        (counts_long, counts_wide)
            counts_long: tidy with ["period_start", "period_end", "period_label", class_col, "count"]
            counts_wide: wide matrix (#fires per period × class)
    """
    # If no explicit class list is given, default to the eight core MODIS classes
    if valid_classes is None:
        valid_classes = [
            "Deciduous Broadleaf forest",
            "Evergreen Broadleaf forest",
            "Mixed forest",
            "Savannas",
            "Woody savannas",
            "Evergreen Needleleaf forest",
            "Open shrublands",
            "Grasslands",
        ]

    # Work on a copy to avoid mutating caller's DataFrame
    data = df.copy()

    # Coerce year column to numeric and drop rows with invalid years
    data = data[pd.to_numeric(data[year_col], errors="coerce").notna()]
    data[year_col] = data[year_col].astype(int)

    # Drop rows with missing class labels and restrict to valid_classes
    data = data[data[class_col].notna()]
    data = data[data[class_col].isin(valid_classes)]

    # If no rows survive filtering, raise an error for the caller
    if data.empty:
        raise ValueError("No valid rows after filtering by years and classes.")

    # Determine global min/max years given optional start/end bounds
    ymin = data[year_col].min() if start_year is None else int(start_year)
    ymax = data[year_col].max() if end_year is None else int(end_year)

    # Drop incomplete final bin to ensure all bins have equal length
    # Compute the last year that still forms a complete period of width 'period_years'
    max_complete_year = (ymax // period_years) * period_years + (period_years - 1)
    if max_complete_year > ymax:
        max_complete_year -= period_years
    data = data[(data[year_col] >= ymin) & (data[year_col] <= max_complete_year)]

    # Helper to compute the start year of the period in which a given year falls
    def period_start(y: int) -> int:
        offset = (y - ymin) // period_years
        return ymin + offset * period_years

    # Compute period start, end, and label for each row
    data["_period_start"] = data[year_col].apply(period_start)
    data["_period_end"] = data["_period_start"] + period_years - 1
    data["_period_label"] = data["_period_start"].astype(str) + "–" + data["_period_end"].astype(str)

    # Aggregate counts by period and class in long/tidy format
    counts = (
        data.groupby(["_period_start", "_period_end", "_period_label", class_col], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["_period_start", class_col])
    )

    # Pivot to wide format: periods as rows, classes as columns
    counts_wide = counts.pivot_table(
        index="_period_label", columns=class_col, values="count", aggfunc="sum", fill_value=0
    ).reindex(columns=sorted(valid_classes), fill_value=0)

    # Optionally add a "Total" column summarizing counts across all classes
    if include_total:
        counts_wide["Total"] = counts_wide.sum(axis=1)

    # Prepare a tidy long-form version with friendlier column names
    counts_long = counts.rename(
        columns={"_period_start": "period_start", "_period_end": "period_end", "_period_label": "period_label"}
    ).reset_index(drop=True)

    # If requested, produce a multi-series time series plot by class
    if plot:
        plt.figure(figsize=figsize)
        for cls in counts_wide.columns:
            plt.plot(counts_wide.index, counts_wide[cls], marker="o", label=cls)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(f"Number of fires per {period_years}-year period")
        plt.title(f"MTBS Fires per Land Cover Type — {periods_years}-year periods" if (periods_years := period_years) else
                  f"MTBS Fires per Land Cover Type")
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0, fontsize=9)
        plt.tight_layout()
        plt.show()

    # Return both long and wide forms of the period-aggregated time series
    return counts_long, counts_wide
def fit_poisson_tail_trend_by_biome_highres(
    df: pd.DataFrame | gpd.GeoDataFrame,
    year_col: str = "year",
    area_col: str = "area_km2",
    biome_col: str = "modis_class_static",
    min_years: int = 10,
) -> pd.DataFrame:
    """Fit Poisson GLM of fire frequency (≥ Amin) ~ year across a high-res Amin grid.

    For each biome in a reduced set, uses ~100 log-spaced thresholds between 4 km² and the
    biome's max observed size. Plots β₁ ±1 SE against Amin, marking p<0.05 points.

    Args:
        df: Input data with `[biome_col, year_col, area_col]`.
        year_col: Name of year column.
        area_col: Fire size column (km²).
        biome_col: Biome label column.
        min_years: Minimum unique years required to fit per biome.

    Returns:
        Long DataFrame of per-biome trend estimates over thresholds.
    """
    # Suppress numerical warnings from GLM fitting to keep output clean
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Restrict to a reduced set of biomes where we want to scan thresholds
    valid_classes = [
        "Deciduous Broadleaf forest",
        "Evergreen Broadleaf forest",
        "Mixed forest",
        "Evergreen Needleleaf forest",
        "Open shrublands",
        "Grasslands",
    ]
    df = df[df[biome_col].isin(valid_classes)].copy()

    # Basic cleaning: keep only relevant columns, drop missing values, and enforce positive areas
    df = df[[biome_col, year_col, area_col]].dropna()
    df = df[df[area_col] > 0]

    # Build a log-spaced grid of minimum area thresholds between 4 km² and the max observed size
    amax = df[area_col].max()
    thresholds = np.logspace(np.log10(4), np.log10(amax), num=100)

    # Determine biomes present and initialize containers for results and skipped biomes
    biomes = sorted(df[biome_col].unique())
    all_results: list[pd.DataFrame] = []
    skipped: list[tuple[str, int]] = []

    # Set up faceted figure: one panel per biome, with independent y-scale
    ncols = 3
    nrows = int(np.ceil(len(biomes) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(4.8 * ncols, 3.6 * nrows),
        sharex=False, sharey=False
    )
    axes = axes.flatten()

    # Loop over each biome and fit a Poisson GLM of yearly fire counts vs year at each threshold
    for i, biome in enumerate(biomes):
        # Subset to a single biome and check that enough years are present
        sub = df[df[biome_col] == biome]
        n_years = sub[year_col].nunique()
        if n_years < min_years:
            skipped.append((biome, n_years))
            continue

        trends: list[dict] = []
        # Scan across all Amin thresholds
        for Amin in thresholds:
            # Count fires per year above the current threshold
            yearly_counts = (
                sub.loc[sub[area_col] >= Amin]
                .groupby(year_col)
                .size()
                .reset_index(name="count")
            )
            # If there are no fires at this threshold, skip
            if yearly_counts["count"].sum() == 0:
                continue

            # Center year for numerical stability, then build design matrix
            yearly_counts["year_c"] = yearly_counts[year_col] - yearly_counts[year_col].mean()
            X = sm.add_constant(yearly_counts["year_c"])

            # Fit Poisson GLM to log-rate vs centered year
            model = sm.GLM(yearly_counts["count"], X, family=sm.families.Poisson())
            res = model.fit()

            # Store the slope, standard error, p-value, and number of years for this Amin
            trends.append(
                {
                    "biome": biome,
                    "Amin_km2": Amin,
                    "beta1": float(res.params["year_c"]),
                    "beta1_se": float(res.bse["year_c"]),
                    "pval": float(res.pvalues["year_c"]),
                    "n_years": int(len(yearly_counts)),
                }
            )

        # If no thresholds produced valid fits, mark this biome as skipped
        if not trends:
            skipped.append((biome, n_years))
            continue

        # Convert trends list into a DataFrame for this biome and store in the overall list
        df_trend = pd.DataFrame(trends)
        all_results.append(df_trend)

        # Plot β₁ and its SE vs Amin, highlighting significant (p<0.05) trends
        ax = axes[i]
        sig = df_trend["pval"] < 0.05

        ax.errorbar(
            df_trend["Amin_km2"],
            df_trend["beta1"],
            yerr=df_trend["beta1_se"],
            fmt="o-",
            capsize=3,
            color="C0",
            alpha=0.8,
        )
        ax.scatter(
            df_trend.loc[sig, "Amin_km2"],
            df_trend.loc[sig, "beta1"],
            color="red",
            s=25,
            label="p<0.05",
        )
        ax.axhline(0, color="k", lw=1, ls="--", alpha=0.6)
        ax.set_xscale("log")

        # Allow each facet to choose its own y-scale based on its data
        ax.relim()
        ax.autoscale(axis="y")

        # Constrain x-range to the range of Amin values used for this biome
        ax.set_xlim(4, df_trend["Amin_km2"].max())
        ax.set_title(biome, fontsize=11)
        ax.grid(True, alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel("Trend β₁ (log-rate per year)")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Minimum fire size (km²)")

    # Remove any unused subplot axes when there are fewer biomes than panels
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add overall figure title and layout adjustments
    fig.suptitle("Poisson Trend in Fire Frequency vs. Size Threshold by Biome", fontsize=14, y=1.02)
    fig.tight_layout()
    plt.show()

    # Concatenate per-biome results into a single long DataFrame
    results_all = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    # If some biomes were skipped due to insufficient years, report them
    if skipped:
        skipped_df = pd.DataFrame(skipped, columns=["biome", "n_unique_years"])
        print("Skipped due to too few unique years:")
        try:
            from IPython.display import display  # type: ignore
            display(skipped_df)
        except Exception:
            print(skipped_df.to_string(index=False))

    return results_all


# Note: The following function uses an external `wfpl` module for fitting and plotting.
# Ensure `import wildfire_powerlaw as wfpl` is available in your environment.
def plot_modis_category_ccdf_cdf_qq(
    gfa_df: pd.DataFrame | gpd.GeoDataFrame,
    modis_class: str,
    xmin: float = 4,
    which: tuple[str, ...] = ("power_law",),
    save_dir: str = "../data",
):
    """Three-panel plot for a given MODIS class:
        1) CCDF with selected fits (via `wfpl`)
        2) CDF with selected fits (via `wfpl`)
        3) Q–Q plot vs selected best-fit distribution (numeric inversion if no .ppf)

    Args:
        gfa_df: Fire area dataset with `modis_class_static` and `area_km2`.
        modis_class: Class name to subset.
        xmin: Tail cutoff.
        which: Tuple of `wfpl` distribution names to overlay.
        save_dir: Directory to write the PNG.

    Returns:
        Matplotlib Figure, or None if skipped.
    """
    # Restrict to the target MODIS class and check that data exist
    subset = gfa_df[gfa_df["modis_class_static"] == modis_class]
    if subset.empty:
        print(f"No fires found for MODIS class: {modis_class}")
        return None

    # Extract and filter fire size data to the tail region (≥ xmin)
    data = subset["area_km2"].dropna().to_numpy()
    data = data[data >= xmin]
    if len(data) == 0:
        print(f"No valid area data above xmin={xmin} for {modis_class}")
        return None

    # Sort data and build empirical CCDF for plotting limits
    data = np.sort(data)
    n = len(data)
    empirical_ccdf = 1 - np.arange(1, n + 1) / (n + 1)
    ymin = max(np.min(empirical_ccdf), 1e-5)

    # Set up three side-by-side panels: CCDF, CDF, and Q–Q
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax_ccdf, ax_cdf, ax_qq = axes

    # (1) CCDF panel: use wfpl helper to overlay selected parametric fits
    ax_ccdf = wfpl.plot_ccdf_with_selected_fits(data, xmin=xmin, which=which, ax=ax_ccdf)  # type: ignore[name-defined]
    ax_ccdf.set_ylim(ymin, 1)
    ax_ccdf.set_title(f"{modis_class}: Fire Size CCDF")
    ax_ccdf.set_xlabel("Fire size (km²)")
    ax_ccdf.set_ylabel("CCDF")

    # (2) CDF panel: empirical CDF plus fitted parametric CDFs in log-x space
    try:
        # Fit full distribution suite once and let wfpl plot the empirical CDF
        fit = wfpl.Fit(data, xmin=xmin, discrete=False)  # type: ignore[name-defined]
        fit.plot_cdf(color="k", linewidth=2, ax=ax_cdf, label="Empirical data")

        # Overlay each selected distribution's CDF using the wfpl accessors
        for name in which:
            try:
                dist = getattr(fit, name)
                dist.plot_cdf(ax=ax_cdf, linestyle="--", label=f"{name.replace('_',' ').title()} fit")
            except Exception as e:
                print(f"Skipped {name} in CDF: {e}")

        ax_cdf.set_xscale("log")
        ax_cdf.set_ylim(0, 1)
        ax_cdf.set_xlabel("Fire size (km²)")
        ax_cdf.set_ylabel("CDF")
        ax_cdf.set_title(f"{modis_class}: Fire Size CDF")
        ax_cdf.legend(fontsize=8)
    except Exception as e:
        print(f"Could not plot CDFs: {e}")

    # (3) Q–Q panel: empirical quantiles vs theoretical quantiles from a chosen fit
    try:
        dist_name = which[0]
        fit = wfpl.Fit(data, xmin=xmin, discrete=False)  # type: ignore[name-defined]
        model = getattr(fit, dist_name)

        # Compute empirical quantiles and associated plotting positions
        empirical_q = np.sort(data)
        probs = (np.arange(1, len(empirical_q) + 1) - 0.5) / len(empirical_q)

        # If the model exposes a ppf, use it directly; otherwise invert its CDF numerically
        if hasattr(model, "ppf") and callable(model.ppf):
            theoretical_q = model.ppf(probs)
        else:
            xgrid = np.logspace(np.log10(data.min()), np.log10(data.max()), 4000)
            try:
                cdf_vals = model.cdf(xgrid)
                if np.ndim(cdf_vals) > 1:
                    cdf_vals = np.asarray(cdf_vals).squeeze()
            except Exception:
                cdf_vals = np.array([float(model.cdf(float(x))) for x in xgrid])

            # Enforce well-behaved CDF and invert to get quantiles
            cdf_vals = np.clip(cdf_vals, 1e-12, 1 - 1e-12)
            cdf_vals = np.maximum.accumulate(cdf_vals)
            theoretical_q = np.interp(probs, cdf_vals, xgrid)

        # Scatter plot of theoretical vs empirical quantiles plus 1:1 reference line
        ax_qq.scatter(theoretical_q, empirical_q, s=15, alpha=0.6, color="tab:blue", label=f"{dist_name.title()} fit")
        ax_qq.plot(
            [empirical_q.min(), empirical_q.max()],
            [empirical_q.min(), empirical_q.max()],
            "r--",
            lw=1,
            label="1:1 line",
        )
        ax_qq.set_xscale("log")
        ax_qq.set_yscale("log")
        ax_qq.set_xlabel("Theoretical Quantiles")
        ax_qq.set_ylabel("Empirical Quantiles")
        ax_qq.set_title(f"{modis_class}: Q–Q Plot")
        ax_qq.legend(fontsize=8)
    except Exception as e:
        print(f"Could not generate Q–Q plot: {e}")

    # Final formatting and writing figure to disk
    plt.tight_layout()
    save_path = f"{save_dir}/{modis_class.replace(' ', '_').lower()}_cdf_ccdf_qq.png"
    plt.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
    print(f"Figure saved to {save_path}")
    plt.show()
    return fig


def plot_distribution_evolution_ccdf_pdf(
    df_both: pd.DataFrame,
    mtbs_classified: pd.DataFrame | gpd.GeoDataFrame,
    df_p1: pd.DataFrame | None = None,
    df_p2: pd.DataFrame | None = None,
    xmin: float = 12,
    biome_col: str = "modis_class_static",
    year_col: str = "year",
    trend_unit_years: int = 10,
    x_max_log10: int = 3,
    ccdf_ymin: float = 1e-5,
) -> None:
    """For each (biome, distribution) row in `df_both`, draw CCDF (log–log, normalized at xmin)
    and corresponding PDF (linear y, log x). Time evolution only included when the parameter
    slope is significant both in `df_both` and the corresponding single-parameter table.

    Args:
        df_both: Summary table with columns ["biome","distribution","p1","p2","p1'","p2'","p1_slope_sig","p2_slope_sig"].
        mtbs_classified: Input data for deriving per-biome year ranges.
        df_p1: Single-parameter (p1-only) significance table (optional).
        df_p2: Single-parameter (p2-only) significance table (optional).
        xmin: Tail cutoff for distributions.
        biome_col: Column with biome labels in `mtbs_classified`.
        year_col: Year column in `mtbs_classified`.
        trend_unit_years: Years corresponding to “+1” in slope multipliers.
        x_max_log10: Max exponent for logspace grid (10^x_max_log10).
        ccdf_ymin: Lower limit for CCDF plots (default 1e-5).

    Returns:
        None. Displays plots for each row in `df_both`.
    """
    # Build lookup tables for significance of p1 and p2 slopes in single-parameter fits
    sig_lookup_p1: dict[tuple[str, str], bool] = {}
    sig_lookup_p2: dict[tuple[str, str], bool] = {}
    if df_p1 is not None:
        for _, r in df_p1.iterrows():
            sig_lookup_p1[(r["biome"], r["distribution"])] = int(r.get("p1_slope_sig", 0)) == 1
    if df_p2 is not None:
        for _, r in df_p2.iterrows():
            sig_lookup_p2[(r["biome"], r["distribution"])] = int(r.get("p2_slope_sig", 0)) == 1

    # Precompute min/max year per biome for scaling and selecting a small set of time slices
    year_summary = (
        mtbs_classified.dropna(subset=[biome_col, year_col]).groupby(biome_col)[year_col].agg(["min", "max"]).to_dict("index")
    )

    # Common x-grid used for all distributions and time slices
    x = np.logspace(np.log10(xmin), x_max_log10, 500)
    cmap = cm.viridis

    # Iterate through each (biome, distribution) row in the combined table
    for _, row in df_both.iterrows():
        biome = row["biome"]
        dist = row["distribution"]

        # Skip biomes without year info or with degenerate time ranges
        if biome not in year_summary:
            print(f"Skipping {biome} — no year data.")
            continue
        y0, y1 = int(year_summary[biome]["min"]), int(year_summary[biome]["max"])
        if y0 >= y1:
            print(f"Skipping {biome} — only one year.")
            continue

        # Choose a small set of evenly spaced years across the observed range
        years = np.linspace(y0, y1, 5)
        year_center = np.mean(years)
        norm = plt.Normalize(vmin=y0, vmax=y1)

        # Determine whether p1 and p2 slopes are significant both in df_both and single-parameter tables
        key = (biome, dist)
        sig_p1_both = int(row.get("p1_slope_sig", 0)) == 1 and sig_lookup_p1.get(key, False)
        sig_p2_both = int(row.get("p2_slope_sig", 0)) == 1 and sig_lookup_p2.get(key, False)

        # Helper to compute time-varying p1 at a given year, rescaled by trend_unit_years
        def p1(year: float) -> float:
            base = float(row["p1"])
            slope = float(row.get("p1'", 0) or 0)
            if not sig_p1_both or not np.isfinite(slope):
                return base
            return base + slope * ((year - year_center) / trend_unit_years)

        # Helper to compute time-varying p2 (or return None if does not exist)
        def p2(year: float) -> float | None:
            if not np.isfinite(row.get("p2", np.nan)):
                return None
            base = float(row["p2"])
            slope = float(row.get("p2'", 0) or 0)
            if not sig_p2_both or not np.isfinite(slope):
                return base
            return base + slope * ((year - year_center) / trend_unit_years)

        # Set up two-panel figure: CCDF (log–log) and approximate PDF (from CCDF derivative)
        fig, (ax_ccdf, ax_pdf) = plt.subplots(1, 2, figsize=(11, 4))
        good = False

        # Loop over years, compute CCDF for each time slice, and approximate PDF via finite difference
        for yr in years:
            try:
                if dist == "exponential":
                    y_ccdf = ccdf_exponential(x, p1(yr), xmin)
                    y_pdf = np.gradient(1 - y_ccdf, x)
                elif dist == "power_law":
                    y_ccdf = ccdf_power_law(x, p1(yr), xmin)
                    y_pdf = np.gradient(1 - y_ccdf, x)
                elif dist == "truncated_power_law":
                    y_ccdf = ccdf_truncated_power_law(x, p1(yr), max(p2(yr), 1e-8), xmin)
                    y_pdf = np.gradient(1 - y_ccdf, x)
                elif dist == "stretched_exponential":
                    y_ccdf = ccdf_stretched_exponential(x, p1(yr), p2(yr), xmin)  # type: ignore[arg-type]
                    y_pdf = np.gradient(1 - y_ccdf, x)
                elif dist == "weibull":
                    y_ccdf = ccdf_weibull(x, p1(yr), p2(yr), xmin)  # type: ignore[arg-type]
                    y_pdf = np.gradient(1 - y_ccdf, x)
                elif dist == "lognormal":
                    y_ccdf = ccdf_lognormal(x, p1(yr), p2(yr), xmin)  # type: ignore[arg-type]
                    y_pdf = np.gradient(1 - y_ccdf, x)
                elif dist == "generalized_pareto":
                    y_ccdf = ccdf_genpareto(x, p1(yr), p2(yr), xmin)  # type: ignore[arg-type]
                    y_pdf = np.gradient(1 - y_ccdf, x)
                else:
                    continue

                # Plot CCDF and PDF for this year with a color encoding time
                ax_ccdf.plot(x, y_ccdf, color=cmap(norm(yr)), lw=1.5, label=f"{int(yr)}")
                ax_pdf.plot(x, y_pdf, color=cmap(norm(yr)), lw=1.5)
                good = True
            except Exception:
                continue

        # If none of the curves were valid, skip this biome–distribution combination
        if not good:
            plt.close()
            print(f"Skipped {biome} ({dist}) — invalid CCDF/PDF.")
            continue

        # Configure CCDF panel: log–log axes and common labels
        ax_ccdf.set_xscale("log")
        ax_ccdf.set_yscale("log")
        ax_ccdf.set_ylim(ccdf_ymin, 1.0)
        ax_ccdf.set_xlabel("Fire size (km²)")
        ax_ccdf.set_ylabel("CCDF")
        ax_ccdf.set_title(f"{biome}\n{dist.replace('_',' ').title()} (CCDF)")
        ax_ccdf.grid(True, alpha=0.3)

        # Configure approximate PDF panel: log-x, linear y
        ax_pdf.set_xscale("log")
        ax_pdf.set_xlabel("Fire size (km²)")
        ax_pdf.set_ylabel("PDF")
        ax_pdf.set_title(f"{biome}\n{dist.replace('_',' ').title()} (PDF)")
        ax_pdf.grid(True, alpha=0.3)

        # Add a colorbar indicating which curve corresponds to which year
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cbar.set_label("Year", rotation=270, labelpad=15)
        plt.show()


def summarize_all_fits(
    df_both: pd.DataFrame,
    df_p1: pd.DataFrame,
    df_p2: pd.DataFrame,
    overall_results: dict,
) -> pd.DataFrame:
    """Summarize all fits per ecoregion (primary + other), grouped by biome.

    The first row of each biome group shows its *primary* distribution (from `dist_map`),
    followed by other fits for the same biome. Includes Δloglik and trend terms.

    Args:
        df_both: Joint-mode summary results (with p1 & p2 fits).
        df_p1: p1-only mode summary.
        df_p2: p2-only mode summary.
        overall_results: Static fit results for Δloglik lookup.

    Returns:
        DataFrame combining all fits, grouped by biome with primaries first.
    """
    # Mapping from biome to primary distribution selected based on previous analysis
    dist_map = {
        "Deciduous Broadleaf forest": "generalized_pareto",
        "Evergreen Broadleaf forest": "generalized_pareto",
        "Mixed forest": "generalized_pareto",
        "Savannas": "truncated_power_law",
        "Woody savannas": "truncated_power_law",
        "Evergreen Needleleaf forest": "lognormal",
        "Grasslands": "lognormal",
        "Open shrublands": "lognormal",
    }

    records: list[dict] = []

    # Iterate over each biome and construct a biome-specific, ordered list of distributions
    for biome, primary_dist in dist_map.items():
        biome_rows = df_both[df_both["biome"] == biome]
        if biome_rows.empty:
            print(f"No entries for {biome}")
            continue

        # For each biome, ensure that its primary distribution appears first
        dists_in_biome = [primary_dist] + [
            d for d in biome_rows["distribution"].unique() if d != primary_dist
        ]

        # For each distribution (primary first, then others), merge static and dynamic info
        for dist in dists_in_biome:
            # Look up rows in the three mode-specific DataFrames
            row_both = biome_rows[biome_rows["distribution"] == dist]
            row_p1 = df_p1[(df_p1["biome"] == biome) & (df_p1["distribution"] == dist)]
            row_p2 = df_p2[(df_p2["biome"] == biome) & (df_p2["distribution"] == dist)]

            if row_both.empty:
                continue

            r_b = row_both.iloc[0]
            n = int(r_b["n"])

            # Extract mean parameter estimates and SEs from the both-parameters-vary fit
            p1, p1_se = r_b["p1"], r_b["p1_se"]
            p2, p2_se = r_b["p2"], r_b["p2_se"]

            # Look up Δloglik from static fits (overall_results)
            delta_ll = np.nan
            if biome in overall_results:
                ll_df: pd.DataFrame = overall_results[biome]["likelihood_matrix"]
                if dist in ll_df.index:
                    delta_ll = ll_df.loc[dist].iloc[0]

            # Initialize trend terms (slopes) for p1 and p2 as missing
            p1_trend = p1_trend_se = p2_trend = p2_trend_se = np.nan

            # If p1-slope significant both in df_both and df_p1, keep the combined-mode slope
            if not row_p1.empty:
                if (r_b.get("p1_slope_sig") == 1) and (row_p1.iloc[0].get("p1_slope_sig") == 1):
                    p1_trend = r_b["p1'"]
                    p1_trend_se = r_b["p1'_se"]

            # Analogous logic for p2 slope significance
            if not row_p2.empty:
                if (r_b.get("p2_slope_sig") == 1) and (row_p2.iloc[0].get("p2_slope_sig") == 1):
                    p2_trend = r_b["p2'"]
                    p2_trend_se = r_b["p2'_se"]

            # Append a fully formatted record for this biome–distribution pair
            records.append(
                {
                    "biome": biome,
                    "distribution": dist,
                    "n": n,
                    "Δloglik": delta_ll,
                    "p1 ± se": f"{p1:.3f} ± {p1_se:.3f}" if pd.notna(p1) else np.nan,
                    "p2 ± se": f"{p2:.3f} ± {p2_se:.3f}" if pd.notna(p2) else np.nan,
                    "Δp1 ± se": f"{p1_trend:.3f} ± {p1_trend_se:.3f}" if pd.notna(p1_trend) else "",
                    "Δp2 ± se": f"{p2_trend:.3f} ± {p2_trend_se:.3f}" if pd.notna(p2_trend) else "",
                }
            )

    # Convert list of records to a DataFrame and return
    return pd.DataFrame(records)


def fit_poisson_tail_trend_by_biome_highres(
    df: pd.DataFrame | gpd.GeoDataFrame,
    year_col: str = "year",
    area_col: str = "area_km2",
    biome_col: str = "modis_class_static",
    min_years: int = 10,
) -> pd.DataFrame:
    """Fit Poisson GLM of fire frequency (≥ Amin) ~ year across a high-res Amin grid.

    For each biome in a reduced set, uses ~100 log-spaced thresholds between 4 km² and the
    biome's max observed size. Plots β₁ ±1 SE against Amin, marking p<0.05 points.

    Args:
        df: Input data with `[biome_col, year_col, area_col]`.
        year_col: Name of year column.
        area_col: Fire size column (km²).
        biome_col: Biome label column.
        min_years: Minimum unique years required to fit per biome.

    Returns:
        Long DataFrame of per-biome trend estimates over thresholds.
    """
    # Suppress warnings produced by the GLM fitting process
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Restrict to the subset of biomes of interest
    valid_classes = [
        "Deciduous Broadleaf forest",
        "Evergreen Broadleaf forest",
        "Mixed forest",
        "Evergreen Needleleaf forest",
        "Open shrublands",
        "Grasslands",
    ]
    df = df[df[biome_col].isin(valid_classes)].copy()

    # Clean up the data: keep core columns, drop missing, enforce positive areas
    df = df[[biome_col, year_col, area_col]].dropna()
    df = df[df[area_col] > 0]

    # Create log-spaced thresholds between 4 km² and the maximum observed area
    amax = df[area_col].max()
    thresholds = np.logspace(np.log10(4), np.log10(amax), num=100)

    # Identify biomes and initialize containers for results and skipped cases
    biomes = sorted(df[biome_col].unique())
    all_results: list[pd.DataFrame] = []
    skipped: list[tuple[str, int]] = []

    # Prepare facet grid for plotting (one panel per biome)
    ncols = 3
    nrows = int(np.ceil(len(biomes) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(4.8 * ncols, 3.6 * nrows),
        sharex=False, sharey=False
    )
    axes = axes.flatten()

    # For each biome, fit trends of yearly counts vs year at each Amin threshold
    for i, biome in enumerate(biomes):
        # Subset data and check that there are enough distinct years to fit a trend
        sub = df[df[biome_col] == biome]
        n_years = sub[year_col].nunique()
        if n_years < min_years:
            skipped.append((biome, n_years))
            continue

        trends: list[dict] = []
        # Loop over the threshold grid
        for Amin in thresholds:
            # For this threshold, compute counts per year
            yearly_counts = (
                sub.loc[sub[area_col] >= Amin]
                .groupby(year_col)
                .size()
                .reset_index(name="count")
            )
            # If no events remain, no GLM can be fitted at this threshold
            if yearly_counts["count"].sum() == 0:
                continue

            # Center year for numerical stability, then build design matrix
            yearly_counts["year_c"] = yearly_counts[year_col] - yearly_counts[year_col].mean()
            X = sm.add_constant(yearly_counts["year_c"])

            # Poisson GLM: log-rate as linear function of centered year
            model = sm.GLM(yearly_counts["count"], X, family=sm.families.Poisson())
            res = model.fit()

            # Collect slope estimate (β1), SE, p-value, and number of years
            trends.append(
                {
                    "biome": biome,
                    "Amin_km2": Amin,
                    "beta1": float(res.params["year_c"]),
                    "beta1_se": float(res.bse["year_c"]),
                    "pval": float(res.pvalues["year_c"]),
                    "n_years": int(len(yearly_counts)),
                }
            )

        # If no thresholds yielded valid fits, mark biome as skipped
        if not trends:
            skipped.append((biome, n_years))
            continue

        # Convert trend list to DataFrame and add to overall results
        df_trend = pd.DataFrame(trends)
        all_results.append(df_trend)

        # Plot β1 vs Amin for this biome, with error bars and significant points marked
        ax = axes[i]
        sig = df_trend["pval"] < 0.05

        ax.errorbar(
            df_trend["Amin_km2"],
            df_trend["beta1"],
            yerr=df_trend["beta1_se"],
            fmt="o-",
            capsize=3,
            color="C0",
            alpha=0.8,
        )
        ax.scatter(
            df_trend.loc[sig, "Amin_km2"],
            df_trend.loc[sig, "beta1"],
            color="red",
            s=25,
            label="p<0.05",
        )
        ax.axhline(0, color="k", lw=1, ls="--", alpha=0.6)
        ax.set_xscale("log")

        # Let each facet autoscale its y-range
        ax.relim()
        ax.autoscale(axis="y")

        # Limit Amin axis to the range used for this biome
        ax.set_xlim(4, df_trend["Amin_km2"].max())
        ax.set_title(biome, fontsize=11)
        ax.grid(True, alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel("Trend β₁ (log-rate per year)")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Minimum fire size (km²)")

    # Remove unused axes when the number of biomes is not a multiple of ncols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add shared title and tighten layout
    fig.suptitle("Poisson Trend in Fire Frequency vs. Size Threshold by Biome", fontsize=14, y=1.02)
    fig.tight_layout()
    plt.show()

    # Concatenate per-biome trend DataFrames
    results_all = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    # Report biomes that could not be analyzed due to insufficient years
    if skipped:
        skipped_df = pd.DataFrame(skipped, columns=["biome", "n_unique_years"])
        print("Skipped due to too few unique years:")
        try:
            from IPython.display import display
            display(skipped_df)
        except Exception:
            print(skipped_df.to_string(index=False))

    return results_all


def analyze_time_varying_mle_refined(
    mtbs_classified: pd.DataFrame | gpd.GeoDataFrame,
    overall_results: dict,
    year_col: str = "year",
    xmin: float = 4,
    llhr_cutoff: float = 2.0,
    R_boot: int = 20,
    relerr_cutoff: float = 1.0,
    min_total: int = 400,
    verbose: bool = True,
    prior_weight: float = 1e-3,
) -> dict:
    """Time-varying MLE with Differential Evolution for refined biome groupings.

    This routine mirrors the logic of `analyze_time_varying_mle` but operates on a
    set of refined biome groupings that split broad categories by longitude.

    Refined biome groupings:
        1. Deciduous Broadleaf forest
        2. Evergreen Broadleaf forest
        3. Evergreen Broadleaf forest E of 100W
        4. Evergreen Broadleaf forest W of 100W
        5. Grasslands
        6. Open shrublands
        7. Open shrublands W of 130W
        8. Open shrublands E of 130W
        9. Grasslands + Open shrublands E of 130W

    The function:
      - Builds these refined biome labels from `modis_class_static` and longitude.
      - Restricts to fires above `xmin`.
      - For each refined biome, filters candidate distributions based on
        static `overall_results` (likelihood comparisons and reductions).
      - For each candidate, fits time-varying parameters via Differential
        Evolution, with a weak prior nudging parameters toward the static fit.
      - Uses bootstrap resampling to estimate standard errors of transformed
        parameters (e.g., α, λ and their slopes for truncated power law).

    Args:
        mtbs_classified: Fire records with columns including `modis_class_static`,
            LONGITUDE, `year_col`, and `area_km2`.
        overall_results: Static fit results per refined biome, with entries:
            overall_results[biome]['params'] (DataFrame)
            overall_results[biome]['likelihood_matrix'] (DataFrame)
        year_col: Name of the year column.
        xmin: Minimum size threshold (km²) for the tail analysis.
        llhr_cutoff: Δlog-likelihood threshold for filtering candidate dists.
        R_boot: Number of bootstrap replicates for SE estimation.
        relerr_cutoff: Retained for API compatibility (not used here).
        min_total: Minimum number of tail observations per refined biome.
        verbose: If True, print progress and filtering diagnostics.
        prior_weight: Strength of weak Gaussian prior on transformed parameters.

    Returns:
        Nested dict: results[biome_refined][dist_name][mode] with keys:
          - "coeffs": transformed coefficients (p1, p1', p2, p2') or subset
          - "ses": bootstrap SEs for the transformed coefficients
          - "loglik": best penalized log-likelihood (after negation in optimizer)
          - "n": number of tail observations used for this biome
    """
    # Local imports (kept here so the function can be transplanted if needed)
    import warnings
    import numpy as np
    import pandas as pd
    from scipy.optimize import differential_evolution
    from scipy.special import gamma, gammaincc

    # Silence numerical warnings from likelihood evaluations
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # ----------------------------------------------------------------------
    # STEP 1. Create refined biome groups (exactly matching your static fits)
    # ----------------------------------------------------------------------
    # Restrict to a subset of base MODIS biomes relevant for refined groupings
    valid_biomes = [
        "Deciduous Broadleaf forest",
        "Evergreen Broadleaf forest",
        "Open shrublands",
        "Grasslands",
    ]
    df = mtbs_classified.copy()
    df = df[df["modis_class_static"].isin(valid_biomes)].copy()

    # Ensure longitude is present for east/west splits
    if "LONGITUDE" not in df.columns:
        raise ValueError("Expected 'LONGITUDE' column for east/west splits.")

    # Map original biome and longitude to refined biome label
    def assign_refined_label(biome: str, lon: float) -> str:
        if biome == "Deciduous Broadleaf forest":
            return "Deciduous Broadleaf forest"
        elif biome == "Evergreen Broadleaf forest":
            return (
                "Evergreen Broadleaf forest W of 100W"
                if lon < -100
                else "Evergreen Broadleaf forest E of 100W"
            )
        elif biome == "Open shrublands":
            return (
                "Open shrublands W of 130W"
                if lon < -130
                else "Open shrublands E of 130W"
            )
        elif biome == "Grasslands":
            return "Grasslands"

    # Create initial refined labels based on MODIS class and longitude
    df["biome_refined"] = [
        assign_refined_label(b, lon)
        for b, lon in zip(df["modis_class_static"], df["LONGITUDE"])
    ]

    # Helper to merge a set of refined labels into a new combined group
    def subset_union(df, members, new_label):
        sub = df[df["biome_refined"].isin(members)].copy()
        sub["biome_refined"] = new_label
        return sub

    # Build the nine explicit refined groupings as separate DataFrames
    group_frames = [
        df[df["biome_refined"] == "Deciduous Broadleaf forest"].copy(),
        subset_union(
            df,
            ["Evergreen Broadleaf forest E of 100W", "Evergreen Broadleaf forest W of 100W"],
            "Evergreen Broadleaf forest",
        ),
        df[df["biome_refined"] == "Evergreen Broadleaf forest E of 100W"].copy(),
        df[df["biome_refined"] == "Evergreen Broadleaf forest W of 100W"].copy(),
        df[df["biome_refined"] == "Grasslands"].copy(),
        subset_union(
            df,
            ["Open shrublands E of 130W", "Open shrublands W of 130W"],
            "Open shrublands",
        ),
        df[df["biome_refined"] == "Open shrublands W of 130W"].copy(),
        df[df["biome_refined"] == "Open shrublands E of 130W"].copy(),
        subset_union(
            df,
            ["Grasslands", "Open shrublands E of 130W"],
            "Grasslands + Open shrublands E of 130W",
        ),
    ]

    # Combine all refined groups into a single table and center years (in decades)
    df_refined = pd.concat(group_frames, ignore_index=True)
    df_refined["year_c"] = (df_refined[year_col] - df_refined[year_col].mean()) / 10.0

    if verbose:
        print("\nRefined group counts for time-varying MLE:")
        print(df_refined["biome_refined"].value_counts())

    # ----------------------------------------------------------------------
    # STEP 2. Log-PDF helpers
    # ----------------------------------------------------------------------
    # The following functions implement log-PDFs for each candidate distribution.
    # They are written to handle vectorized x, and clip parameters to safe ranges.

    def logpdf_lognormal(x, mu, sigma, xmin=0):
        x = np.asarray(x)
        valid = x > xmin
        pdf = -np.inf * np.ones_like(x, dtype=float)
        sigma = np.clip(sigma, 1e-6, 50)
        pdf[valid] = (
            -np.log(x[valid]) - np.log(sigma)
            - 0.5 * ((np.log(x[valid]) - mu) / sigma) ** 2
            - np.log(np.sqrt(2 * np.pi))
        )
        return pdf

    def logpdf_powerlaw(x, alpha, xmin=1):
        x = np.asarray(x)
        valid = x >= xmin
        pdf = -np.inf * np.ones_like(x, dtype=float)
        alpha = np.clip(alpha, 0.0, 5)
        C = (alpha - 1) / xmin if alpha != 1 else 1 / xmin
        pdf[valid] = np.log(np.abs(C)) - alpha * np.log(x[valid] / xmin)
        return pdf

    def logpdf_trunc_powerlaw(x, alpha, lambd, xmin=1):
        x = np.asarray(x)
        valid = x >= xmin
        pdf = -np.inf * np.ones_like(x, dtype=float)
        alpha = np.clip(alpha, 0, 5)
        lambd = np.clip(lambd, 0, 5)
        try:
            Z = (
                (lambd ** (1 - alpha))
                * np.exp(lambd * xmin)
                * gammaincc(1 - alpha, lambd * xmin)
                * gamma(1 - alpha)
            )
            Z = np.clip(Z, 1e-300, 1e300)
            logZ = np.log(Z)
        except Exception:
            logZ = 0.0
        pdf[valid] = -alpha * np.log(x[valid]) - lambd * (x[valid] - xmin) - logZ
        return pdf

    def logpdf_genpareto(x, xi, sigma, xmin=0):
        x = np.asarray(x)
        y = x - xmin
        valid = (sigma > 0) & (1 + xi * y / sigma > 0)
        pdf = -np.inf * np.ones_like(x, dtype=float)
        xi = np.clip(xi, -1, 2)
        sigma = np.clip(sigma, 1e-6, 10)
        pdf[valid] = -np.log(sigma) - (1 / xi + 1) * np.log(1 + xi * y[valid] / sigma)
        return pdf

    def logpdf_weibull(x, k, lam, xmin=0):
        x = np.asarray(x)
        valid = x >= xmin
        pdf = -np.inf * np.ones_like(x, dtype=float)
        k = np.clip(k, 1e-6, 50)
        lam = np.clip(lam, 1e-6, 50)
        z = np.clip(x[valid] / lam, 1e-12, 1e6)
        pdf[valid] = np.log(k) - np.log(lam) + (k - 1) * np.log(z) - z ** k
        return pdf

    def logpdf_stretched_exp(x, lam, beta, xmin=0):
        x = np.asarray(x)
        valid = x >= xmin
        pdf = -np.inf * np.ones_like(x, dtype=float)
        lam = np.clip(lam, 1e-6, 50)
        beta = np.clip(beta, 0.1, 5)
        pdf[valid] = (
            np.log(beta)
            + np.log(lam)
            + (beta - 1) * (np.log(x[valid]) + np.log(lam))
            + (lam * xmin) ** beta
            - (lam * x[valid]) ** beta
        )
        return pdf

    # Map distribution names to their log-PDF implementations
    dist_logpdfs = {
        "lognormal": logpdf_lognormal,
        "power_law": logpdf_powerlaw,
        "truncated_power_law": logpdf_trunc_powerlaw,
        "generalized_pareto": logpdf_genpareto,
        "weibull": logpdf_weibull,
        "stretched_exponential": logpdf_stretched_exp,
    }

    # ----------------------------------------------------------------------
    # STEP 3. Fit time-varying parameters for each refined biome
    # ----------------------------------------------------------------------
    timevary_results = {}
    rng_global = np.random.default_rng(42)

    # Loop over each refined biome group and perform time-varying MLE
    for biome, subset in df_refined.groupby("biome_refined"):
        # Extract tail data for the current refined biome
        data = subset["area_km2"].values
        data = data[data >= xmin]
        if len(data) < min_total:
            if verbose:
                print(f"\n=== {biome} skipped: only {len(data)} fires ≥ {xmin} ===")
            continue

        # Use centered years (in decades) for time-varying parameterization
        years = subset.loc[subset["area_km2"] >= xmin, "year_c"].values
        if verbose:
            print(f"\n=== {biome} (n={len(data)} fires ≥ {xmin}) ===")

        # Retrieve static result for this refined biome from overall_results
        res = overall_results.get(biome, {})
        if not res:
            if verbose:
                print(f"No static results found for {biome}")
            continue

        params_df = res["params"]
        llhr = res["likelihood_matrix"]

        # Filter candidate distributions based on availability, reduction flags, and Δloglik
        candidates = []
        for dist, row in params_df.iterrows():
            if dist not in dist_logpdfs:
                continue
            if isinstance(row.get("reduces_to"), str):
                continue
            if dist in llhr.index and llhr.loc[dist].min() > llhr_cutoff:
                continue
            candidates.append(dist)

        if not candidates:
            if verbose:
                print(f"No viable candidates for {biome}")
            continue

        biome_res = {}

        # For each candidate distribution, fit time-varying parameters in multiple modes
        for dist_name in candidates:
            fit_modes = {}
            fit_configs = {"both": [True, True], "p1_only": [True, False], "p2_only": [False, True]}

            static_row = params_df.loc[dist_name]
            p1_static = float(static_row.get("p1", 1.0))
            p2_static = float(static_row.get("p2", 1.0)) if not pd.isna(static_row.get("p2", np.nan)) else 1.0

            # Iterate over the three modes (both, p1_only, p2_only)
            for mode, (fit_p1, fit_p2) in fit_configs.items():

                # Negative log-likelihood including weak prior toward static parameters
                def neg_loglik(params, data=data):
                    try:
                        # Unpack time-varying parameters depending on mode and distribution
                        if dist_name == "truncated_power_law":
                            if mode == "both":
                                a1, b1, a2, b2 = params
                            elif mode == "p1_only":
                                a1, b1, a2 = params
                                b2 = 0.0
                            elif mode == "p2_only":
                                a1, a2, b2 = params
                                b1 = 0.0
                        else:
                            if mode == "both":
                                a1, b1, a2, b2 = params
                            elif mode == "p1_only":
                                a1, b1, a2 = params
                                b2 = 0.0
                            elif mode == "p2_only":
                                a1, a2, b2 = params
                                b1 = 0.0

                        # Build time-varying parameters for each distribution
                        if dist_name == "truncated_power_law":
                            alpha = 1.0 + np.exp(a1 + b1 * years)
                            lambd = np.exp(a2 + b2 * years)
                            ll = logpdf_trunc_powerlaw(data, alpha, lambd, xmin)
                        elif dist_name == "lognormal":
                            mu = a1 + b1 * years
                            sigma = np.clip(a2 + b2 * years, 1e-6, 50)
                            ll = logpdf_lognormal(data, mu, sigma, xmin)
                        elif dist_name == "generalized_pareto":
                            xi = np.clip(a1 + b1 * years, -1, 2)
                            sigma = np.clip(a2 + b2 * years, 1e-6, 10)
                            ll = logpdf_genpareto(data, xi, sigma, xmin)
                        elif dist_name == "power_law":
                            alpha = np.clip(a1 + b1 * years, 0.0, 5)
                            ll = logpdf_powerlaw(data, alpha, xmin)
                        elif dist_name == "weibull":
                            k = np.clip(a1 + b1 * years, 1e-3, 10)
                            lam = np.clip(a2 + b2 * years, 1e-3, 50)
                            ll = logpdf_weibull(data, k, lam, xmin)
                        elif dist_name == "stretched_exponential":
                            lam = np.clip(a1 + b1 * years, 1e-4, 50)
                            beta = np.clip(a2 + b2 * years, 0.1, 5)
                            ll = logpdf_stretched_exp(data, lam, beta, xmin)
                        else:
                            return np.inf

                        # Restrict to finite log-likelihood values
                        valid_ll = ll[np.isfinite(ll)]

                        # Weak prior: encourage a1, a2 to stay near static log-parameter values
                        if dist_name == "truncated_power_law":
                            # For truncated power law, α is parameterized via log(α-1)
                            prior_center_a1 = np.log(max(p1_static - 1, 1e-3))
                        else:
                            prior_center_a1 = np.log(max(p1_static, 1e-6))
                        prior_center_a2 = np.log(max(p2_static, 1e-6))
                        prior_penalty = prior_weight * (
                            (a1 - prior_center_a1) ** 2 + (a2 - prior_center_a2) ** 2
                        )
                        return -np.sum(valid_ll) + prior_penalty
                    except Exception:
                        return np.inf

                # Bounds on transformed parameters (a1, b1, a2, b2) by distribution type
                if dist_name == "truncated_power_law":
                    bounds = [(-2, 2), (-1, 1), (-9, 0), (-1, 1)]
                elif dist_name == "generalized_pareto":
                    bounds = [(-1, 2), (-1, 1), (1e-6, 10), (-1, 1)]
                elif dist_name == "power_law":
                    bounds = [(0.0, 5), (-1, 1)]
                elif dist_name == "weibull":
                    bounds = [(1e-3, 10), (-1, 1), (1e-3, 50), (-1, 10)]
                elif dist_name == "stretched_exponential":
                    bounds = [(1e-4, 50), (-1, 1), (0.1, 5), (-1, 1)]
                else:
                    bounds = [(-5, 5), (-1, 1), (-5, 5), (-1, 1)]

                # Subselect bounds depending on which parameters are allowed to vary
                if mode == "both":
                    bnds = bounds
                elif mode == "p1_only":
                    bnds = [bounds[0], bounds[1], bounds[2]]
                elif mode == "p2_only":
                    bnds = [bounds[0], bounds[2], bounds[3]]

                # Global optimization via Differential Evolution for this mode
                opt = differential_evolution(
                    neg_loglik,
                    bounds=bnds,
                    strategy="best1bin",
                    maxiter=600,
                    popsize=15,
                    polish=True,
                    seed=42,
                    updating="deferred",
                    workers=1,
                    init="latinhypercube",
                )
                coeffs = opt.x
                ll_max = -opt.fun

                # Bootstrap: resample data, re-fit via DE, and store parameter vectors
                boot_params = []
                for _ in range(R_boot):
                    idx = rng_global.choice(len(data), size=len(data), replace=True)
                    boot_data = data[idx]
                    try:
                        opt_b = differential_evolution(
                            lambda p: neg_loglik(p, data=boot_data),
                            bounds=bnds,
                            maxiter=300,
                            polish=False,
                            seed=None,
                            updating="deferred",
                            workers=1,
                        )
                        boot_params.append(opt_b.x)
                    except Exception:
                        continue

                # Transform parameters for truncated power law to (p1, p1', p2, p2')
                if dist_name == "truncated_power_law":
                    if mode == "both":
                        a1, b1, a2, b2 = coeffs
                    elif mode == "p1_only":
                        a1, b1, a2 = coeffs
                        b2 = 0.0
                    elif mode == "p2_only":
                        a1, a2, b2 = coeffs
                        b1 = 0.0

                    # Forward transform: α=1+exp(a1), λ=exp(a2), slopes scaled accordingly
                    p1 = 1 + np.exp(a1)
                    p1p = b1 * np.exp(a1)
                    p2 = np.exp(a2)
                    p2p = b2 * np.exp(a2)

                    # Bootstrap SEs in the transformed parameter space
                    if boot_params:
                        boot_p1, boot_p1p, boot_p2, boot_p2p = [], [], [], []
                        for bp in boot_params:
                            if mode == "both":
                                ba1, bb1, ba2, bb2 = bp
                            elif mode == "p1_only":
                                ba1, bb1, ba2 = bp
                                bb2 = 0.0
                            elif mode == "p2_only":
                                ba1, ba2, bb2 = bp
                                bb1 = 0.0
                            boot_p1.append(1 + np.exp(ba1))
                            boot_p1p.append(bb1 * np.exp(ba1))
                            boot_p2.append(np.exp(ba2))
                            boot_p2p.append(bb2 * np.exp(ba2))
                        p1_se, p1p_se = np.std(boot_p1), np.std(boot_p1p)
                        p2_se, p2p_se = np.std(boot_p2), np.std(boot_p2p)
                    else:
                        p1_se = p1p_se = p2_se = p2p_se = 0.0

                    coeffs_trans = [p1, p1p, p2, p2p]
                    ses_trans = [p1_se, p1p_se, p2_se, p2p_se]
                else:
                    # For non-TPL distributions, retain coefficients and SEs in their native scale
                    ses = np.std(boot_params, axis=0) if boot_params else np.zeros_like(coeffs)
                    coeffs_trans = coeffs
                    ses_trans = ses

                # Store results for this mode and distribution
                fit_modes[mode] = {
                    "coeffs": coeffs_trans,
                    "ses": ses_trans,
                    "loglik": ll_max,
                    "n": len(data),
                }

            # If any mode was successfully fitted, register the distribution for this biome
            if fit_modes:
                biome_res[dist_name] = fit_modes

        # Store refined-biome results if non-empty
        if biome_res:
            timevary_results[biome] = biome_res

    return timevary_results


def summarize_all_fits_general(
    df_both: pd.DataFrame,
    df_p1: pd.DataFrame,
    df_p2: pd.DataFrame,
    overall_results: dict,
    ll_col: str = "loglik",
) -> pd.DataFrame:
    """Summarize all dynamic fits across biomes and distributions.

    Merges the results from 'both', 'p1_only', and 'p2_only' modes with
    static Δloglik info from `overall_results`. Each row corresponds to one
    biome × distribution pair. Only includes slope (Δp₁, Δp₂) estimates if
    they are statistically significant in both the 'both' and corresponding
    single-parameter fits.

    Args:
        df_both: DataFrame of time-varying fits where both parameters vary.
        df_p1: DataFrame where only p₁ varies with time.
        df_p2: DataFrame where only p₂ varies with time.
        overall_results: Dict of static fit results (each containing 'likelihood_matrix').
        ll_col: Column name of log-likelihood in df_both.

    Returns:
        pd.DataFrame summarizing all fits with formatted mean ± SE columns.
    """
    # Collect all unique biomes represented in any of the dynamic-fit tables
    records: list[dict] = []
    biomes = sorted(
        set(df_both["biome"].unique())
        | set(df_p1["biome"].unique())
        | set(df_p2["biome"].unique())
    )

    # Process each biome independently
    for biome in biomes:
        # Determine all candidate distributions observed for this biome across modes
        biome_dists = sorted(
            set(df_both.loc[df_both["biome"] == biome, "distribution"].unique())
            | set(df_p1.loc[df_p1["biome"] == biome, "distribution"].unique())
            | set(df_p2.loc[df_p2["biome"] == biome, "distribution"].unique())
        )
        if not biome_dists:
            print(f"No distributions found for {biome}")
            continue

        # Pull static Δloglik values (min across comparisons) to rank distributions
        delta_lls = {}
        if biome in overall_results:
            ll_df = overall_results[biome].get("likelihood_matrix")
            if isinstance(ll_df, pd.DataFrame):
                delta_lls = {dist: ll_df.loc[dist].min() for dist in ll_df.index}

        # Order distributions from best to worst based on static Δloglik (ascending)
        biome_dists = sorted(biome_dists, key=lambda d: delta_lls.get(d, np.inf))

        # Build a summarized record for each distribution
        for dist in biome_dists:
            # Extract matching rows from each mode's summary
            row_b = df_both[
                (df_both["biome"] == biome) & (df_both["distribution"] == dist)
            ]
            row_p1 = df_p1[
                (df_p1["biome"] == biome) & (df_p1["distribution"] == dist)
            ]
            row_p2 = df_p2[
                (df_p2["biome"] == biome) & (df_p2["distribution"] == dist)
            ]
            if row_b.empty:
                continue

            r_b = row_b.iloc[0]
            n = int(r_b.get("n", np.nan))
            p1, p1_se = r_b.get("p1", np.nan), r_b.get("p1_se", np.nan)
            p2, p2_se = r_b.get("p2", np.nan), r_b.get("p2_se", np.nan)

            # Static Δloglik (min across comparisons) to assess relative performance
            delta_ll = np.nan
            if biome in overall_results:
                ll_df = overall_results[biome].get("likelihood_matrix")
                if isinstance(ll_df, pd.DataFrame) and dist in ll_df.index:
                    delta_ll = ll_df.loc[dist].min()

            # Initialize dynamic trend terms as NaN
            p1_trend = p1_trend_se = p2_trend = p2_trend_se = np.nan

            # Only retain p1 trends if slopes are significant in both combined and p1-only fits
            if (
                not row_p1.empty
                and "p1'" in r_b
                and (r_b.get("p1_slope_sig") == 1)
                and (row_p1.iloc[0].get("p1_slope_sig") == 1)
            ):
                p1_trend = r_b.get("p1'", np.nan)
                p1_trend_se = r_b.get("p1'_se", np.nan)

            # Only retain p2 trends if slopes are significant in both combined and p2-only fits
            if (
                not row_p2.empty
                and "p2'" in r_b
                and (r_b.get("p2_slope_sig") == 1)
                and (row_p2.iloc[0].get("p2_slope_sig") == 1)
            ):
                p2_trend = r_b.get("p2'", np.nan)
                p2_trend_se = r_b.get("p2'_se", np.nan)

            # Append a formatted record summarizing this biome–distribution combination
            records.append(
                {
                    "biome": biome,
                    "distribution": dist,
                    "n": n,
                    "Δloglik": delta_ll,
                    "p1 ± se": f"{p1:.3f} ± {p1_se:.3f}"
                    if pd.notna(p1)
                    else np.nan,
                    "p2 ± se": f"{p2:.3f} ± {p2_se:.3f}"
                    if pd.notna(p2)
                    else np.nan,
                    "Δp1 ± se": f"{p1_trend:.3f} ± {p1_trend_se:.3f}"
                    if pd.notna(p1_trend)
                    else "",
                    "Δp2 ± se": f"{p2_trend:.3f} ± {p2_trend_se:.3f}"
                    if pd.notna(p2_trend)
                    else "",
                }
            )

    # Convert to DataFrame and order rows within each biome by Δloglik
    df_out = pd.DataFrame(records)
    if not df_out.empty:
        df_out.sort_values(["biome", "Δloglik"], inplace=True, na_position="last")

    return df_out


def fit_poisson_tail_trend_by_biome_refined(
    mtbs_classified: gpd.GeoDataFrame | pd.DataFrame,
    year_col: str = "year",
    area_col: str = "area_km2",
    lon_col: str = "LONGITUDE",
    biome_col: str = "modis_class_static",
    min_years: int = 10,
) -> pd.DataFrame:
    """Fit Poisson GLM of fire frequency (≥ Amin) ~ year for refined biome groupings.

    Uses nine explicit biome groupings (broadleaf, evergreen subsets, shrubland splits,
    and a grasslands+shrubland-east combo). Each facet has independent y-scaling.

    Args:
        mtbs_classified: Input GeoDataFrame with biome + geometry + year + size.
        year_col: Column for fire year.
        area_col: Column for fire size (km²).
        lon_col: Column for longitude (used for east/west splits).
        biome_col: Original MODIS biome classification.
        min_years: Minimum number of unique years required to fit.

    Returns:
        pd.DataFrame of per-biome trend estimates across size thresholds.
    """
    # Suppress runtime warnings from Poisson GLM and log-scale plotting
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # --- Step 1: Base filter ---
    # Keep only the subset of MODIS biomes used for refined grouping
    valid_biomes = [
        "Deciduous Broadleaf forest",
        "Evergreen Broadleaf forest",
        "Open shrublands",
        "Grasslands",
    ]
    df = mtbs_classified.copy()
    df = df[df[biome_col].isin(valid_biomes)].copy()

    # Ensure longitude is present for splitting east vs west
    if lon_col not in df.columns:
        raise ValueError(f"Expected longitude column '{lon_col}' not found.")

    # --- Step 2: Assign refined biome labels ---
    # Assign preliminary refined labels based on biome and longitude
    def assign_refined_label(biome: str, lon: float) -> str:
        if biome == "Deciduous Broadleaf forest":
            return "Deciduous Broadleaf forest"
        elif biome == "Evergreen Broadleaf forest":
            return (
                "Evergreen Broadleaf forest W of 100W"
                if lon < -100
                else "Evergreen Broadleaf forest E of 100W"
            )
        elif biome == "Open shrublands":
            return (
                "Open shrublands W of 130W"
                if lon < -130
                else "Open shrublands E of 130W"
            )
        elif biome == "Grasslands":
            return "Grasslands"
        else:
            return None

    df["biome_refined"] = [
        assign_refined_label(b, lon) for b, lon in zip(df[biome_col], df[lon_col])
    ]

    # --- Step 3: Explicit 9 group construction ---
    # Helper to merge a set of refined labels into a higher-level group
    def subset_union(df, members, new_label):
        sub = df[df["biome_refined"].isin(members)].copy()
        sub["biome_refined"] = new_label
        return sub

    # Build explicit refined biome groups, preserving consistency with static fits
    group_frames = [
        df[df["biome_refined"] == "Deciduous Broadleaf forest"].copy(),
        subset_union(
            df,
            ["Evergreen Broadleaf forest E of 100W", "Evergreen Broadleaf forest W of 100W"],
            "Evergreen Broadleaf forest",
        ),
        df[df["biome_refined"] == "Evergreen Broadleaf forest E of 100W"].copy(),
        df[df["biome_refined"] == "Evergreen Broadleaf forest W of 100W"].copy(),
        df[df["biome_refined"] == "Grasslands"].copy(),
        subset_union(
            df,
            ["Open shrublands E of 130W", "Open shrublands W of 130W"],
            "Open shrublands",
        ),
        df[df["biome_refined"] == "Open shrublands W of 130W"].copy(),
        df[df["biome_refined"] == "Open shrublands E of 130W"].copy(),
        subset_union(
            df,
            ["Grasslands", "Open shrublands E of 130W"],
            "Grasslands + Open shrublands E of 130W",
        ),
    ]
    df_refined = pd.concat(group_frames, ignore_index=True)

    # Print diagnostic counts for the refined groups
    print("Final refined grouping counts:")
    print(df_refined["biome_refined"].value_counts())
    print(f"\nTotal fires included: {len(df_refined)}")

    # --- Step 4: Poisson GLM fitting ---
    # Clean to core columns and enforce positive areas
    df_refined = df_refined[[year_col, area_col, "biome_refined"]].dropna()
    df_refined = df_refined[df_refined[area_col] > 0]

    # Define size thresholds (log-spaced) for tail frequency trends
    thresholds = np.logspace(np.log10(4), np.log10(df_refined[area_col].max()), num=100)
    biomes = sorted(df_refined["biome_refined"].unique())
    all_results = []
    skipped = []

    # Set up faceted figure with one panel per refined biome
    ncols = 3
    nrows = int(np.ceil(len(biomes) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.8 * ncols, 3.6 * nrows),
        sharex=False,
        sharey=False,
    )
    axes = axes.flatten()

    # For each refined biome, fit Poisson GLM to yearly counts above each threshold
    for i, biome in enumerate(biomes):
        sub = df_refined[df_refined["biome_refined"] == biome]
        n_years = sub[year_col].nunique()
        # Skip if too few unique years to estimate a trend
        if n_years < min_years:
            skipped.append((biome, n_years))
            continue

        trends = []
        # Scan the grid of Amin thresholds
        for Amin in thresholds:
            yearly_counts = (
                sub.loc[sub[area_col] >= Amin]
                .groupby(year_col)
                .size()
                .reset_index(name="count")
            )
            # If there are no fires above this threshold, skip
            if yearly_counts["count"].sum() == 0:
                continue

            # Center year for stability and build design matrix
            yearly_counts["year_c"] = yearly_counts[year_col] - yearly_counts[year_col].mean()
            X = sm.add_constant(yearly_counts["year_c"])
            model = sm.GLM(yearly_counts["count"], X, family=sm.families.Poisson())
            res = model.fit()

            # Capture slope, SE, p-value, and number of years for this threshold
            trends.append(
                {
                    "biome_refined": biome,
                    "Amin_km2": Amin,
                    "beta1": float(res.params["year_c"]),
                    "beta1_se": float(res.bse["year_c"]),
                    "pval": float(res.pvalues["year_c"]),
                    "n_years": int(len(yearly_counts)),
                }
            )

        # If no thresholds produced a valid fit for this biome, mark as skipped
        if not trends:
            skipped.append((biome, n_years))
            continue

        # Store trend results and generate a panel in the facet grid
        df_trend = pd.DataFrame(trends)
        all_results.append(df_trend)

        ax = axes[i]
        sig = df_trend["pval"] < 0.05

        ax.errorbar(
            df_trend["Amin_km2"],
            df_trend["beta1"],
            yerr=df_trend["beta1_se"],
            fmt="o-",
            capsize=3,
            color="C0",
            alpha=0.8,
        )
        ax.scatter(
            df_trend.loc[sig, "Amin_km2"],
            df_trend.loc[sig, "beta1"],
            color="red",
            s=25,
            label="p<0.05",
        )
        ax.axhline(0, color="k", lw=1, ls="--", alpha=0.6)
        ax.set_xscale("log")
        ax.relim()
        ax.autoscale(axis="y")

        ax.set_xlim(4, df_trend["Amin_km2"].max())
        ax.set_title(biome, fontsize=11)
        ax.grid(True, alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel("Trend β₁ (log-rate per year)")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Minimum fire size (km²)")

    # Remove unused panels if the number of biomes is less than grid capacity
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Final layout and title for the refined biome trend figure
    fig.suptitle("Poisson Trend in Fire Frequency vs. Size Threshold (Refined Biomes)", fontsize=14, y=1.02)
    fig.tight_layout()
    plt.show()

    # Combine per-biome DataFrames into a single table
    results_all = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    # Report any refined biomes that were skipped due to too few unique years
    if skipped:
        skipped_df = pd.DataFrame(skipped, columns=["biome_refined", "n_unique_years"])
        print("Skipped due to too few unique years:")
        try:
            from IPython.display import display
            display(skipped_df)
        except Exception:
            print(skipped_df.to_string(index=False))

    return results_all


def plot_refined_distribution_evolution_ccdf_pdf(
    df_both: pd.DataFrame,
    df_refined: pd.DataFrame | gpd.GeoDataFrame,
    df_p1: pd.DataFrame | None = None,
    df_p2: pd.DataFrame | None = None,
    xmin: float = 12,
    biome_col: str = "biome_refined",
    year_col: str = "year",
    trend_unit_years: int = 10,
    x_max_log10: int = 3,
    ccdf_ymin: float = 1e-5,
) -> None:
    """For each (refined biome, distribution) fit, plot CCDF + PDF evolution across time.

    Args:
        df_both: Combined-mode time-varying fit summary table.
        df_refined: Fire dataset with refined biome labels and year info.
        df_p1: p1-only mode table (optional, for significance).
        df_p2: p2-only mode table (optional).
        xmin: Tail cutoff for CCDF normalization.
        biome_col: Refined biome label column.
        year_col: Year column in dataset.
        trend_unit_years: Time unit scaling for slope effects.
        x_max_log10: Max exponent for logspace grid.
        ccdf_ymin: Minimum CCDF value for log scaling.

    Returns:
        None (shows plots).
    """
    # Build significance lookups for p1 and p2 slopes from single-parameter fits
    sig_lookup_p1, sig_lookup_p2 = {}, {}
    if df_p1 is not None:
        for _, r in df_p1.iterrows():
            sig_lookup_p1[(r["biome"], r["distribution"])] = int(r.get("p1_slope_sig", 0)) == 1
    if df_p2 is not None:
        for _, r in df_p2.iterrows():
            sig_lookup_p2[(r["biome"], r["distribution"])] = int(r.get("p2_slope_sig", 0)) == 1

    # Summarize year ranges per refined biome for selecting time slices and color scale
    year_summary = (
        df_refined.dropna(subset=[biome_col, year_col])
        .groupby(biome_col)[year_col]
        .agg(["min", "max"])
        .to_dict("index")
    )

    # Common x-grid for CCDF/PDF plotting
    x = np.logspace(np.log10(xmin), x_max_log10, 500)
    cmap = cm.viridis

    # Loop through dynamic fits (refined biome × distribution)
    for _, row in df_both.iterrows():
        biome = row["biome"]
        dist = row["distribution"]

        # Skip if refined biome does not have a year range or is degenerate
        if biome not in year_summary:
            print(f"Skipping {biome} — no year range.")
            continue
        y0, y1 = int(year_summary[biome]["min"]), int(year_summary[biome]["max"])
        if y0 >= y1:
            continue

        # Pick several evenly spaced years to visualize temporal evolution
        years = np.linspace(y0, y1, 5)
        year_center = np.mean(years)
        norm = plt.Normalize(vmin=y0, vmax=y1)
        key = (biome, dist)

        # Determine if p1 and p2 slopes are significant in both combined and single-parameter tables
        sig_p1_both = int(row.get("p1_slope_sig", 0)) == 1 and sig_lookup_p1.get(key, False)
        sig_p2_both = int(row.get("p2_slope_sig", 0)) == 1 and sig_lookup_p2.get(key, False)

        # Helper: time-varying p1 at a given year
        def p1(year: float) -> float:
            base = float(row["p1"])
            slope = float(row.get("p1'", 0) or 0)
            if not sig_p1_both or not np.isfinite(slope):
                return base
            return base + slope * ((year - year_center) / trend_unit_years)

        # Helper: time-varying p2 at a given year (if available)
        def p2(year: float) -> float | None:
            if not np.isfinite(row.get("p2", np.nan)):
                return None
            base = float(row["p2"])
            slope = float(row.get("p2'", 0) or 0)
            if not sig_p2_both or not np.isfinite(slope):
                return base
            return base + slope * ((year - year_center) / trend_unit_years)

        # Two-panel figure: CCDF and approximate PDF across multiple years
        fig, (ax_ccdf, ax_pdf) = plt.subplots(1, 2, figsize=(11, 4))
        good = False

        # Evaluate CCDF/PDF at each selected year and plot in color-coded fashion
        for yr in years:
            try:
                if dist == "exponential":
                    y_ccdf = ccdf_exponential(x, p1(yr), xmin)
                    y_pdf = np.gradient(1 - y_ccdf, x)
                elif dist == "power_law":
                    y_ccdf = ccdf_power_law(x, p1(yr), xmin)
                    y_pdf = np.gradient(1 - y_ccdf, x)
                elif dist == "truncated_power_law":
                    y_ccdf = ccdf_truncated_power_law(x, p1(yr), max(p2(yr), 1e-8), xmin)
                    y_pdf = np.gradient(1 - y_ccdf, x)
                elif dist == "stretched_exponential":
                    y_ccdf = ccdf_stretched_exponential(x, p1(yr), p2(yr), xmin)
                    y_pdf = np.gradient(1 - y_ccdf, x)
                elif dist == "weibull":
                    y_ccdf = ccdf_weibull(x, p1(yr), p2(yr), xmin)
                    y_pdf = np.gradient(1 - y_ccdf, x)
                elif dist == "lognormal":
                    y_ccdf = ccdf_lognormal(x, p1(yr), p2(yr), xmin)
                    y_pdf = np.gradient(1 - y_ccdf, x)
                elif dist == "generalized_pareto":
                    y_ccdf = ccdf_genpareto(x, p1(yr), p2(yr), xmin)
                    y_pdf = np.gradient(1 - y_ccdf, x)
                else:
                    continue

                # Plot the CCDF and PDF curves for this year
                ax_ccdf.plot(x, y_ccdf, color=cmap(norm(yr)), lw=1.5, label=f"{int(yr)}")
                ax_pdf.plot(x, y_pdf, color=cmap(norm(yr)), lw=1.5)
                good = True
            except Exception:
                continue

        # If nothing was successfully computed, skip this refined biome–distribution pair
        if not good:
            plt.close()
            print(f"Skipped {biome} ({dist}) — invalid CCDF/PDF.")
            continue

        # Configure CCDF panel (log–log)
        ax_ccdf.set_xscale("log")
        ax_ccdf.set_yscale("log")
        ax_ccdf.set_ylim(ccdf_ymin, 1.0)
        ax_ccdf.set_xlabel("Fire size (km²)")
        ax_ccdf.set_ylabel("CCDF")
        ax_ccdf.set_title(f"{biome}\n{dist.replace('_',' ').title()} (CCDF)")
        ax_ccdf.grid(True, alpha=0.3)

        # Configure PDF panel (log x)
        ax_pdf.set_xscale("log")
        ax_pdf.set_xlabel("Fire size (km²)")
        ax_pdf.set_ylabel("PDF")
        ax_pdf.set_title(f"{biome}\n{dist.replace('_',' ').title()} (PDF)")
        ax_pdf.grid(True, alpha=0.3)

        # Add colorbar indicating which color corresponds to which year
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cbar.set_label("Year", rotation=270, labelpad=15)
        plt.show()


def plot_refined_biome_tail_scan_ccdf_cdf_qq(
    df_refined: pd.DataFrame,
    biome_refined: str,
    xmin0: float = 4,
    which: tuple[str, ...] = ("power_law",),
    candidate_percentiles: list[float] | None = None,
    min_tail_n: int = 100,
    llhr_cutoff: float = 2.0,
    R_boot_tail: int = 20,
    save_dir: str = "../data",
):
    """
    For a given *refined biome* group, do two things:

      1) Search over increasing tail thresholds and find a robust power-law tail:
         - At each candidate percentile p, define xmin = quantile(data, p).
         - Fit wfpl on tail data (X >= xmin) and check if 'power_law' is a
           "good" fit by your logic:
             * present in params,
             * reduces_to is NOT a string,
             * row-min(LLR) for 'power_law' <= llhr_cutoff.
         - Impose a stronger condition: choose the smallest xmin such that
           power-law passes at that xmin AND at all higher candidate thresholds.

      2) Produce your original 3-panel diagnostic plot:
           (1) CCDF with selected fits (via wfpl) + the tail power-law overlay
           (2) CDF with selected fits (via wfpl) + the tail power-law overlay
           (3) Q–Q plot using ONLY the tail data (X >= xmin_tail) vs the
               best-fitting power-law on that tail.

    Plotting choices (scales, colors, labels) follow your original
    `plot_refined_biome_ccdf_cdf_qq` as closely as possible.

    Args:
        df_refined: Fire dataset with 'biome_refined' and 'area_km2' columns.
        biome_refined: Refined biome name (e.g., 'Open shrublands E of 130W').
        xmin0: Lower cutoff for the *main* fits (as before, default=4 km²).
        which: Distributions to overlay in the CCDF/CDF via wfpl.
        candidate_percentiles: Percentiles to consider for tail thresholds.
                               If None, uses [50, 60, 70, 80, 85, 90, 92, 94, 96, 97, 98].
        min_tail_n: Minimum number of fires in a tail to attempt a fit.
        llhr_cutoff: LLR cutoff defining a "good" power-law fit.
        R_boot_tail: Bootstrap replicates for wfpl tail fits.
        save_dir: Directory for saving PNGs.

    Returns:
        (fig, tail_info) where:
          - fig is the Matplotlib Figure object (or None on failure)
          - tail_info is a dict with keys:
                biome, xmin_tail, percentile, n_tail, alpha_hat, row_min_llhr,
                robust_all_above
    """
    # -------------------------------
    # 0. Setup and data selection
    # -------------------------------
    # Filter to the specified refined biome and ensure data are present
    subset = df_refined[df_refined["biome_refined"] == biome_refined]
    if subset.empty:
        print(f"No fires found for refined biome: {biome_refined}")
        return None, None

    # Extract and filter fire size data to the main tail region (≥ xmin0)
    data_all = subset["area_km2"].dropna().to_numpy()
    data_all = data_all[data_all >= xmin0]
    if len(data_all) == 0:
        print(f"No valid area data above xmin0={xmin0} for {biome_refined}")
        return None, None

    data_all = np.sort(data_all)
    n_total = len(data_all)

    # -------------------------------
    # 1. Tail search: robust power law
    # -------------------------------
    # Define candidate percentiles for the tail threshold search if not supplied
    if candidate_percentiles is None:
        candidate_percentiles = [50, 60, 70, 80, 85, 90, 92, 94, 96, 97, 98]
    candidate_percentiles = sorted(candidate_percentiles)

    candidates: list[dict] = []

    print(f"\n=== {biome_refined} — total fires ≥ {xmin0} km²: n={n_total} ===")

    # Loop over candidate percentiles, defining candidate tail thresholds
    for perc in candidate_percentiles:
        xmin_tail = float(np.quantile(data_all, perc / 100.0))
        tail = data_all[data_all >= xmin_tail]
        n_tail = len(tail)
        if n_tail < min_tail_n:
            continue

        print(
            f"  Trying tail threshold ≈ {xmin_tail:.2f} km² "
            f"({perc:.1f}th pct, n_tail={n_tail})"
        )

        # Run wfpl fits on this tail and record whether power-law passes your criteria
        try:
            params_tail = wfpl.summarize_parameters_bootstrap(
                tail, R=R_boot_tail, xmin=xmin_tail, random_state=42
            )
            llhr_tail, best_tail = wfpl.likelihood_matrix_and_best(
                tail, xmin=xmin_tail
            )
        except Exception as e:
            print(f"wfpl failed at this threshold: {e}")
            candidates.append(
                dict(
                    percentile=perc,
                    xmin=xmin_tail,
                    n_tail=n_tail,
                    alpha_hat=np.nan,
                    row_min_llhr=np.nan,
                    best_fit_tail=None,
                    passes=False,
                )
            )
            continue

        # Require power law to be present and not reduced to another distribution
        if "power_law" not in params_tail.index or "power_law" not in llhr_tail.index:
            print("'power_law' not present in params/llhr, skipping.")
            candidates.append(
                dict(
                    percentile=perc,
                    xmin=xmin_tail,
                    n_tail=n_tail,
                    alpha_hat=np.nan,
                    row_min_llhr=np.nan,
                    best_fit_tail=best_tail,
                    passes=False,
                )
            )
            continue

        row_pl = params_tail.loc["power_law"]
        if isinstance(row_pl.get("reduces_to"), str):
            print(f"'power_law' reduces_to={row_pl['reduces_to']}, skipping.")
            candidates.append(
                dict(
                    percentile=perc,
                    xmin=xmin_tail,
                    n_tail=n_tail,
                    alpha_hat=np.nan,
                    row_min_llhr=np.nan,
                    best_fit_tail=best_tail,
                    passes=False,
                )
            )
            continue

        # Compute minimum row-wise likelihood ratio and check against cutoff
        row_min_llhr = float(llhr_tail.loc["power_law"].min())
        alpha_hat = float(row_pl.get("p1", np.nan))
        passes = row_min_llhr <= llhr_cutoff

        print(
            f"    power_law: alpha_hat={alpha_hat:.3f}, "
            f"row_min_llhr={row_min_llhr:.3f}, passes={passes}"
        )

        # Record candidate tail info (whether it passes your criteria or not)
        candidates.append(
            dict(
                percentile=perc,
                xmin=xmin_tail,
                n_tail=n_tail,
                alpha_hat=alpha_hat,
                row_min_llhr=row_min_llhr,
                best_fit_tail=best_tail,
                passes=passes,
            )
        )

    # If there were no viable candidates, abort
    if not candidates:
        print(f"{biome_refined}: no candidate tail with ≥ {min_tail_n} fires.")
        return None, None

    # Compute suffix condition: for robust tail, we want all higher thresholds to pass
    passes_array = np.array([c["passes"] for c in candidates], dtype=bool)
    suffix_all_pass = np.zeros_like(passes_array, dtype=bool)
    running = True
    for i in range(len(candidates) - 1, -1, -1):
        running = running and passes_array[i]
        suffix_all_pass[i] = running

    # Find smallest percentile where it passes and all larger percentiles pass
    robust_idx = None
    for i in range(len(candidates)):
        if passes_array[i] and suffix_all_pass[i]:
            robust_idx = i
            break

    # Either choose a fully robust threshold or fall back to the highest passing candidate
    if robust_idx is not None:
        chosen = candidates[robust_idx]
        robust_all_above = True
        print(
            f"{biome_refined}: robust tail at ≈ {chosen['xmin']:.2f} km² "
            f"({chosen['percentile']:.1f}th pct), valid for all higher thresholds."
        )
    else:
        any_pass = [i for i, c in enumerate(candidates) if c["passes"]]
        if any_pass:
            idx = any_pass[-1]
        else:
            idx = len(candidates) - 1
        chosen = candidates[idx]
        robust_all_above = False
        print(
            f"{biome_refined}: no percentile where all higher thresholds pass; "
            f"using ≈ {chosen['xmin']:.2f} km² "
            f"({chosen['percentile']:.1f}th pct)."
        )

    xmin_tail = chosen["xmin"]
    perc_tail = chosen["percentile"]
    n_tail = chosen["n_tail"]
    alpha_hat = chosen["alpha_hat"]
    row_min_llhr = chosen["row_min_llhr"]

    # Package tail summary information to return
    tail_info = dict(
        biome=biome_refined,
        xmin_tail=xmin_tail,
        percentile=perc_tail,
        n_tail=n_tail,
        n_total=n_total,
        alpha_hat=alpha_hat,
        row_min_llhr=row_min_llhr,
        robust_all_above=robust_all_above,
    )

    # -------------------------------
    # 2. Plotting (match original style)
    # -------------------------------
    # Prepare full tail data and empirical CCDF for global CCDF/CDF plotting
    data = data_all
    n = len(data)
    empirical_ccdf = 1 - np.arange(1, n + 1) / (n + 1)
    ymin = max(np.min(empirical_ccdf), 1e-5)

    # Three-panel diagnostic figure: CCDF, CDF, and tail Q–Q
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax_ccdf, ax_cdf, ax_qq = axes

    # (1) CCDF with wfpl fits plus tail power-law overlay rescaled by tail mass
    ax_ccdf = wfpl.plot_ccdf_with_selected_fits(
        data, xmin=xmin0, which=which, ax=ax_ccdf
    )
    ax_ccdf.set_ylim(ymin, 1)
    ax_ccdf.set_title(
        f"{biome_refined}: Fire Size CCDF\n"
        f"Tail xmin={xmin_tail:.1f} km² ({perc_tail:.1f}th pct, n_tail={n_tail}, "
        f"row_min_llhr={row_min_llhr:.2f}, robust={'yes' if robust_all_above else 'no'})"
    )
    ax_ccdf.set_xlabel("Fire size (km²)")
    ax_ccdf.set_ylabel("CCDF")

    # Overlay tail-only power-law CCDF restricted to x>=xmin_tail and scaled by tail mass
    if np.isfinite(alpha_hat):
        mass_tail = n_tail / n_total
        xgrid = np.logspace(np.log10(data.min()), np.log10(data.max()), 500)
        mask = xgrid >= xmin_tail
        if np.any(mask):
            y_tail = mass_tail * ccdf_power_law(xgrid[mask], alpha_hat, xmin_tail)
            ax_ccdf.plot(
                xgrid[mask],
                y_tail,
                linestyle="--",
                color="tab:green",
                linewidth=2,
                label="Tail power law",
            )
            ax_ccdf.legend(fontsize=8)

    # (2) CDF (log-x) with wfpl fits and corresponding tail power-law overlay
    try:
        # Fit full model for the main tail and plot empirical CDF
        fit_full = wfpl.Fit(data, xmin=xmin0, discrete=False)
        fit_full.plot_cdf(color="k", linewidth=2, ax=ax_cdf, label="Empirical data")

        # Overlay the CDF of each selected distribution
        for name in which:
            try:
                dist = getattr(fit_full, name)
                dist.plot_cdf(
                    ax=ax_cdf,
                    linestyle="--",
                    label=f"{name.replace('_',' ').title()} fit",
                )
            except Exception as e:
                print(f"Skipped {name} in CDF: {e}")

        # Tail power-law CDF overlay: account for probability mass below xmin_tail
        if np.isfinite(alpha_hat):
            xgrid = np.logspace(np.log10(data.min()), np.log10(data.max()), 500)
            mask = xgrid >= xmin_tail
            tail_cdf = None
            if np.any(mask):
                ccdf_tail = ccdf_power_law(xgrid[mask], alpha_hat, xmin_tail)
                mass_tail = n_tail / n_total
                p_below = 1.0 - mass_tail
                tail_cdf = np.empty_like(xgrid[mask])
                tail_cdf[:] = p_below + mass_tail * (1.0 - ccdf_tail)
                ax_cdf.plot(
                    xgrid[mask],
                    tail_cdf,
                    linestyle="--",
                    color="tab:green",
                    linewidth=2,
                    label="Tail power law",
                )

        ax_cdf.set_xscale("log")
        ax_cdf.set_ylim(0, 1)
        ax_cdf.set_xlabel("Fire size (km²)")
        ax_cdf.set_ylabel("CDF")
        ax_cdf.set_title(f"{biome_refined}: Fire Size CDF")
        ax_cdf.legend(fontsize=8)
    except Exception as e:
        print(f"Could not plot CDFs: {e}")

    # (3) Q–Q: only tail points versus the tail power-law fit
    try:
        tail = data[data >= xmin_tail]
        if len(tail) == 0:
            raise ValueError("No data in tail for QQ plot.")
        tail = np.sort(tail)
        n_tail = len(tail)

        # Fit power-law to the tail again (for consistency with wfpl's parameterization)
        fit_tail = wfpl.Fit(tail, xmin=xmin_tail, discrete=False)
        model = fit_tail.power_law

        empirical_q = tail.copy()
        probs = (np.arange(1, n_tail + 1) - 0.5) / n_tail

        # Use model ppf if available; otherwise invert CDF numerically via interpolation
        if hasattr(model, "ppf") and callable(model.ppf):
            theoretical_q = model.ppf(probs)
        else:
            xgrid = np.logspace(np.log10(tail.min()), np.log10(tail.max()), 4000)
            try:
                cdf_vals = model.cdf(xgrid)
                if np.ndim(cdf_vals) > 1:
                    cdf_vals = np.asarray(cdf_vals).squeeze()
            except Exception:
                cdf_vals = np.array([float(model.cdf(float(x))) for x in xgrid])
            cdf_vals = np.clip(cdf_vals, 1e-12, 1 - 1e-12)
            cdf_vals = np.maximum.accumulate(cdf_vals)
            theoretical_q = np.interp(probs, cdf_vals, xgrid)

        ax_qq.scatter(
            theoretical_q,
            empirical_q,
            s=15,
            alpha=0.6,
            color="tab:blue",
            label="Power law (tail) fit",
        )
        ax_qq.plot(
            [empirical_q.min(), empirical_q.max()],
            [empirical_q.min(), empirical_q.max()],
            "r--",
            lw=1,
            label="1:1 line",
        )
        ax_qq.set_xscale("log")
        ax_qq.set_yscale("log")
        ax_qq.set_xlabel("Theoretical Quantiles")
        ax_qq.set_ylabel("Empirical Quantiles")
        ax_qq.set_title(f"{biome_refined}: Q–Q Plot (tail only)")
        ax_qq.legend(fontsize=8)
    except Exception as e:
        print(f"Could not generate Q–Q plot: {e}")

    # Finalize layout and write figure to disk
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f"{biome_refined.replace(' ', '_').lower()}_tail_scan_cdf_ccdf_qq.png",
    )
    plt.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
    print(f"Figure saved to {save_path}")
    plt.show()

    return fig, tail_info


def plot_refined_biome_ccdf_cdf_qq(
    df_refined,
    biome_refined,
    xmin=4,
    which=("power_law",),
    save_dir="../data",
    year_col="year",
    timevary_summary=None,
    timevary_dist=None,
    trend_unit_years=10.0,
):
    """
    Four-panel diagnostic plot for a given *refined biome* group:
        1) CCDF with selected fits (via `wfpl`)
        2) CDF with selected fits (via `wfpl`)
        3) Stationary Q–Q plot vs selected best-fit distribution
        4) Nonstationary Q–Q plot vs time-varying distribution (from timevary_summary)

    Nonstationary QQ logic:
      - For biome_refined and distribution dist_tv (timevary_dist or which[0]),
        find the row in timevary_summary with that (biome, distribution).
      - Parse parameters from columns:
            'p1 ± se', 'p2 ± se', 'Δp1 ± se', 'Δp2 ± se'
        giving p1, p2, Δp1, Δp2. If Δ columns blank/NaN -> 0.
      - Let year_center = mean(year) in this biome, and t_scaled = (year - year_center)/trend_unit_years.
        Then for each fire with year t_i:
            p1(t_i) = p1 + Δp1 * t_scaled_i
            p2(t_i) = p2 + Δp2 * t_scaled_i
      - For each sorted fire with size x_i (≥ xmin), define empirical probability
            p_i = (rank_i - 0.5) / n
        and compute theoretical quantile
            q_i = F^{-1}(p_i; p1(t_i), p2(t_i), xmin)
        using distribution-specific quantile formulas.
      - Panel 4 scatters (q_i, x_i) in log–log with a 1:1 line.

    Supported distributions for the nonstationary QQ:
        'lognormal', 'generalized_pareto', 'power_law',
        'stretched_exponential', 'weibull', 'exponential'.
    """

    # ---------- helpers defined inside so you only paste one function ----------

    def parse_val_pm_se(val):
        """From 'x.xxx ± y.yyy' or float -> float(x.xxx)."""
        if isinstance(val, str) and "±" in val:
            return float(val.split("±")[0].strip())
        try:
            return float(val)
        except Exception:
            return np.nan

    def q_trunc_lognormal(p, mu, sigma, xmin_local):
        """Quantile for lognormal truncated below xmin."""
        p = float(np.clip(p, 1e-10, 1 - 1e-10))
        sigma = float(max(sigma, 1e-8))
        z_min = (np.log(xmin_local) - mu) / sigma
        c0 = norm.cdf(z_min)
        target = c0 + p * (1.0 - c0)
        z = norm.ppf(target)
        return float(np.exp(mu + sigma * z))

    def q_gpd_tail(p, xi, sigma, xmin_local):
        """Quantile for GPD tail above xmin: X = xmin + Y, Y~GPD(xi,sigma)."""
        p = float(np.clip(p, 1e-10, 1 - 1e-10))
        sigma = float(max(sigma, 1e-10))
        xi = float(xi)
        u = -np.log(1.0 - p)  # >= 0

        if abs(xi) < 1e-6:
            y = sigma * u
        else:
            z = np.clip(xi * u, -50.0, 50.0)
            y = sigma / xi * (np.exp(z) - 1.0)

        y = max(y, 0.0)
        return float(xmin_local + y)

    def q_power_law(p, alpha, xmin_local):
        """Quantile for X>=xmin: CCDF(x) = (x/xmin)^(1-alpha)."""
        p = float(np.clip(p, 1e-10, 1 - 1e-10))
        if alpha == 1:
            return float(xmin_local)
        x = xmin_local * (1.0 - p) ** (1.0 / (1.0 - alpha))
        return float(max(x, xmin_local))

    def q_stretched_exp(p, lam, beta, xmin_local):
        """
        Quantile for stretched exponential tail:
            CCDF(x) = exp(-(lam*x)^beta + (lam*xmin)^beta), x>=xmin.
        """
        p = float(np.clip(p, 1e-10, 1 - 1e-10))
        lam = float(max(lam, 1e-12))
        beta = float(max(beta, 1e-6))
        term0 = (lam * xmin_local) ** beta
        term1 = -np.log(1.0 - p)
        base = term0 + term1
        base = max(base, 1e-12)
        x = (base ** (1.0 / beta)) / lam
        return float(max(x, xmin_local))

    def q_weibull(p, k, lam, xmin_local):
        """
        Quantile for Weibull tail conditional on X>=xmin:
            CCDF(x) = exp(-(x/lam)^k + (xmin/lam)^k).
        """
        p = float(np.clip(p, 1e-10, 1 - 1e-10))
        lam = float(max(lam, 1e-12))
        k = float(max(k, 1e-6))
        term0 = (xmin_local / lam) ** k
        term1 = -np.log(1.0 - p)
        base = term0 + term1
        base = max(base, 1e-12)
        x = lam * (base ** (1.0 / k))
        return float(max(x, xmin_local))

    def q_exponential(p, lam, xmin_local):
        """Quantile for exponential tail above xmin: X = xmin + Exp(lam)."""
        p = float(np.clip(p, 1e-10, 1 - 1e-10))
        lam = float(max(lam, 1e-12))
        y = -np.log(1.0 - p) / lam
        return float(xmin_local + max(y, 0.0))

    def quantile_for_dist(p, dist_name, p1_t, p2_t, xmin_local):
        """Dispatch to the correct quantile formula."""
        if dist_name == "lognormal":
            return q_trunc_lognormal(p, p1_t, p2_t, xmin_local)
        elif dist_name == "generalized_pareto":
            return q_gpd_tail(p, p1_t, p2_t, xmin_local)
        elif dist_name == "power_law":
            return q_power_law(p, p1_t, xmin_local)
        elif dist_name == "stretched_exponential":
            return q_stretched_exp(p, p1_t, p2_t, xmin_local)
        elif dist_name == "weibull":
            return q_weibull(p, p1_t, p2_t, xmin_local)
        elif dist_name == "exponential":
            return q_exponential(p, p1_t, xmin_local)
        elif dist_name == "truncated_power_law":
            raise ValueError("Nonstationary QQ for 'truncated_power_law' not implemented.")
        else:
            raise ValueError(f"Unsupported distribution '{dist_name}' for nonstationary QQ.")

    # ---------- main body starts here ----------

    # Subset to the chosen refined biome and ensure observations exist
    subset = df_refined[df_refined["biome_refined"] == biome_refined]
    if subset.empty:
        print(f"No fires found for refined biome: {biome_refined}")
        return None

    # Extract years if present; this is required for nonstationary QQ
    if year_col not in subset.columns:
        print(f"year_col='{year_col}' not in df_refined; nonstationary QQ will be skipped.")
        years = None
    else:
        years = subset[year_col].to_numpy()

    # Extract fire size data and enforce alignment with year values
    data = subset["area_km2"].dropna().to_numpy()
    if years is not None:
        mask_valid = np.isfinite(data) & np.isfinite(years)
        data = data[mask_valid]
        years = years[mask_valid]

    # Restrict to tail data above xmin
    data = data[data >= xmin]
    if years is not None:
        years = years[data >= xmin]

    if len(data) == 0:
        print(f"No valid area data above xmin={xmin} for {biome_refined}")
        return None

    # Sort data (and years, if present) so Q–Q plots are well-defined
    idx = np.argsort(data)
    data = data[idx]
    if years is not None:
        years = years[idx]

    # Build empirical CCDF for y-axis limits in CCDF plot
    n = len(data)
    empirical_ccdf = 1 - np.arange(1, n + 1) / (n + 1)
    ymin = max(np.min(empirical_ccdf), 1e-5)

    # Create 4-panel figure: CCDF, CDF, stationary Q–Q, nonstationary Q–Q
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    ax_ccdf, ax_cdf, ax_qq_stat, ax_qq_nonstat = axes

    # (1) CCDF – reuse wfpl helper for overlaying multiple fits
    ax_ccdf = wfpl.plot_ccdf_with_selected_fits(
        data, xmin=xmin, which=which, ax=ax_ccdf
    )
    ax_ccdf.set_ylim(ymin, 1)
    ax_ccdf.set_title(f"{biome_refined}: Fire Size CCDF")
    ax_ccdf.set_xlabel("Fire size (km²)")
    ax_ccdf.set_ylabel("CCDF")

    # (2) CDF – empirical and fitted distributions in log-x space
    try:
        fit = wfpl.Fit(data, xmin=xmin, discrete=False)
        fit.plot_cdf(color="k", linewidth=2, ax=ax_cdf, label="Empirical data")
        for name in which:
            try:
                dist = getattr(fit, name)
                dist.plot_cdf(
                    ax=ax_cdf,
                    linestyle="--",
                    label=f"{name.replace('_',' ').title()} fit",
                )
            except Exception as e:
                print(f"Skipped {name} in CDF: {e}")
        ax_cdf.set_xscale("log")
        ax_cdf.set_ylim(0, 1)
        ax_cdf.set_xlabel("Fire size (km²)")
        ax_cdf.set_ylabel("CDF")
        ax_cdf.set_title(f"{biome_refined}: Fire Size CDF")
        ax_cdf.legend(fontsize=8)
    except Exception as e:
        print(f"Could not plot CDFs: {e}")

    # (3) Stationary Q–Q plot vs selected distribution
    try:
        dist_name = which[0]
        fit = wfpl.Fit(data, xmin=xmin, discrete=False)
        model = getattr(fit, dist_name)

        # Empirical quantiles and plotting positions
        empirical_q = np.sort(data)
        probs = (np.arange(1, len(empirical_q) + 1) - 0.5) / len(empirical_q)

        # Use ppf if available, otherwise invert CDF numerically
        if hasattr(model, "ppf") and callable(model.ppf):
            theoretical_q = model.ppf(probs)
        else:
            xgrid = np.logspace(np.log10(data.min()), np.log10(data.max()), 4000)
            try:
                cdf_vals = model.cdf(xgrid)
                if np.ndim(cdf_vals) > 1:
                    cdf_vals = np.asarray(cdf_vals).squeeze()
            except Exception:
                cdf_vals = np.array([float(model.cdf(float(x))) for x in xgrid])
            cdf_vals = np.clip(cdf_vals, 1e-12, 1 - 1e-12)
            cdf_vals = np.maximum.accumulate(cdf_vals)
            theoretical_q = np.interp(probs, cdf_vals, xgrid)

        # Plot stationary Q–Q with 1:1 reference line
        ax_qq_stat.scatter(
            theoretical_q,
            empirical_q,
            s=15,
            alpha=0.6,
            color="tab:blue",
            label=f"{dist_name.title()} fit",
        )
        ax_qq_stat.plot(
            [empirical_q.min(), empirical_q.max()],
            [empirical_q.min(), empirical_q.max()],
            "r--",
            lw=1,
            label="1:1 line",
        )
        ax_qq_stat.set_xscale("log")
        ax_qq_stat.set_yscale("log")
        ax_qq_stat.set_xlabel("Theoretical Quantiles")
        ax_qq_stat.set_ylabel("Empirical Quantiles")
        ax_qq_stat.set_title(f"{biome_refined}: Q–Q Plot (stationary)")
        ax_qq_stat.legend(fontsize=8)
    except Exception as e:
        print(f"Could not generate stationary Q–Q plot: {e}")

    # (4) Nonstationary Q–Q using time-varying parameters from timevary_summary
    try:
        if timevary_summary is None:
            raise ValueError("timevary_summary is None; skipping nonstationary QQ.")
        if years is None:
            raise ValueError("year_col not available; skipping nonstationary QQ.")

        # Distribution to use for time-varying QQ; default to first in `which`
        dist_tv = timevary_dist if timevary_dist is not None else which[0]

        # Look up time-varying summary row corresponding to this biome and distribution
        mask = (
            (timevary_summary["biome"] == biome_refined)
            & (timevary_summary["distribution"] == dist_tv)
        )
        if not mask.any():
            raise ValueError(
                f"No row in timevary_summary for biome='{biome_refined}', "
                f"distribution='{dist_tv}'."
            )
        row = timevary_summary[mask].iloc[0]

        # Parse p1, p2, and their slopes (Δp1, Δp2) from formatted columns
        p1 = parse_val_pm_se(row.get("p1 ± se", np.nan))
        p2 = parse_val_pm_se(row.get("p2 ± se", np.nan))
        dp1 = parse_val_pm_se(row.get("Δp1 ± se", np.nan))
        dp2 = parse_val_pm_se(row.get("Δp2 ± se", np.nan))

        if not np.isfinite(p1):
            raise ValueError("Non-finite p1 in timevary_summary.")
        if not np.isfinite(p2):
            p2 = 0.0
        if not np.isfinite(dp1):
            dp1 = 0.0
        if not np.isfinite(dp2):
            dp2 = 0.0

        # Build empirical probabilities for the Q–Q plot
        empirical_q_ns = data.copy()
        n_ns = len(empirical_q_ns)
        probs_ns = (np.arange(1, n_ns + 1) - 0.5) / n_ns

        # Center years and rescale to match timevary trend units
        year_center = float(np.nanmean(years))
        t_scaled = (years - year_center) / trend_unit_years

        # Compute time-varying parameters for each observation
        p1_t = p1 + dp1 * t_scaled
        p2_t = p2 + dp2 * t_scaled

        # Compute theoretical quantiles under the time-varying model
        theoretical_q_ns = np.empty_like(empirical_q_ns, dtype=float)
        for i in range(n_ns):
            theoretical_q_ns[i] = quantile_for_dist(
                probs_ns[i], dist_tv, p1_t[i], p2_t[i], xmin
            )

        # Scatter plot of time-varying theoretical quantiles vs empirical sizes
        ax_qq_nonstat.scatter(
            theoretical_q_ns,
            empirical_q_ns,
            s=15,
            alpha=0.6,
            color="tab:purple",
            label=f"Nonstationary {dist_tv}",
        )
        qmin = min(empirical_q_ns.min(), theoretical_q_ns.min())
        qmax = max(empirical_q_ns.max(), theoretical_q_ns.max())
        ax_qq_nonstat.plot(
            [qmin, qmax],
            [qmin, qmax],
            "r--",
            lw=1,
            label="1:1 line",
        )
        ax_qq_nonstat.set_xscale("log")
        ax_qq_nonstat.set_yscale("log")
        ax_qq_nonstat.set_xlabel("Theoretical Quantiles (time-varying)")
        ax_qq_nonstat.set_ylabel("Empirical Quantiles")
        ax_qq_nonstat.set_title(
            f"{biome_refined}: Q–Q Plot (nonstationary {dist_tv})"
        )
        ax_qq_nonstat.legend(fontsize=8)
    except Exception as e:
        print(f"Could not generate nonstationary Q–Q plot: {e}")

    # Finalize layout and save figure to disk
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = (
        f"{save_dir}/{biome_refined.replace(' ', '_').lower()}"
        "_cdf_ccdf_qq_nonstationary.png"
    )
    plt.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
    print(f"Figure saved to {save_path}")
    plt.show()
    return fig
