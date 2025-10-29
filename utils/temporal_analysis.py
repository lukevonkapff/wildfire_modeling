import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm
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


def ccdf_exponential(x: np.ndarray, lam: float, xmin: int = 12) -> np.ndarray:
    """Compute the complementary CDF (CCDF) of an exponential distribution.

    Args:
        x: Input array of fire sizes.
        lam: Exponential rate parameter λ.
        xmin: Lower cutoff value.

    Returns:
        CCDF values normalized to 1 at xmin.
    """
    mask = x >= xmin
    ccdf = np.exp(-lam * (x - xmin))
    ccdf /= max(ccdf[mask][0], 1e-12)
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
    mask = x >= xmin
    ccdf = (x / xmin) ** (1 - alpha)
    ccdf /= max(ccdf[mask][0], 1e-12)
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
    x = np.asarray(x, dtype=float)
    mask = x >= xmin
    z = lambd * x[mask]

    # if λ is negligible or α large, behave as pure power law
    if lambd < 1e-3 or np.all(z < 1e-2) or alpha > 5:
        return ccdf_power_law(x, alpha, xmin)

    try:
        val = gammaincc(1 - alpha, z) / gamma(1 - alpha)
        if not np.any(np.isfinite(val)) or np.all(val < 1e-12):
            # fallback to power law if gamma underflows
            return ccdf_power_law(x, alpha, xmin)
        ccdf = np.zeros_like(x)
        ccdf[mask] = np.clip(val, 1e-12, 1.0)
        ccdf /= max(ccdf[mask][0], 1e-12)
    except Exception:
        # final fallback on numerical errors
        return ccdf_power_law(x, alpha, xmin)

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
    mask = x >= xmin
    ccdf = np.exp(-(lam * x) ** beta + (lam * xmin) ** beta)
    ccdf /= max(ccdf[mask][0], 1e-12)
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
    mask = x >= xmin
    ccdf = np.exp(-((x / lam) ** k - (xmin / lam) ** k))
    ccdf /= max(ccdf[mask][0], 1e-12)
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
    mask = x >= xmin
    ccdf = lognorm.sf(x, sigma, scale=np.exp(mu))
    ccdf /= max(ccdf[mask][0], 1e-12)
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
    mask = x >= xmin
    y = x[mask] - xmin
    if abs(xi) < 1e-6:
        ccdf = np.exp(-y / sigma)
    else:
        ccdf = (1 + xi * y / sigma) ** (-1 / xi)
    full_ccdf = np.zeros_like(x)
    full_ccdf[mask] = ccdf
    full_ccdf /= max(full_ccdf[mask][0], 1e-12)
    return np.clip(full_ccdf, 1e-12, 1.0)

def plot_distribution_evolution_ccdf(df: pd.DataFrame, xmin: int = 12) -> None:
    """Plot CCDF evolution over relative time for each fitted distribution.

    Degenerate truncated power laws (tiny λ) are drawn as pure power laws.

    Args:
        df: DataFrame with columns ["distribution", "biome", "p1", "p2", "p1'", "p2'", "p1_slope_sig", "p2_slope_sig"].
        xmin: Minimum fire size cutoff.
    """
    x = np.logspace(np.log10(xmin), 3, 500)
    time_steps = np.linspace(-1, 1, 5)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=time_steps.min(), vmax=time_steps.max())

    for _, row in df.iterrows():
        dist = row["distribution"]
        biome = row["biome"]

        # parameter slope helper functions
        def p1(t: float) -> float:
            if row.get("p1_slope_sig", 0) == 0:
                return row["p1"]
            return row["p1"] + row["p1'"] * t

        def p2(t: float) -> float | None:
            if np.isnan(row.get("p2", np.nan)):
                return None
            if row.get("p2_slope_sig", 0) == 0:
                return row["p2"]
            return row["p2"] + row["p2'"] * t

        plt.figure(figsize=(6, 4))
        success = False

        for t in time_steps:
            y = None
            try:
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
                y = None

            if y is None or np.all(np.isnan(y)):
                continue

            plt.plot(x, y, color=cmap(norm(t)), label=f"t={t:+.1f}")
            success = True

        if not success:
            plt.close()
            print(f"⚠️ Skipped {biome} ({dist}) — all NaN CCDFs.")
            continue

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
    rows: list[dict] = []

    for biome, dists in timevary_results.items():
        for dist, modes in dists.items():
            if mode not in modes:
                continue
            res = modes[mode]
            coeffs = np.array(res.get("coeffs", []), dtype=float)
            ses = np.array(res.get("ses", []), dtype=float)
            n = res.get("n", np.nan)

            p1 = p1_se = p1s = p1s_se = np.nan
            p2 = p2_se = p2s = p2s_se = np.nan
            p1_sig = p2_sig = 0

            # interpret coefficients by mode length
            if mode == "both":
                if len(coeffs) == 2:
                    p1, p1s = coeffs
                    p1_se, p1s_se = ses
                    ci_low, ci_high = p1s - 1.96 * p1s_se, p1s + 1.96 * p1s_se
                    p1_sig = 1 if (ci_low > 0 or ci_high < 0) else 0
                elif len(coeffs) == 4:
                    p1, p1s, p2, p2s = coeffs
                    p1_se, p1s_se, p2_se, p2s_se = ses
                    ci_low1, ci_high1 = p1s - 1.96 * p1s_se, p1s + 1.96 * p1s_se
                    ci_low2, ci_high2 = p2s - 1.96 * p2s_se, p2s + 1.96 * p2s_se
                    p1_sig = 1 if (ci_low1 > 0 or ci_high1 < 0) else 0
                    p2_sig = 1 if (ci_low2 > 0 or ci_high2 < 0) else 0

            elif mode == "p1_only":
                if len(coeffs) == 2:
                    p1, p1s = coeffs
                    p1_se, p1s_se = ses
                    ci_low, ci_high = p1s - 1.96 * p1s_se, p1s + 1.96 * p1s_se
                    p1_sig = 1 if (ci_low > 0 or ci_high < 0) else 0
                elif len(coeffs) == 3:
                    p1, p1s, p2 = coeffs
                    p1_se, p1s_se, p2_se = ses
                    ci_low, ci_high = p1s - 1.96 * p1s_se, p1s + 1.96 * p1s_se
                    p1_sig = 1 if (ci_low > 0 or ci_high < 0) else 0

            elif mode == "p2_only":
                if len(coeffs) == 2:
                    p1, p2s = coeffs
                    p1_se, p2s_se = ses
                    ci_low, ci_high = p2s - 1.96 * p2s_se, p2s + 1.96 * p2s_se
                    p2_sig = 1 if (ci_low > 0 or ci_high < 0) else 0
                elif len(coeffs) == 3:
                    p1, p2, p2s = coeffs
                    p1_se, p2_se, p2s_se = ses
                    ci_low, ci_high = p2s - 1.96 * p2s_se, p2s + 1.96 * p2s_se
                    p2_sig = 1 if (ci_low > 0 or ci_high < 0) else 0

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
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # --------- Log-PDF helpers (vectorized, stable ranges) ----------
    def logpdf_lognormal(x: np.ndarray, mu: float, sigma: float, xmin: float = 0) -> np.ndarray:
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

    def logpdf_powerlaw(x: np.ndarray, alpha: float, xmin: float = 1) -> np.ndarray:
        x = np.asarray(x)
        valid = x >= xmin
        pdf = -np.inf * np.ones_like(x, dtype=float)
        alpha = np.clip(alpha, 0.0, 5)
        C = (alpha - 1) / xmin if alpha != 1 else 1 / xmin
        pdf[valid] = np.log(np.abs(C)) - alpha * np.log(x[valid] / xmin)
        return pdf

    def logpdf_trunc_powerlaw(x: np.ndarray, alpha: float, lambd: float, xmin: float = 1) -> np.ndarray:
        """Properly normalized truncated power law; stable for small/large α, λ."""
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
            Z = np.clip(Z, 1e-300, 1e300)  # avoid log under/overflow
            logZ = np.log(Z)
        except Exception:
            logZ = 0.0
        pdf[valid] = -alpha * np.log(x[valid]) - lambd * (x[valid] - xmin) - logZ
        return pdf

    def logpdf_genpareto(x: np.ndarray, xi: float, sigma: float, xmin: float = 0) -> np.ndarray:
        x = np.asarray(x)
        y = x - xmin
        valid = (sigma > 0) & (1 + xi * y / sigma > 0)
        pdf = -np.inf * np.ones_like(x, dtype=float)
        xi = np.clip(xi, -1, 2)
        sigma = np.clip(sigma, 1e-6, 10)
        pdf[valid] = -np.log(sigma) - (1 / xi + 1) * np.log(1 + xi * y[valid] / sigma)
        return pdf

    def logpdf_weibull(x: np.ndarray, k: float, lam: float, xmin: float = 0) -> np.ndarray:
        x = np.asarray(x)
        valid = x >= xmin
        pdf = -np.inf * np.ones_like(x, dtype=float)
        k = np.clip(k, 1e-6, 50)
        lam = np.clip(lam, 1e-6, 50)
        z = np.clip(x[valid] / lam, 1e-12, 1e6)
        pdf[valid] = np.log(k) - np.log(lam) + (k - 1) * np.log(z) - z ** k
        return pdf

    def logpdf_stretched_exp(x: np.ndarray, lam: float, beta: float, xmin: float = 0) -> np.ndarray:
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

    dist_logpdfs: dict[str, callable] = {
        "lognormal": logpdf_lognormal,
        "power_law": logpdf_powerlaw,
        "truncated_power_law": logpdf_trunc_powerlaw,
        "generalized_pareto": logpdf_genpareto,
        "weibull": logpdf_weibull,
        "stretched_exponential": logpdf_stretched_exp,
    }

    timevary_results: dict = {}
    mtbs_classified = mtbs_classified.copy()
    # center years by mean and scale to decades for slope interpretability
    mtbs_classified["year_c"] = (mtbs_classified[year_col] - mtbs_classified[year_col].mean()) / 10.0
    rng_global = np.random.default_rng(42)

    for biome, subset in mtbs_classified.groupby("modis_class_static"):
        data = subset["area_km2"].values
        data = data[data >= xmin]
        if len(data) < min_total:
            if verbose:
                print(f"\n=== {biome} skipped: only {len(data)} fires above xmin ({xmin}) ===")
            continue

        years = subset.loc[subset["area_km2"] >= xmin, "year_c"].values
        if verbose:
            print(f"\n=== {biome} (n={len(data)} fires ≥ {xmin}) ===")

        res = overall_results.get(biome, {})
        if not res:
            if verbose:
                print(f"⚠️ No overall_results found for {biome}")
            continue

        params_df: pd.DataFrame = res["params"]
        llhr: pd.DataFrame = res["likelihood_matrix"]

        # Filter candidate distributions by availability, reductions, and Δloglik
        candidates: list[str] = []
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
                print(f"⚠️ No viable candidates for {biome}")
            continue

        biome_res: dict = {}

        for dist_name in candidates:
            fit_modes: dict = {}
            # Which parameters may vary with time in each "mode"
            fit_configs = {"both": [True, True], "p1_only": [True, False], "p2_only": [False, True]}

            static_row = params_df.loc[dist_name]
            p1_static = float(static_row.get("p1", 1.0))
            p2_static = float(static_row.get("p2", 1.0)) if not pd.isna(static_row.get("p2", np.nan)) else 1.0

            for mode, (fit_p1, fit_p2) in fit_configs.items():
                # -------- Negative log-likelihood with weak prior ----------
                def neg_loglik(params: np.ndarray, data: np.ndarray = data) -> float:
                    try:
                        # unpack by distribution + mode
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

                        # time-varying parameterization (per distribution)
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

                        valid_ll = ll[np.isfinite(ll)]

                        # weak prior → center around static p1/p2 (on stable transforms)
                        if dist_name == "truncated_power_law":
                            prior_center_a1 = np.log(max(p1_static - 1, 1e-3))  # α → log(α-1)
                        else:
                            prior_center_a1 = np.log(max(p1_static, 1e-6))      # p1 → log(p1)
                        prior_center_a2 = np.log(max(p2_static, 1e-6))          # p2 → log(p2)

                        prior_penalty = prior_weight * ((a1 - prior_center_a1) ** 2 + (a2 - prior_center_a2) ** 2)

                        return -np.sum(valid_ll) + prior_penalty
                    except Exception:
                        return np.inf

                # --------- Bounds by distribution and mode ----------
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

                if mode == "both":
                    bnds = bounds
                elif mode == "p1_only":
                    bnds = [bounds[0], bounds[1], bounds[2]]
                elif mode == "p2_only":
                    bnds = [bounds[0], bounds[2], bounds[3]]

                # -------- Differential Evolution global search ----------
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
                boot_params: list[np.ndarray] = []
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
                        # allow some failures (ill-conditioned resamples)
                        continue

                # -------- Transform coefficients to (p1, p1', p2, p2') --------
                if dist_name == "truncated_power_law":
                    if mode == "both":
                        a1, b1, a2, b2 = coeffs
                    elif mode == "p1_only":
                        a1, b1, a2 = coeffs
                        b2 = 0.0
                    elif mode == "p2_only":
                        a1, a2, b2 = coeffs
                        b1 = 0.0

                    p1 = 1 + np.exp(a1)
                    p1_prime = b1 * np.exp(a1)
                    p2 = np.exp(a2)
                    p2_prime = b2 * np.exp(a2)

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
                        p1_se = p1p_se = p2_se = p2p_se = 0.0

                    coeffs_trans = [p1, p1_prime, p2, p2_prime]
                    ses_trans = [p1_se, p1p_se, p2_se, p2p_se]
                else:
                    ses = np.std(boot_params, axis=0) if boot_params else np.zeros_like(coeffs)
                    coeffs_trans = coeffs
                    ses_trans = ses

                fit_modes[mode] = {
                    "coeffs": coeffs_trans,
                    "ses": ses_trans,
                    "loglik": ll_max,
                    "n": len(data),
                }

            if fit_modes:
                biome_res[dist_name] = fit_modes

        if biome_res:
            timevary_results[biome] = biome_res

    return timevary_results


def plot_savanna_fires(mtbs_classified: pd.DataFrame | gpd.GeoDataFrame, biome: str = "both") -> None:
    """Plot Savanna / Woody Savanna fires on a Cartopy basemap (includes Alaska fallback).

    Args:
        mtbs_classified: Dataset with columns LATITUDE, LONGITUDE, modis_class_static, geometry (if GeoDataFrame).
        biome: One of {"Savannas", "Woody savannas", "both"}.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    valid_biomes = ["Savannas", "Woody savannas", "both"]
    if biome not in valid_biomes:
        raise ValueError(f"biome must be one of {valid_biomes}")

    target_biomes = ["Savannas", "Woody savannas"] if biome == "both" else [biome]

    subset = mtbs_classified[mtbs_classified["modis_class_static"].isin(target_biomes)].copy()
    subset = subset[subset["LATITUDE"].apply(np.isfinite) & subset["LONGITUDE"].apply(np.isfinite)]

    if subset.empty:
        print(f"⚠️ No valid fires found for {', '.join(target_biomes)}.")
        return

    # ensure geospatial types
    if not isinstance(subset, gpd.GeoDataFrame):
        subset = gpd.GeoDataFrame(
            subset,
            geometry=gpd.points_from_xy(subset.LONGITUDE, subset.LATITUDE),
            crs="EPSG:4326",
        )

    subset = subset[subset.geometry.notna() & (~subset.geometry.is_empty) & subset.geometry.is_valid].copy()
    if subset.empty:
        print(f"⚠️ No valid geometries found for {', '.join(target_biomes)}.")
        return

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.add_feature(cfeature.LAND, facecolor="lightgrey")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    colors = {"Savannas": "tab:orange", "Woody savannas": "tab:green"}
    for biome_name, group in subset.groupby("modis_class_static"):
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
        minx, miny, maxx, maxy = subset.total_bounds
        if np.all(np.isfinite([minx, miny, maxx, maxy])) and (minx < maxx and miny < maxy):
            pad_x = max((maxx - minx) * 0.05, 5)
            pad_y = max((maxy - miny) * 0.05, 5)
            ax.set_extent([minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y], crs=ccrs.PlateCarree())
        else:
            raise ValueError("Invalid bounds")
    except Exception:
        ax.set_extent([-170, -50, 15, 75], crs=ccrs.PlateCarree())

    title_str = "Savanna & Woody Savanna Fires (MTBS)" if biome == "both" else f"{biome} Fires (MTBS)"
    ax.set_title(title_str, fontsize=14, pad=10)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="Biome", loc="upper right", frameon=True)

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
    yearly_counts = mtbs_classified.groupby(year_col).size()
    category_counts = mtbs_classified.groupby([year_col, static_col]).size().unstack(fill_value=0)

    categories = category_counts.columns.tolist()
    n_categories = len(categories)
    n_plots = n_categories + 1
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()

    # per-category panels
    for i, cat in enumerate(categories):
        ax = axes[i]
        category_counts[cat].plot(kind="bar", ax=ax, color="tab:blue", alpha=0.7)
        ax.set_title(cat)
        ax.set_xlabel("")
        ax.set_ylabel("Fires")

    # overall panel
    ax = axes[n_categories]
    yearly_counts.plot(kind="bar", ax=ax, color="black", alpha=0.8)
    ax.set_title("Overall Fires")
    ax.set_xlabel("")
    ax.set_ylabel("Fires")

    # remove any extra subplot axes
    for j in range(n_categories + 1, len(axes)):
        fig.delaxes(axes[j])

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
    yearly_counts_by_thresh: dict[float, pd.Series] = {}
    for t in thresholds:
        filtered = mtbs_classified[mtbs_classified[area_col] >= t]
        yearly_counts = filtered.groupby(year_col).size()
        yearly_counts_by_thresh[t] = yearly_counts

    counts_df = pd.DataFrame(yearly_counts_by_thresh).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(12, 6))
    for t in thresholds:
        ax.plot(counts_df.index, counts_df[t], marker="o", label=f"≥ {t} km²")

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
    os.makedirs(out_dir, exist_ok=True)
    tile_pattern = re.compile(r"h\d{2}v\d{2}")

    # collect rasters by MODIS tile id
    tile_files: dict[str, list[str]] = {}
    for year, files in modis_by_year.items():
        for f in files:
            m = tile_pattern.search(os.path.basename(f))
            if not m:
                continue
            tile_id = m.group()
            tile_files.setdefault(tile_id, []).append(f)

    out_paths: dict[str, str] = {}

    for tile_id, files in tile_files.items():
        arrays: list[np.ndarray] = []
        ref_profile: dict | None = None
        nodata = 255

        for f in files:
            with rasterio.open(f) as src:
                if ref_profile is None:
                    ref_profile = src.profile.copy()
                    nodata = ref_profile.get("nodata", 255)
                arr = src.read(1).astype(float)
                arrays.append(arr)

        # stack and mask nodata + class 17 as NaN (per your convention)
        stack = np.stack(arrays, axis=0)
        stack = np.where((stack == nodata) | (stack == 17), np.nan, stack)

        # pixelwise mode across time
        mode_map, _ = stats.mode(stack, axis=0, nan_policy="omit")
        static_map = np.squeeze(mode_map).astype(np.uint8)

        out_path = os.path.join(out_dir, f"static_modis_mode_{tile_id}.tif")
        assert ref_profile is not None
        ref_profile.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=nodata)
        with rasterio.open(out_path, "w", **ref_profile) as dst:
            dst.write(static_map, 1)
        out_paths[tile_id] = out_path

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
    results: list[tuple[int, str]] = []

    for tile_id, raster_path in static_tile_dict.items():
        with rasterio.open(raster_path) as src:
            fires_proj = fires_gdf.to_crs(src.crs)

            for idx, fire in fires_proj.iterrows():
                try:
                    out_image, _ = rasterio.mask.mask(src, [fire.geometry], crop=True)
                except ValueError:
                    continue

                data = out_image[0]
                data = data[(data != src.nodata) & (data != 255)]
                if data.size > 0:
                    mapped = [modis_to_gfa.get(int(val), "Other") for val in data]
                    majority = Counter(mapped).most_common(1)[0][0]
                    results.append((idx, majority))

    fires_out = fires_gdf.copy()
    fires_out["modis_class_static"] = "Unknown"
    for idx, cls in results:
        fires_out.at[idx, "modis_class_static"] = cls
    return fires_out

def classify_with_static_savanna_check(
    fires_gdf: gpd.GeoDataFrame,
    static_tile_dict: dict[str, str],
    modis_to_gfa: dict[int, str],
) -> gpd.GeoDataFrame:
    """Classify fire polygons using static (per-tile) MODIS rasters,
    except: if the majority class is 'Savannas' but ≤90% of area,
    assign the 2nd-most common class instead.

    Notes:
        - The static MODIS rasters are assumed to live under:
          /Users/lukevonkapff/Desktop/wildfires_github/wildfire_modeling/data/static_modis_tiles
    """
    base_dir = "../data/static_modis_tiles"
    results: list[tuple[int, str]] = []

    for tile_id, raster_path in static_tile_dict.items():
        # build full path
        raster_path_full = os.path.join(base_dir, os.path.basename(raster_path))

        try:
            with rasterio.open(raster_path_full) as src:
                fires_proj = fires_gdf.to_crs(src.crs)

                for idx, fire in fires_proj.iterrows():
                    try:
                        out_image, _ = rasterio.mask.mask(src, [fire.geometry], crop=True)
                    except ValueError:
                        continue

                    data = out_image[0]
                    data = data[(data != src.nodata) & (data != 255)]
                    if data.size == 0:
                        continue

                    # Map MODIS numeric codes → GFA class names
                    mapped = [modis_to_gfa.get(int(val), "Other") for val in data]

                    counts = Counter(mapped)
                    total = sum(counts.values())
                    if total == 0:
                        continue

                    ordered = counts.most_common()
                    top_class, top_count = ordered[0]
                    top_frac = top_count / total

                    # Savanna override logic
                    if top_class == "Savannas" and top_frac <= 0.9 and len(ordered) > 1:
                        alt_class = ordered[1][0]
                        results.append((idx, alt_class))
                    else:
                        results.append((idx, top_class))
        except rasterio.errors.RasterioIOError:
            print(f"⚠️ Skipping missing tile {tile_id}: {raster_path_full}")
            continue

    fires_out = fires_gdf.copy()
    fires_out["modis_class_static_no_savanna"] = "Unknown"
    for idx, cls in results:
        fires_out.at[idx, "modis_class_static_no_savanna"] = cls

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
    results: list[tuple[int, str]] = []
    for year, fires in fires_gdf.groupby(fires_gdf[year_col]):
        if year not in modis_by_year:
            continue
        for modis_file in modis_by_year[year]:
            with rasterio.open(modis_file) as src:
                fires_proj = fires.to_crs(src.crs)
                for idx, fire in fires_proj.iterrows():
                    try:
                        out_image, _ = rasterio.mask.mask(src, [fire.geometry], crop=True)
                    except ValueError:
                        continue
                    data = out_image[0]
                    data = data[(data != src.nodata) & (data != 255)]
                    if data.size > 0:
                        mapped = [modis_to_gfa.get(int(val), "Other") for val in data]
                        majority = Counter(mapped).most_common(1)[0][0]
                        results.append((idx, majority))

    fires_out = fires_gdf.copy()
    fires_out["modis_class_timevary"] = "Unknown"
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
    gdf = gpd.read_file(shp_path)
    gdf = gdf.set_crs(projection)
    gdf = gdf.to_crs("EPSG:6933")
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
    for biome, best_dist in dist_map.items():
        row_both = df_both[(df_both["biome"] == biome) & (df_both["distribution"] == best_dist)]
        row_p1 = df_p1[(df_p1["biome"] == biome) & (df_p1["distribution"] == best_dist)]
        row_p2 = df_p2[(df_p2["biome"] == biome) & (df_p2["distribution"] == best_dist)]

        if row_both.empty:
            print(f"⚠️ No match for {biome} ({best_dist}) in df_both.")
            continue

        r_b = row_both.iloc[0]
        n = int(r_b["n"])

        p1, p1_se = r_b["p1"], r_b["p1_se"]
        p2, p2_se = r_b["p2"], r_b["p2_se"]
        delta_ll = np.nan
        if biome in overall_results:
            ll_df: pd.DataFrame = overall_results[biome]["likelihood_matrix"]
            if best_dist in ll_df.index:
                delta_ll = ll_df.loc[best_dist].iloc[0]

        p1_trend = p1_trend_se = p2_trend = p2_trend_se = np.nan
        if not row_p1.empty:
            if (r_b["p1_slope_sig"] == 1) and (row_p1.iloc[0]["p1_slope_sig"] == 1):
                p1_trend = r_b["p1'"]
                p1_trend_se = r_b["p1'_se"]
        if not row_p2.empty:
            if (r_b["p2_slope_sig"] == 1) and (row_p2.iloc[0]["p2_slope_sig"] == 1):
                p2_trend = r_b["p2'"]
                p2_trend_se = r_b["p2'_se"]

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

    records: list[dict] = []
    for _, r_b in df_both.iterrows():
        biome = r_b["biome"]
        dist = r_b["distribution"]
        if biome in primary_map and dist == primary_map[biome]:
            continue  # skip primaries

        row_p1 = df_p1[(df_p1["biome"] == biome) & (df_p1["distribution"] == dist)]
        row_p2 = df_p2[(df_p2["biome"] == biome) & (df_p2["distribution"] == dist)]

        n = int(r_b["n"])
        p1, p1_se = r_b["p1"], r_b["p1_se"]
        p2, p2_se = r_b["p2"], r_b["p2_se"]
        delta_ll = np.nan
        if biome in overall_results:
            ll_df: pd.DataFrame = overall_results[biome]["likelihood_matrix"]
            if dist in ll_df.index:
                delta_ll = ll_df.loc[dist].iloc[0]

        p1_trend = p1_trend_se = p2_trend = p2_trend_se = np.nan
        if not row_p1.empty:
            if (r_b["p1_slope_sig"] == 1) and (row_p1.iloc[0]["p1_slope_sig"] == 1):
                p1_trend = r_b["p1'"]
                p1_trend_se = r_b["p1'_se"]
        if not row_p2.empty:
            if (r_b["p2_slope_sig"] == 1) and (row_p2.iloc[0]["p2_slope_sig"] == 1):
                p2_trend = r_b["p2'"]
                p2_trend_se = r_b["p2'_se"]

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

    return pd.DataFrame(records)


def plot_biome_facets(mtbs_classified: pd.DataFrame | gpd.GeoDataFrame) -> None:
    """Create a 3-panel faceted map of MTBS fire locations by biome group.

    Panels:
      (1) Deciduous Broadleaf + Evergreen Broadleaf + Mixed Forests
      (2) Savannas + Woody Savannas
      (3) Evergreen Needleleaf + Grasslands + Open Shrublands
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    biome_groups: dict[str, list[str]] = {
        "Group 1 MTBS Fire Locations": [
            "Deciduous Broadleaf forest",
            "Evergreen Broadleaf forest",
            "Mixed forest",
        ],
        "Group 2 MTBS Fire Locations": ["Savannas", "Woody savannas"],
        "Group 3 MTBS Fire Locations": [
            "Evergreen Needleleaf forest",
            "Grasslands",
            "Open shrublands",
        ],
    }

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

    df = mtbs_classified.copy()
    df = df[df["modis_class_static"].notna()].copy()
    df = df[df["LATITUDE"].apply(np.isfinite) & df["LONGITUDE"].apply(np.isfinite)]

    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE), crs="EPSG:4326")

    df = df[df.geometry.notna() & (~df.geometry.is_empty) & df.geometry.is_valid].copy()
    if df.empty:
        print("No valid fire locations found for plotting.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": ccrs.PlateCarree()})

    for ax, (title, classes) in zip(axes, biome_groups.items()):
        subset = df[df["modis_class_static"].isin(classes)].copy()
        if subset.empty:
            ax.set_title(f"{title}\n(no data)")
            ax.set_extent([-170, -50, 15, 75])
            continue

        ax.add_feature(cfeature.LAND, facecolor="lightgrey")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)

        for biome, group in subset.groupby("modis_class_static"):
            ax.scatter(
                group["LONGITUDE"],
                group["LATITUDE"],
                color=biome_colors.get(biome, "black"),
                label=biome,
                s=15,
                alpha=0.6,
                transform=ccrs.PlateCarree(),
            )

        try:
            minx, miny, maxx, maxy = subset.total_bounds
            pad_x = max((maxx - minx) * 0.05, 5)
            pad_y = max((maxy - miny) * 0.05, 5)
            ax.set_extent([minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y], crs=ccrs.PlateCarree())
        except Exception:
            ax.set_extent([-170, -50, 15, 75], crs=ccrs.PlateCarree())

        ax.set_title(title, fontsize=13, pad=10)
        ax.legend(title="Biome", loc="upper right", frameon=True, fontsize=8)
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.3)
        gl.top_labels = gl.right_labels = False

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

    data = df.copy()
    data = data[pd.to_numeric(data[year_col], errors="coerce").notna()]
    data[year_col] = data[year_col].astype(int)
    data = data[data[class_col].notna()]
    data = data[data[class_col].isin(valid_classes)]

    if data.empty:
        raise ValueError("No valid rows after filtering by years and classes.")

    ymin = data[year_col].min() if start_year is None else int(start_year)
    ymax = data[year_col].max() if end_year is None else int(end_year)

    # Drop incomplete final bin to ensure all bins have equal length
    max_complete_year = (ymax // period_years) * period_years + (period_years - 1)
    if max_complete_year > ymax:
        max_complete_year -= period_years
    data = data[(data[year_col] >= ymin) & (data[year_col] <= max_complete_year)]

    def period_start(y: int) -> int:
        offset = (y - ymin) // period_years
        return ymin + offset * period_years

    data["_period_start"] = data[year_col].apply(period_start)
    data["_period_end"] = data["_period_start"] + period_years - 1
    data["_period_label"] = data["_period_start"].astype(str) + "–" + data["_period_end"].astype(str)

    counts = (
        data.groupby(["_period_start", "_period_end", "_period_label", class_col], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["_period_start", class_col])
    )

    counts_wide = counts.pivot_table(
        index="_period_label", columns=class_col, values="count", aggfunc="sum", fill_value=0
    ).reindex(columns=sorted(valid_classes), fill_value=0)

    if include_total:
        counts_wide["Total"] = counts_wide.sum(axis=1)

    counts_long = counts.rename(
        columns={"_period_start": "period_start", "_period_end": "period_end", "_period_label": "period_label"}
    ).reset_index(drop=True)

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
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    valid_classes = [
        "Deciduous Broadleaf forest",
        "Evergreen Broadleaf forest",
        "Mixed forest",
        "Evergreen Needleleaf forest",
        "Open shrublands",
        "Grasslands",
    ]
    df = df[df[biome_col].isin(valid_classes)].copy()

    # Clean
    df = df[[biome_col, year_col, area_col]].dropna()
    df = df[df[area_col] > 0]

    # grid of thresholds
    amax = df[area_col].max()
    thresholds = np.logspace(np.log10(4), np.log10(amax), num=100)

    biomes = sorted(df[biome_col].unique())
    all_results: list[pd.DataFrame] = []
    skipped: list[tuple[str, int]] = []

    ncols = 3
    nrows = int(np.ceil(len(biomes) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.8 * ncols, 3.6 * nrows), sharey=True)
    axes = axes.flatten()

    for i, biome in enumerate(biomes):
        sub = df[df[biome_col] == biome]
        n_years = sub[year_col].nunique()
        if n_years < min_years:
            skipped.append((biome, n_years))
            continue

        trends: list[dict] = []
        for Amin in thresholds:
            yearly_counts = (
                sub.loc[sub[area_col] >= Amin].groupby(year_col).size().reset_index(name="count")
            )
            if yearly_counts["count"].sum() == 0:
                continue

            yearly_counts["year_c"] = yearly_counts[year_col] - yearly_counts[year_col].mean()
            X = sm.add_constant(yearly_counts["year_c"])
            model = sm.GLM(yearly_counts["count"], X, family=sm.families.Poisson())
            res = model.fit()

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

        if not trends:
            skipped.append((biome, n_years))
            continue

        df_trend = pd.DataFrame(trends)
        all_results.append(df_trend)

        ax = axes[i]
        sig = df_trend["pval"] < 0.05

        ax.errorbar(df_trend["Amin_km2"], df_trend["beta1"], yerr=df_trend["beta1_se"], fmt="o-", capsize=3, color="C0", alpha=0.8)
        ax.scatter(df_trend.loc[sig, "Amin_km2"], df_trend.loc[sig, "beta1"], color="red", s=25, label="p<0.05")
        ax.axhline(0, color="k", lw=1, ls="--", alpha=0.6)
        ax.set_xscale("log")

        # Adjust x-limits based on data range
        ax.set_xlim(4, df_trend["Amin_km2"].max())
        ax.set_title(biome, fontsize=11)
        ax.grid(True, alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel("Trend β₁ (log-rate per year)")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Minimum fire size (km²)")

    # remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Poisson Trend in Fire Frequency vs. Size Threshold by Biome", fontsize=14, y=1.02)
    fig.tight_layout()
    plt.show()

    results_all = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    if skipped:
        skipped_df = pd.DataFrame(skipped, columns=["biome", "n_unique_years"])
        print("⚠️ Skipped due to too few unique years:")
        try:
            # in notebooks this would display; keep print as fallback
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
    subset = gfa_df[gfa_df["modis_class_static"] == modis_class]
    if subset.empty:
        print(f"No fires found for MODIS class: {modis_class}")
        return None

    data = subset["area_km2"].dropna().to_numpy()
    data = data[data >= xmin]
    if len(data) == 0:
        print(f"No valid area data above xmin={xmin} for {modis_class}")
        return None

    data = np.sort(data)
    n = len(data)
    empirical_ccdf = 1 - np.arange(1, n + 1) / (n + 1)
    ymin = max(np.min(empirical_ccdf), 1e-5)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax_ccdf, ax_cdf, ax_qq = axes

    # (1) CCDF
    ax_ccdf = wfpl.plot_ccdf_with_selected_fits(data, xmin=xmin, which=which, ax=ax_ccdf)  # type: ignore[name-defined]
    ax_ccdf.set_ylim(ymin, 1)
    ax_ccdf.set_title(f"{modis_class}: Fire Size CCDF")
    ax_ccdf.set_xlabel("Fire size (km²)")
    ax_ccdf.set_ylabel("CCDF")

    # (2) CDF (log-x)
    try:
        fit = wfpl.Fit(data, xmin=xmin, discrete=False)  # type: ignore[name-defined]
        fit.plot_cdf(color="k", linewidth=2, ax=ax_cdf, label="Empirical data")
        for name in which:
            try:
                dist = getattr(fit, name)
                dist.plot_cdf(ax=ax_cdf, linestyle="--", label=f"{name.replace('_',' ').title()} fit")
            except Exception as e:
                print(f"⚠️ Skipped {name} in CDF: {e}")
        ax_cdf.set_xscale("log")
        ax_cdf.set_ylim(0, 1)
        ax_cdf.set_xlabel("Fire size (km²)")
        ax_cdf.set_ylabel("CDF")
        ax_cdf.set_title(f"{modis_class}: Fire Size CDF")
        ax_cdf.legend(fontsize=8)
    except Exception as e:
        print(f"⚠️ Could not plot CDFs: {e}")

    # (3) Q–Q
    try:
        dist_name = which[0]
        fit = wfpl.Fit(data, xmin=xmin, discrete=False)  # type: ignore[name-defined]
        model = getattr(fit, dist_name)

        empirical_q = np.sort(data)
        probs = (np.arange(1, len(empirical_q) + 1) - 0.5) / len(empirical_q)
        if hasattr(model, "ppf") and callable(model.ppf):
            theoretical_q = model.ppf(probs)
        else:
            # numeric inversion from CDF for robustness
            xgrid = np.logspace(np.log10(data.min()), np.log10(data.max()), 4000)
            try:
                cdf_vals = model.cdf(xgrid)
                if np.ndim(cdf_vals) > 1:
                    cdf_vals = np.asarray(cdf_vals).squeeze()
            except Exception:
                cdf_vals = np.array([float(model.cdf(float(x))) for x in xgrid])

            cdf_vals = np.clip(cdf_vals, 1e-12, 1 - 1e-12)
            cdf_vals = np.maximum.accumulate(cdf_vals)  # enforce monotonicity
            theoretical_q = np.interp(probs, cdf_vals, xgrid)

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
        print(f"⚠️ Could not generate Q–Q plot: {e}")

    plt.tight_layout()
    save_path = f"{save_dir}/{modis_class.replace(' ', '_').lower()}_cdf_ccdf_qq.png"
    plt.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
    print(f"✅ Figure saved to {save_path}")
    plt.show()
    return fig


from matplotlib import cm  # for the next function's colorbar


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
    # lookup maps for confirming significance in both combined and single-parameter fits
    sig_lookup_p1: dict[tuple[str, str], bool] = {}
    sig_lookup_p2: dict[tuple[str, str], bool] = {}
    if df_p1 is not None:
        for _, r in df_p1.iterrows():
            sig_lookup_p1[(r["biome"], r["distribution"])] = int(r.get("p1_slope_sig", 0)) == 1
    if df_p2 is not None:
        for _, r in df_p2.iterrows():
            sig_lookup_p2[(r["biome"], r["distribution"])] = int(r.get("p2_slope_sig", 0)) == 1

    # per-biome year ranges for coloring timetable
    year_summary = (
        mtbs_classified.dropna(subset=[biome_col, year_col]).groupby(biome_col)[year_col].agg(["min", "max"]).to_dict("index")
    )

    x = np.logspace(np.log10(xmin), x_max_log10, 500)
    cmap = cm.viridis

    for _, row in df_both.iterrows():
        biome = row["biome"]
        dist = row["distribution"]

        if biome not in year_summary:
            print(f"⚠️ Skipping {biome} — no year data.")
            continue
        y0, y1 = int(year_summary[biome]["min"]), int(year_summary[biome]["max"])
        if y0 >= y1:
            print(f"⚠️ Skipping {biome} — only one year.")
            continue

        years = np.linspace(y0, y1, 5)
        year_center = np.mean(years)
        norm = plt.Normalize(vmin=y0, vmax=y1)

        key = (biome, dist)
        sig_p1_both = int(row.get("p1_slope_sig", 0)) == 1 and sig_lookup_p1.get(key, False)
        sig_p2_both = int(row.get("p2_slope_sig", 0)) == 1 and sig_lookup_p2.get(key, False)

        def p1(year: float) -> float:
            base = float(row["p1"])
            slope = float(row.get("p1'", 0) or 0)
            if not sig_p1_both or not np.isfinite(slope):
                return base
            return base + slope * ((year - year_center) / trend_unit_years)

        def p2(year: float) -> float | None:
            if not np.isfinite(row.get("p2", np.nan)):
                return None
            base = float(row["p2"])
            slope = float(row.get("p2'", 0) or 0)
            if not sig_p2_both or not np.isfinite(slope):
                return base
            return base + slope * ((year - year_center) / trend_unit_years)

        fig, (ax_ccdf, ax_pdf) = plt.subplots(1, 2, figsize=(11, 4))
        good = False

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

                ax_ccdf.plot(x, y_ccdf, color=cmap(norm(yr)), lw=1.5, label=f"{int(yr)}")
                ax_pdf.plot(x, y_pdf, color=cmap(norm(yr)), lw=1.5)
                good = True
            except Exception:
                continue

        if not good:
            plt.close()
            print(f"⚠️ Skipped {biome} ({dist}) — invalid CCDF/PDF.")
            continue

        # CCDF panel (log–log)
        ax_ccdf.set_xscale("log")
        ax_ccdf.set_yscale("log")
        ax_ccdf.set_ylim(ccdf_ymin, 1.0)
        ax_ccdf.set_xlabel("Fire size (km²)")
        ax_ccdf.set_ylabel("CCDF")
        ax_ccdf.set_title(f"{biome}\n{dist.replace('_',' ').title()} (CCDF)")
        ax_ccdf.grid(True, alpha=0.3)

        # PDF panel (log x)
        ax_pdf.set_xscale("log")
        ax_pdf.set_xlabel("Fire size (km²)")
        ax_pdf.set_ylabel("PDF")
        ax_pdf.set_title(f"{biome}\n{dist.replace('_',' ').title()} (PDF)")
        ax_pdf.grid(True, alpha=0.3)

        # Colorbar by year
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cbar.set_label("Year", rotation=270, labelpad=15)
        plt.show()
