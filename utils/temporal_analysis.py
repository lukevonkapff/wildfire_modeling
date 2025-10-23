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

def ccdf_exponential(x, lam, xmin=12):
    mask = x >= xmin
    ccdf = np.exp(-lam * (x - xmin))
    ccdf /= max(ccdf[mask][0], 1e-12)
    return np.clip(ccdf, 1e-12, 1.0)


def ccdf_power_law(x, alpha, xmin=12):
    mask = x >= xmin
    ccdf = (x / xmin) ** (1 - alpha)
    ccdf /= max(ccdf[mask][0], 1e-12)
    return np.clip(ccdf, 1e-12, 1.0)


def ccdf_truncated_power_law(x, alpha, lambd, xmin=4):
    """
    Stable truncated power-law CCDF.
    Automatically switches to pure power law when λ is effectively zero.
    """
    x = np.asarray(x, dtype=float)
    mask = x >= xmin
    z = lambd * x[mask]

    # if λ is negligible or α is large -> behaves as pure power law
    if lambd < 1e-3 or np.all(z < 1e-2) or alpha > 5:
        return ccdf_power_law(x, alpha, xmin)

    try:
        val = gammaincc(1 - alpha, z) / gamma(1 - alpha)
        if not np.any(np.isfinite(val)) or np.all(val < 1e-12):
            # fallback if gamma underflows
            return ccdf_power_law(x, alpha, xmin)
        ccdf = np.zeros_like(x)
        ccdf[mask] = np.clip(val, 1e-12, 1.0)
        ccdf /= max(ccdf[mask][0], 1e-12)
    except Exception:
        # fallback in case of numerical failure
        return ccdf_power_law(x, alpha, xmin)

    return np.clip(ccdf, 1e-12, 1.0)


def ccdf_stretched_exponential(x, lam, beta, xmin=12):
    mask = x >= xmin
    ccdf = np.exp(-(lam * x) ** beta + (lam * xmin) ** beta)
    ccdf /= max(ccdf[mask][0], 1e-12)
    return np.clip(ccdf, 1e-12, 1.0)


def ccdf_weibull(x, k, lam, xmin=12):
    mask = x >= xmin
    ccdf = np.exp(-((x / lam) ** k - (xmin / lam) ** k))
    ccdf /= max(ccdf[mask][0], 1e-12)
    return np.clip(ccdf, 1e-12, 1.0)


def ccdf_lognormal(x, mu, sigma, xmin=12):
    mask = x >= xmin
    ccdf = lognorm.sf(x, sigma, scale=np.exp(mu))
    ccdf /= max(ccdf[mask][0], 1e-12)
    return np.clip(ccdf, 1e-12, 1.0)


def ccdf_genpareto(x, xi, sigma, xmin=12):
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

def plot_distribution_evolution_ccdf(df, xmin=12):
    """
    Plot analytical CCDF (P(X >= x)) over relative time for all fitted distributions.
    Degenerate truncated power laws (tiny λ) are plotted as pure power laws.
    """
    x = np.logspace(np.log10(xmin), 3, 500)
    time_steps = np.linspace(-1, 1, 5)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=time_steps.min(), vmax=time_steps.max())

    for _, row in df.iterrows():
        dist = row["distribution"]
        biome = row["biome"]

        def p1(t):
            if row.get("p1_slope_sig", 0) == 0:
                return row["p1"]
            return row["p1"] + row["p1'"] * t

        def p2(t):
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

def summarize_timevary_results_mode(timevary_results, mode="both"):
    """
    Convert one mode ('both', 'p1_only', or 'p2_only') into a DataFrame.

    Columns:
    biome, distribution, n, p1, p1_se, p1', p1'_se, p2, p2_se, p2', p2'_se,
    p1_slope_sig, p2_slope_sig
    """
    rows = []

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

            rows.append({
                "biome": biome, "distribution": dist, "n": n,
                "p1": p1, "p1_se": p1_se, "p1'": p1s, "p1'_se": p1s_se,
                "p2": p2, "p2_se": p2_se, "p2'": p2s, "p2'_se": p2s_se,
                "p1_slope_sig": p1_sig, "p2_slope_sig": p2_sig
            })

    df = pd.DataFrame(rows)
    df = df[[
        "biome", "distribution", "n",
        "p1", "p1_se", "p1'", "p1'_se",
        "p2", "p2_se", "p2'", "p2'_se",
        "p1_slope_sig", "p2_slope_sig"
    ]]
    return df

def analyze_time_varying_mle(
    mtbs_classified, overall_results,
    year_col="year", xmin=4,
    llhr_cutoff=2.0, R_boot=20,
    relerr_cutoff=1.0, min_total=400,
    verbose=True, prior_weight=1e-3
):
    """
    Time-varying MLE with global Differential Evolution optimization.
    Adds stable reparameterization for truncated power-law:
        α(t) = 1 + exp(a1 + b1*t)
        λ(t) = exp(a2 + b2*t)
    Applies weak prior regularization toward the static case for *all* distributions
    (no generic L2 regularization).
    """

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    def logpdf_lognormal(x, mu, sigma, xmin=0):
        x = np.asarray(x)
        valid = x > xmin
        pdf = -np.inf * np.ones_like(x)
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
        pdf = -np.inf * np.ones_like(x)
        alpha = np.clip(alpha, 0.0, 5)
        C = (alpha - 1) / xmin if alpha != 1 else 1 / xmin
        pdf[valid] = np.log(np.abs(C)) - alpha * np.log(x[valid] / xmin)
        return pdf

    def logpdf_trunc_powerlaw(x, alpha, lambd, xmin=1):
        """Properly normalized truncated power-law."""
        x = np.asarray(x)
        valid = x >= xmin
        pdf = -np.inf * np.ones_like(x)
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
            logZ = 0
        pdf[valid] = -alpha * np.log(x[valid]) - lambd * (x[valid] - xmin) - logZ
        return pdf

    def logpdf_genpareto(x, xi, sigma, xmin=0):
        x = np.asarray(x)
        y = x - xmin
        valid = (sigma > 0) & (1 + xi * y / sigma > 0)
        pdf = -np.inf * np.ones_like(x)
        xi = np.clip(xi, -1, 2)
        sigma = np.clip(sigma, 1e-6, 10)
        pdf[valid] = -np.log(sigma) - (1 / xi + 1) * np.log(1 + xi * y[valid] / sigma)
        return pdf

    def logpdf_weibull(x, k, lam, xmin=0):
        x = np.asarray(x)
        valid = x >= xmin
        pdf = -np.inf * np.ones_like(x)
        k = np.clip(k, 1e-6, 50)
        lam = np.clip(lam, 1e-6, 50)
        z = np.clip(x[valid] / lam, 1e-12, 1e6)
        pdf[valid] = np.log(k) - np.log(lam) + (k - 1) * np.log(z) - z ** k
        return pdf

    def logpdf_stretched_exp(x, lam, beta, xmin=0):
        x = np.asarray(x)
        valid = x >= xmin
        pdf = -np.inf * np.ones_like(x)
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

    dist_logpdfs = {
        "lognormal": logpdf_lognormal,
        "power_law": logpdf_powerlaw,
        "truncated_power_law": logpdf_trunc_powerlaw,
        "generalized_pareto": logpdf_genpareto,
        "weibull": logpdf_weibull,
        "stretched_exponential": logpdf_stretched_exp,
    }

    timevary_results = {}
    mtbs_classified = mtbs_classified.copy()
    mtbs_classified["year_c"] = (
        (mtbs_classified[year_col] - mtbs_classified[year_col].mean()) / 10.0
    )

    rng_global = np.random.default_rng(42)

    for biome, subset in mtbs_classified.groupby("modis_cl_1"):
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
            continue

        params_df = res["params"]
        llhr = res["likelihood_matrix"]

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
                print(f"⚠️ No viable candidates for {biome}")
            continue

        biome_res = {}

        for dist_name in candidates:
            fit_modes = {}
            fit_configs = {"both": [True, True], "p1_only": [True, False], "p2_only": [False, True]}

            static_row = params_df.loc[dist_name]
            p1_static = float(static_row.get("p1", 1.0))
            p2_static = float(static_row.get("p2", 1.0)) if not pd.isna(static_row.get("p2", np.nan)) else 1.0

            for mode, (fit_p1, fit_p2) in fit_configs.items():

                def neg_loglik(params, data=data):
                    try:
                        if dist_name == "truncated_power_law":
                            if mode == "both":
                                a1, b1, a2, b2 = params
                            elif mode == "p1_only":
                                a1, b1, a2 = params; b2 = 0.0
                            elif mode == "p2_only":
                                a1, a2, b2 = params; b1 = 0.0
                        else:
                            if mode == "both":
                                a1, b1, a2, b2 = params
                            elif mode == "p1_only":
                                a1, b1, a2 = params; b2 = 0.0
                            elif mode == "p2_only":
                                a1, a2, b2 = params; b1 = 0.0

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

                        if dist_name == "truncated_power_law":
                            prior_center_a1 = np.log(max(p1_static - 1, 1e-3))
                        else:
                            prior_center_a1 = np.log(max(p1_static, 1e-6))
                        prior_center_a2 = np.log(max(p2_static, 1e-6))

                        prior_penalty = prior_weight * (
                            (a1 - prior_center_a1) ** 2 +
                            (a2 - prior_center_a2) ** 2
                        )

                        return -np.sum(valid_ll) + prior_penalty

                    except Exception:
                        return np.inf

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

                if mode == "both":
                    bnds = bounds
                elif mode == "p1_only":
                    bnds = [bounds[0], bounds[1], bounds[2]]
                elif mode == "p2_only":
                    bnds = [bounds[0], bounds[2], bounds[3]]

                opt = differential_evolution(
                    neg_loglik, bounds=bnds,
                    strategy="best1bin", maxiter=600,
                    popsize=15, polish=True,
                    seed=42, updating="deferred",
                    workers=1, init="latinhypercube"
                )
                coeffs = opt.x
                ll_max = -opt.fun

                boot_params = []
                for _ in range(R_boot):
                    idx = rng_global.choice(len(data), size=len(data), replace=True)
                    boot_data = data[idx]
                    try:
                        opt_b = differential_evolution(
                            lambda p: neg_loglik(p, data=boot_data),
                            bounds=bnds, maxiter=300, polish=False,
                            seed=None, updating="deferred", workers=1
                        )
                        boot_params.append(opt_b.x)
                    except Exception:
                        continue

                if dist_name == "truncated_power_law":
                    if mode == "both":
                        a1, b1, a2, b2 = coeffs
                    elif mode == "p1_only":
                        a1, b1, a2 = coeffs; b2 = 0.0
                    elif mode == "p2_only":
                        a1, a2, b2 = coeffs; b1 = 0.0

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
                                ba1, bb1, ba2 = bp; bb2 = 0.0
                            elif mode == "p2_only":
                                ba1, ba2, bb2 = bp; bb1 = 0.0
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

def plot_savanna_fires(mtbs_classified, biome="both"):
    """
    Plot Savanna or Woody Savanna fires on a Cartopy basemap (incl. Alaska).

    Parameters
    ----------
    mtbs_classified : GeoDataFrame
        MTBS fire dataset with columns LATITUDE, LONGITUDE, modis_cl_1, geometry.
    biome : str, optional
        Which biome to plot: "Savannas", "Woody savannas", or "both" (default)
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    valid_biomes = ["Savannas", "Woody savannas", "both"]
    if biome not in valid_biomes:
        raise ValueError(f"biome must be one of {valid_biomes}")

    if biome == "both":
        target_biomes = ["Savannas", "Woody savannas"]
    else:
        target_biomes = [biome]

    subset = mtbs_classified[
        mtbs_classified["modis_cl_1"].isin(target_biomes)
    ].copy()

    subset = subset[
        subset["LATITUDE"].apply(np.isfinite) & subset["LONGITUDE"].apply(np.isfinite)
    ]

    if subset.empty:
        print(f"⚠️ No valid fires found for {', '.join(target_biomes)}.")
        return

    if not isinstance(subset, gpd.GeoDataFrame):
        subset = gpd.GeoDataFrame(
            subset,
            geometry=gpd.points_from_xy(subset.LONGITUDE, subset.LATITUDE),
            crs="EPSG:4326",
        )

    subset = subset[
        subset.geometry.notna() & 
        (~subset.geometry.is_empty) & 
        subset.geometry.is_valid
    ].copy()

    if subset.empty:
        print(f"⚠️ No valid geometries found for {', '.join(target_biomes)}.")
        return

    fig, ax = plt.subplots(
        figsize=(10, 6),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    ax.add_feature(cfeature.LAND, facecolor="lightgrey")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    colors = {"Savannas": "tab:orange", "Woody savannas": "tab:green"}

    for biome_name, group in subset.groupby("modis_cl_1"):
        ax.scatter(
            group["LONGITUDE"],
            group["LATITUDE"],
            color=colors.get(biome_name, "red"),
            label=biome_name,
            s=20,
            alpha=0.6,
            transform=ccrs.PlateCarree(),
        )

    try:
        minx, miny, maxx, maxy = subset.total_bounds
        if np.all(np.isfinite([minx, miny, maxx, maxy])) and (minx < maxx and miny < maxy):
            pad_x = max((maxx - minx) * 0.05, 5)
            pad_y = max((maxy - miny) * 0.05, 5)
            ax.set_extent([minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y],
                          crs=ccrs.PlateCarree())
        else:
            raise ValueError("Invalid bounds")
    except Exception:
        ax.set_extent([-170, -50, 15, 75], crs=ccrs.PlateCarree())

    title_str = (
        "Savanna & Woody Savanna Fires (MTBS)"
        if biome == "both"
        else f"{biome} Fires (MTBS)"
    )
    ax.set_title(title_str, fontsize=14, pad=10)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="Biome", loc="upper right", frameon=True)

    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.3)
    gl.top_labels = gl.right_labels = False

    plt.tight_layout()
    plt.show()

def plot_fire_counts_faceted(mtbs_classified, year_col="year", static_col="modis_cl_1",
                             ncols=3, panel_width=6, panel_height=4):
    """
    Faceted barplots of fire counts per year by MODIS class + overall.

    Parameters
    ----------
    mtbs_classified : GeoDataFrame
        Must have columns [year_col, static_col].
    year_col : str
        Column with year values.
    static_col : str
        Column with MODIS class labels.
    ncols : int
        Number of columns in the facet grid.
    panel_width : float
        Width of each panel in inches.
    panel_height : float
        Height of each panel in inches.
    """
    yearly_counts = mtbs_classified.groupby(year_col).size()

    category_counts = (
        mtbs_classified.groupby([year_col, static_col])
        .size()
        .unstack(fill_value=0)
    )

    categories = category_counts.columns.tolist()
    n_categories = len(categories)
    n_plots = n_categories + 1

    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        sharex=True, sharey=True
    )
    axes = axes.flatten()

    for i, cat in enumerate(categories):
        ax = axes[i]
        category_counts[cat].plot(kind="bar", ax=ax, color="tab:blue", alpha=0.7)
        ax.set_title(cat)
        ax.set_xlabel("")
        ax.set_ylabel("Fires")

    ax = axes[n_categories]
    yearly_counts.plot(kind="bar", ax=ax, color="black", alpha=0.8)
    ax.set_title("Overall Fires")
    ax.set_xlabel("")
    ax.set_ylabel("Fires")

    for j in range(n_categories + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("MTBS Fires by Year and Static MODIS Classification", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()

def fire_threshold_analysis(mtbs_classified, year_col="year", area_col="area_km2",
                            thresholds=[0, 4, 10, 20, 50]):
    """
    Plot yearly fire counts at different minimum fire size thresholds.

    Parameters
    ----------
    mtbs_classified : DataFrame/GeoDataFrame
        Must include [year_col, area_col].
    year_col : str
        Column with fire years.
    area_col : str
        Column with fire size (km²).
    thresholds : list of floats
        Minimum fire size thresholds to test (km²).
    """
    yearly_counts_by_thresh = {}

    for t in thresholds:
        filtered = mtbs_classified[mtbs_classified[area_col] >= t]
        yearly_counts = filtered.groupby(year_col).size()
        yearly_counts_by_thresh[t] = yearly_counts

    counts_df = pd.DataFrame(yearly_counts_by_thresh).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(12, 6))
    for t in thresholds:
        ax.plot(counts_df.index, counts_df[t],
                marker="o", label=f"≥ {t} km²")

    ax.set_title("Yearly Fire Counts by Threshold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Fires")
    ax.legend(title="Min fire size")
    plt.tight_layout()
    plt.show()

def build_static_modis_tiles(modis_by_year, out_dir="static_modis_tiles"):
    """
    Build static MODIS rasters (per tile), by taking pixelwise mode across years.
    Returns dict: {tile_id: out_path}
    """
    os.makedirs(out_dir, exist_ok=True)
    tile_pattern = re.compile(r"h\d{2}v\d{2}")

    tile_files = {}
    for year, files in modis_by_year.items():
        for f in files:
            m = tile_pattern.search(os.path.basename(f))
            if not m:
                continue
            tile_id = m.group()
            tile_files.setdefault(tile_id, []).append(f)

    out_paths = {}

    for tile_id, files in tile_files.items():
        arrays = []
        ref_profile = None

        for f in files:
            with rasterio.open(f) as src:
                if ref_profile is None:
                    ref_profile = src.profile.copy()
                    nodata = ref_profile.get("nodata", 255)
                arr = src.read(1).astype(float)
                arrays.append(arr)

        stack = np.stack(arrays, axis=0)
        stack = np.where((stack == nodata) | (stack == 17), np.nan, stack)

        mode_map, _ = stats.mode(stack, axis=0, nan_policy="omit")
        static_map = np.squeeze(mode_map).astype(np.uint8)
        
        out_path = os.path.join(out_dir, f"static_modis_mode_{tile_id}.tif")
        ref_profile.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=nodata)
        with rasterio.open(out_path, "w", **ref_profile) as dst:
            dst.write(static_map, 1)
        out_paths[tile_id] = out_path

    return out_paths

def classify_with_static(fires_gdf, static_tile_dict, modis_to_gfa):
    """
    Classify fires with static MODIS rasters (per tile).
    
    Parameters
    ----------
    fires_gdf : GeoDataFrame
        Fire perimeters (must have geometry).
    static_tile_dict : dict
        {tile_id: raster_path} from build_static_modis_tiles.
    modis_to_gfa : dict
        Mapping from MODIS class integer -> GFA category (string).
    
    Returns
    -------
    fires_gdf : GeoDataFrame
        Copy with new column 'modis_class_static'.
    """
    results = []
    
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

    fires_gdf = fires_gdf.copy()
    fires_gdf["modis_class_static"] = "Unknown"
    for idx, cls in results:
        fires_gdf.at[idx, "modis_class_static"] = cls
    
    return fires_gdf

def classify_with_modis(fires_gdf, year_col, modis_by_year, modis_to_gfa):
    results = []
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
    fires_gdf = fires_gdf.copy()
    fires_gdf["modis_class_timevary"] = "Unknown"
    for idx, cls in results:
        fires_gdf.at[idx, "modis_class_timevary"] = cls
    return fires_gdf

def load_shapefile(shp_path, projection, area_col=None):
    gdf = gpd.read_file(shp_path)
    gdf = gdf.set_crs(projection)
    gdf = gdf.to_crs("EPSG:6933")
    if area_col is not None and area_col not in gdf.columns:
        gdf[area_col] = gdf.geometry.area / 1e6 
    return gdf