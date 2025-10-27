import numpy as np
import geopandas as gpd
import pandas as pd
import wildfire_powerlaw as wfpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry import box
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
from matplotlib.colors import LogNorm
from matplotlib.patches import Wedge
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Ellipse
import matplotlib as mpl
from matplotlib.gridspec import GridSpec


def analyze_gfa_by_grid(gfa_all, xmin=4, R=100, min_n=400, random_state=42):
    """Fit distributions in 5°×5° lat–lon tiles by biome.

    Converts input GFA geometries to EPSG:4326, computes centroid-based
    5-degree bins, and for each biome×tile with at least `min_n` events,
    fits size distributions using `wfpl`. Stores bootstrap summaries and a
    likelihood-ratio matrix to identify the best-fitting family.

    Args:
        gfa_all (geopandas.GeoDataFrame): Input per-fire GeoDataFrame. Must
            have columns: ``geometry``, ``area_km2``, and ``landcover_s``.
        xmin (float, optional): Lower cutoff (area in km²) for fitting tails.
            Defaults to 4.
        R (int, optional): Number of bootstrap replicates. Defaults to 100.
        min_n (int, optional): Minimum number of fires required in a tile to
            attempt fitting. Defaults to 400.
        random_state (int, optional): Random seed for reproducibility for wfpl
            bootstrap. Defaults to 42.

    Returns:
        dict: Nested structure with per-biome results. The mapping is:
            ``results[biome][(lat_bin, lon_bin)] = { "params": DataFrame,
                                                     "likelihood_matrix": DataFrame,
                                                     "best_fit": str,
                                                     "n": int }``.
            - ``params``: wfpl bootstrap summary (index = distribution names).
            - ``likelihood_matrix``: per-distribution Δlog-likelihoods vs others.
            - ``best_fit``: distribution label (string) from wfpl.
            - ``n``: number of observations used in the tile.

    Notes:
        - Tiles are defined by flooring centroid lat/lon to 5-degree steps.
        - Empty or < `min_n` tiles are skipped.
    """
    # Work in geographic CRS for consistent lat/lon binning
    gfa_latlon = gfa_all.to_crs("EPSG:4326").copy()

    # Centroids (safe as scalar points for binning; geometry remains intact)
    gfa_latlon["lat"] = gfa_latlon.geometry.centroid.y
    gfa_latlon["lon"] = gfa_latlon.geometry.centroid.x

    # 5-degree bins (floored)
    gfa_latlon["lat_bin"] = np.floor(gfa_latlon["lat"] / 5) * 5
    gfa_latlon["lon_bin"] = np.floor(gfa_latlon["lon"] / 5) * 5

    results = {}

    # Fit per biome → per tile
    for biome, biome_df in gfa_latlon.groupby("landcover_s"):
        biome_results = {}

        for (lat_bin, lon_bin), box_df in biome_df.groupby(["lat_bin", "lon_bin"]):
            if len(box_df) < min_n:
                # Skip tiles with insufficient sample size
                continue

            data = box_df["area_km2"].dropna().values
            if len(data) == 0:
                continue

            print(f"\n=== {biome} @ box ({lat_bin}, {lon_bin}) (n={len(data)}) ===")

            # Bootstrap parameter summary & likelihood matrix (relative comparison)
            params = wfpl.summarize_parameters_bootstrap(
                data, R=R, xmin=xmin, random_state=random_state
            )
            Rmat, best = wfpl.likelihood_matrix_and_best(data, xmin=xmin)

            biome_results[(lat_bin, lon_bin)] = {
                "params": params,
                "likelihood_matrix": Rmat,
                "best_fit": best,
                "n": len(data),
            }

        if biome_results:
            results[biome] = biome_results

    return results


def filter_best_fits(results_gfa, llhr_cutoff=2.0):
    """Filter fitted results to retain high-quality, non-redundant fits.

    Applies several criteria to each distribution fit in every biome×tile:
      1) Δlog-likelihood threshold: keep fits that are not strongly rejected
         (i.e., at least one pairwise comparison ≤ `llhr_cutoff`).
      2) Exclude reductions: skip fits that reduce to a simpler family
         (`reduces_to` non-null).
      3) Relative error bounds: reject if max relative error on (p1, p2)
         exceeds 1.0 (with log/absolute handling for certain families).
      4) Parameter sanity checks: exclude implausible/out-of-range estimates.

    Args:
        results_gfa (dict): Output from :func:`analyze_gfa_by_grid`.
        llhr_cutoff (float, optional): Δlog-likelihood cutoff; entries whose
            minimum row value exceeds this are dropped. Defaults to 2.0.

    Returns:
        dict: Filtered nested mapping like input structure, with each tile
            keeping:
              ``{ "best_fits": list[str], "params": DataFrame, "n": int }``
            where ``best_fits`` are distribution names that passed filters.

    Notes:
        - Relative-error handling uses absolute/log-based ratios depending
          on family parameterization (e.g., lognormal).
        - This function *excludes* reductions (i.e., skips fits with a
          non-null ``reduces_to``).
    """
    filtered = {}

    for biome, tiles in results_gfa.items():
        biome_filtered = {}

        for tile, res in tiles.items():
            params = res["params"]
            llhr = res["likelihood_matrix"]
            good_dists = []

            # Evaluate each fitted distribution
            for dist, row in params.iterrows():
                p1, p2 = row.get("p1", np.nan), row.get("p2", np.nan)
                p1_se, p2_se = row.get("p1_se", np.nan), row.get("p2_se", np.nan)
                reduces_to = row.get("reduces_to", np.nan)

                # Skip fits that reduce to simpler families
                if pd.notna(reduces_to):
                    continue

                # Δlog-likelihood cutoff (retain only if at least one comparison is small)
                if dist in llhr.index and llhr.loc[dist].min() > llhr_cutoff:
                    continue

                # Compute relative errors with special cases handled in log space
                rel_err_ok = True
                rel_err_p1 = np.nan
                rel_err_p2 = np.nan

                # p1 relative error
                if np.isfinite(p1) and np.isfinite(p1_se) and p1 != 0:
                    if dist in ("lognormal", "lognormal_excess", "stretched_exponential"):
                        rel_err_p1 = abs(p1_se / abs(p1))
                    else:
                        rel_err_p1 = abs(p1_se / p1)

                # p2 relative error
                if np.isfinite(p2) and np.isfinite(p2_se) and p2 != 0:
                    if dist in ("lognormal", "lognormal_excess", "weibull", "weibull_excess"):
                        rel_err_p2 = abs(p2_se / abs(p2))
                    else:
                        rel_err_p2 = abs(p2_se / p2)

                # Reject if either parameter’s relative error > 1
                if ((not np.isnan(rel_err_p1) and rel_err_p1 > 1.0) or
                        (not np.isnan(rel_err_p2) and rel_err_p2 > 1.0)):
                    rel_err_ok = False

                if not rel_err_ok:
                    continue

                # Coarse parameter sanity checks by family
                outlier = False
                if dist in ("generalized_pareto",):
                    if (p1 < -10) or (p1 > 10) or (p2 <= 0):
                        outlier = True
                elif dist in ("lognormal", "lognormal_excess"):
                    if (p2 <= 0) or (p2 > 10) or (p1 <= -2.5):
                        outlier = True
                elif dist in ("power_law", "truncated_power_law"):
                    if p1 <= 1.05:
                        outlier = True
                elif dist in ("weibull", "weibull_excess", "stretched_exponential"):
                    if (p1 <= 0) or (p2 <= 0):
                        outlier = True
                elif dist == "exponential":
                    if p1 <= 0:
                        outlier = True

                if outlier:
                    continue

                good_dists.append(dist)

            if good_dists:
                biome_filtered[tile] = {
                    "best_fits": good_dists,
                    "params": params.loc[good_dists],
                    "n": res["n"],
                }

        if biome_filtered:
            filtered[biome] = biome_filtered

    return filtered


def filter_best_fits_include_reductions(results_gfa, llhr_cutoff=2.0):
    """Filter high-quality fits but **keep** reductions (no `reduces_to` filter).

    Same as :func:`filter_best_fits` except it does not drop distributions
    that reduce to simpler families. This is useful for diagnostic workflows
    where you want to see the *full* set of viable fits, including those that
    wfpl flags as reducible.

    Args:
        results_gfa (dict): Output from :func:`analyze_gfa_by_grid`.
        llhr_cutoff (float, optional): Δlog-likelihood cutoff; entries whose
            minimum row value exceeds this are dropped. Defaults to 2.0.

    Returns:
        dict: Filtered nested mapping like input structure, with each tile
            keeping:
              ``{ "best_fits": list[str], "params": DataFrame, "n": int }``.

    Notes:
        - Relative error thresholds and parameter sanity checks are applied
          identically to :func:`filter_best_fits`.
        - Reductions are **not** excluded in this variant.
    """
    filtered = {}

    for biome, tiles in results_gfa.items():
        biome_filtered = {}

        for tile, res in tiles.items():
            params = res["params"]
            llhr = res["likelihood_matrix"]
            good_dists = []

            for dist, row in params.iterrows():
                p1, p2 = row.get("p1", np.nan), row.get("p2", np.nan)
                p1_se, p2_se = row.get("p1_se", np.nan), row.get("p2_se", np.nan)

                # Δlog-likelihood cutoff
                if dist in llhr.index and llhr.loc[dist].min() > llhr_cutoff:
                    continue

                # Relative error handling
                rel_err_ok = True
                if all(np.isfinite([p1, p1_se])) and p1 != 0:
                    if dist in ("lognormal", "lognormal_excess", "stretched_exponential"):
                        # Robust interpretation for wide-scale parameters
                        rel_err_p1 = p1_se / abs(p1)
                    else:
                        rel_err_p1 = abs(p1_se / p1)
                else:
                    rel_err_p1 = np.nan

                if all(np.isfinite([p2, p2_se])) and p2 != 0:
                    if dist in ("lognormal", "lognormal_excess", "weibull", "weibull_excess"):
                        rel_err_p2 = abs(p2_se / abs(p2))
                    else:
                        rel_err_p2 = abs(p2_se / p2)
                else:
                    rel_err_p2 = np.nan

                if ((not np.isnan(rel_err_p1) and rel_err_p1 > 1.0) or
                        (not np.isnan(rel_err_p2) and rel_err_p2 > 1.0)):
                    rel_err_ok = False

                if not rel_err_ok:
                    continue

                # Parameter sanity checks by family
                outlier = False
                if dist in ("generalized_pareto",):
                    if (p1 < -10) or (p1 > 10) or (p2 <= 0):
                        outlier = True
                elif dist in ("lognormal", "lognormal_excess"):
                    if (p2 <= 0) or (p2 > 10) or (p1 <= -2.5):
                        outlier = True
                elif dist in ("power_law", "truncated_power_law"):
                    if p1 <= 1.05:
                        outlier = True
                elif dist in ("weibull", "weibull_excess", "stretched_exponential"):
                    if (p1 <= 0) or (p2 <= 0):
                        outlier = True
                elif dist == "exponential":
                    if p1 <= 0:
                        outlier = True

                if outlier:
                    continue

                good_dists.append(dist)

            if good_dists:
                biome_filtered[tile] = {
                    "best_fits": good_dists,
                    "params": params.loc[good_dists],
                    "n": res["n"],
                }

        if biome_filtered:
            filtered[biome] = biome_filtered

    return filtered


def plot_distribution_fractions_cells(best_fits_gfa):
    """Plot percent of tiles per biome that accept each distribution.

    Builds a grouped bar chart (one group per biome) showing the fraction
    of 5°×5° tiles where each distribution is included among the accepted
    “best fits”. A tile can count toward multiple distributions.

    Args:
        best_fits_gfa (dict): Output from :func:`filter_best_fits` (or the
            reductions-including variant).

    Returns:
        pandas.DataFrame: Pivot table (index: biome, columns: distribution,
            values: percent of cells).

    Notes:
        - Biomes are labeled with total tile counts to aid interpretation.
        - Returns the pivot for downstream table/figure export.
    """
    # Flatten tile→distribution membership
    records = []
    biome_totals = {}

    for biome, tiles in best_fits_gfa.items():
        total_cells = len(tiles)
        biome_totals[biome] = total_cells
        for (lat_bin, lon_bin), res in tiles.items():
            for dist in res.get("best_fits", []):
                records.append(
                    {
                        "biome": biome,
                        "distribution": dist,
                        "cell_id": f"{lat_bin}_{lon_bin}",
                    }
                )

    df = pd.DataFrame(records)

    # Unique tile counts by (biome, distribution)
    summary = (
        df.groupby(["biome", "distribution"])["cell_id"]
        .nunique()
        .reset_index(name="n_cells")
    )

    # Convert to percentages within each biome
    summary["percent"] = summary.apply(
        lambda row: 100 * row["n_cells"] / biome_totals[row["biome"]],
        axis=1,
    )

    # Pivot to wide format (biomes × distributions)
    pivot = summary.pivot(index="biome", columns="distribution", values="percent").fillna(0)

    # Order biomes by total number of tiles (descending)
    sorted_biomes = sorted(biome_totals.keys(), key=lambda b: biome_totals[b], reverse=True)
    pivot = pivot.loc[sorted_biomes]

    # Label with counts
    new_index = [f"{biome}\n(n={biome_totals[biome]})" for biome in pivot.index]
    pivot.index = new_index

    # Plot
    ax = pivot.plot(kind="bar", stacked=False, figsize=(12, 6))
    plt.ylabel("Percent of Grid Cells (%)")
    plt.title("Best-Fit Distribution Fractions by Biome (by Cell Count)")
    plt.legend(title="Distribution", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return pivot


def plot_parameter_heatmap(best_fits_gfa, biome, distribution, vmin=None, vmax=None):
    """(Deprecated by the presentation variant below) Simple parameter heatmap.

    Creates one or two choropleths (p1, p2) for a chosen biome and
    distribution across 5°×5° tiles. Retained to match original module’s
    behavior; overridden by a later function of the same name.

    Args:
        best_fits_gfa (dict): Filtered results from :func:`filter_best_fits`.
        biome (str): Biome key to visualize.
        distribution (str): Distribution label (e.g., 'truncated_power_law').
        vmin (float, optional): Lower bound for color scale. If None, uses data
            minimum. Defaults to None.
        vmax (float, optional): Upper bound for color scale. If None, uses data
            maximum. Defaults to None.

    Returns:
        None: Displays matplotlib figures.

    Notes:
        - If the specified distribution has both p1 and p2, this draws two
          side-by-side maps.
        - For a more comprehensive figure (with Δ-parameters and scatter),
          prefer the later `plot_parameter_heatmap` (presentation variant).
    """
    # Collect tile polygons plus parameter values
    records = []
    for (lat_bin, lon_bin), res in best_fits_gfa.get(biome, {}).items():
        if distribution not in res["best_fits"]:
            continue
        params = res["params"].loc[distribution]
        geom = box(lon_bin, lat_bin, lon_bin + 5, lat_bin + 5)
        records.append(
            {
                "lat_bin": lat_bin,
                "lon_bin": lon_bin,
                "geometry": geom,
                "p1": params.get("p1"),
                "p2": params.get("p2"),
            }
        )

    if not records:
        print(f"No data for {biome} – {distribution}")
        return

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    # Determine which parameter columns are present and non-empty
    param_cols = [c for c in ["p1", "p2"] if c in gdf.columns and not gdf[c].isna().all()]
    n_params = len(param_cols)

    # Prepare figure
    fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 6), constrained_layout=True)
    if n_params == 1:
        axes = [axes]

    # Draw each parameter’s heatmap
    for ax, p in zip(axes, param_cols):
        gdf.plot(
            column=p,
            cmap="viridis",
            legend=True,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            edgecolor="k",
            linewidth=0.2,
        )
        ax.set_title(f"{biome} – {distribution} ({p})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.show()


def plot_distribution_params_biome(best_fits_gfa, biome, color_by="lat", log_axes=False):
    """Visualize parameter uncertainty ellipses by distribution within one biome.

    For each tile×distribution accepted in ``best_fits_gfa[biome]``, this plots:
      - An error representation (ellipse or x-errorbar) for (p1, p2) using
        bootstrap standard errors.
      - Colors points by either latitude or longitude to reveal spatial
        gradients.

    Args:
        best_fits_gfa (dict): Output of :func:`filter_best_fits`.
        biome (str): Biome key to visualize.
        color_by (str, optional): Which tile coordinate to use for color
            mapping; one of ``{"lat", "lon"}``. Defaults to "lat".
        log_axes (bool, optional): If True, set both axes to log scale for
            distributions where parameters are strictly positive. Defaults to
            False.

    Returns:
        None: Displays a matplotlib figure.

    Notes:
        - For families like 'exponential' and 'power_law' with a single
          parameter, this draws asymmetric x-errorbars for p1 only.
        - For two-parameter families, plots an ellipse with 1-SE radii
          (scaled visually).
    """
    records = []
    # Collect per-tile parameter estimates and SEs across accepted distributions
    for (lat_bin, lon_bin), res in best_fits_gfa.get(biome, {}).items():
        for dist_name in res["best_fits"]:
            row = res["params"].loc[dist_name]
            p1, p1_se = row.get("p1", np.nan), row.get("p1_se", np.nan)
            p2, p2_se = row.get("p2", np.nan), row.get("p2_se", np.nan)

            if np.isnan(p1):
                continue

            records.append(
                {
                    "dist": dist_name,
                    "p1": p1,
                    "p1_se": p1_se,
                    "p2": p2,
                    "p2_se": p2_se,
                    "lat": lat_bin,
                    "lon": lon_bin,
                }
            )

    if not records:
        print(f"No usable fits for biome: {biome}")
        return

    dists = sorted(set(r["dist"] for r in records))
    ncols, nrows = 3, int(np.ceil(len(dists) / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True)
    axes = axes.flatten()

    # Color by requested coordinate (symmetric diverging palette around 0)
    vals = [r[color_by] for r in records]
    vmax = max(abs(min(vals)), abs(max(vals)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r

    # Draw per-distribution panels
    for r in records:
        ax = axes[dists.index(r["dist"])]
        color = cmap(norm(r[color_by]))

        if r["dist"] in ("exponential", "power_law"):
            # Single-parameter families: horizontal error bars on p1
            p1_low = max(r["p1"] - r["p1_se"], 1e-8)
            p1_high = r["p1"] + r["p1_se"]
            ax.errorbar(
                r["p1"],
                0,
                xerr=[[r["p1"] - p1_low], [p1_high - r["p1"]]],
                fmt="o",
                color=color,
            )
        else:
            # Two-parameter families: ellipse representing 1-SE radii
            theta = np.linspace(0, 2 * np.pi, 200)
            x = r["p1"] + r["p1_se"] * np.cos(theta)
            y = r["p2"] + r["p2_se"] * np.sin(theta) if not np.isnan(r["p2"]) else np.zeros_like(theta)
            x = np.clip(x, 0, None)
            y = np.clip(y, 0, None)

            polygon = Polygon(
                np.column_stack([x, y]),
                closed=True,
                facecolor=color,
                edgecolor="none",
                alpha=0.5,
            )
            ax.add_patch(polygon)
            ax.plot(
                r["p1"],
                r["p2"] if not np.isnan(r["p2"]) else 0,
                "o",
                color=color,
                markersize=4,
            )

    # Axis labels/titles and scaling
    for i, dist in enumerate(dists):
        ax = axes[i]
        ax.set_title(dist, fontsize=12)
        ax.set_xlabel("Parameter 1", fontsize=10)
        ax.set_ylabel("Parameter 2", fontsize=10)
        if log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")

    # Remove any unused axes if grid is larger than needed
    for j in range(len(dists), len(axes)):
        fig.delaxes(axes[j])

    # Add global colorbar for the panel grid
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.08)
    cbar.set_label(f"Color by {color_by}")

    fig.suptitle(f"Parameter fits for {biome}", fontsize=16, y=1.02)
    plt.show()


def plot_parameter_heatmap(best_fits_gfa, biome, distribution, vmin=None, vmax=None):
    """Presentation-quality parameter maps and scatter for one biome+family.

    This function **overrides** the earlier simple `plot_parameter_heatmap` by
    design, matching original module behavior. It draws vertically stacked maps
    for p1 and p2 (or 1/λ proxy for TPL), followed by a p1–p2 scatter panel.

    Special handling:
      * Truncated Power Law ('truncated_power_law'):
          - p2 is visualized as 1/λ (with log normalization).
          - p1 colormap flipped so small α is reddish.
      * Lognormal:
          - Filters extreme/invalid p1 where appropriate for readability.

    Args:
        best_fits_gfa (dict): Filtered results from :func:`filter_best_fits`.
        biome (str): Biome key to visualize.
        distribution (str): Distribution label (e.g., 'truncated_power_law').
        vmin (float, optional): Optional lower bound for colormap. Defaults to None.
        vmax (float, optional): Optional upper bound for colormap. Defaults to None.

    Returns:
        None: Displays a composite matplotlib figure.

    Notes:
        - Adds coastlines and borders for geographic context (Cartopy).
        - Reports covariance/correlation between static p1 and p2 (or 1/λ).
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Build tile polygons and extract parameter estimates
    records = []
    for (lat_bin, lon_bin), res in best_fits_gfa.get(biome, {}).items():
        if distribution not in res["best_fits"]:
            continue
        params = res["params"].loc[distribution]
        geom = box(lon_bin, lat_bin, lon_bin + 5, lat_bin + 5)
        records.append(
            {
                "lat_bin": lat_bin,
                "lon_bin": lon_bin,
                "geometry": geom,
                "p1": params.get("p1", np.nan),
                "p2": params.get("p2", np.nan),
            }
        )

    if not records:
        print(f"No data for {biome} – {distribution}")
        return

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    # Colormaps / labels per family
    log_scale_p2 = False
    cmap_p1 = plt.cm.RdYlGn_r
    cmap_p2 = plt.cm.RdYlGn_r
    p2_label = "p₂"
    plot_p2_col = "p2"

    if distribution == "truncated_power_law":
        # For readability, plot 1/λ and use log-scale normalization
        gdf["p2_inv"] = 1 / gdf["p2"].replace(0, np.nan)
        plot_p2_col = "p2_inv"
        log_scale_p2 = True
        p2_label = "log(1/λ)"
        cmap_p1 = plt.cm.RdYlGn  # Flip so small α appears reddish

    if distribution == "lognormal":
        # Drop extreme/invalid lognormal p1 entries for legibility
        before = len(gdf)
        gdf = gdf[gdf["p1"] > 0]
        filtered = before - len(gdf)
        if filtered > 0:
            print(f"⚠️ Filtered {filtered} extreme lognormal p₁ values (< -2.5)")

    # Summary stats for scatter caption
    valid = gdf.dropna(subset=["p1", "p2"])
    if len(valid) > 1:
        cov = np.cov(valid["p1"], valid["p2"])[0, 1]
        corr = np.corrcoef(valid["p1"], valid["p2"])[0, 1]
    else:
        cov = corr = np.nan

    # Figure with two maps + scatter
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 1.8], hspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax3 = fig.add_subplot(gs[2, 0])

    # Map extent with slight padding
    minx, miny, maxx, maxy = gdf.total_bounds
    extent = [minx - 2, maxx + 2, miny - 2, maxy + 2]

    def format_map(ax, title):
        """Add land/coast/border context and annotate axes."""
        ax.add_feature(cfeature.LAND, facecolor="lightgrey", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=0)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=0)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)
        ax.tick_params(labelsize=9)

    # (1) p1 map
    gdf.plot(
        column="p1",
        cmap=cmap_p1,
        ax=ax1,
        edgecolor="k",
        linewidth=0.2,
        alpha=1.0,
        vmin=vmin or np.nanmin(gdf["p1"]),
        vmax=vmax or np.nanmax(gdf["p1"]),
        legend=True,
        legend_kwds={"shrink": 0.6},
    )
    format_map(ax1, f"{biome} – {distribution} (p₁)")

    # (2) p2 (or 1/λ) map
    if log_scale_p2:
        gdf.plot(
            column=plot_p2_col,
            cmap=cmap_p2,
            ax=ax2,
            edgecolor="k",
            linewidth=0.2,
            alpha=1.0,
            norm=LogNorm(
                vmin=vmin or np.nanmin(gdf[plot_p2_col]),
                vmax=vmax or np.nanmax(gdf[plot_p2_col]),
            ),
            legend=True,
            legend_kwds={"shrink": 0.6},
        )
    else:
        gdf.plot(
            column=plot_p2_col,
            cmap=cmap_p2,
            ax=ax2,
            edgecolor="k",
            linewidth=0.2,
            alpha=1.0,
            vmin=vmin or np.nanmin(gdf[plot_p2_col]),
            vmax=vmax or np.nanmax(gdf[plot_p2_col]),
            legend=True,
            legend_kwds={"shrink": 0.6},
        )
    format_map(ax2, f"{biome} – {distribution} ({p2_label})")

    # (3) Scatter with latitude color
    x = valid["p1"].values
    y = valid["p2"].values
    c = valid["lat_bin"].values
    sc = ax3.scatter(x, y, c=c, cmap="viridis", s=40, edgecolor="k", alpha=0.9)
    plt.colorbar(sc, ax=ax3, label="Latitude", shrink=0.7)
    if log_scale_p2:
        ax3.set_yscale("log")
    ax3.set_xlabel("p₁", fontsize=11)
    ax3.set_ylabel(f"{p2_label}", fontsize=11)
    ax3.set_title(f"p₁ vs {p2_label} — ρ = {corr:.2f}, cov = {cov:.2f}", fontsize=12)
    ax3.grid(True, linestyle="--", alpha=0.6, linewidth=0.5)
    ax3.tick_params(labelsize=10)

    plt.suptitle(f"{biome} – {distribution} Parameter Maps", fontsize=16, y=0.98)
    plt.show()


def plot_biome_bestfit_pies(filtered_results, biome_name):
    """Draw multi-segment “pie markers” showing best-fit families per tile.

    Each 5°×5° tile is drawn as a small pie whose slices represent the set
    of accepted best-fitting distributions in that tile.

    Args:
        filtered_results (dict): Output from :func:`filter_best_fits` (or
            :func:`filter_best_fits_include_reductions`).
        biome_name (str): Biome key to visualize.

    Returns:
        geopandas.GeoDataFrame | None: GDF of tile geometries with list-valued
            ``best_fits``; None if there is no data.

    Notes:
        - Colors are hard-coded by family for quick visual scanning.
        - Intended for static map contexts; not CRS-aware rendering.
    """
    # Family → color mapping for pie slices
    dist_to_color = {
        "power_law": "red",
        "truncated_power_law": "blue",
        "weibull": "green",
        "lognormal": "orange",
        "generalized_pareto": "purple",
        "stretched_exponential": "cyan",
        "exponential": "brown",
    }

    biome_tiles = filtered_results.get(biome_name, {})
    if not biome_tiles:
        print(f"No data found for biome: {biome_name}")
        return None

    # Assemble tile geometries and their best-fit sets
    records = []
    for (lat_bin, lon_bin), res in biome_tiles.items():
        best_fits = res.get("best_fits", [])
        if not best_fits:
            continue
        geom = box(lon_bin, lat_bin, lon_bin + 5, lat_bin + 5)
        records.append(
            {
                "lat_bin": lat_bin,
                "lon_bin": lon_bin,
                "best_fits": best_fits,
                "geometry": geom,
            }
        )

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    # Render boundary and overlay pies
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.boundary.plot(ax=ax, color="lightgray", linewidth=0.5)

    for _, row in gdf.iterrows():
        best_fits = row["best_fits"]
        lon_c = row["lon_bin"] + 2.5
        lat_c = row["lat_bin"] + 2.5
        n = len(best_fits)
        if n == 0:
            continue

        # Equal-slice pie for that tile
        start_angle = 0
        for dist in best_fits:
            color = dist_to_color.get(dist, "gray")
            wedge = Wedge(
                center=(lon_c, lat_c),
                r=2.0,
                theta1=start_angle,
                theta2=start_angle + 360 / n,
                facecolor=color,
                edgecolor="k",
                linewidth=0.3,
            )
            ax.add_patch(wedge)
            start_angle += 360 / n

    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Best-Fit Distributions – {biome_name}", fontsize=15)

    # Legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="k",
            markersize=8,
            label=dist,
        )
        for dist, color in dist_to_color.items()
    ]
    ax.legend(
        handles=handles,
        title="Best-fit Distributions",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=False,
    )
    plt.subplots_adjust(bottom=0.15, top=0.95)

    plt.show()
    return gdf


def plot_gp_vs_tpl(filtered_results, biome_name):
    """Map tiles where Generalized Pareto (GP) and/or TPL are accepted.

    Colors:
        - Yellowish (RGB ≈ [1, 1, 0]): generalized_pareto only
        - Blue-ish  (RGB ≈ [0, 0.4, 1]): truncated_power_law only
        - Green-ish (RGB ≈ [0, 0.8, 0]): both GP and TPL

    Args:
        filtered_results (dict): Output of :func:`filter_best_fits`.
        biome_name (str): Biome key to visualize.

    Returns:
        geopandas.GeoDataFrame | None: GDF of tiles with a per-row RGB color,
            or None if no relevant tiles exist.
    """
    # RGB triplets for category colors
    color_gp = np.array([1, 1, 0])     # GP only (yellowish)
    color_tpl = np.array([0, 0.4, 1])  # TPL only (blue-ish)
    color_both = np.array([0, 0.8, 0])  # Both (green-ish)

    biome_tiles = filtered_results.get(biome_name, {})
    if not biome_tiles:
        print(f"No data found for biome: {biome_name}")
        return None

    # Select only tiles where GP/TPL appear among accepted fits
    records = []
    for (lat_bin, lon_bin), res in biome_tiles.items():
        best_fits = res.get("best_fits", [])
        if not best_fits:
            continue

        has_gp = "generalized_pareto" in best_fits
        has_tpl = "truncated_power_law" in best_fits

        if has_gp and has_tpl:
            color = color_both
        elif has_gp:
            color = color_gp
        elif has_tpl:
            color = color_tpl
        else:
            continue

        geom = box(lon_bin, lat_bin, lon_bin + 5, lat_bin + 5)
        records.append(
            {
                "lat_bin": lat_bin,
                "lon_bin": lon_bin,
                "best_fits": best_fits,
                "color": color,
                "geometry": geom,
            }
        )

    if not records:
        print(f"No cells with generalized_pareto or truncated_power_law for {biome_name}.")
        return None

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    # Render colored tiles
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(ax=ax, color=[r["color"] for _, r in gdf.iterrows()], edgecolor="k", linewidth=0.3)

    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{biome_name}: Generalized Pareto vs Truncated Power Law", fontsize=15)

    # Legend patches
    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(color=color_gp, label="Generalized Pareto"),
        mpatches.Patch(color=color_tpl, label="Truncated Power Law"),
        mpatches.Patch(color=color_both, label="Both"),
    ]
    ax.legend(
        handles=handles,
        title="Best-fit Distributions",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
    )

    plt.subplots_adjust(bottom=0.15, top=0.95)
    plt.show()

    return gdf


def make_adaptive_global_bins(
    df,
    xmin=4,
    min_per_bin=400,
    start_year=None,
    end_year=None,
    max_bin_width=2,
    min_valid_fraction=0.75,
):
    """Create adaptive time bins guaranteeing sufficient tail counts.

    Finds the smallest integer bin width (1–10 years) such that at least
    `min_valid_fraction` of bins each contain ≥ `min_per_bin` events with
    area ≥ `xmin`. Only bins meeting the criterion are returned.

    Args:
        df (pandas.DataFrame or geopandas.GeoDataFrame): Must contain columns
            ``'YEAR'`` (int) and ``'area_km2'`` (float).
        xmin (float, optional): Tail threshold (km²). Defaults to 4.
        min_per_bin (int, optional): Minimum number of tail events per bin.
            Defaults to 400.
        start_year (int, optional): If provided, override min year in df.
            Defaults to None (uses data minimum).
        end_year (int, optional): If provided, override max year in df.
            Defaults to None (uses data maximum).
        max_bin_width (int, optional): Max allowed bin width. Defaults to 2.
        min_valid_fraction (float, optional): Target fraction (0–1) of bins
            that must meet the min_per_bin criterion. Defaults to 0.75.

    Returns:
        list[tuple[int, int]] | None: List of (start_year, end_year) bin
            intervals that satisfy the criterion. Returns None if no bin
            width achieves the required fraction.

    Notes:
        - Bins are adjacent, non-overlapping, and cover [start, end].
        - If the minimal width achieving the target fraction is > max_bin_width,
          the function returns None and logs a warning.
    """
    years = np.sort(df["YEAR"].unique())
    start = start_year or years.min()
    end = end_year or years.max()

    # Try widths from 1 to 10 years, stop at the first satisfying the fraction
    for width in range(1, 11):
        bins = [(y, min(y + width - 1, end)) for y in range(start, end + 1, width)]
        counts = [
            np.sum((df["YEAR"].between(y0, y1)) & (df["area_km2"] >= xmin))
            for (y0, y1) in bins
        ]

        valid_mask = np.array(counts) >= min_per_bin
        valid_fraction = np.mean(valid_mask)

        if valid_fraction >= min_valid_fraction:
            if width > max_bin_width:
                print(f"⚠️ Required bin width {width} > {max_bin_width} → insufficient data.")
                return None

            valid_bins = [b for b, ok in zip(bins, valid_mask) if ok]
            print(
                f"Using {width}-year bins ({valid_fraction*100:.0f}% valid; "
                f"≥{min_per_bin} above xmin={xmin}): {valid_bins}"
            )
            return valid_bins

    print(
        f"⚠️ No bin width satisfied ≥{min_valid_fraction*100:.0f}% valid bins with "
        f"≥{min_per_bin} fires above xmin={xmin}."
    )
    return None


def ols_with_se(x, y):
    """Ordinary least squares with standard errors for slope/intercept.

    Fits y = a + b x using scikit-learn’s LinearRegression, then computes
    classical OLS standard errors for the slope and intercept.

    Args:
        x (array-like): 2D array of shape (n, 1) or (n, p) (only 1D used here).
        y (array-like): 1D array of length n.

    Returns:
        tuple: (intercept, slope, se_intercept, se_slope); NaNs returned if
            degenerate cases (n ≤ 2 or Sxx = 0).

    Notes:
        - This is a minimal OLS wrapper used in time-trend summaries.
        - Assumes homoskedastic errors for SE derivation.
    """
    lr = LinearRegression().fit(x, y)
    y_pred = lr.predict(x)
    residuals = y - y_pred
    n = len(y)
    if n <= 2:
        return (np.nan, np.nan, np.nan, np.nan)
    s2 = np.sum(residuals**2) / (n - 2)
    x_mean = np.mean(x)
    Sxx = np.sum((x - x_mean) ** 2)
    if Sxx == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    se_slope = np.sqrt(s2 / Sxx)
    se_intercept = np.sqrt(s2 * (1 / n + x_mean**2 / Sxx))
    return lr.intercept_, lr.coef_[0], se_intercept, se_slope


def analyze_time_varying_fits_grid_single_biome(
    gfa_all,
    results_gfa,
    best_fits_gfa,
    target_biome,
    xmin=4,
    R_boot=100,
    min_per_bin=400,
    target_distribution="truncated_power_law",
    max_bin_width=2,
    min_valid_fraction=0.75,
    max_tiles=None,
):
    """Estimate time trends in parameters per tile for one biome.

    For each 5°×5° tile in the target biome where `target_distribution`
    is accepted as a “good fit” (via `best_fits_gfa`), this function:
      1) Builds adaptive time bins using :func:`make_adaptive_global_bins`
         with >= `min_per_bin` tail events (area ≥ `xmin`).
      2) Re-fits the target distribution within each valid bin using wfpl
         bootstrap summaries.
      3) Fits OLS lines to p1(t) and p2(t), capturing trend slope and SE.

    Args:
        gfa_all (geopandas.GeoDataFrame): Full GFA dataset; must include
            ``'landcover_s'``, area column, and 'YEAR'.
        results_gfa (dict): Output from :func:`analyze_gfa_by_grid`.
        best_fits_gfa (dict): Output from :func:`filter_best_fits`.
        target_biome (str): Biome to analyze.
        xmin (float, optional): Tail threshold (km²). Defaults to 4.
        R_boot (int, optional): wfpl bootstrap replicates per bin. Defaults to 100.
        min_per_bin (int, optional): Minimum tail events per time bin. Defaults to 400.
        target_distribution (str, optional): Distribution family to re-fit
            over time. Defaults to "truncated_power_law".
        max_bin_width (int, optional): Maximum allowed time-bin width (years)
            to accept when building adaptive bins. Defaults to 2.
        min_valid_fraction (float, optional): Required fraction of bins meeting
            the min_per_bin constraint to proceed. Defaults to 0.75.
        max_tiles (int, optional): Optional cap on number of tiles processed
            (useful for testing). Defaults to None.

    Returns:
        dict: Mapping keyed by tile (lat_bin, lon_bin) with:
            {
              "binwise_params": {
                  (y0, y1): {
                      "p1": (estimate, se),
                      "p2": (estimate, se),
                      "n_tail": int
                  }, ...
              },
              "coeffs": {"p1": (intercept, slope), "p2": (intercept, slope)},
              "ses":    {"p1": (se_intercept, se_slope), "p2": (se_intercept, se_slope)}
            }

    Notes:
        - Skips bins with degenerate estimates (e.g., p1 ~ 1 for TPL).
        - Requires at least two valid bins to fit time trends.
        - Lat/lon bins are re-derived from the provided `gfa_all` (EPSG:4326).
    """
    # Restrict to the requested biome (copy to avoid accidental mutation)
    biome_df = gfa_all[gfa_all["landcover_s"] == target_biome].copy()
    if biome_df.empty:
        print(f"⚠️ No data for biome {target_biome}")
        return {}

    best_tiles = best_fits_gfa.get(target_biome, {})
    results_tiles = results_gfa.get(target_biome, {})

    # Valid analysis tiles: target dist accepted, enough tail events, and present in results
    valid_tiles = [
        (lat_bin, lon_bin)
        for (lat_bin, lon_bin), res in best_tiles.items()
        if target_distribution in res.get("best_fits", [])
        and res.get("n", 0) >= min_per_bin
        and (lat_bin, lon_bin) in results_tiles
    ]

    if max_tiles is not None:
        valid_tiles = valid_tiles[:max_tiles]

    print(
        f"Found {len(valid_tiles)} valid 5°×5° tiles for {target_biome} "
        f"where '{target_distribution}' is a good fit"
    )

    time_fits = {}
    skipped_degenerate = 0

    for i, (lat_bin, lon_bin) in enumerate(valid_tiles, 1):
        print(f"\n🔹 Tile {i}/{len(valid_tiles)} → ({lat_bin}, {lon_bin})")

        # Identify this tile’s records by recomputing the 5° binning
        tile_mask = ((np.floor(biome_df["lat"] / 5) * 5 == lat_bin) &
                     (np.floor(biome_df["lon"] / 5) * 5 == lon_bin))
        tile_df = biome_df[tile_mask]
        if tile_df.empty:
            continue

        # Build adaptive bins ensuring sufficient tail counts per bin
        bins = make_adaptive_global_bins(
            tile_df,
            xmin=xmin,
            min_per_bin=min_per_bin,
            max_bin_width=max_bin_width,
            min_valid_fraction=min_valid_fraction,
        )
        if bins is None:
            continue

        # Fit the target family in each valid bin
        binwise = {}
        for (y0, y1) in bins:
            sub = tile_df[(tile_df["YEAR"] >= y0) & (tile_df["YEAR"] <= y1)]
            data = sub["area_km2"].dropna().values
            data_tail = data[data >= xmin]
            if len(data_tail) < min_per_bin:
                continue

            # Bootstrap summary for this bin
            with np.errstate(all="ignore"):
                fit = wfpl.summarize_parameters_bootstrap(data_tail, R=R_boot, xmin=xmin)

            if target_distribution not in fit.index:
                continue
            row = fit.loc[target_distribution]

            # Skip degenerate p1 values (e.g., α ≈ 1 for power-law-like families)
            if np.isnan(row["p1"]) or np.isnan(row["p2"]) or np.isclose(row["p1"], 1.0, atol=0.05):
                skipped_degenerate += 1
                continue

            binwise[(y0, y1)] = {
                "p1": (row["p1"], row["p1_se"]),
                "p2": (row["p2"], row["p2_se"]),
                "n_tail": len(data_tail),
            }

        # Require at least two time bins with valid fits to estimate a trend
        if len(binwise) < 2:
            continue

        # Prepare arrays for OLS
        years = np.array([np.mean(b) for b in binwise.keys()])
        p1_vals = np.array([v["p1"][0] for v in binwise.values()])
        p2_vals = np.array([v["p2"][0] for v in binwise.values()])
        mask = np.isfinite(p1_vals) & np.isfinite(p2_vals)
        if mask.sum() < 2:
            continue

        years = years[mask].reshape(-1, 1)
        p1_vals = p1_vals[mask]
        p2_vals = p2_vals[mask]

        # OLS fits and standard errors
        b1_int, b1_slope, b1_se_int, b1_se_slope = ols_with_se(years, p1_vals)
        b2_int, b2_slope, b2_se_int, b2_se_slope = ols_with_se(years, p2_vals)

        coeffs = {"p1": (b1_int, b1_slope), "p2": (b2_int, b2_slope)}
        ses = {"p1": (b1_se_int, b1_se_slope), "p2": (b2_se_int, b2_se_slope)}

        time_fits[(lat_bin, lon_bin)] = {
            "binwise_params": binwise,
            "coeffs": coeffs,
            "ses": ses,
        }

    print(f"\nCompleted {len(time_fits)} grid-cell time fits for {target_distribution}")
    print(f"Skipped {skipped_degenerate} degenerate bins (p₁≈1) across all tiles.")
    return time_fits


def time_fits_grid_to_df(time_fits):
    """Convert per-tile time-fit summaries to a tidy DataFrame.

    Flattens the dictionary output returned by
    :func:`analyze_time_varying_fits_grid_single_biome` into one row per tile,
    capturing OLS intercepts, slopes, standard errors, and simple significance
    flags for each parameter.

    Args:
        time_fits (dict): Output of
            :func:`analyze_time_varying_fits_grid_single_biome`.

    Returns:
        pandas.DataFrame: Columns include:
            - lat_bin, lon_bin
            - p1_intercept, p1_slope, p1_se_intercept, p1_se_slope, p1_sig
            - p2_intercept, p2_slope, p2_se_intercept, p2_se_slope, p2_sig

    Notes:
        - Significance flag (`*_sig`) is 1 if the 95% CI on the slope excludes 0.
        - CI uses ±1.96 × SE; no multiple testing correction is applied.
    """
    rows = []
    for (lat_bin, lon_bin), vals in time_fits.items():
        row = {"lat_bin": lat_bin, "lon_bin": lon_bin}
        for param in ["p1", "p2"]:
            if param in vals["coeffs"]:
                intercept, slope = vals["coeffs"][param]
                se_int, se_slope = vals["ses"][param]
                row[f"{param}_intercept"] = intercept
                row[f"{param}_slope"] = slope
                row[f"{param}_se_intercept"] = se_int
                row[f"{param}_se_slope"] = se_slope
                if not np.isnan(slope) and not np.isnan(se_slope):
                    lo, hi = slope - 1.96 * se_slope, slope + 1.96 * se_slope
                    row[f"{param}_sig"] = int(not (lo <= 0 <= hi))
                else:
                    row[f"{param}_sig"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_parameter_heatmap(
    best_fits_gfa,
    df_tv,
    biome="Savannas",
    distribution="truncated_power_law",
    vmin=None,
    vmax=None,
):
    """Presentation-quality, multi-panel static + trend maps + scatter.

    Builds a 5-row composite figure for a given biome and distribution:
      1) Static p1 map (or α for TPL)
      2) Static p2 map (or 1/λ for TPL; log-normalized)
      3) Significant Δp1 per decade map (non-significant tiles shaded gray)
      4) Significant Δp2 per decade map (non-significant tiles shaded gray)
      5) p1–p2 (or α–1/λ) scatter colored by latitude, with ρ and cov

    Additional rules:
      - For TPL, skip tiles with static α < 1.05 (unstable/degenerate).
      - Converts slopes from per-year to per-decade by multiplying by 10.
      - Flips the Δp2/Δλ colormap so red = positive changes.

    Args:
        best_fits_gfa (dict): Output of :func:`filter_best_fits`.
        df_tv (pandas.DataFrame): Output of :func:`time_fits_grid_to_df`, or a
            DataFrame with compatible columns, providing slopes and significance.
        biome (str, optional): Biome key to visualize. Defaults to "Savannas".
        distribution (str, optional): Target distribution (e.g., 'truncated_power_law').
            Defaults to "truncated_power_law".
        vmin (float, optional): Optional colormap lower bound. Defaults to None.
        vmax (float, optional): Optional colormap upper bound. Defaults to None.

    Returns:
        None: Saves a high-resolution PNG and displays the figure.

    Notes:
        - This function writes a PNG to ``~/Desktop`` using a name derived
          from biome and distribution (as in the original code).
        - Expects Cartopy data/cache to be available for features.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Assemble static parameters across tiles for the biome/family
    records = []
    for (lat_bin, lon_bin), res in best_fits_gfa.get(biome, {}).items():
        if distribution not in res["best_fits"]:
            continue
        params = res["params"].loc[distribution]

        # Skip degenerate α for TPL
        if distribution == "truncated_power_law" and params.get("p1", np.nan) < 1.05:
            continue

        geom = box(lon_bin, lat_bin, lon_bin + 5, lat_bin + 5)
        records.append(
            {
                "lat_bin": lat_bin,
                "lon_bin": lon_bin,
                "geometry": geom,
                "p1": params.get("p1", np.nan),
                "p2": params.get("p2", np.nan),
            }
        )

    if not records:
        print(f"No data for {biome} – {distribution}")
        return

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    # Align time-varying fits to 5° bins and merge onto static params
    df_tv_aligned = df_tv.copy()
    df_tv_aligned["lat_bin"] = np.floor(df_tv_aligned["lat_bin"] / 5) * 5
    df_tv_aligned["lon_bin"] = np.floor(df_tv_aligned["lon_bin"] / 5) * 5
    df_slopes = df_tv_aligned[
        ["lat_bin", "lon_bin", "p1_slope", "p2_slope", "p1_sig", "p2_sig"]
    ].copy()
    gdf = gdf.merge(df_slopes, on=["lat_bin", "lon_bin"], how="left")

    # Convert per-year slopes to per-decade
    gdf["p1_slope"] = gdf["p1_slope"] * 10
    gdf["p2_slope"] = gdf["p2_slope"] * 10

    print(f"Matched {gdf['p1_slope'].notna().sum()} time-varying fits out of {len(gdf)} tiles.")

    # Family-specific plotting knobs
    if distribution == "truncated_power_law":
        gdf["p2_plot_static"] = 1 / gdf["p2"].replace(0, np.nan)
        plot_p2_col = "p2_plot_static"
        use_log_norm = True
        cmap_p1, cmap_p2 = plt.cm.RdYlGn, plt.cm.RdYlGn_r
        cmap_dp2 = plt.cm.RdYlGn  # flipped (red = positive)
        title_p1, title_p2 = "α", "1/λ"
        title_dp1, title_dp2 = "Δα per decade", "Δλ per decade"
        scatter_ylabel = "1/λ"
    else:
        gdf["p2_plot_static"] = gdf["p2"]
        plot_p2_col = "p2_plot_static"
        use_log_norm = False
        cmap_p1 = cmap_p2 = plt.cm.RdYlGn_r
        cmap_dp2 = plt.cm.RdYlGn_r  # flipped (red = positive)
        title_p1, title_p2 = "p₁", "p₂"
        title_dp1, title_dp2 = "Δp₁ per decade", "Δp₂ per decade"
        scatter_ylabel = "p₂"

    # Stats for scatter caption
    valid = gdf.dropna(subset=["p1", "p2_plot_static"])
    corr = cov = np.nan
    if len(valid) > 1:
        corr = np.corrcoef(valid["p1"], valid["p2_plot_static"])[0, 1]
        cov = np.cov(valid["p1"], valid["p2_plot_static"])[0, 1]

    # Figure with 5 stacked panels
    fig = plt.figure(figsize=(12, 15))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 0.8], hspace=0.25)
    axes = [
        fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree()) if i < 4 else fig.add_subplot(gs[4, 0])
        for i in range(5)
    ]

    # Map extent
    minx, miny, maxx, maxy = gdf.total_bounds
    extent = [minx - 2, maxx + 2, miny - 2, maxy + 2]

    def format_map(ax, title):
        """Common map formatting for the first four panels."""
        ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=0)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=0)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_title(title, fontsize=16, pad=5)
        ax.tick_params(labelsize=10)

    # (1) p1 / α map
    m1 = gdf.plot(
        column="p1",
        cmap=cmap_p1,
        ax=axes[0],
        edgecolor="k",
        linewidth=0.2,
        vmin=vmin or np.nanmin(gdf["p1"]),
        vmax=vmax or np.nanmax(gdf["p1"]),
        legend=False,
    )
    plt.colorbar(m1.collections[0], ax=axes[0], orientation="vertical", fraction=0.046, pad=0.02)
    format_map(axes[0], title_p1)

    # (2) p2 / 1/λ map
    norm = (
        LogNorm(
            vmin=vmin or np.nanmin(gdf[plot_p2_col][gdf[plot_p2_col] > 0]),
            vmax=vmax or np.nanmax(gdf[plot_p2_col]),
        )
        if use_log_norm
        else None
    )
    m2 = gdf.plot(
        column=plot_p2_col,
        cmap=cmap_p2,
        ax=axes[1],
        edgecolor="k",
        linewidth=0.2,
        norm=norm,
        legend=False,
    )
    plt.colorbar(m2.collections[0], ax=axes[1], orientation="vertical", fraction=0.046, pad=0.02)
    format_map(axes[1], title_p2)

    # (3) Δp1 / Δα map (only significant tiles colored)
    gdf["p1_plot"] = np.where(gdf["p1_sig"] == 1, gdf["p1_slope"], np.nan)
    vmax_dp1 = (
        np.nanmax(np.abs(gdf["p1_slope"].dropna())) if np.any(np.isfinite(gdf["p1_slope"])) else 1e-6
    )
    m3 = gdf.plot(
        column="p1_plot",
        cmap=cmap_p1,
        ax=axes[2],
        linewidth=0.3,
        vmin=-vmax_dp1,
        vmax=vmax_dp1,
        legend=False,
        missing_kwds={"color": (0, 0, 0, 0), "edgecolor": None},
    )
    gdf[gdf["p1_sig"] == 0].plot(ax=axes[2], color="grey", linewidth=0.1, alpha=0.9, zorder=1)
    plt.colorbar(m3.collections[0], ax=axes[2], orientation="vertical", fraction=0.046, pad=0.02)
    format_map(axes[2], title_dp1)

    # (4) Δp2 / Δλ map (only significant tiles colored; flipped colormap)
    gdf["p2_plot"] = np.where(gdf["p2_sig"] == 1, gdf["p2_slope"], np.nan)
    vmax_dp2 = (
        np.nanmax(np.abs(gdf["p2_slope"].dropna())) if np.any(np.isfinite(gdf["p2_slope"])) else 1e-6
    )
    m4 = gdf.plot(
        column="p2_plot",
        cmap=cmap_dp2,
        ax=axes[3],
        linewidth=0.3,
        vmin=-vmax_dp2,
        vmax=vmax_dp2,
        legend=False,
        missing_kwds={"color": (0, 0, 0, 0), "edgecolor": None},
    )
    gdf[gdf["p2_sig"] == 0].plot(ax=axes[3], color="grey", linewidth=0.1, alpha=0.9, zorder=1)
    plt.colorbar(m4.collections[0], ax=axes[3], orientation="vertical", fraction=0.046, pad=0.02)
    format_map(axes[3], title_dp2)

    # (5) Scatter
    ax5 = axes[4]
    x, y, c = valid["p1"].values, valid["p2_plot_static"].values, valid["lat_bin"].values
    sc = ax5.scatter(x, y, c=c, cmap="viridis", s=40, edgecolor="k", alpha=0.9)
    plt.colorbar(sc, ax=ax5, label="Latitude", shrink=0.8)
    if use_log_norm:
        ax5.set_yscale("log")
    ax5.set_xlabel(title_p1, fontsize=12)
    ax5.set_ylabel(scatter_ylabel, fontsize=12)
    ax5.set_title(f"{title_p1} vs {scatter_ylabel} — ρ = {corr:.2f}, cov = {cov:.2f}", fontsize=13)
    ax5.grid(True, linestyle="--", alpha=0.6, linewidth=0.5)
    ax5.tick_params(labelsize=10)

    fig.suptitle(
        f"{biome} — {distribution.replace('_', ' ').title()}: Spatiotemporal Variation",
        fontsize=21,
        y=0.98,
    )
    fig.text(0.5, 0.04, "Grey = not significant change", ha="center", fontsize=12, color="dimgray")

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    save_path = f"/Users/lukevonkapff/Desktop/{biome.lower()}_{distribution}_map.png"
    plt.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
    print(f"Figure saved to {save_path}")
    plt.show()


def plot_timevary_param_space_grid(
    time_fits,
    tile_key,
    dist_name="truncated_power_law",
    param_x="p1",
    param_y="p2",
):
    """Plot binwise parameter evolution in (p1, p2) space for one tile.

    Draws:
      - Main panel of ellipses representing 95% (±1.96 SE) around the
        binwise (p1, p2) estimates colored by time (bin midpoint year),
        plus OLS trendline in parameter space.
      - Two side panels showing parameter vs. time with SE errorbars.

    Args:
        time_fits (dict): Output of :func:`analyze_time_varying_fits_grid_single_biome`.
        tile_key (tuple[float, float]): (lat_bin, lon_bin) of the tile to plot.
        dist_name (str, optional): Distribution name for title text. Defaults to
            "truncated_power_law".
        param_x (str, optional): Name of parameter on x-axis. Defaults to "p1".
        param_y (str, optional): Name of parameter on y-axis. Defaults to "p2".

    Returns:
        None: Displays a matplotlib figure.

    Notes:
        - Requires that ``time_fits[tile_key]["binwise_params"]`` exists and
          contains entries for p1 and p2 with SE estimates.
    """
    if tile_key not in time_fits:
        print(f"Tile {tile_key} not found in time_fits.")
        return

    entry = time_fits[tile_key]
    binwise = entry.get("binwise_params", {})
    coeffs = entry.get("coeffs", {})

    # Extract (t, p1, p2, SEs) per bin midpoint
    records = []
    for (y0, y1), params in binwise.items():
        px, px_se = params.get(param_x, (np.nan, np.nan))
        py, py_se = params.get(param_y, (np.nan, np.nan))
        if np.isfinite(px) and np.isfinite(py):
            records.append(
                {
                    "t": 0.5 * (y0 + y1),
                    param_x: px,
                    f"{param_x}_se": px_se,
                    param_y: py,
                    f"{param_y}_se": py_se,
                }
            )

    if not records:
        print(f"No valid bins found for {tile_key}.")
        return

    years = [r["t"] for r in records]
    year_min, year_max = min(years), max(years)

    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=year_min, vmax=year_max)

    # Layout: main parameter-space panel + two small time-series + colorbar
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = GridSpec(2, 3, width_ratios=[2, 1, 0.05], height_ratios=[3, 1], figure=fig)

    ax_main = fig.add_subplot(gs[:, 0])
    ax_p1 = fig.add_subplot(gs[0, 1])
    ax_p2 = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[:, 2])

    # Ellipses (±1.96 SE) colored by time bin midpoint
    for rec in records:
        color = cmap(norm(rec["t"]))
        ell = Ellipse(
            (rec[param_x], rec[param_y]),
            width=2 * 1.96 * rec[f"{param_x}_se"],
            height=2 * 1.96 * rec[f"{param_y}_se"],
            facecolor=color,
            edgecolor="k",
            alpha=0.5,
        )
        ax_main.add_patch(ell)
        ax_main.plot(rec[param_x], rec[param_y], "o", color=color, markersize=4)

    # Trendline in parameter space if both param OLS were computed
    if param_x in coeffs and param_y in coeffs:
        t_range = np.linspace(year_min, year_max, 200)
        x_line = coeffs[param_x][0] + coeffs[param_x][1] * t_range
        y_line = coeffs[param_y][0] + coeffs[param_y][1] * t_range
        ax_main.plot(x_line, y_line, color="red", lw=2.5, label="OLS trend")
        ax_main.legend()

    lat, lon = tile_key
    ax_main.set_xlabel(param_x)
    ax_main.set_ylabel(param_y)
    ax_main.set_title(f"Tile ({lat:.0f}°, {lon:.0f}°) – {dist_name}\nParameter space over time")

    # Side panels: parameter vs time with errorbars and OLS lines (if present)
    times = [r["t"] for r in records]

    ax_p1.errorbar(
        times,
        [r[param_x] for r in records],
        yerr=[r[f"{param_x}_se"] for r in records],
        fmt="o-",
        color="blue",
        alpha=0.7,
    )
    ax_p1.set_title(f"{param_x} over time")
    ax_p1.set_ylabel(param_x)
    if param_x in coeffs:
        ax_p1.plot(t_range, coeffs[param_x][0] + coeffs[param_x][1] * t_range, "r--", lw=2)

    ax_p2.errorbar(
        times,
        [r[param_y] for r in records],
        yerr=[r[f"{param_y}_se"] for r in records],
        fmt="o-",
        color="green",
        alpha=0.7,
    )
    ax_p2.set_title(f"{param_y} over time")
    ax_p2.set_xlabel("Year")
    ax_p2.set_ylabel(param_y)
    if param_y in coeffs:
        ax_p2.plot(t_range, coeffs[param_y][0] + coeffs[param_y][1] * t_range, "r--", lw=2)

    # Colorbar for time encoding
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Bin midpoint year")

    plt.show()
