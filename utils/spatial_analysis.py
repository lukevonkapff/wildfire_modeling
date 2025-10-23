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
    """
    Segment globe into 5°x5° lat-lon boxes and fit distributions by landcover_s.
    
    Parameters
    ----------
    gfa_all : GeoDataFrame
        Must have columns: 'geometry', 'area_km2', 'landcover_s'.
    xmin : float
        Minimum area threshold for fitting.
    R : int
        Number of bootstrap reps.
    min_n : int
        Minimum number of fires required in bin to fit.
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    results : dict
        Nested dict: results[landcover_s][(lat_bin, lon_bin)] = {...}
    """
    gfa_latlon = gfa_all.to_crs("EPSG:4326").copy()
    gfa_latlon["lat"] = gfa_latlon.geometry.centroid.y
    gfa_latlon["lon"] = gfa_latlon.geometry.centroid.x
    
    gfa_latlon["lat_bin"] = np.floor(gfa_latlon["lat"] / 5) * 5
    gfa_latlon["lon_bin"] = np.floor(gfa_latlon["lon"] / 5) * 5

    results = {}

    for biome, biome_df in gfa_latlon.groupby("landcover_s"):
        biome_results = {}
        
        for (lat_bin, lon_bin), box_df in biome_df.groupby(["lat_bin", "lon_bin"]):
            if len(box_df) < min_n:
                continue

            data = box_df["area_km2"].dropna().values
            if len(data) == 0:
                continue

            print(f"\n=== {biome} @ box ({lat_bin}, {lon_bin}) (n={len(data)}) ===")

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

def filter_best_fits(results_gfa, llhr_cutoff=2.0, max_relerr=1.0):
    """
    Filter results_gfa to keep only good fits:
    - Δloglikelihood < cutoff
    - no reduces_to (except if GP→PL reduction is not statistically valid)
    - max relative error < threshold
    - parameters within plausible (non-outlier) ranges

    Special handling:
    - If a Generalized Pareto 'reduces_to' Power Law, check whether ξ's 95% CI
      includes 1/(α_PL - 1). Only then is it truly degenerate.
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

                rel1 = abs(p1_se / p1) if p1 and not pd.isna(p1) else np.inf
                rel2 = abs(p2_se / p2) if p2 and not pd.isna(p2) else 0
                if max(rel1, rel2) > max_relerr:
                    continue

                reduces_to = row.get("reduces_to", np.nan)

                if isinstance(reduces_to, str):
                    if dist == "generalized_pareto" and reduces_to == "power_law":
                        if "power_law" in params.index:
                            pl_row = params.loc["power_law"]
                            alpha_pl = pl_row.get("p1", np.nan)
                            if pd.notna(alpha_pl) and alpha_pl > 1:
                                xi_equiv = 1.0 / (alpha_pl - 1.0)
                                ci_low = p1 - 1.96 * p1_se
                                ci_high = p1 + 1.96 * p1_se
                                if ci_low <= xi_equiv <= ci_high:
                                    continue
                                else:
                                    pass
                            else:
                                pass
                        else:
                            pass 
                    else:
                        continue

                if dist in llhr.index and llhr.loc[dist].min() > llhr_cutoff:
                    continue

                outlier = False
                if dist in ("generalized_pareto",):
                    if (p1 < -10) or (p1 > 10) or (p2 <= 0):
                        outlier = True
                elif dist in ("lognormal", "lognormal_excess"):
                    if (p2 <= 0) or (p2 > 10):
                        outlier = True
                elif dist in ("power_law", "truncated_power_law"):
                    if p1 <= 1:
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
                    "n": res["n"]
                }

        if biome_filtered:
            filtered[biome] = biome_filtered

    return filtered

def filter_best_fits_include_reductions(results_gfa, llhr_cutoff=2.0, max_relerr=1.0):
    """
    Filter results_gfa to keep only good fits:
    - Δloglikelihood < cutoff
    - no reduces_to
    - max relative error < threshold
    
    Parameters
    ----------
    results_gfa : dict
        Nested dict from analyze_gfa_by_grid
    llhr_cutoff : float
        Δloglikelihood threshold (default 2.0)
    max_relerr : float
        Maximum allowed relative error (default 1.0)
    
    Returns
    -------
    filtered : dict
        filtered[biome][(lat_bin, lon_bin)] = {
            "best_fits": [list of dist names],
            "params": {dist: df row with parameters},
            "n": sample size
        }
    """
    filtered = {}

    for biome, tiles in results_gfa.items():
        biome_filtered = {}
        for tile, res in tiles.items():
            params = res["params"]
            llhr = res["likelihood_matrix"]
            good_dists = []

            for dist, row in params.iterrows():
                rel1 = abs(row["p1_se"]/row["p1"]) if row["p1"] and not pd.isna(row["p1"]) else np.inf
                rel2 = abs(row["p2_se"]/row["p2"]) if row["p2"] and not pd.isna(row["p2"]) else 0
                if max(rel1, rel2) > max_relerr:
                    continue

                if dist in llhr.index and llhr.loc[dist].min() > llhr_cutoff:
                    continue

                good_dists.append(dist)

            if good_dists:
                biome_filtered[tile] = {
                    "best_fits": good_dists,
                    "params": params.loc[good_dists],
                    "n": res["n"]
                }
        if biome_filtered:
            filtered[biome] = biome_filtered
    return filtered

def plot_distribution_fractions_cells(best_fits_gfa):
    """
    Grouped bar chart: percent of grid cells in each biome fit by each distribution.
    A cell can count toward multiple distributions if more than one is a "best fit".
    Biomes are sorted (descending) by number of cells.
    """
    records = []
    biome_totals = {}

    for biome, tiles in best_fits_gfa.items():
        total_cells = len(tiles)
        biome_totals[biome] = total_cells
        for (lat_bin, lon_bin), res in tiles.items():
            for dist in res.get("best_fits", []):
                records.append({
                    "biome": biome,
                    "distribution": dist,
                    "cell_id": f"{lat_bin}_{lon_bin}"
                })

    df = pd.DataFrame(records)

    summary = (
        df.groupby(["biome", "distribution"])["cell_id"]
        .nunique()
        .reset_index(name="n_cells")
    )

    summary["percent"] = summary.apply(
        lambda row: 100 * row["n_cells"] / biome_totals[row["biome"]],
        axis=1
    )

    pivot = summary.pivot(index="biome", columns="distribution", values="percent").fillna(0)
    sorted_biomes = sorted(biome_totals.keys(), key=lambda b: biome_totals[b], reverse=True)
    pivot = pivot.loc[sorted_biomes]

    new_index = [f"{biome}\n(n={biome_totals[biome]})" for biome in pivot.index]
    pivot.index = new_index

    ax = pivot.plot(kind="bar", stacked=False, figsize=(12, 6))
    plt.ylabel("Percent of Grid Cells (%)")
    plt.title("Best-Fit Distribution Fractions by Biome (by Cell Count)")
    plt.legend(title="Distribution", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return pivot




def plot_parameter_heatmap(best_fits_gfa, biome, distribution, vmin=None, vmax=None):
    """
    Heatmap of parameter values for a given biome + distribution.
    If the distribution has multiple parameters (e.g. p1, p2),
    creates side-by-side subplots.
    """
    records = []
    for (lat_bin, lon_bin), res in best_fits_gfa.get(biome, {}).items():
        if distribution not in res["best_fits"]:
            continue
        params = res["params"].loc[distribution]
        geom = box(lon_bin, lat_bin, lon_bin + 5, lat_bin + 5)
        records.append({
            "lat_bin": lat_bin,
            "lon_bin": lon_bin,
            "geometry": geom,
            "p1": params.get("p1"),
            "p2": params.get("p2")
        })
    
    if not records:
        print(f"No data for {biome} – {distribution}")
        return
    
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    
    param_cols = [c for c in ["p1", "p2"] if c in gdf.columns and not gdf[c].isna().all()]
    n_params = len(param_cols)
    
    fig, axes = plt.subplots(1, n_params, figsize=(6*n_params, 6), constrained_layout=True)
    if n_params == 1:
        axes = [axes]
    
    for ax, p in zip(axes, param_cols):
        gdf.plot(column=p, cmap="viridis", legend=True,
                 vmin=vmin, vmax=vmax, ax=ax, edgecolor="k", linewidth=0.2)
        ax.set_title(f"{biome} – {distribution} ({p})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    
    plt.show()

def plot_distribution_params_biome(best_fits_gfa, biome, color_by="lat",
                                   max_rel_error=1.0, log_axes=False):
    records = []
    for (lat_bin, lon_bin), res in best_fits_gfa.get(biome, {}).items():
        for dist_name in res["best_fits"]:
            row = res["params"].loc[dist_name]
            p1, p1_se = row.get("p1", np.nan), row.get("p1_se", np.nan)
            p2, p2_se = row.get("p2", np.nan), row.get("p2_se", np.nan)

            if np.isnan(p1):
                continue
            if (not np.isnan(p1_se) and abs(p1_se/p1) > max_rel_error) or \
               (not np.isnan(p2) and not np.isnan(p2_se) and abs(p2_se/p2) > max_rel_error):
                continue

            records.append({
                "dist": dist_name,
                "p1": p1, "p1_se": p1_se,
                "p2": p2, "p2_se": p2_se,
                "lat": lat_bin, "lon": lon_bin
            })

    if not records:
        print(f"No usable fits for biome: {biome}")
        return

    dists = sorted(set(r["dist"] for r in records))
    ncols, nrows = 3, int(np.ceil(len(dists) / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             constrained_layout=True)
    axes = axes.flatten()

    vals = [r[color_by] for r in records]
    vmax = max(abs(min(vals)), abs(max(vals)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r

    for r in records:
        ax = axes[dists.index(r["dist"])]
        color = cmap(norm(r[color_by]))

        if r["dist"] in ("exponential", "power_law"):
            p1_low = max(r["p1"] - r["p1_se"], 1e-8)
            p1_high = r["p1"] + r["p1_se"]
            ax.errorbar(
                r["p1"], 0,
                xerr=[[r["p1"] - p1_low], [p1_high - r["p1"]]],
                fmt="o", color=color
            )
        else:
            theta = np.linspace(0, 2 * np.pi, 200)
            x = r["p1"] + r["p1_se"] * np.cos(theta)
            y = r["p2"] + r["p2_se"] * np.sin(theta) if not np.isnan(r["p2"]) else np.zeros_like(theta)
            x = np.clip(x, 0, None)
            y = np.clip(y, 0, None)

            polygon = Polygon(np.column_stack([x, y]), closed=True,
                              facecolor=color, edgecolor="none", alpha=0.5)
            ax.add_patch(polygon)
            ax.plot(r["p1"], r["p2"] if not np.isnan(r["p2"]) else 0,
                    "o", color=color, markersize=4)

    for i, dist in enumerate(dists):
        ax = axes[i]
        ax.set_title(dist, fontsize=12)
        ax.set_xlabel("Parameter 1", fontsize=10)
        ax.set_ylabel("Parameter 2", fontsize=10)
        if log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")

    for j in range(len(dists), len(axes)):
        fig.delaxes(axes[j])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                        fraction=0.05, pad=0.08)
    cbar.set_label(f"Color by {color_by}")

    fig.suptitle(f"Parameter fits for {biome}", fontsize=16, y=1.02)
    plt.show()

def plot_parameter_heatmap(best_fits_gfa, biome, distribution, vmin=None, vmax=None):
    """
    Vertically stacked parameter maps (p1, p2/log(1/p2)) with a single p1–p2 scatter plot at the bottom.
    - Flips p1 color scale for truncated power law (low α = red)
    - Filters only extreme negative p1 values for lognormal (p1 < -2.5)
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    records = []
    for (lat_bin, lon_bin), res in best_fits_gfa.get(biome, {}).items():
        if distribution not in res["best_fits"]:
            continue
        params = res["params"].loc[distribution]
        geom = box(lon_bin, lat_bin, lon_bin + 5, lat_bin + 5)
        records.append({
            "lat_bin": lat_bin,
            "lon_bin": lon_bin,
            "geometry": geom,
            "p1": params.get("p1", np.nan),
            "p2": params.get("p2", np.nan),
        })

    if not records:
        print(f"No data for {biome} – {distribution}")
        return

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    log_scale_p2 = False
    cmap_p1 = plt.cm.RdYlGn_r
    cmap_p2 = plt.cm.RdYlGn_r
    p2_label = "p₂"
    plot_p2_col = "p2"

    if distribution == "truncated_power_law":
        gdf["p2_inv"] = 1 / gdf["p2"].replace(0, np.nan)
        plot_p2_col = "p2_inv"
        log_scale_p2 = True
        p2_label = "log(1/λ)"
        cmap_p1 = plt.cm.RdYlGn 

    if distribution == "lognormal":
        before = len(gdf)
        gdf = gdf[gdf["p1"] > 0]
        filtered = before - len(gdf)
        if filtered > 0:
            print(f"⚠️ Filtered {filtered} extreme lognormal p₁ values (< -2.5)")

    valid = gdf.dropna(subset=["p1", "p2"])
    if len(valid) > 1:
        cov = np.cov(valid["p1"], valid["p2"])[0, 1]
        corr = np.corrcoef(valid["p1"], valid["p2"])[0, 1]
    else:
        cov = corr = np.nan

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 1.8], hspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax3 = fig.add_subplot(gs[2, 0])

    minx, miny, maxx, maxy = gdf.total_bounds
    extent = [minx - 2, maxx + 2, miny - 2, maxy + 2]

    def format_map(ax, title):
        ax.add_feature(cfeature.LAND, facecolor="lightgrey", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=0)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=0)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)
        ax.tick_params(labelsize=9)

    # ---------- P1 MAP ----------
    gdf.plot(
        column="p1", cmap=cmap_p1, ax=ax1, edgecolor="k", linewidth=0.2, alpha=1.0,
        vmin=vmin or np.nanmin(gdf["p1"]), vmax=vmax or np.nanmax(gdf["p1"]),
        legend=True, legend_kwds={'shrink': 0.6}
    )
    format_map(ax1, f"{biome} – {distribution} (p₁)")

    # ---------- P2 MAP ----------
    if log_scale_p2:
        gdf.plot(
            column=plot_p2_col, cmap=cmap_p2, ax=ax2, edgecolor="k", linewidth=0.2, alpha=1.0,
            norm=LogNorm(vmin=vmin or np.nanmin(gdf[plot_p2_col]),
                         vmax=vmax or np.nanmax(gdf[plot_p2_col])),
            legend=True, legend_kwds={'shrink': 0.6}
        )
    else:
        gdf.plot(
            column=plot_p2_col, cmap=cmap_p2, ax=ax2, edgecolor="k", linewidth=0.2, alpha=1.0,
            vmin=vmin or np.nanmin(gdf[plot_p2_col]),
            vmax=vmax or np.nanmax(gdf[plot_p2_col]),
            legend=True, legend_kwds={'shrink': 0.6}
        )
    format_map(ax2, f"{biome} – {distribution} ({p2_label})")

    # ---------- SCATTER ----------
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
    """
    Plot best-fit distributions for a single biome using multi-segment pie markers.
    Each tile is represented by a small pie divided among all its best-fitting distributions.
    """
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

    records = []
    for (lat_bin, lon_bin), res in biome_tiles.items():
        best_fits = res.get("best_fits", [])
        if not best_fits:
            continue
        geom = box(lon_bin, lat_bin, lon_bin + 5, lat_bin + 5)
        records.append({
            "lat_bin": lat_bin,
            "lon_bin": lon_bin,
            "best_fits": best_fits,
            "geometry": geom
        })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")


    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.boundary.plot(ax=ax, color="lightgray", linewidth=0.5)

    for _, row in gdf.iterrows():
        best_fits = row["best_fits"]
        lon_c = row["lon_bin"] + 2.5
        lat_c = row["lat_bin"] + 2.5
        n = len(best_fits)
        if n == 0:
            continue

        start_angle = 0
        for dist in best_fits:
            color = dist_to_color.get(dist, "gray")
            wedge = Wedge(center=(lon_c, lat_c),
                          r=2.0,
                          theta1=start_angle,
                          theta2=start_angle + 360 / n,
                          facecolor=color,
                          edgecolor="k",
                          linewidth=0.3)
            ax.add_patch(wedge)
            start_angle += 360 / n

    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Best-Fit Distributions – {biome_name}", fontsize=15)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color, markeredgecolor="k",
                   markersize=8, label=dist)
        for dist, color in dist_to_color.items()
    ]

    legend = ax.legend(
        handles=handles,
        title="Best-fit Distributions",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=False
    )
    plt.subplots_adjust(bottom=0.15, top=0.95)

    plt.show()
    return gdf

def plot_gp_vs_tpl(filtered_results, biome_name):
    """
    Plot map showing where generalized_pareto and/or truncated_power_law
    are best fits within a single biome.

    - Red  = generalized_pareto only
    - Blue = truncated_power_law only
    - Purple = both
    """
    color_gp = np.array([1, 1, 0])  
    color_tpl = np.array([0, .4, 1])   
    color_both = np.array([0, 0.8, 0]) 

    biome_tiles = filtered_results.get(biome_name, {})
    if not biome_tiles:
        print(f"No data found for biome: {biome_name}")
        return None

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
        records.append({
            "lat_bin": lat_bin,
            "lon_bin": lon_bin,
            "best_fits": best_fits,
            "color": color,
            "geometry": geom
        })

    if not records:
        print(f"No cells with generalized_pareto or truncated_power_law for {biome_name}.")
        return None

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(ax=ax, color=[r["color"] for _, r in gdf.iterrows()],
             edgecolor="k", linewidth=0.3)

    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{biome_name}: Generalized Pareto vs Truncated Power Law", fontsize=15)

    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=color_gp, label="Generalized Pareto"),
        mpatches.Patch(color=color_tpl, label="Truncated Power Law"),
        mpatches.Patch(color=color_both, label="Both")
    ]
    ax.legend(handles=handles, title="Best-fit Distributions",
              loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False)

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
    min_valid_fraction=0.75
):
    """
    Creates adaptive time bins such that at least `min_valid_fraction`
    (default: 75%) of bins contain ≥ min_per_bin *fires above xmin*.
    Returns only the bins that actually meet the min_per_bin criterion.
    """
    years = np.sort(df["YEAR"].unique())
    start = start_year or years.min()
    end = end_year or years.max()

    for width in range(1, 11):
        bins = [(y, min(y + width - 1, end)) for y in range(start, end + 1, width)]
        counts = [
            np.sum(
                (df["YEAR"].between(y0, y1)) &
                (df["area_km2"] >= xmin)
            )
            for (y0, y1) in bins
        ]

        valid_mask = np.array(counts) >= min_per_bin
        valid_fraction = np.mean(valid_mask)

        if valid_fraction >= min_valid_fraction:
            if width > max_bin_width:
                print(f"⚠️ Required bin width {width} > {max_bin_width} → insufficient data.")
                return None

            valid_bins = [b for b, ok in zip(bins, valid_mask) if ok]
            print(f"Using {width}-year bins ({valid_fraction*100:.0f}% valid; ≥{min_per_bin} above xmin={xmin}): {valid_bins}")
            return valid_bins

    print(f"⚠️ No bin width satisfied ≥{min_valid_fraction*100:.0f}% valid bins with ≥{min_per_bin} fires above xmin={xmin}.")
    return None

def ols_with_se(x, y):
    lr = LinearRegression().fit(x, y)
    y_pred = lr.predict(x)
    residuals = y - y_pred
    n = len(y)
    if n <= 2:
        return (np.nan, np.nan, np.nan, np.nan)
    s2 = np.sum(residuals ** 2) / (n - 2)
    x_mean = np.mean(x)
    Sxx = np.sum((x - x_mean) ** 2)
    if Sxx == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    se_slope = np.sqrt(s2 / Sxx)
    se_intercept = np.sqrt(s2 * (1/n + x_mean**2 / Sxx))
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
    max_tiles=None
):
    """
    Runs time-varying fits for each 5°×5° grid cell in a biome,
    restricted to cells where the target_distribution is a *good fit*
    according to best_fits_gfa[biome].

    Binning rule:
        - At least `min_valid_fraction` (default 0.75) of bins must have ≥ min_per_bin fires *above xmin*.
        - Only those valid bins are used for WFPL fitting.
    """
    biome_df = gfa_all[gfa_all["landcover_s"] == target_biome].copy()
    if biome_df.empty:
        print(f"⚠️ No data for biome {target_biome}")
        return {}

    best_tiles = best_fits_gfa.get(target_biome, {})
    results_tiles = results_gfa.get(target_biome, {})

    valid_tiles = [
        (lat_bin, lon_bin)
        for (lat_bin, lon_bin), res in best_tiles.items()
        if target_distribution in res.get("best_fits", [])
        and res.get("n", 0) >= min_per_bin
        and (lat_bin, lon_bin) in results_tiles
    ]

    if max_tiles is not None:
        valid_tiles = valid_tiles[:max_tiles]

    print(f"Found {len(valid_tiles)} valid 5°×5° tiles for {target_biome} "
          f"where '{target_distribution}' is a good fit")

    time_fits = {}
    skipped_degenerate = 0

    for i, (lat_bin, lon_bin) in enumerate(valid_tiles, 1):
        print(f"\n🔹 Tile {i}/{len(valid_tiles)} → ({lat_bin}, {lon_bin})")
        tile_mask = (
            (np.floor(biome_df["lat"] / 5) * 5 == lat_bin)
            & (np.floor(biome_df["lon"] / 5) * 5 == lon_bin)
        )
        tile_df = biome_df[tile_mask]
        if tile_df.empty:
            continue

        bins = make_adaptive_global_bins(
            tile_df,
            xmin=xmin,
            min_per_bin=min_per_bin,
            max_bin_width=max_bin_width,
            min_valid_fraction=min_valid_fraction
        )
        if bins is None:
            continue

        binwise = {}
        for (y0, y1) in bins:
            sub = tile_df[(tile_df["YEAR"] >= y0) & (tile_df["YEAR"] <= y1)]
            data = sub["area_km2"].dropna().values
            data_tail = data[data >= xmin]
            if len(data_tail) < min_per_bin:
                continue

            with np.errstate(all="ignore"):
                fit = wfpl.summarize_parameters_bootstrap(data_tail, R=R_boot, xmin=xmin)

            if target_distribution not in fit.index:
                continue
            row = fit.loc[target_distribution]

            if np.isnan(row["p1"]) or np.isnan(row["p2"]) or np.isclose(row["p1"], 1.0, atol=0.05):
                skipped_degenerate += 1
                continue

            binwise[(y0, y1)] = {
                "p1": (row["p1"], row["p1_se"]),
                "p2": (row["p2"], row["p2_se"]),
                "n_tail": len(data_tail)
            }

        if len(binwise) < 2:
            continue

        years = np.array([np.mean(b) for b in binwise.keys()])
        p1_vals = np.array([v["p1"][0] for v in binwise.values()])
        p2_vals = np.array([v["p2"][0] for v in binwise.values()])
        mask = np.isfinite(p1_vals) & np.isfinite(p2_vals)
        if mask.sum() < 2:
            continue

        years = years[mask].reshape(-1, 1)
        p1_vals = p1_vals[mask]
        p2_vals = p2_vals[mask]

        b1_int, b1_slope, b1_se_int, b1_se_slope = ols_with_se(years, p1_vals)
        b2_int, b2_slope, b2_se_int, b2_se_slope = ols_with_se(years, p2_vals)

        coeffs = {"p1": (b1_int, b1_slope), "p2": (b2_int, b2_slope)}
        ses = {"p1": (b1_se_int, b1_se_slope), "p2": (b2_se_int, b2_se_slope)}

        time_fits[(lat_bin, lon_bin)] = {
            "binwise_params": binwise,
            "coeffs": coeffs,
            "ses": ses
        }

    print(f"\nCompleted {len(time_fits)} grid-cell time fits for {target_distribution}")
    print(f"Skipped {skipped_degenerate} degenerate bins (p₁≈1) across all tiles.")
    return time_fits

def time_fits_grid_to_df(time_fits):
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

def plot_parameter_heatmap(best_fits_gfa, df_tv,
                                        biome="Savannas",
                                        distribution="truncated_power_law",
                                        vmin=None, vmax=None):
    """
    Presentation-quality version:
      - For truncated_power_law: (α, 1/λ, Δα, Δλ)
      - For other distributions: (p₁, p₂, Δp₁, Δp₂)
      - Skips tiles where static α (p1) < 1.05 (TPL only)
      - Converts slopes to per decade
      - Grey = not significant change
    """

    records = []
    for (lat_bin, lon_bin), res in best_fits_gfa.get(biome, {}).items():
        if distribution not in res["best_fits"]:
            continue
        params = res["params"].loc[distribution]

        # Only apply α < 1.05 cutoff for truncated_power_law
        if distribution == "truncated_power_law" and params.get("p1", np.nan) < 1.05:
            continue

        geom = box(lon_bin, lat_bin, lon_bin + 5, lat_bin + 5)
        records.append({
            "lat_bin": lat_bin,
            "lon_bin": lon_bin,
            "geometry": geom,
            "p1": params.get("p1", np.nan),
            "p2": params.get("p2", np.nan),
        })

    if not records:
        print(f"No data for {biome} – {distribution}")
        return

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    df_tv_aligned = df_tv.copy()
    df_tv_aligned["lat_bin"] = np.floor(df_tv_aligned["lat_bin"] / 5) * 5
    df_tv_aligned["lon_bin"] = np.floor(df_tv_aligned["lon_bin"] / 5) * 5

    df_slopes = df_tv_aligned[["lat_bin", "lon_bin",
                               "p1_slope", "p2_slope",
                               "p1_sig", "p2_sig"]].copy()
    gdf = gdf.merge(df_slopes, on=["lat_bin", "lon_bin"], how="left")

    gdf["p1_slope"] = gdf["p1_slope"] * 10
    gdf["p2_slope"] = gdf["p2_slope"] * 10

    print(f"Matched {gdf['p1_slope'].notna().sum()} time-varying fits out of {len(gdf)} tiles.")

    cmap_main = plt.cm.RdYlGn
    cmap_secondary = plt.cm.RdYlGn_r if distribution == "truncated_power_law" else cmap_main

    # --- Handle λ inversion only for truncated_power_law ---
    if distribution == "truncated_power_law":
        gdf["p2_plot_static"] = 1 / gdf["p2"].replace(0, np.nan)
        plot_p2_col = "p2_plot_static"
        use_log_norm = True
        title_p1, title_p2 = "α", "1/λ"
        title_dp1, title_dp2 = "Δα per decade", "Δλ per decade"
    else:
        gdf["p2_plot_static"] = gdf["p2"]
        plot_p2_col = "p2_plot_static"
        use_log_norm = False
        title_p1, title_p2 = "p₁", "p₂"
        title_dp1, title_dp2 = "Δp₁ per decade", "Δp₂ per decade"

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.18)
    axes = [fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree()) for i in range(4)]

    minx, miny, maxx, maxy = gdf.total_bounds
    extent = [minx - 2, maxx + 2, miny - 2, maxy + 2]

    def format_map(ax, title):
        ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=0)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=0)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_title(title, fontsize=17, pad=6)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=11)

    # ---------- (1) p1 / α MAP ----------
    m1 = gdf.plot(
        column="p1", cmap=cmap_main, ax=axes[0], edgecolor="k", linewidth=0.2,
        vmin=vmin or np.nanmin(gdf["p1"]), vmax=vmax or np.nanmax(gdf["p1"]),
        legend=False
    )
    cb1 = plt.colorbar(m1.collections[0], ax=axes[0], orientation="vertical",
                       fraction=0.046, pad=0.02)
    cb1.ax.tick_params(labelsize=11)
    format_map(axes[0], title_p1)

    # ---------- (2) p2 / 1/λ MAP ----------
    norm = LogNorm(vmin=vmin or np.nanmin(gdf[plot_p2_col][gdf[plot_p2_col] > 0]),
                   vmax=vmax or np.nanmax(gdf[plot_p2_col])) if use_log_norm else None
    m2 = gdf.plot(
        column=plot_p2_col, cmap=cmap_secondary, ax=axes[1], edgecolor="k", linewidth=0.2,
        norm=norm, legend=False
    )
    cb2 = plt.colorbar(m2.collections[0], ax=axes[1], orientation="vertical",
                       fraction=0.046, pad=0.02)
    cb2.ax.tick_params(labelsize=11)
    format_map(axes[1], title_p2)

    # ---------- (3) Δp1 / Δα MAP ----------
    gdf["p1_plot"] = np.where(gdf["p1_sig"] == 1, gdf["p1_slope"], np.nan)
    vmax_dp1 = np.nanmax(np.abs(gdf["p1_slope"].dropna())) if np.any(np.isfinite(gdf["p1_slope"])) else 1e-6
    m3 = gdf.plot(
        column="p1_plot", cmap=cmap_main, ax=axes[2], linewidth=0.3,
        vmin=-vmax_dp1, vmax=vmax_dp1, legend=False,
        missing_kwds={'color': (0, 0, 0, 0), 'edgecolor': None}
    )
    gdf[gdf["p1_sig"] == 0].plot(ax=axes[2], color="grey", linewidth=0.1, alpha=0.9, zorder=1)
    cb3 = plt.colorbar(m3.collections[0], ax=axes[2], orientation="vertical",
                       fraction=0.046, pad=0.02)
    cb3.ax.tick_params(labelsize=11)
    format_map(axes[2], title_dp1)

    # ---------- (4) Δp2 / Δλ MAP ----------
    gdf["p2_plot"] = np.where(gdf["p2_sig"] == 1, gdf["p2_slope"], np.nan)
    vmax_dp2 = np.nanmax(np.abs(gdf["p2_slope"].dropna())) if np.any(np.isfinite(gdf["p2_slope"])) else 1e-6
    cmap_dp2 = plt.cm.RdYlGn if distribution != "truncated_power_law" else plt.cm.RdYlGn 
    m4 = gdf.plot(
        column="p2_plot", cmap=cmap_dp2, ax=axes[3], linewidth=0.3,
        vmin=-vmax_dp2, vmax=vmax_dp2, legend=False,
        missing_kwds={'color': (0, 0, 0, 0), 'edgecolor': None}
    )
    gdf[gdf["p2_sig"] == 0].plot(ax=axes[3], color="grey", linewidth=0.1, alpha=0.9, zorder=1)
    cb4 = plt.colorbar(m4.collections[0], ax=axes[3], orientation="vertical",
                       fraction=0.046, pad=0.02)
    cb4.ax.tick_params(labelsize=11)
    format_map(axes[3], title_dp2)

    fig.suptitle(f"{biome} — {distribution.replace('_', ' ').title()}: Spatiotemporal Variation",
                 fontsize=21, y=0.97)
    fig.text(0.5, 0.075, "Grey = not significant change",
             ha="center", fontsize=12, color="dimgray")

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    save_path = f"/Users/lukevonkapff/Desktop/{biome.lower()}_{distribution}_map.png"
    plt.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
    print(f"Figure saved to {save_path}")
    plt.show()

def plot_timevary_param_space_grid(time_fits, tile_key, dist_name="truncated_power_law",
                                   param_x="p1", param_y="p2"):
    """
    Plot binwise parameter estimates for a single grid cell:
    - Ellipses in parameter space (p1 vs p2) with time-colored shading + trendline
    - Side plots for parameter evolution over time
    """

    if tile_key not in time_fits:
        print(f"Tile {tile_key} not found in time_fits.")
        return

    entry = time_fits[tile_key]
    binwise = entry.get("binwise_params", {})
    coeffs = entry.get("coeffs", {})

    records = []
    for (y0, y1), params in binwise.items():
        px, px_se = params.get(param_x, (np.nan, np.nan))
        py, py_se = params.get(param_y, (np.nan, np.nan))
        if np.isfinite(px) and np.isfinite(py):
            records.append({
                "t": 0.5*(y0+y1),
                param_x: px, f"{param_x}_se": px_se,
                param_y: py, f"{param_y}_se": py_se
            })

    if not records:
        print(f"No valid bins found for {tile_key}.")
        return

    years = [r["t"] for r in records]
    year_min, year_max = min(years), max(years)

    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=year_min, vmax=year_max)

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = GridSpec(2, 3, width_ratios=[2, 1, 0.05], height_ratios=[3, 1], figure=fig)

    ax_main = fig.add_subplot(gs[:, 0])
    ax_p1   = fig.add_subplot(gs[0, 1])
    ax_p2   = fig.add_subplot(gs[1, 1])
    cax     = fig.add_subplot(gs[:, 2])

    for rec in records:
        color = cmap(norm(rec["t"]))
        ell = Ellipse(
            (rec[param_x], rec[param_y]),
            width=2*1.96*rec[f"{param_x}_se"],
            height=2*1.96*rec[f"{param_y}_se"],
            facecolor=color, edgecolor="k", alpha=0.5
        )
        ax_main.add_patch(ell)
        ax_main.plot(rec[param_x], rec[param_y], "o", color=color, markersize=4)

    if param_x in coeffs and param_y in coeffs:
        t_range = np.linspace(year_min, year_max, 200)
        x_line = coeffs[param_x][0] + coeffs[param_x][1]*t_range
        y_line = coeffs[param_y][0] + coeffs[param_y][1]*t_range
        ax_main.plot(x_line, y_line, color="red", lw=2.5, label="OLS trend")
        ax_main.legend()

    lat, lon = tile_key
    ax_main.set_xlabel(param_x)
    ax_main.set_ylabel(param_y)
    ax_main.set_title(f"Tile ({lat:.0f}°, {lon:.0f}°) – {dist_name}\nParameter space over time")

    times = [r["t"] for r in records]

    ax_p1.errorbar(times, [r[param_x] for r in records],
                   yerr=[r[f"{param_x}_se"] for r in records],
                   fmt="o-", color="blue", alpha=0.7)
    ax_p1.set_title(f"{param_x} over time")
    ax_p1.set_ylabel(param_x)
    if param_x in coeffs:
        ax_p1.plot(t_range, coeffs[param_x][0] + coeffs[param_x][1]*t_range,
                   "r--", lw=2)

    ax_p2.errorbar(times, [r[param_y] for r in records],
                   yerr=[r[f"{param_y}_se"] for r in records],
                   fmt="o-", color="green", alpha=0.7)
    ax_p2.set_title(f"{param_y} over time")
    ax_p2.set_xlabel("Year")
    ax_p2.set_ylabel(param_y)
    if param_y in coeffs:
        ax_p2.plot(t_range, coeffs[param_y][0] + coeffs[param_y][1]*t_range,
                   "r--", lw=2)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Bin midpoint year")

    plt.show()