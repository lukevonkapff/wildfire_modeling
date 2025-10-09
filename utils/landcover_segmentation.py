import geopandas as gpd
import rasterio
from collections import Counter
import rasterio.mask
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import wildfire_powerlaw as wfpl

def load_shapefile(shp_path, projection, area_col=None):
    gdf = gpd.read_file(shp_path)
    gdf = gdf.set_crs(projection)
    gdf = gdf.to_crs("EPSG:6933")
    if area_col is not None and area_col not in gdf.columns:
        gdf[area_col] = gdf.geometry.area / 1e6 
    return gdf

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
    fires_gdf["modis_class"] = "Unknown"
    
    for idx, cls in results:
        fires_gdf.at[idx, "modis_class"] = cls

    return fires_gdf

def plot_distribution_params(results_dicts, counts_dicts, 
                             min_count=200, llhr_cutoff=2.0, 
                             max_rel_error=1.0,
                             log_axes=False,
                             datasets=("GFA", "Idaho", "MTBS")):
    """
    Faceted figure with one panel per distribution.
    - Ellipses for 2-parameter fits (truncated to valid domain).
    - Horizontal error bars for 1-parameter fits (exponential, power_law).
    - Skips points with huge relative error (SE/param > max_rel_error).
    - Optionally log-scale axes.
    """
    markers = {"GFA": "o", "Idaho": "s", "MTBS": "^"}
    colors = plt.cm.tab10.colors
    
    all_dists = set()
    for dset, res in results_dicts.items():
        for biome, out in res.items():
            all_dists.update(out["params"].index)
    all_dists = sorted(all_dists)

    ncols, nrows = 3, int(np.ceil(len(all_dists)/3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()

    biome_to_color = {}
    panels_with_points = set()

    for dset, res in results_dicts.items():
        for biome, out in res.items():
            if biome == "Unknown" or counts_dicts[dset].get(biome, 0) < min_count:
                continue

            likelihoods = out["likelihood_matrix"]

            for dist_name, row in out["params"].iterrows():
                if isinstance(row.get("reduces_to"), str):
                    continue
                if dist_name in likelihoods.index:
                    if likelihoods.loc[dist_name].min() > llhr_cutoff:
                        continue

                if biome not in biome_to_color:
                    biome_to_color[biome] = colors[len(biome_to_color) % len(colors)]
                ax = axes[all_dists.index(dist_name)]

                p1, p1_se = row.get("p1", np.nan), row.get("p1_se", np.nan)
                p2, p2_se = row.get("p2", np.nan), row.get("p2_se", np.nan)
                if np.isnan(p1):
                    continue

                if not np.isnan(p1_se) and abs(p1) > 0 and p1_se/abs(p1) > max_rel_error:
                    continue
                if not np.isnan(p2) and not np.isnan(p2_se) and abs(p2) > 0 and p2_se/abs(p2) > max_rel_error:
                    continue

                valid = True
                min_p1, min_p2 = -np.inf, -np.inf
                if dist_name in ("weibull", "weibull_excess", "stretched_exponential"):
                    if p1 <= 0 or (not np.isnan(p2) and p2 <= 0): valid = False
                    min_p1, min_p2 = 0, 0
                elif dist_name in ("lognormal", "lognormal_excess"):
                    if not np.isnan(p2) and p2 <= 0: valid = False
                    min_p2 = 0
                elif dist_name == "generalized_pareto":
                    if p2 <= 0 or p1 < 0: valid = False
                    min_p1, min_p2 = 0, 0
                elif dist_name == "truncated_power_law":
                    if p1 <= 1 or p2 <= 0: valid = False
                    min_p1, min_p2 = 1, 0
                elif dist_name == "exponential":
                    if p1 <= 0: valid = False
                    min_p1 = 0
                elif dist_name == "power_law":
                    if p1 <= 1: valid = False
                    min_p1 = 1

                if not valid:
                    continue

                if dist_name in ("exponential", "power_law"):
                    p1_low, p1_high = max(p1 - p1_se, min_p1 + 1e-8), p1 + p1_se
                    ax.errorbar(
                        p1, 0,
                        xerr=[[p1 - p1_low], [p1_high - p1]],
                        fmt=markers.get(dset, "o"),
                        color=biome_to_color[biome]
                    )
                else:
                    theta = np.linspace(0, 2*np.pi, 200)
                    x = p1 + p1_se * np.cos(theta)
                    y = p2 + p2_se * np.sin(theta) if not np.isnan(p2) else np.zeros_like(theta)

                    x = np.clip(x, min_p1, None)
                    y = np.clip(y, min_p2, None)

                    polygon = Polygon(np.column_stack([x, y]),
                                      closed=True,
                                      facecolor=biome_to_color[biome],
                                      edgecolor="none",
                                      alpha=0.3)
                    ax.add_patch(polygon)

                    ax.plot(p1, p2 if not np.isnan(p2) else 0,
                            markers.get(dset, "o"),
                            color=biome_to_color[biome], markersize=4)

                panels_with_points.add(dist_name)

    for i, dist_name in enumerate(all_dists):
        ax = axes[i]
        if dist_name in panels_with_points:
            ax.set_title(dist_name)
            ax.set_xlabel("Parameter 1")
            ax.set_ylabel("Parameter 2")
            if log_axes:
                ax.set_xscale("log")
                ax.set_yscale("log")
        else:
            fig.delaxes(ax)

    biome_handles = [
        plt.Line2D([0],[0], marker="o", color="w",
                   markerfacecolor=color, markersize=10, label=biome)
        for biome, color in biome_to_color.items()
    ]
    dataset_handles = [
        plt.Line2D([0],[0], marker=markers[dset], color="k",
                   linestyle="None", markersize=8, label=dset)
        for dset in datasets
    ]

    fig.legend(handles=biome_handles, title="Biomes", loc="upper right", bbox_to_anchor=(1.15, 1.0))
    fig.legend(handles=dataset_handles, title="Datasets", loc="lower right", bbox_to_anchor=(1.15, 0.0))

    plt.tight_layout()
    plt.show()

def build_comparison_table(results_dict, counts_dict, drop_classes=None,
                           threshold=2.0, min_n=400, max_relerr=1.0):
    rows_by_class = {}

    for dataset, classes in results_dict.items():
        counts = counts_dict.get(dataset, {})
        for modis_class, res in classes.items():
            n = counts.get(modis_class, np.nan)
            if pd.isna(n) or n < min_n:  # skip small classes
                continue

            best_fit = res.get("best_fit", None)
            llmat = res.get("likelihood_matrix", None)
            params = res.get("params", None)

            if best_fit is None or llmat is None or params is None:
                continue

            # helper: check filtering rules
            def valid_fit(dist):
                if dist not in params.index:
                    return False
                # must not reduce_to something else
                if pd.notna(params.loc[dist, "reduces_to"]):
                    return False
                # relative error check
                for p, se in [("p1", "p1_se"), ("p2", "p2_se")]:
                    if p in params.columns and se in params.columns:
                        val, err = params.loc[dist, p], params.loc[dist, se]
                        if pd.notna(val) and pd.notna(err) and val != 0:
                            if abs(err / val) > max_relerr:
                                return False
                return True

            # Δloglik column
            col = [c for c in llmat.columns if best_fit in c]
            if not col:
                continue
            deltas = llmat[col[0]].copy()

            fits = []
            # include best fit if valid
            if valid_fit(best_fit):
                fits.append((best_fit, "", ""))

            # include others within threshold
            for dist, val in deltas.items():
                if dist == best_fit:
                    continue
                if val <= threshold and valid_fit(dist):
                    fits.append((dist, round(val, 3) if not np.isinf(val) else np.inf, ""))

            if not fits:
                continue

            rows_by_class.setdefault(modis_class, {}).setdefault(dataset, {
                "n": n, "fits": fits
            })

    # unify across datasets
    final_rows = []
    all_classes = sorted(rows_by_class.keys())
    for modis in all_classes:
        mtbs = rows_by_class[modis].get("MTBS", {"n": "", "fits": []})
        gfa = rows_by_class[modis].get("GFA", {"n": "", "fits": []})
        idaho = rows_by_class[modis].get("Idaho", {"n": "", "fits": []})

        nrows = max(len(mtbs["fits"]), len(gfa["fits"]), len(idaho["fits"]))
        for i in range(nrows):
            row = {"modis_class": modis}
            for dataset, d in [("MTBS", mtbs), ("GFA", gfa), ("Idaho", idaho)]:
                row[f"{dataset}_n"] = d["n"] if i == 0 else ""
                if i < len(d["fits"]):
                    dist, val, reduces = d["fits"][i]
                    row[f"{dataset}_fit"] = dist
                    row[f"{dataset}_loglik"] = val
                    row[f"{dataset}_reduces_to"] = reduces
                else:
                    row[f"{dataset}_fit"] = ""
                    row[f"{dataset}_loglik"] = ""
                    row[f"{dataset}_reduces_to"] = ""
            final_rows.append(row)

    df = pd.DataFrame(final_rows)

    # drop unwanted
    if drop_classes:
        df = df[~df["modis_class"].isin(drop_classes)]

    # collapse class names (show only once per group)
    def keep_first(series):
        return [series.iloc[0]] + [""] * (len(series) - 1)
    df["modis_class"] = df.groupby("modis_class")["modis_class"].transform(keep_first)

    return df.reset_index(drop=True)

def plot_modis_category_ccdf(gfa_df, modis_class, xmin=4, which=("power_law")):
    """
    Filter gfa_classified for a given MODIS category and plot CCDF with selected fits.
    
    Parameters
    ----------
    gfa_df : pandas.DataFrame
        Your classified GFA dataframe, must include 'modis_class' and 'area_km2'.
    modis_class : str
        MODIS category name to filter (e.g., 'Croplands').
    xmin : float, optional
        Minimum cutoff for fitting (default=4).
    which : list of str, optional
        List of distributions to overlay on the CCDF plot.
    """
    subset = gfa_df[gfa_df["modis_class"] == modis_class]
    
    if subset.empty:
        print(f"No fires found for MODIS class: {modis_class}")
        return None

    data = subset["area_km2"].dropna().to_numpy()
    
    if len(data) == 0:
        print(f"No valid area data for MODIS class: {modis_class}")
        return None
    
    ax = wfpl.plot_ccdf_with_selected_fits(data, xmin=xmin, which=which)
    ax.set_title(f"CCDF of fire size in {modis_class}")
    plt.show()
    return ax