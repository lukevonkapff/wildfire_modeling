import numpy as np
import matplotlib.pyplot as plt


def print_summary(name, gdf, year_col):
    """Print a summary of key fire statistics for a GeoDataFrame.

    Args:
        name (str): Descriptive name of the dataset (e.g., "MTBS" or "Idaho").
        gdf (GeoDataFrame): Dataset containing fire information, including area and year columns.
        year_col (str): Name of the column in `gdf` that stores fire year information.

    Prints:
        Summary statistics such as number of fires, year range, and basic size metrics.
    """
    print(f"\n{name} dataset:")
    print(f"  Number of fires: {len(gdf)}")
    print(f"  Year range: {gdf[year_col].min()} - {gdf[year_col].max()}")
    print(f"  Mean fire size: {gdf['area_km2'].mean():.2f} km²")
    print(f"  Median fire size: {gdf['area_km2'].median():.2f} km²")
    print(f"  Min fire size: {gdf['area_km2'].min():.2f} km²")
    print(f"  Max fire size: {gdf['area_km2'].max():.2f} km²")
    print(f"  Total burned area: {gdf['area_km2'].sum():.2f} km²")


def kuiper_statistic(data1, data2):
    """Compute the Kuiper statistic between two one-dimensional samples.

    The Kuiper statistic is similar to the Kolmogorov–Smirnov statistic but
    is equally sensitive at the tails and the center of the distributions.

    Args:
        data1 (np.ndarray): First dataset (1D array-like).
        data2 (np.ndarray): Second dataset (1D array-like).

    Returns:
        float: Kuiper statistic value measuring distributional difference.
    """
    # Sort both datasets
    data1 = np.sort(data1)
    data2 = np.sort(data2)

    # Compute empirical CDFs across combined range
    n1, n2 = len(data1), len(data2)
    all_data = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, all_data, side='right') / n1
    cdf2 = np.searchsorted(data2, all_data, side='right') / n2

    # Compute maximum positive and negative differences
    d_plus = np.max(cdf1 - cdf2)
    d_minus = np.max(cdf2 - cdf1)

    # Kuiper statistic is the sum of both
    return d_plus + d_minus


def plot_cdf_vs_area(mtbs_df, idaho_df, area_col="area_km2"):
    """Plot cumulative distribution functions (CDFs) of fire sizes for two datasets.

    Args:
        mtbs_df (DataFrame): MTBS dataset containing fire size column.
        idaho_df (DataFrame): Idaho dataset containing fire size column.
        area_col (str, optional): Name of the column representing fire area (in km²).
            Defaults to "area_km2".

    Displays:
        A log-scaled CDF plot comparing MTBS and Idaho fire size distributions.
    """
    plt.figure(figsize=(8, 6))
    
    # Compute and plot empirical CDFs
    for data, label in [(mtbs_df, "MTBS"), (idaho_df, "Idaho")]:
        sizes = np.sort(data[area_col].values)
        cdf = np.arange(1, len(sizes) + 1) / len(sizes)
        plt.plot(sizes, cdf, label=label)
    
    plt.xscale("log")
    plt.xlabel("Fire size (km², log scale)")
    plt.ylabel("CDF")
    plt.title("CDF of Fire Sizes")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()


def kuiper_for_alpha(mtbs_df, idaho_df, alpha, area_col="area_km2"):
    """Compute the Kuiper statistic after filtering datasets by a minimum fire size.

    Args:
        mtbs_df (DataFrame): MTBS dataset containing fire size column.
        idaho_df (DataFrame): Idaho dataset containing fire size column.
        alpha (float): Fire size threshold (km²) used to filter both datasets.
        area_col (str, optional): Name of the fire size column. Defaults to "area_km2".

    Returns:
        float: Kuiper statistic computed for the filtered datasets.
    """
    # Filter both datasets for fires larger than alpha
    mtbs_filt = mtbs_df[mtbs_df[area_col] > alpha]
    idaho_filt = idaho_df[idaho_df[area_col] > alpha]

    # Compute and return the Kuiper statistic
    return kuiper_statistic(mtbs_filt[area_col].values, idaho_filt[area_col].values)


def plot_alpha_vs_kuiper_with_min_cdf(mtbs_df, idaho_df, alphas, area_col="area_km2"):
    """Analyze sensitivity of Kuiper statistic across fire size thresholds.

    For a range of `alpha` thresholds, this function computes the Kuiper statistic,
    plots the statistic vs. alpha, and highlights the alpha with the minimum Kuiper value.
    It also plots CDFs of the two datasets filtered at that minimum alpha.

    Args:
        mtbs_df (DataFrame): MTBS dataset containing fire size column.
        idaho_df (DataFrame): Idaho dataset containing fire size column.
        alphas (Iterable[float]): Sequence of threshold values (km²) to evaluate.
        area_col (str, optional): Name of the fire size column. Defaults to "area_km2".

    Returns:
        tuple:
            - float: Alpha value that minimizes the Kuiper statistic.
            - float: Minimum Kuiper statistic value.
    """
    kuipers = []

    # Compute Kuiper statistic for each alpha threshold
    for a in alphas:
        ku = kuiper_for_alpha(mtbs_df, idaho_df, a, area_col)
        kuipers.append(ku)
    
    kuipers = np.array(kuipers)
    min_idx = np.argmin(kuipers)
    min_alpha = alphas[min_idx]

    # Plot alpha vs. Kuiper statistic
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, kuipers, marker="o")
    plt.axvline(min_alpha, color="red", linestyle="--", label=f"Min Kuiper at alpha={min_alpha:.2f}")
    plt.xlabel("Alpha (km² threshold)")
    plt.ylabel("Kuiper Statistic")
    plt.title("Alpha vs. Kuiper Statistic")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.show()
    
    print(f"Minimum Kuiper statistic {kuipers[min_idx]:.4f} occurs at alpha = {min_alpha:.4f} km²")
    
    # Filter datasets using the alpha that minimizes Kuiper
    mtbs_filt = mtbs_df[mtbs_df[area_col] > min_alpha]
    idaho_filt = idaho_df[idaho_df[area_col] > min_alpha]
    
    # Plot CDFs for filtered datasets
    plt.figure(figsize=(8, 6))
    for data, label in [(mtbs_filt, "MTBS"), (idaho_filt, "Idaho")]:
        sizes = np.sort(data[area_col].values)
        cdf = np.arange(1, len(sizes) + 1) / len(sizes)
        plt.plot(sizes, cdf, label=label)
    
    plt.xscale("log")
    plt.xlabel("Fire size (km², log scale)")
    plt.ylabel("CDF")
    plt.title(f"CDF of Fire Sizes for alpha > {min_alpha:.4f} km² (min Kuiper)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()
    
    return min_alpha, kuipers[min_idx]
