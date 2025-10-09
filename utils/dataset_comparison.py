import numpy as np
import matplotlib.pyplot as plt

def print_summary(name, gdf, year_col):
    print(f"\n{name} dataset:")
    print(f"  Number of fires: {len(gdf)}")
    print(f"  Year range: {gdf[year_col].min()} - {gdf[year_col].max()}")
    print(f"  Mean fire size: {gdf['area_km2'].mean():.2f} km²")
    print(f"  Median fire size: {gdf['area_km2'].median():.2f} km²")
    print(f"  Min fire size: {gdf['area_km2'].min():.2f} km²")
    print(f"  Max fire size: {gdf['area_km2'].max():.2f} km²")
    print(f"  Total burned area: {gdf['area_km2'].sum():.2f} km²")

def kuiper_statistic(data1, data2):
    """Compute Kuiper statistic between two 1D numpy arrays."""
    data1 = np.sort(data1)
    data2 = np.sort(data2)

    n1, n2 = len(data1), len(data2)
    all_data = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, all_data, side='right') / n1
    cdf2 = np.searchsorted(data2, all_data, side='right') / n2

    d_plus = np.max(cdf1 - cdf2)
    d_minus = np.max(cdf2 - cdf1)
    return d_plus + d_minus

def plot_cdf_vs_area(mtbs_df, idaho_df, area_col="area_km2"):
    """Plot CDF vs fire size for MTBS and Idaho datasets."""
    plt.figure(figsize=(8,6))
    
    for data, label in [(mtbs_df, "MTBS"), (idaho_df, "Idaho")]:
        sizes = np.sort(data[area_col].values)
        cdf = np.arange(1, len(sizes)+1) / len(sizes)
        plt.plot(sizes, cdf, label=label)
    
    plt.xscale("log")
    plt.xlabel("Fire size (km², log scale)")
    plt.ylabel("CDF")
    plt.title("CDF of Fire Sizes")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()


def kuiper_for_alpha(mtbs_df, idaho_df, alpha, area_col="area_km2"):
    """Filter datasets by alpha threshold and compute Kuiper statistic."""
    mtbs_filt = mtbs_df[mtbs_df[area_col] > alpha]
    idaho_filt = idaho_df[idaho_df[area_col] > alpha]
    return kuiper_statistic(mtbs_filt[area_col].values, idaho_filt[area_col].values)


def plot_alpha_vs_kuiper_with_min_cdf(mtbs_df, idaho_df, alphas, area_col="area_km2"):
    """Compute Kuiper statistics for a range of alphas, plot alpha vs Kuiper, 
    and also plot CDF for the alpha with the minimum Kuiper statistic."""
    
    kuipers = []
    for a in alphas:
        ku = kuiper_for_alpha(mtbs_df, idaho_df, a, area_col)
        kuipers.append(ku)
    
    kuipers = np.array(kuipers)
    min_idx = np.argmin(kuipers)
    min_alpha = alphas[min_idx]

    plt.figure(figsize=(8,6))
    plt.plot(alphas, kuipers, marker="o")
    plt.axvline(min_alpha, color="red", linestyle="--", label=f"Min Kuiper at alpha={min_alpha:.2f}")
    plt.xlabel("Alpha (km² threshold)")
    plt.ylabel("Kuiper Statistic")
    plt.title("Alpha vs. Kuiper Statistic")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.show()
    
    print(f"Minimum Kuiper statistic {kuipers[min_idx]:.4f} occurs at alpha = {min_alpha:.4f} km²")
    
    mtbs_filt = mtbs_df[mtbs_df[area_col] > min_alpha]
    idaho_filt = idaho_df[idaho_df[area_col] > min_alpha]
    
    plt.figure(figsize=(8,6))
    for data, label in [(mtbs_filt, "MTBS"), (idaho_filt, "Idaho")]:
        sizes = np.sort(data[area_col].values)
        cdf = np.arange(1, len(sizes)+1) / len(sizes)
        plt.plot(sizes, cdf, label=label)
    
    plt.xscale("log")
    plt.xlabel("Fire size (km², log scale)")
    plt.ylabel("CDF")
    plt.title(f"CDF of Fire Sizes for alpha > {min_alpha:.4f} km² (min Kuiper)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()
    
    return min_alpha, kuipers[min_idx]