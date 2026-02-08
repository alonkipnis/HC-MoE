import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.integrate import trapz, cumtrapz
import pandas as pd

mpl.style.use('ggplot')

# Load data
path = "measurements/"
# df_pvals = pd.read_csv(path + "logit_1_p_values.csv") # Not strictly needed for the loop, but good to have if needed
df = pd.read_parquet(path + "layer_1.parquet")

def fit_empirical_null(data, bins=113, poly_deg=2, lower_range=0.25, upper_range=0.75, ax=None):
    """
    Fits an empirical null distribution using Efron's central matching method on standardized data.
    Returns the fitted polynomial, median, and MAD.
    """
    # Standardize data
    med = np.median(data)
    # scale='normal' makes it consistent with sigma for normal distribution
    mad = stats.median_abs_deviation(data, scale='normal') 
    data_std = (data - med) / mad

    # 1. Compute histogram
    counts, bin_edges = np.histogram(data_std, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    lower_range_bin = np.quantile(data_std, lower_range)
    upper_range_bin = np.quantile(data_std, upper_range)
    
    fraction = np.mean((data_std >= lower_range_bin) & (data_std <= upper_range_bin))
    print(f"Range (std): [{lower_range_bin:.4f}, {upper_range_bin:.4f}]")
    print(f"Fraction of data in range: {fraction:.4f}")

    # 2. Fit polynomial to log-density
    mask = (counts > 0) & (bin_centers >= lower_range_bin) & (bin_centers <= upper_range_bin)
    x = bin_centers[mask]
    y = np.log(counts[mask])
    
    coeffs = np.polyfit(x, y, poly_deg)
    poly = np.poly1d(coeffs)
    
    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    ax.hist(data_std, bins=bins, density=True, alpha=0.5, color='gray', label='Histogram', edgecolor='black', linestyle='--')
    
    # Shade used bins
    bin_width = bin_edges[1] - bin_edges[0]
    ax.bar(bin_centers[mask], counts[mask], width=bin_width, color='green', alpha=0.3, label='Used in Fit', align='center')

    # Plot fitted polynomial
    x_grid = np.linspace(data_std.min(), data_std.max(), 500)
    # Note: exp(poly) is the estimated density
    ax.plot(x_grid, np.exp(poly(x_grid)), 'b-', linewidth=2, label=f'Poly Fit (deg={poly_deg})')
    
    ax.set_title(f"Empirical Null Fit (Standardized)")
    ax.set_xlabel("Standardized Value")
    ax.set_ylabel("Density")
    ax.legend()
    
    return poly, med, mad

def get_pvalues_from_poly(poly, data):
    """
    Calculate two-sided p-values for data based on the density exp(poly(x)).
    """
    # Define a grid for integration covering the data range
    # Extend slightly beyond min/max to cover tails
    x_min, x_max = data.min(), data.max()
    margin = (x_max - x_min) * 0.1
    grid = np.linspace(x_min - margin, x_max + margin, 10000)
    
    # Calculate density
    pdf = np.exp(poly(grid))
    
    # Normalize density
    normalization = trapz(pdf, grid)
    if normalization == 0 or np.isnan(normalization) or np.isinf(normalization):
        # Fallback if integration fails (e.g. diverging poly)
        return np.ones_like(data)
        
    pdf_norm = pdf / normalization
    
    # Calculate CDF
    cdf_grid = cumtrapz(pdf_norm, grid, initial=0)
    # Ensure CDF goes from 0 to 1
    cdf_grid /= cdf_grid[-1]
    
    # Interpolate CDF for data points
    cdf_vals = np.interp(data, grid, cdf_grid)
    
    # Two-sided p-value: 2 * min(CDF, 1-CDF)
    p_values = 2 * np.minimum(cdf_vals, 1 - cdf_vals)
    # Clip to avoid log(0) issues
    p_values = np.maximum(p_values, 1e-300)
    
    return p_values

# Run the fit
if __name__ == "__main__":
    block_nums = np.random.choice(np.arange(1, 128), size=10, replace=False)

    for i in block_nums:
        print(f"Block: logit_{i}" )
        logits = df[f"logit_{i}"]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Empirical Null Fit
        poly, med, mad = fit_empirical_null(logits, lower_range=0.1, upper_range=0.9, ax=axes[0])
        
        # Right: QQ Plot of -log(P-values)
        # Standardize logits
        logits_std = (logits - med) / mad
        
        # Calculate P-values under the fitted polynomial null
        p_values = get_pvalues_from_poly(poly, logits_std)
        
        neg_log_pvals = -np.log(p_values)
        
        # Sort
        sorted_indices = np.argsort(neg_log_pvals)
        sample_quantiles = neg_log_pvals[sorted_indices]
        
        # Theoretical quantiles for Exp(1)
        n = len(logits)
        theoretical_quantiles = -np.log(1 - (np.arange(1, n + 1) / (n + 1)))
        
        # Determine points in range (using standardized values)
        lower_cutoff = np.quantile(logits_std, 0.05)
        upper_cutoff = np.quantile(logits_std, 0.95)
        
        # We need to find which sorted quantiles correspond to logits in the range
        sorted_logits_std = logits_std.iloc[sorted_indices] if hasattr(logits_std, 'iloc') else logits_std[sorted_indices]
        in_range_mask = (sorted_logits_std >= lower_cutoff) & (sorted_logits_std <= upper_cutoff)
        
        # Plot
        axes[1].scatter(theoretical_quantiles[~in_range_mask], sample_quantiles[~in_range_mask], 
                        s=5, c='gray', alpha=0.5, label='Out of Range')
        axes[1].scatter(theoretical_quantiles[in_range_mask], sample_quantiles[in_range_mask], 
                        s=5, c='blue', alpha=0.5, label='In Range')
        
        # y=x line
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        axes[1].plot([0, max_val], [0, max_val], 'k--', label='Exp(1)')
        
        # limit x and y axes to [0,20]
        axes[1].set_xlim(0, 20)
        axes[1].set_ylim(0, 20)
        axes[1].set_xlabel('Theoretical Quantiles (Exp(1))')
        axes[1].set_ylabel('Sample Quantiles (-log(P))')
        axes[1].set_title(f"QQ Plot of -log(P-values)")
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        print(f"Fit: Median={med:.4f}, MAD={mad:.4f}")
