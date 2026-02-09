import numpy as np
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.stats import t
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import trapz, cumulative_trapezoid
import pandas as pd

mpl.style.use('ggplot')

# Load data
file_path = "measurements/layer_1.parquet"
df = pd.read_parquet(file_path)

def fit_empirical_null_lindsey(data, bins=71, quantile_range=(0.2, 0.8), fit_config=None, ax=None):
    """
    Fits an empirical null distribution using Lindsey's Method (Poisson Regression) or curve fit.
    
    Parameters:
        fit_config (dict): {'type': 'student_t' | 'laplace' | 'gaussian'}
    """
    if fit_config is None:
        fit_config = {'type': 'laplace'}
    
    data = np.array(data)
    n_total = len(data)
    null_pdf = None
    
    # 1. Bin the Data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # 2. Define "Null Domain" (Central Mass)
    q_low, q_high = np.percentile(data, [100 * q for q in quantile_range])
    mask = (bin_centers >= q_low) & (bin_centers <= q_high)
    
    y_train = counts[mask]
    x_train = bin_centers[mask]
    
    fit_type = fit_config.get('type', 'laplace').lower()
    
    # ==========================================
    # BRANCH 1: GLM Methods (Laplace / Gaussian)
    # ==========================================
    if fit_type in ['laplace', 'gaussian']:
        mu = fit_config.get('center', np.median(data))
        
        # Define the design matrix (X_train) based on the shape
        if fit_type == 'laplace':
            # Exponential decay = Linear in log-space with absolute value
            X_train = np.column_stack([np.ones_like(x_train), np.abs(x_train - mu)])
            def make_exog(x): 
                return np.column_stack([np.ones_like(x), np.abs(x - mu)])
        else: 
            # Gaussian decay = Quadratic in log-space
            X_train = np.column_stack([np.ones_like(x_train), (x_train - mu)**2])
            def make_exog(x): 
                return np.column_stack([np.ones_like(x), (x - mu)**2])

        # Fit the Poisson GLM
        glm_model = sm.GLM(y_train, X_train, family=sm.families.Poisson())
        results = glm_model.fit()
        
        # Return the PDF function immediately
        def null_pdf_glm(x):
            return results.predict(make_exog(np.array(x))) / (n_total * bin_width)
        null_pdf = null_pdf_glm

    # ==========================================
    # BRANCH 2: Non-Linear Fit (Student's t or Skewed t)
    # ==========================================
    elif fit_type in ['student_t', 'skewed_t']:
        
        if fit_type == 'student_t':
            # Model: Count = Amplitude * PDF(x)
            def t_model_counts(x, df, loc, scale, amp):
                return amp * t.pdf(x, df, loc, scale)

            # Initial Guesses
            p0 = [
                fit_config.get('df', 4.0),      
                np.median(data),                
                np.std(data),                   
                np.max(y_train) * 5             # Amplitude guess
            ]
            
            # Bounds: df>2, scale>0, amp>0
            bounds = ([2, -np.inf, 1e-6, 0], [10, np.inf, np.inf, np.inf])

            try:
                popt, _ = curve_fit(t_model_counts, x_train, y_train, p0=p0, bounds=bounds)
            except RuntimeError:
                print("Curve fit failed to converge.")
                return None

            df_fit, loc_fit, scale_fit, amp_fit = popt
            print(f"Fitted Student's t: df={df_fit:.2f}, loc={loc_fit:.2f}, scale={scale_fit:.2f}")

            # Return the pure PDF function immediately
            def null_pdf_t(x):
                return t.pdf(x, df_fit, loc_fit, scale_fit)
                
            null_pdf = null_pdf_t
            
        elif fit_type == 'skewed_t':
            from scipy.stats import nct
            
            # Model: Count = Amplitude * PDF(x) using Non-central t distribution
            # Parameters: df (degrees of freedom), nc (non-centrality), loc, scale, amp
            def skewed_t_model_counts(x, df, nc, loc, scale, amp):
                return amp * nct.pdf(x, df, nc, loc=loc, scale=scale)

            # Initial Guesses
            # We guess df=4, nc=0 (symmetric start), loc=median, scale=std
            p0 = [
                fit_config.get('df', 4.0),
                0.0,                            # nc (skewness parameter)
                np.median(data),                
                np.std(data),                   
                np.max(y_train) * 5             # Amplitude guess
            ]
            
            # Bounds: df>2, scale>0, amp>0
            # nc is unbounded
            bounds = ([2, -np.inf, -np.inf, 1e-6, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])

            try:
                popt, _ = curve_fit(skewed_t_model_counts, x_train, y_train, p0=p0, bounds=bounds)
            except RuntimeError:
                print("Curve fit failed to converge.")
                return None

            df_fit, nc_fit, loc_fit, scale_fit, amp_fit = popt
            print(f"Fitted Skewed t: df={df_fit:.2f}, nc={nc_fit:.2f}, loc={loc_fit:.2f}, scale={scale_fit:.2f}")

            # Return the pure PDF function immediately
            def null_pdf_skewed_t(x):
                return nct.pdf(x, df_fit, nc_fit, loc=loc_fit, scale=scale_fit)
                
            null_pdf = null_pdf_skewed_t

    else:
        raise ValueError(f"Unknown fit type: {fit_type}")


    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    ax.hist(data, bins=bins, density=True, alpha=0.5, color='gray', label='Histogram', edgecolor='black', linestyle='--')
    
    # Shade used bins
    density_train = counts[mask] / (n_total * bin_width)
    ax.bar(bin_centers[mask], density_train, width=bin_width, color='green', alpha=0.3, label='Used in Fit', align='center')

    # Plot fitted PDF
    x_grid = np.linspace(data.min(), data.max(), 500)
    y_grid = null_pdf(x_grid)
    ax.plot(x_grid, y_grid, 'b-', linewidth=2, label=f'GLM Fit ({fit_type})')
    
    ax.set_title(f"Empirical Null Fit (Lindsey's Method)")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    

    return null_pdf


def get_pvalues_from_pdf_func(pdf_func, data):
    """
    Calculate two-sided p-values for data based on a PDF function.
    """
    # Define a grid for integration covering the data range
    # Extend slightly beyond min/max to cover tails
    x_min, x_max = data.min(), data.max()
    margin = (x_max - x_min) * 0.1
    grid = np.linspace(x_min - margin, x_max + margin, 10000)
    
    # Calculate density
    pdf_vals = pdf_func(grid)
    
    # Normalize density (just in case the function isn't perfectly normalized)
    normalization = trapz(pdf_vals, grid)
    if normalization == 0 or np.isnan(normalization) or np.isinf(normalization):
        # Fallback if integration fails
        return np.ones_like(data)
        
    pdf_norm = pdf_vals / normalization
    
    # Calculate CDF
    cdf_grid = cumulative_trapezoid(pdf_norm, grid, initial=0)
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
        null_pdf = fit_empirical_null_lindsey(logits, bins=71, quantile_range=(0.05, 0.95), ax=axes[0],
         fit_config={'type': 'skewed_t'},
         #fit_config={'type': 'student_t'},
         #fit_config={'type': 'laplace'},
         #fit_config={'type': 'gaussian'},
         )
        

        # Calculate P-values under the fitted polynomial null
        # We integrate the PDF to get the CDF and then 2-sided p-values
        if null_pdf is not None:
             p_values = get_pvalues_from_pdf_func(null_pdf, logits)
        else:
             p_values = np.ones_like(logits)
        
        neg_log_pvals = -np.log(p_values)
        
        # Sort
        sorted_indices = np.argsort(neg_log_pvals)
        sample_quantiles = neg_log_pvals[sorted_indices]
        
        # Theoretical quantiles for Exp(1)
        n = len(logits)
        theoretical_quantiles = -np.log(1 - (np.arange(1, n + 1) / (n + 1)))
        
        # Determine points in range (using standardized values)
        lower_cutoff = np.quantile(logits, 0.05)
        upper_cutoff = np.quantile(logits, 0.95)
        
        # We need to find which sorted quantiles correspond to logits in the range
        sorted_logits_std = logits.iloc[sorted_indices] if hasattr(logits, 'iloc') else logits[sorted_indices]
        in_range_mask = (sorted_logits_std >= lower_cutoff) & (sorted_logits_std <= upper_cutoff)
        
        # Plot
        axes[1].scatter(theoretical_quantiles[~in_range_mask], sample_quantiles[~in_range_mask], 
                        s=5, c='gray', alpha=0.5, label='Out of Range')
        axes[1].scatter(theoretical_quantiles[in_range_mask], sample_quantiles[in_range_mask], 
                        s=5, c='blue', alpha=0.5, label='In Range')
        
        # y=x line
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        axes[1].plot([0, max_val], [0, max_val], 'k--', label='Exp(1)')
        
        # limit x and y axes to [0,7]
        axes[1].set_xlim(0, 7)
        axes[1].set_ylim(0, 7)
        axes[1].set_xlabel('Theoretical Quantiles (Exp(1))')
        axes[1].set_ylabel('Sample Quantiles (-log(P))')
        axes[1].set_title(f"QQ Plot of -log(P-values)")
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f"figures/qq_plot_logit_{i}.png")
        plt.show()
