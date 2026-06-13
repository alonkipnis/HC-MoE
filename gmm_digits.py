import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = digits.data  # shape (1797, 64)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One GMM component per digit class × 2 modes each
gmm = GaussianMixture(n_components=30, covariance_type='full', random_state=42, max_iter=200)
gmm.fit(X_scaled)

print(f"GMM converged: {gmm.converged_}")
print(f"Log-likelihood: {gmm.lower_bound_:.3f}")

samples_scaled, _ = gmm.sample(16)
samples = scaler.inverse_transform(samples_scaled)
samples = np.clip(samples, 0, 16)

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for ax, sample in zip(axes.flat, samples):
    ax.imshow(sample.reshape(8, 8), cmap='gray_r', interpolation='nearest')
    ax.axis('off')

fig.suptitle('GMM-generated digit samples', fontsize=13)
plt.tight_layout()
plt.savefig('gmm_digits_samples.png', dpi=120)
plt.show()
print("Saved gmm_digits_samples.png")
