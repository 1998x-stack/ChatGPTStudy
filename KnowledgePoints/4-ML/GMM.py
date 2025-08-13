# GMM hands-on demo with visualization (no seaborn, single-plot figures, default colors only)
# This script will:
# 1) Generate a synthetic 2D dataset from 3 Gaussians.
# 2) Fit GMMs for K in [1..6], compute AIC/BIC, and choose K via BIC.
# 3) Visualize (a) AIC/BIC vs K, (b) GMM clustering with soft-probability contours,
#    and (c) K-Means clustering for comparison.
# 4) Print key diagnostics and the chosen K.

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap  # not used for explicit colors; kept for potential extensions
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1) Data generation
# ----------------------------
rng = np.random.RandomState(42)

# Three Gaussian components in 2D
means = np.array([[0, 0], [5, 5], [-5, 5]], dtype=float)
covs = np.array([
    [[1.0, 0.4], [0.4, 1.2]],
    [[1.3, -0.2], [-0.2, 0.8]],
    [[0.7, 0.3], [0.3, 1.5]],
], dtype=float)
sizes = [400, 300, 300]

X_list = []
for m, S, n in zip(means, covs, sizes):
    X_list.append(rng.multivariate_normal(mean=m, cov=S, size=n))
X = np.vstack(X_list)

# Optional standardization (many real datasets benefit from this)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 2) Model selection by AIC/BIC
# ----------------------------
k_range = range(1, 7)
aics = []
bics = []
gmms = {}

for k in k_range:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        init_params="kmeans",
        n_init=5,
        reg_covar=1e-6,
        random_state=42,
        max_iter=500,
        tol=1e-4,
    )
    gmm.fit(X_scaled)
    aics.append(gmm.aic(X_scaled))
    bics.append(gmm.bic(X_scaled))
    gmms[k] = gmm

best_k = int(k_range[np.argmin(bics)])
best_gmm = gmms[best_k]

print("=== Model selection summary ===")
print("K values:", list(k_range))
print("AIC per K:", [round(v, 2) for v in aics])
print("BIC per K:", [round(v, 2) for v in bics])
print("Chosen K by BIC:", best_k)
print()

# ----------------------------
# 3a) Plot AIC/BIC vs K
# ----------------------------
plt.figure(figsize=(6, 4))
plt.plot(list(k_range), aics, marker="o", label="AIC")
plt.plot(list(k_range), bics, marker="s", label="BIC")
plt.xlabel("Number of components K")
plt.ylabel("Information criterion")
plt.title("AIC/BIC vs K (lower is better)")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# 3b) GMM clustering visualization (soft boundaries)
# ----------------------------
# Create a meshgrid over the feature space
x_min, x_max = X_scaled[:, 0].min() - 1.0, X_scaled[:, 0].max() + 1.0
y_min, y_max = X_scaled[:, 1].min() - 1.0, X_scaled[:, 1].max() + 1.0

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300),
)
grid = np.c_[xx.ravel(), yy.ravel()]

probas = best_gmm.predict_proba(grid)
labels_grid = np.argmax(probas, axis=1)

# Predict labels for the actual points
labels_points = best_gmm.predict(X_scaled)

plt.figure(figsize=(6, 6))
# Background: decision regions via contourf (default colormap)
plt.contourf(xx, yy, labels_grid.reshape(xx.shape), alpha=0.25)

# Data points
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], s=10, alpha=0.7)

# Optional: plot component means (crosses)
means_scaled = best_gmm.means_
plt.scatter(means_scaled[:, 0], means_scaled[:, 1], marker="x", s=100)

plt.title(f"GMM clustering (K={best_k}) with soft regions")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.tight_layout()
plt.show()

# ----------------------------
# 3c) K-Means comparison
# ----------------------------
km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
km.fit(X_scaled)

# Decision regions for K-Means
probas_km = km.transform(grid)  # distances to cluster centers
labels_grid_km = np.argmin(probas_km, axis=1)
labels_points_km = km.predict(X_scaled)

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, labels_grid_km.reshape(xx.shape), alpha=0.25)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], s=10, alpha=0.7)
centers = km.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker="x", s=100)
plt.title(f"K-Means clustering (K={best_k})")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.tight_layout()
plt.show()

# ----------------------------
# 4) Print key diagnostics
# ----------------------------
print("=== Best GMM diagnostics ===")
print("Weights (pi):", np.round(best_gmm.weights_, 3))
print("Means (scaled):\n", np.round(best_gmm.means_, 3))
print("Converged:", best_gmm.converged_)
print("Iterations:", best_gmm.n_iter_)