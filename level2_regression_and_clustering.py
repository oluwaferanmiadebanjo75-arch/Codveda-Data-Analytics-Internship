# %% [markdown]
# # Codveda Data Analytics Internship - Level 2 (Intermediate)
# **Task 1: Regression Analysis** (Diabetes Progression dataset, 442 patients)
# **Task 3: Clustering Analysis, K-Means** (UCI Wine dataset, 178 wines, 13 chemical measurements)

# %% Setup
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes, load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (adjusted_rand_score, mean_absolute_error, mean_squared_error,
                             r2_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

# %% ================= TASK 1: REGRESSION ANALYSIS =================
diab = load_diabetes(as_frame=True, scaled=False).frame.rename(columns={
    "s1": "total_cholesterol", "s2": "ldl", "s3": "hdl", "s4": "tch_ratio", "s5": "log_triglycerides",
    "s6": "glucose", "target": "disease_progression"})

# Simple linear regression: predict disease progression from BMI
X, y = diab[["bmi"]], diab["disease_progression"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
log(f"Train size: {len(X_tr)} | Test size: {len(X_te)}")

model = LinearRegression().fit(X_tr, y_tr)
pred_tr, pred_te = model.predict(X_tr), model.predict(X_te)
mse = mean_squared_error(y_te, pred_te)
res = {"intercept": model.intercept_, "coef_bmi": model.coef_[0],
       "R2_train": r2_score(y_tr, pred_tr), "R2_test": r2_score(y_te, pred_te),
       "MSE_test": mse, "RMSE_test": np.sqrt(mse), "MAE_test": mean_absolute_error(y_te, pred_te)}
log("\n=== Simple Linear Regression: disease_progression ~ bmi ===")
for k, v in res.items(): log(f"{k:10s}: {v:10.3f}")
log(f"\nInterpretation: each +1 unit of BMI is associated with a {model.coef_[0]:.2f}-point rise in the "
    f"disease progression score. BMI alone explains {res['R2_test']*100:.1f}% of test-set variance.")
pd.Series(res).round(3).to_csv(HERE / "regression_metrics.csv", header=["value"])

# Fit plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X_tr, y_tr, alpha=.5, label="Training data", color="#2a9d8f")
axes[0].scatter(X_te, y_te, alpha=.9, label="Test data", color="#e76f51")
xs = np.linspace(X["bmi"].min(), X["bmi"].max(), 100).reshape(-1, 1)
axes[0].plot(xs, model.predict(pd.DataFrame(xs, columns=["bmi"])), color="black", lw=2.5, label="Fitted line")
axes[0].set(title="Simple Linear Regression: BMI → Disease Progression", xlabel="BMI (kg/m²)",
            ylabel="Disease Progression Score"); axes[0].legend()
resid = y_te - pred_te
axes[1].scatter(pred_te, resid, alpha=.8, color="#264653"); axes[1].axhline(0, color="red", ls="--")
axes[1].set(title="Residual Plot (Test Set)", xlabel="Predicted Value", ylabel="Residual")
fig.tight_layout(); fig.savefig(FIG / "regression_fit_and_residuals.png", dpi=250); plt.close(fig)

# Actual vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_te, pred_te, alpha=.8, color="#2a9d8f")
lims = [y_te.min(), y_te.max()]; plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Actual vs Predicted (Test Set)"); plt.legend()
plt.tight_layout(); plt.savefig(FIG / "regression_actual_vs_predicted.png", dpi=250); plt.close()

# Bonus: multiple regression using all features (standardised so coefficients are comparable)
Xa = diab.drop(columns="disease_progression"); Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
    Xa, y, test_size=0.2, random_state=42)
sc = StandardScaler().fit(Xa_tr); m2 = LinearRegression().fit(sc.transform(Xa_tr), ya_tr)
p2 = m2.predict(sc.transform(Xa_te))
log(f"\nBonus - multiple regression (all 10 features): R2_test={r2_score(ya_te, p2):.3f}, "
    f"RMSE_test={np.sqrt(mean_squared_error(ya_te, p2)):.2f}")
coefs = pd.Series(m2.coef_, index=Xa.columns).sort_values()
log("Standardised coefficients:\n", coefs.round(2).to_string())
plt.figure(figsize=(8, 5))
coefs.plot(kind="barh", color=np.where(coefs > 0, "#e76f51", "#2a9d8f"))
plt.title("Multiple Regression: Standardised Coefficients"); plt.xlabel("Effect on progression score (per 1 SD)")
plt.tight_layout(); plt.savefig(FIG / "regression_multiple_coefficients.png", dpi=250); plt.close()

# %% ================= TASK 3: CLUSTERING (K-MEANS) =================
wine = load_wine(as_frame=True)
Xw = wine.data; true_labels = wine.target  # labels are NOT used to fit, only to sanity-check at the end
log(f"\n\nWine dataset shape: {Xw.shape}; missing values: {int(Xw.isna().sum().sum())}")

scaler = StandardScaler(); Xs = scaler.fit_transform(Xw)  # standardise

ks = range(1, 11); inertia, sil = [], {}
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    inertia.append(km.inertia_)
    if k > 1: sil[k] = silhouette_score(Xs, km.labels_)

# Elbow point = point furthest from the line joining first and last inertia values
pts = np.column_stack([np.arange(1, 11), inertia]).astype(float)
pn = (pts - pts.min(0)) / (pts.max(0) - pts.min(0))
d = np.abs((pn[-1, 0] - pn[0, 0]) * (pn[0, 1] - pn[:, 1]) - (pn[0, 0] - pn[:, 0]) * (pn[-1, 1] - pn[0, 1]))
k_opt = int(pts[np.argmax(d), 0])
log("Inertia by k:", [round(v, 1) for v in inertia])
log("Silhouette by k:", {k: round(v, 3) for k, v in sil.items()})
log(f"Optimal k (elbow): {k_opt} | best silhouette k: {max(sil, key=sil.get)}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].plot(list(ks), inertia, "o-", color="#264653", lw=2)
axes[0].axvline(k_opt, color="red", ls="--", label=f"Elbow at k={k_opt}")
axes[0].set(title="Elbow Method", xlabel="Number of clusters (k)", ylabel="Inertia (WCSS)"); axes[0].legend()
axes[1].plot(list(sil), list(sil.values()), "s-", color="#e76f51", lw=2)
axes[1].set(title="Silhouette Score by k", xlabel="Number of clusters (k)", ylabel="Silhouette score")
fig.tight_layout(); fig.savefig(FIG / "kmeans_elbow_silhouette.png", dpi=250); plt.close(fig)

km = KMeans(n_clusters=k_opt, n_init=10, random_state=42).fit(Xs)
Xw = Xw.assign(cluster=km.labels_ + 1)
log(f"\nFinal model: k={k_opt}, silhouette={silhouette_score(Xs, km.labels_):.3f}")
log("Cluster sizes:", Xw["cluster"].value_counts().sort_index().to_dict())
profile = Xw.groupby("cluster").mean().round(2)
profile.to_csv(HERE / "cluster_profiles.csv"); log("\nCluster profiles (feature means):\n", profile.T.to_string())
ari = adjusted_rand_score(true_labels, km.labels_)
log(f"\nSanity check vs. true wine cultivars (not used in fitting): Adjusted Rand Index = {ari:.3f}")

# 2D scatter plots
pca = PCA(n_components=2, random_state=42); Z = pca.fit_transform(Xs)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=Xw["cluster"], palette="Set1", s=60, ax=axes[0], edgecolor="white")
cen = pca.transform(km.cluster_centers_); axes[0].scatter(cen[:, 0], cen[:, 1], c="black", marker="X", s=250, label="Centroids")
axes[0].set(title="K-Means Clusters (PCA projection)",
            xlabel=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)",
            ylabel=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)"); axes[0].legend(title="Cluster")
sns.scatterplot(data=Xw, x="flavanoids", y="alcohol", hue="cluster", palette="Set1", s=60, ax=axes[1], edgecolor="white")
axes[1].set(title="Clusters on Two Original Features", xlabel="Flavanoids", ylabel="Alcohol (%)")
axes[1].legend(title="Cluster")
fig.tight_layout(); fig.savefig(FIG / "kmeans_cluster_scatter.png", dpi=250); plt.close(fig)

(HERE / "level2_results.txt").write_text("\n".join(LOG))
