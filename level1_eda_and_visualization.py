# %% [markdown]
# # Codveda Data Analytics Internship - Level 1 (Basic)
# **Task 2: Exploratory Data Analysis (EDA)** and **Task 3: Basic Data Visualization**
# Dataset: Diabetes Progression Dataset (442 patients, 10 baseline measures + 1-year disease progression).
# Source: Efron, Hastie, Johnstone & Tibshirani (2004), "Least Angle Regression", Annals of Statistics.
# Loaded in raw (unscaled) form via scikit-learn.

# %% Setup
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes

HERE = Path(__file__).parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook", palette="deep")
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

# %% Load data
raw = load_diabetes(as_frame=True, scaled=False)
df = raw.frame.rename(columns={
    "s1": "total_cholesterol", "s2": "ldl", "s3": "hdl", "s4": "tch_ratio",
    "s5": "log_triglycerides", "s6": "glucose", "target": "disease_progression"})
df["sex"] = df["sex"].map({1.0: "Group 1", 2.0: "Group 2"})  # dataset codes sex as 1/2
log("Shape:", df.shape)
log("\nMissing values per column:\n", df.isna().sum().to_string())
log("Duplicate rows:", df.duplicated().sum())
df.to_csv(HERE / "diabetes_dataset.csv", index=False)

# %% ================= TASK 2: EDA =================
num = df.select_dtypes("number")
summary = pd.DataFrame({"mean": num.mean(), "median": num.median(),
                        "mode": num.mode().iloc[0], "std_dev": num.std(),
                        "min": num.min(), "max": num.max()}).round(2)
log("\n=== Summary statistics (mean, median, mode, std) ===\n", summary.to_string())
summary.to_csv(HERE / "summary_statistics.csv")
log("\nSkewness:\n", num.skew().round(2).to_string())

def grid(n, cols=4):
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    return fig, axes

# Histograms
fig, axes = grid(len(num.columns))
for ax, c in zip(axes, num.columns):
    sns.histplot(num[c], kde=True, ax=ax, color="#2a9d8f")
    ax.set_title(c.replace("_", " ").title()); ax.set_xlabel("")
for ax in axes[len(num.columns):]: ax.axis("off")
fig.suptitle("Distribution of Numerical Features", fontsize=16, y=1.01)
fig.tight_layout(); fig.savefig(FIG / "eda_histograms.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# Boxplots
fig, axes = grid(len(num.columns))
for ax, c in zip(axes, num.columns):
    sns.boxplot(y=num[c], ax=ax, color="#e9c46a"); ax.set_title(c.replace("_", " ").title()); ax.set_ylabel("")
for ax in axes[len(num.columns):]: ax.axis("off")
fig.suptitle("Boxplots: Spread and Outliers", fontsize=16, y=1.01)
fig.tight_layout(); fig.savefig(FIG / "eda_boxplots.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# Outlier count (IQR rule)
q1, q3 = num.quantile(.25), num.quantile(.75); iqr = q3 - q1
outliers = ((num < q1 - 1.5 * iqr) | (num > q3 + 1.5 * iqr)).sum()
log("\nOutliers per column (1.5*IQR rule):\n", outliers.to_string())

# Scatter plots of key predictors vs outcome
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, c in zip(axes, ["bmi", "bp", "log_triglycerides"]):
    sns.scatterplot(data=df, x=c, y="disease_progression", ax=ax, alpha=.6, color="#264653")
    sns.regplot(data=df, x=c, y="disease_progression", scatter=False, ax=ax, color="#e76f51")
    ax.set_title(f"{c.replace('_',' ').title()} vs Disease Progression")
fig.tight_layout(); fig.savefig(FIG / "eda_scatter_plots.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# Correlation
corr = num.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=.5)
plt.title("Correlation Matrix of Numerical Features", fontsize=14)
plt.tight_layout(); plt.savefig(FIG / "eda_correlation_heatmap.png", dpi=200); plt.close()
t = corr["disease_progression"].drop("disease_progression").sort_values(ascending=False)
log("\nCorrelation with disease_progression:\n", t.round(3).to_string())
pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().sort_values(key=abs, ascending=False)
log("\nStrongest feature-feature correlations:\n", pairs.head(5).round(3).to_string())

# %% ================= TASK 3: BASIC DATA VISUALIZATION =================
df["bmi_category"] = pd.cut(df["bmi"], [0, 18.5, 25, 30, 100],
                            labels=["Underweight", "Normal", "Overweight", "Obese"], right=False)
df["age_group"] = pd.cut(df["age"], [18, 30, 40, 50, 60, 70, 80],
                         labels=["19-30", "31-40", "41-50", "51-60", "61-70", "71-79"])

# 1. Bar plot
bar = df.groupby("bmi_category", observed=True)["disease_progression"].agg(["mean", "count"]).reset_index()
log("\nMean progression by BMI category:\n", bar.round(1).to_string(index=False))
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=bar, x="bmi_category", y="mean", hue="bmi_category", palette="YlOrRd", legend=False, ax=ax)
for i, r in bar.iterrows():
    ax.text(i, r["mean"] + 2, f"{r['mean']:.0f}\n(n={int(r['count'])})", ha="center", fontsize=10)
ax.set_title("Average Disease Progression by BMI Category", fontsize=14, weight="bold")
ax.set_xlabel("BMI Category"); ax.set_ylabel("Mean Disease Progression Score"); ax.set_ylim(0, bar["mean"].max() * 1.2)
fig.tight_layout(); fig.savefig(FIG / "viz_bar_bmi_category.png", dpi=300); plt.close(fig)

# 2. Line chart
line = df.pivot_table(index="age_group", columns="sex", values="disease_progression", aggfunc="mean", observed=True)
fig, ax = plt.subplots(figsize=(8, 5))
for col, m in zip(line.columns, ["o", "s"]):
    ax.plot(line.index.astype(str), line[col], marker=m, linewidth=2.2, label=col)
ax.set_title("Mean Disease Progression Across Age Groups", fontsize=14, weight="bold")
ax.set_xlabel("Age Group (years)"); ax.set_ylabel("Mean Disease Progression Score")
ax.legend(title="Sex code (1/2)"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "viz_line_age_group.png", dpi=300); plt.close(fig)

# 3. Scatter plot
fig, ax = plt.subplots(figsize=(8, 5.5))
sns.scatterplot(data=df, x="bmi", y="disease_progression", hue="bp", size="age", sizes=(20, 120),
                palette="viridis", alpha=.75, ax=ax)
ax.set_title("BMI vs Disease Progression (colour = blood pressure, size = age)", fontsize=13, weight="bold")
ax.set_xlabel("Body Mass Index (kg/m²)"); ax.set_ylabel("Disease Progression Score")
ax.legend(title="BP / Age", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "viz_scatter_bmi_progression.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# 4. Distribution of outcome by sex (seaborn)
fig, ax = plt.subplots(figsize=(8, 5))
sns.kdeplot(data=df, x="disease_progression", hue="sex", fill=True, alpha=.4, common_norm=False, ax=ax)
ax.set_title("Disease Progression Distribution by Sex Code", fontsize=14, weight="bold")
ax.set_xlabel("Disease Progression Score"); ax.set_ylabel("Density")
fig.tight_layout(); fig.savefig(FIG / "viz_kde_by_sex.png", dpi=300); plt.close(fig)

(HERE / "level1_results.txt").write_text("\n".join(LOG))
log("\nSaved figures:", sorted(p.name for p in FIG.glob("*.png")))
