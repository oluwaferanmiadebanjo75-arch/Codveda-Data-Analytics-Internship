# %% [markdown]
# # Codveda Data Analytics Internship - Level 3 (Advanced)
# **Task 1: Predictive Modeling (Classification)**
# Dataset: Wisconsin Diagnostic Breast Cancer (569 tumours, 30 numeric features from cell-nucleus images).
# Goal: predict whether a tumour is malignant (positive class) or benign.
# Models: Logistic Regression, Decision Tree, Random Forest - compared before and after grid-search tuning.

# %% Setup
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

# %% Load and preprocess
bc = load_breast_cancer(as_frame=True)
X = bc.data
y = (bc.target == 0).astype(int)  # original coding: 0 = malignant. Recode so 1 = malignant (positive class)
log("Shape:", X.shape, "| missing values:", int(X.isna().sum().sum()), "| duplicates:", int(X.duplicated().sum()))
log("Class balance (1 = malignant):", y.value_counts().to_dict())
log("Categorical variables: none (all 30 features numeric); target already binary. Feature scaling: StandardScaler in pipeline.")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
log(f"Train: {len(X_tr)} | Test: {len(X_te)}")

# Class distribution plot
plt.figure(figsize=(5, 4))
sns.countplot(x=y.map({0: "Benign", 1: "Malignant"}), hue=y.map({0: "Benign", 1: "Malignant"}),
              palette=["#2a9d8f", "#e76f51"], legend=False)
plt.title("Target Class Distribution"); plt.xlabel(""); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(FIG / "class_distribution.png", dpi=250); plt.close()

# %% Models and grids
def pipe(clf): return Pipeline([("scale", StandardScaler()), ("clf", clf)])
models = {
    "Logistic Regression": (pipe(LogisticRegression(max_iter=5000, random_state=42)),
                            {"clf__C": [0.01, 0.1, 1, 10, 100]}),
    "Decision Tree": (pipe(DecisionTreeClassifier(random_state=42)),
                      {"clf__max_depth": [2, 3, 4, 5, 8, None], "clf__min_samples_leaf": [1, 3, 5, 10],
                       "clf__criterion": ["gini", "entropy"]}),
    "Random Forest": (pipe(RandomForestClassifier(random_state=42, n_jobs=-1)),
                      {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 5, 10],
                       "clf__min_samples_leaf": [1, 3]}),
}

def metrics(model, name, stage):
    p = model.predict(X_te); s = model.predict_proba(X_te)[:, 1]
    return {"Model": name, "Stage": stage, "Accuracy": accuracy_score(y_te, p),
            "Precision": precision_score(y_te, p), "Recall": recall_score(y_te, p),
            "F1": f1_score(y_te, p), "ROC_AUC": roc_auc_score(y_te, s)}

rows, tuned = [], {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, (base, grid) in models.items():
    base.fit(X_tr, y_tr); rows.append(metrics(base, name, "Baseline"))
    gs = GridSearchCV(base, grid, cv=cv, scoring="f1", n_jobs=-1).fit(X_tr, y_tr)
    tuned[name] = gs.best_estimator_; rows.append(metrics(gs.best_estimator_, name, "Tuned"))
    log(f"\n{name}: best params = { {k.replace('clf__',''): v for k, v in gs.best_params_.items()} } | CV F1 = {gs.best_score_:.4f}")

res = pd.DataFrame(rows); res.iloc[:, 2:] = res.iloc[:, 2:].round(4)
res.to_csv(HERE / "model_comparison.csv", index=False)
log("\n=== Test-set performance (positive class = malignant) ===\n", res.to_string(index=False))

# %% Plots
# 1. Metric comparison (tuned models)
t = res[res.Stage == "Tuned"].set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]]
ax = t.plot(kind="bar", figsize=(9, 5), colormap="viridis", width=.8, rot=0)
ax.set_ylim(0.8, 1.02); ax.set_title("Tuned Model Comparison on Test Set", fontsize=14, weight="bold")
ax.set_ylabel("Score"); ax.legend(loc="lower right", ncol=4, fontsize=9)
for c in ax.containers: ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
plt.tight_layout(); plt.savefig(FIG / "model_comparison_tuned.png", dpi=250); plt.close()

# 2. Baseline vs tuned F1
f = res.pivot(index="Model", columns="Stage", values="F1")[["Baseline", "Tuned"]]
ax = f.plot(kind="bar", figsize=(8, 5), color=["#adb5bd", "#2a9d8f"], rot=0)
ax.set_ylim(0.85, 1.0); ax.set_title("F1-Score: Baseline vs Grid-Search Tuned", fontsize=14, weight="bold"); ax.set_ylabel("F1")
for c in ax.containers: ax.bar_label(c, fmt="%.3f", padding=2)
plt.tight_layout(); plt.savefig(FIG / "f1_baseline_vs_tuned.png", dpi=250); plt.close()

# 3. ROC curves
plt.figure(figsize=(6.5, 6))
for name, m in tuned.items():
    fpr, tpr, _ = roc_curve(y_te, m.predict_proba(X_te)[:, 1])
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=.5); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves (Tuned Models)"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(FIG / "roc_curves.png", dpi=250); plt.close()

# 4. Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, m) in zip(axes, tuned.items()):
    ConfusionMatrixDisplay.from_predictions(y_te, m.predict(X_te), display_labels=["Benign", "Malignant"],
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(name)
fig.suptitle("Confusion Matrices (Tuned Models, Test Set)", fontsize=14)
fig.tight_layout(); fig.savefig(FIG / "confusion_matrices.png", dpi=250); plt.close(fig)

# 5. Random forest feature importance
rf = tuned["Random Forest"].named_steps["clf"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(10)
plt.figure(figsize=(8, 5)); imp.plot(kind="barh", color="#264653")
plt.title("Top 10 Features (Tuned Random Forest)"); plt.xlabel("Importance")
plt.tight_layout(); plt.savefig(FIG / "rf_feature_importance.png", dpi=250); plt.close()
log("\nTop 5 features:", list(imp.sort_values(ascending=False).head(5).round(3).items()))

best = res[res.Stage == "Tuned"].sort_values(["F1", "Recall"], ascending=False).iloc[0]
log(f"\nBest tuned model by F1: {best.Model} (F1={best.F1}, Recall={best.Recall}, Precision={best.Precision})")
(HERE / "level3_results.txt").write_text("\n".join(LOG))
