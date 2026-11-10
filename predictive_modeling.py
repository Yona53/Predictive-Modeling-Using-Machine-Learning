"""
Predictive Modeling Project (scikit-learn)
  Part A: Classification  - breast cancer (malignant vs benign)
  Part B: Regression      - diabetes progression score
Run: python predictive_modeling.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
                             mean_absolute_error, mean_squared_error, r2_score)

OUT = "/mnt/user-data/outputs"
SEED = 42
sns.set_theme(style="whitegrid")

# ==================================================================
# PART A: CLASSIFICATION
# ==================================================================
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target          # target: 1 = benign, 0 = malignant
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, class balance: {y.value_counts(normalize=True).round(2).to_dict()}")

# Hold out a test set FIRST; it is never used for tuning or model selection.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
}

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, fitted = [], {}
for name, m in models.items():
    cv_auc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc")   # train-only CV
    m.fit(X_tr, y_tr)
    fitted[name] = m
    pred, proba = m.predict(X_te), m.predict_proba(X_te)[:, 1]
    rows.append({
        "Model": name,
        "CV AUC (train)": f"{cv_auc.mean():.3f} +/- {cv_auc.std():.3f}",
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred),
        "Recall": recall_score(y_te, pred),
        "F1": f1_score(y_te, pred),
        "Test AUC": roc_auc_score(y_te, proba),
    })
clf_results = pd.DataFrame(rows).set_index("Model")
print("\nCLASSIFICATION RESULTS\n", clf_results.round(3).to_string())

# Tune the random forest (grid search on training data only)
grid = GridSearchCV(RandomForestClassifier(random_state=SEED),
                    {"n_estimators": [100, 300], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 3]},
                    cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
best_rf = grid.best_estimator_
print("\nTuned RF params:", grid.best_params_,
      f"| CV AUC {grid.best_score_:.3f} | Test AUC {roc_auc_score(y_te, best_rf.predict_proba(X_te)[:, 1]):.3f}")

# ---- classification figure ----
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Part A: Predicting Breast Cancer Diagnosis", fontsize=20, fontweight="bold")

for ax, (name, m) in zip(axes[0], fitted.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_te, m.predict(X_te)),
                           display_labels=["Malignant", "Benign"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}\nAccuracy {accuracy_score(y_te, m.predict(X_te)):.1%}", fontweight="bold")
    ax.grid(False)

ax = axes[1][0]
for name, m in fitted.items():
    fpr, tpr, _ = roc_curve(y_te, m.predict_proba(X_te)[:, 1])
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]):.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=.5, label="Random guess")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves (test set)", fontweight="bold"); ax.legend(loc="lower right")

ax = axes[1][1]
imp = pd.Series(fitted["Random Forest"].feature_importances_, index=X.columns).nlargest(10).sort_values()
ax.barh(imp.index, imp.values, color=sns.color_palette("deep")[2])
ax.set_title("Top 10 features (Random Forest)", fontweight="bold")

ax = axes[1][2]
metrics = clf_results[["Accuracy", "Precision", "Recall", "F1", "Test AUC"]].astype(float)
metrics.T.plot.bar(ax=ax, rot=0, width=.8)
ax.set_ylim(0.85, 1.0); ax.set_title("Metric comparison (test set)", fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{OUT}/classification_results.png", dpi=120)
plt.close()

# ==================================================================
# PART B: REGRESSION
# ==================================================================
d = load_diabetes(as_frame=True)
Xr, yr = d.data, d.target
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=SEED)

reg_models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=3, random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED),
}
rrows, rfit = [], {}
for name, m in reg_models.items():
    cv_r2 = cross_val_score(m, Xr_tr, yr_tr, cv=5, scoring="r2")
    m.fit(Xr_tr, yr_tr); rfit[name] = m
    p = m.predict(Xr_te)
    rrows.append({"Model": name, "CV R2 (train)": f"{cv_r2.mean():.3f}",
                  "Test R2": r2_score(yr_te, p), "Test MAE": mean_absolute_error(yr_te, p),
                  "Test RMSE": np.sqrt(mean_squared_error(yr_te, p))})
reg_results = pd.DataFrame(rrows).set_index("Model")
print("\nREGRESSION RESULTS\n", reg_results.round(3).to_string())
baseline_mae = mean_absolute_error(yr_te, np.full_like(yr_te, yr_tr.mean(), dtype=float))
print(f"Baseline (always predict train mean) MAE: {baseline_mae:.1f}")

best_name = reg_results["Test R2"].astype(float).idxmax()
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle("Part B: Predicting Diabetes Progression", fontsize=20, fontweight="bold")
p = rfit[best_name].predict(Xr_te)
axes[0].scatter(yr_te, p, alpha=.7, color=sns.color_palette("deep")[0])
lim = [yr.min(), yr.max()]
axes[0].plot(lim, lim, "r--", label="Perfect prediction")
axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted"); axes[0].legend()
axes[0].set_title(f"{best_name}: predicted vs actual", fontweight="bold")
axes[1].scatter(p, yr_te - p, alpha=.7, color=sns.color_palette("deep")[1])
axes[1].axhline(0, color="r", ls="--")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
axes[1].set_title("Residuals (should be patternless)", fontweight="bold")
reg_results["Test R2"].astype(float).plot.barh(ax=axes[2], color=sns.color_palette("deep")[2])
axes[2].set_title("Test R2 by model", fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(f"{OUT}/regression_results.png", dpi=120)

# ==================================================================
# REPORT
# ==================================================================
best_clf = clf_results["Test AUC"].astype(float).idxmax()
with open(f"{OUT}/model_report.md", "w") as f:
    f.write("# Predictive Modeling Report\n\n## Part A: Classification (breast cancer)\n\n")
    f.write(clf_results.round(3).to_markdown() + "\n\n")
    f.write(f"Tuned Random Forest params: `{grid.best_params_}`\n\n")
    f.write("## Part B: Regression (diabetes progression)\n\n")
    f.write(reg_results.round(3).to_markdown() + f"\n\nBaseline MAE (predict the mean): {baseline_mae:.1f}\n")
print("\nBest classifier by test AUC:", best_clf, "| Best regressor by test R2:", best_name)
