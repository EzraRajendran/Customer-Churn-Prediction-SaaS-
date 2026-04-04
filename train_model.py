"""
train_model.py
Trains a Gradient Boosting classifier on the churn dataset,
evaluates it, and saves the pipeline + feature list to models/.

Run: python src/train_model.py
"""

import os, json, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.preprocessing     import StandardScaler, OneHotEncoder
from sklearn.compose           import ColumnTransformer
from sklearn.pipeline          import Pipeline
from sklearn.ensemble          import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.metrics           import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/churn_data.csv")
print(f"Data loaded: {df.shape}  |  Churn rate: {df['churn'].mean():.1%}\n")

TARGET      = "churn"
NUM_COLS    = ["tenure", "monthly_charges", "total_charges",
               "num_products", "support_tickets", "login_frequency"]
CAT_COLS    = ["contract_type", "payment_method",
               "tech_support", "online_backup"]
BIN_COLS    = ["senior_citizen"]
FEATURE_COLS= NUM_COLS + CAT_COLS + BIN_COLS

X = df[FEATURE_COLS]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 2. Preprocessing pipeline ─────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(),                              NUM_COLS + BIN_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
])

# ── 3. Compare models ─────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, random_state=42),
}

print("── Model Comparison (5-Fold CV ROC-AUC) ──")
results = {}
for name, clf in models.items():
    pipe  = Pipeline([("pre", preprocessor), ("clf", clf)])
    scores= cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
    results[name] = scores.mean()
    print(f"  {name:<25}  AUC = {scores.mean():.4f} ± {scores.std():.4f}")

best_name = max(results, key=results.get)
print(f"\n✅ Best model: {best_name}  (AUC={results[best_name]:.4f})\n")

# ── 4. Train best model ───────────────────────────────────────────────────────
best_pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", models[best_name])
])
best_pipeline.fit(X_train, y_train)

y_pred      = best_pipeline.predict(X_test)
y_proba     = best_pipeline.predict_proba(X_test)[:, 1]
test_auc    = roc_auc_score(y_test, y_proba)

print("── Test Set Results ──")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
print(f"ROC-AUC: {test_auc:.4f}")

# ── 5. Save artefacts ─────────────────────────────────────────────────────────
joblib.dump(best_pipeline, "models/churn_model.pkl")
meta = {
    "best_model":    best_name,
    "test_auc":      round(test_auc, 4),
    "feature_cols":  FEATURE_COLS,
    "num_cols":      NUM_COLS,
    "cat_cols":      CAT_COLS,
    "bin_cols":      BIN_COLS,
    "model_comparison": {k: round(v, 4) for k, v in results.items()},
}
with open("models/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n✅ Model saved → models/churn_model.pkl")
print("✅ Metadata  → models/model_meta.json")

# ── 6. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Customer Churn Prediction — Model Evaluation", fontsize=14, fontweight="bold")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[0].plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {test_auc:.3f}")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
axes[0].legend()

# Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                       display_labels=["No Churn", "Churn"]).plot(ax=axes[1], colorbar=False)
axes[1].set_title("Confusion Matrix")

# Model Comparison Bar
model_names  = list(results.keys())
model_scores = list(results.values())
bars = axes[2].barh(model_names, model_scores, color=["#4C72B0","#55A868","#C44E52"])
axes[2].set(title="Model Comparison (CV AUC)", xlabel="ROC-AUC", xlim=[0.5, 1.0])
for bar, score in zip(bars, model_scores):
    axes[2].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{score:.3f}", va="center")

plt.tight_layout()
plt.savefig("notebooks/model_evaluation.png", dpi=150, bbox_inches="tight")
print("✅ Evaluation plots → notebooks/model_evaluation.png")

# Feature Importance
if hasattr(best_pipeline["clf"], "feature_importances_"):
    ohe_features = list(
        best_pipeline["pre"]
        .named_transformers_["cat"]
        .get_feature_names_out(CAT_COLS)
    )
    all_features = NUM_COLS + BIN_COLS + ohe_features
    importances  = best_pipeline["clf"].feature_importances_
    fi_df = pd.DataFrame({"Feature": all_features, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(12)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=fi_df, x="Importance", y="Feature", palette="Blues_r", ax=ax2)
    ax2.set_title("Top 12 Feature Importances", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("notebooks/feature_importance.png", dpi=150, bbox_inches="tight")
    print("✅ Feature importance → notebooks/feature_importance.png")

print("\n🎉 Training complete!")
