import os
import pandas as pd
import matplotlib.pyplot as plt
import shap

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")


print("=" * 60)
print("SHAP MODEL EXPLAINABILITY")
print("=" * 60)

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)


# ============================================================
# 2. LOAD FINAL XGBOOST MODEL
# ============================================================

model = XGBClassifier()

model.load_model(
    "models/final_xgboost.json"
)

print("\nFinal XGBoost model loaded successfully.")


# ============================================================
# 3. CREATE SHAP EXPLAINER
# ============================================================

print("\n" + "=" * 60)
print("CREATING SHAP EXPLAINER")
print("=" * 60)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

print("SHAP values calculated successfully.")


# ============================================================
# 4. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    "outputs/shap",
    exist_ok=True
)


# ============================================================
# 5. SHAP SUMMARY PLOT
# ============================================================

print("\n" + "=" * 60)
print("CREATING SHAP SUMMARY PLOT")
print("=" * 60)

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.tight_layout()

plt.savefig(
    "outputs/shap/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: outputs/shap/shap_summary.png"
)


# ============================================================
# 6. SHAP BAR PLOT
# ============================================================

print("\n" + "=" * 60)
print("CREATING SHAP FEATURE IMPORTANCE PLOT")
print("=" * 60)

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "outputs/shap/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: outputs/shap/shap_feature_importance.png"
)


# ============================================================
# 7. CALCULATE MEAN ABSOLUTE SHAP VALUES
# ============================================================

mean_shap = pd.DataFrame(
    {
        "feature": X_test.columns,
        "mean_absolute_shap": abs(shap_values).mean(axis=0)
    }
)

mean_shap = mean_shap.sort_values(
    "mean_absolute_shap",
    ascending=False
)


print("\n" + "=" * 60)
print("SHAP FEATURE IMPORTANCE")
print("=" * 60)

print(
    mean_shap.to_string(index=False)
)


# ============================================================
# 8. SAVE SHAP FEATURE IMPORTANCE
# ============================================================

mean_shap.to_csv(
    "data/processed/shap_feature_importance.csv",
    index=False
)

print(
    "\nSaved: data/processed/shap_feature_importance.csv"
)


# ============================================================
# 9. CREATE SHAP VALUES DATASET
# ============================================================

shap_output = X_test.copy()

for i, feature in enumerate(X_test.columns):

    shap_output[
        f"{feature}_shap"
    ] = shap_values[:, i]


# ============================================================
# 10. SAVE SHAP VALUES
# ============================================================

shap_output.to_csv(
    "data/processed/shap_values.csv",
    index=False
)

print(
    "Saved: data/processed/shap_values.csv"
)


# ============================================================
# 11. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETE")
print("=" * 60)