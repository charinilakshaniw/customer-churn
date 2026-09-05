import os
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD TRAINING AND TEST DATA
# ============================================================

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()


print("=" * 60)
print("FINAL XGBOOST MODEL")
print("=" * 60)

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)


# ============================================================
# 2. DEFINE FINAL MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=3,
    random_state=42,
    eval_metric="logloss"
)


# ============================================================
# 3. TRAIN FINAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING FINAL MODEL")
print("=" * 60)

model.fit(
    X_train,
    y_train
)

print("Final model trained successfully.")


# ============================================================
# 4. MAKE TEST PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# 5. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL - CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"]
    )
)


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL - CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# 7. ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n" + "=" * 60)
print("FINAL MODEL - ROC-AUC")
print("=" * 60)

print(f"ROC-AUC: {roc_auc:.4f}")


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame(
    {
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }
)

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FINAL MODEL - FEATURE IMPORTANCE")
print("=" * 60)

print(
    feature_importance.to_string(index=False)
)


# ============================================================
# 9. CREATE MODEL OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# 10. SAVE FINAL MODEL
# ============================================================

model.save_model(
    "models/final_xgboost.json"
)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    "models/final_xgboost.json"
)


# ============================================================
# 11. SAVE FEATURE IMPORTANCE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

feature_importance.to_csv(
    "data/processed/feature_importance.csv",
    index=False
)

print(
    "data/processed/feature_importance.csv"
)