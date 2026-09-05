import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
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
print("XGBOOST HYPERPARAMETER TUNING")
print("=" * 60)

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)


# ============================================================
# 2. DEFINE BASE XGBOOST MODEL
# ============================================================

xgb_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)


# ============================================================
# 3. DEFINE PARAMETER SEARCH SPACE
# ============================================================

param_grid = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7]
}


# ============================================================
# 4. DEFINE CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 5. RANDOMIZED SEARCH
# ============================================================

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=30,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


print("\n" + "=" * 60)
print("STARTING HYPERPARAMETER SEARCH")
print("=" * 60)

random_search.fit(
    X_train,
    y_train
)


# ============================================================
# 6. BEST PARAMETERS
# ============================================================

print("\n" + "=" * 60)
print("BEST PARAMETERS")
print("=" * 60)

print(random_search.best_params_)


# ============================================================
# 7. BEST CROSS-VALIDATION SCORE
# ============================================================

print("\n" + "=" * 60)
print("BEST CROSS-VALIDATION F1")
print("=" * 60)

print(
    f"Best CV F1: {random_search.best_score_:.4f}"
)


# ============================================================
# 8. GET BEST MODEL
# ============================================================

best_model = random_search.best_estimator_


# ============================================================
# 9. TEST SET PREDICTIONS
# ============================================================

y_pred = best_model.predict(X_test)

y_prob = best_model.predict_proba(X_test)[:, 1]


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("TUNED MODEL - CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"]
    )
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("TUNED MODEL - CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# 12. ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n" + "=" * 60)
print("TUNED MODEL - ROC-AUC")
print("=" * 60)

print(f"ROC-AUC: {roc_auc:.4f}")


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame(
    {
        "feature": X_train.columns,
        "importance": best_model.feature_importances_
    }
)

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

print("\n" + "=" * 60)
print("TUNED MODEL - FEATURE IMPORTANCE")
print("=" * 60)

print(
    feature_importance.to_string(index=False)
)