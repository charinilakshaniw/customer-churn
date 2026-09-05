import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()


print("=" * 60)
print("CROSS-VALIDATION MODEL EVALUATION")
print("=" * 60)

print("\nTraining data:")
print(X_train.shape)

print("\nTarget distribution:")
print(y_train.value_counts())


# ============================================================
# 2. DEFINE CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 3. DEFINE MODELS
# ============================================================

# Logistic Regression
logistic_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        )
    ]
)


# XGBoost
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)


# ============================================================
# 4. DEFINE EVALUATION METRICS
# ============================================================

scoring = {
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}


# ============================================================
# 5. LOGISTIC REGRESSION CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION - 5-FOLD CROSS-VALIDATION")
print("=" * 60)

logistic_results = cross_validate(
    logistic_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring
)


print("\nFold ROC-AUC:")
print(logistic_results["test_roc_auc"])

print("\nFold Precision:")
print(logistic_results["test_precision"])

print("\nFold Recall:")
print(logistic_results["test_recall"])

print("\nFold F1:")
print(logistic_results["test_f1"])


# ============================================================
# 6. XGBOOST CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST - 5-FOLD CROSS-VALIDATION")
print("=" * 60)

xgb_results = cross_validate(
    xgb_model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring
)


print("\nFold ROC-AUC:")
print(xgb_results["test_roc_auc"])

print("\nFold Precision:")
print(xgb_results["test_precision"])

print("\nFold Recall:")
print(xgb_results["test_recall"])

print("\nFold F1:")
print(xgb_results["test_f1"])


# ============================================================
# 7. CALCULATE MEAN AND STANDARD DEVIATION
# ============================================================

results = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "XGBoost"
        ],

        "ROC-AUC Mean": [
            logistic_results["test_roc_auc"].mean(),
            xgb_results["test_roc_auc"].mean()
        ],

        "ROC-AUC Std": [
            logistic_results["test_roc_auc"].std(),
            xgb_results["test_roc_auc"].std()
        ],

        "Precision Mean": [
            logistic_results["test_precision"].mean(),
            xgb_results["test_precision"].mean()
        ],

        "Recall Mean": [
            logistic_results["test_recall"].mean(),
            xgb_results["test_recall"].mean()
        ],

        "F1 Mean": [
            logistic_results["test_f1"].mean(),
            xgb_results["test_f1"].mean()
        ]
    }
)


# ============================================================
# 8. DISPLAY FINAL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("CROSS-VALIDATION RESULTS")
print("=" * 60)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 9. INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print(
    "\nThe mean scores show the average performance "
    "across 5 stratified folds."
)

print(
    "The ROC-AUC standard deviation shows how much "
    "performance varies between folds."
)

print(
    "\nLower standard deviation generally indicates "
    "more consistent model performance."
)