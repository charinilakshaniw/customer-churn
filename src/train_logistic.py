import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD TRAINING AND TEST DATA
# ============================================================

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()


print("=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)


# ============================================================
# 2. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 3. TRAIN MODEL
# ============================================================

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(X_train_scaled, y_train)


# ============================================================
# 4. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_scaled)

y_prob = model.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# 5. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
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
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)

print(cm)


# ============================================================
# 7. ROC-AUC
# ============================================================

roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("ROC-AUC")
print("=" * 60)

print(f"ROC-AUC: {roc_auc:.4f}")


# ============================================================
# 8. MODEL COEFFICIENTS
# ============================================================

coefficients = pd.DataFrame(
    {
        "feature": X_train.columns,
        "coefficient": model.coef_[0]
    }
)

coefficients["absolute_coefficient"] = (
    coefficients["coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "absolute_coefficient",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE COEFFICIENTS")
print("=" * 60)

print(
    coefficients[
        ["feature", "coefficient"]
    ].to_string(index=False)
)