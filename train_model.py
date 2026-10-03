import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "diabetes.csv"
MODEL_DIR = BASE / "models"
OUTPUT_DIR = BASE / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"

# Project preprocessing described in the paper:
# zero values in these clinical columns are treated as missing.
ZERO_AS_MISSING = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]


def preprocess(df):
    df = df.copy()

    # Remove duplicate records.
    df = df.drop_duplicates().reset_index(drop=True)

    # Replace clinically invalid zero values with NaN.
    for col in ZERO_AS_MISSING:
        df[col] = df[col].replace(0, np.nan)

    # Median imputation.
    medians = {}
    for col in ZERO_AS_MISSING:
        medians[col] = float(df[col].median())
        df[col] = df[col].fillna(medians[col])

    return df, medians


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "diabetes.csv was not found. Put the Pima Indians Diabetes CSV "
            "in the project root with the expected column names."
        )

    raw = pd.read_csv(DATA_FILE)

    missing = [c for c in FEATURES + [TARGET] if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw = raw[FEATURES + [TARGET]].copy()
    df, medians = preprocess(raw)

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Fixed hyperparameters make the experiment reproducible.
    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_weight=1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset_rows_after_preprocessing": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    # 5-fold stratified cross-validation.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_result = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc"
        },
        n_jobs=-1
    )

    cv_summary = {}
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        values = cv_result[f"test_{metric}"]
        cv_summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "folds": [float(v) for v in values]
        }

    metrics["cross_validation"] = cv_summary

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Non-Diabetic", "Diabetic"],
        yticklabels=["Non-Diabetic", "Diabetic"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("XGBoost Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"XGBoost (AUC = {metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=300)
    plt.close()

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(8, 5))
    importance.plot(kind="barh")
    plt.xlabel("Importance")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300)
    plt.close()

    # CV means with error bars.
    names = list(cv_summary.keys())
    means = [cv_summary[n]["mean"] for n in names]
    stds = [cv_summary[n]["std"] for n in names]
    plt.figure(figsize=(8, 5))
    plt.bar(names, means, yerr=stds, capsize=4)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("5-Fold Cross-Validation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cv_scores.png", dpi=300)
    plt.close()

    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "zero_as_missing": ZERO_AS_MISSING,
        "imputation_medians": medians,
        "threshold": 0.50,
        "hyperparameters": model.get_params()
    }

    joblib.dump(model, MODEL_DIR / "xgboost_diabetes.joblib")
    joblib.dump(metadata, MODEL_DIR / "model_metadata.joblib")

    print("\nTraining complete.")
    print(json.dumps(metrics, indent=2))
    print("\nSaved model, metrics and figures in models/ and outputs/.")


if __name__ == "__main__":
    main()
