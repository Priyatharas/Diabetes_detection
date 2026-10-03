import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "diabetes.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"

# Project preprocessing described in the paper:
# zero values in these clinical columns are treated as missing and replaced with the feature median
ZERO_AS_MISSING = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]

def preprocess(df):
    df = df.copy()
    # Remove duplicate records
    df = df.drop_duplicates().reset_index(drop=True)

    # Replace zero values with NaN for invalid clinical values
    for col in ZERO_AS_MISSING:
        df[col] = df[col].replace(0, np.nan)

    # Median imputation as in paper Section VI.B Fig 3
    medians = {}
    for col in ZERO_AS_MISSING:
        medians[col] = float(df[col].median())
        df[col] = df[col].fillna(medians[col])

    return df, medians

def create_model():
    return XGBClassifier(
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

def train_and_evaluate():
    print("=" * 60)
    print("XGBoost Diabetes Predictive Model Training & Ablation Pipeline")
    print("=" * 60)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing {DATA_FILE}")

    raw = pd.read_csv(DATA_FILE)
    print(f"Loaded raw dataset: {raw.shape[0]} rows, {raw.shape[1]} columns")

    missing = [c for c in FEATURES + [TARGET] if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df, medians = preprocess(raw[FEATURES + [TARGET]])
    print(f"Dataset after preprocessing: {len(df)} rows")
    print(f"Imputation medians: {medians}")

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    # Baseline Model Training
    baseline_model = create_model()
    baseline_model.fit(X_train, y_train)

    y_pred = baseline_model.predict(X_test)
    y_prob = baseline_model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    baseline_metrics = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(round(accuracy_score(y_test, y_pred), 4)),
        "precision": float(round(precision_score(y_test, y_pred, zero_division=0), 4)),
        "recall": float(round(recall_score(y_test, y_pred, zero_division=0), 4)),
        "f1": float(round(f1_score(y_test, y_pred, zero_division=0), 4)),
        "roc_auc": float(round(roc_auc_score(y_test, y_prob), 4)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist()
        }
    }

    # 5-fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
    for tr_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        m_fold = create_model()
        m_fold.fit(X_tr, y_tr)
        f_pred = m_fold.predict(X_val)
        f_prob = m_fold.predict_proba(X_val)[:, 1]
        fold_scores["accuracy"].append(float(round(accuracy_score(y_val, f_pred), 4)))
        fold_scores["precision"].append(float(round(precision_score(y_val, f_pred, zero_division=0), 4)))
        fold_scores["recall"].append(float(round(recall_score(y_val, f_pred, zero_division=0), 4)))
        fold_scores["f1"].append(float(round(f1_score(y_val, f_pred, zero_division=0), 4)))
        fold_scores["roc_auc"].append(float(round(roc_auc_score(y_val, f_prob), 4)))

    cv_summary = {}
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        vals = fold_scores[metric]
        cv_summary[metric] = {
            "mean": float(round(float(np.mean(vals)), 4)),
            "std": float(round(float(np.std(vals)), 4)),
            "folds": vals
        }
    baseline_metrics["cross_validation"] = cv_summary

    # Classification Report
    cr = classification_report(y_test, y_pred, digits=4, zero_division=0, output_dict=True)
    baseline_metrics["classification_report"] = cr
    (OUTPUT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, y_pred, digits=4, zero_division=0), encoding="utf-8"
    )

    # Feature Importance
    importances = baseline_model.feature_importances_
    feat_imp = [
        {"feature": feat, "importance": float(round(imp, 5)), "percentage": float(round(imp * 100, 2))}
        for feat, imp in zip(FEATURES, importances)
    ]
    feat_imp.sort(key=lambda x: x["importance"], reverse=True)
    baseline_metrics["feature_importance"] = feat_imp

    # Save metrics.json
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(baseline_metrics, f, indent=2)

    # Generate Confusion Matrix Chart
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Predicted No Diabetes", "Predicted Diabetes"],
        yticklabels=["Actual No Diabetes", "Actual Diabetes"]
    )
    plt.title(f"XGBoost Confusion Matrix (Accuracy: {baseline_metrics['accuracy']:.1%})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # Generate ROC Curve Chart
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#0d9488", lw=2, label=f"XGBoost (AUC = {baseline_metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=200)
    plt.close()

    # =========================================================================
    # ABLATION EXPERIMENTS (Single Feature Removal & Feature Combinations)
    # =========================================================================
    print("\nRunning ablation experiments...")
    ablation_experiments = []

    # 1. Baseline Experiment (All features)
    ablation_experiments.append({
        "experiment": "Baseline (All Features)",
        "removed_feature": "None",
        "feature_count": len(FEATURES),
        "features_used": FEATURES,
        "accuracy": baseline_metrics["accuracy"],
        "precision": baseline_metrics["precision"],
        "recall": baseline_metrics["recall"],
        "f1": baseline_metrics["f1"],
        "roc_auc": baseline_metrics["roc_auc"],
        "accuracy_delta": 0.0,
        "f1_delta": 0.0
    })

    # 2. Individual Feature Removals
    for feat in FEATURES:
        sub_features = [f for f in FEATURES if f != feat]
        sub_X_train = X_train[sub_features]
        sub_X_test = X_test[sub_features]

        m = create_model()
        m.fit(sub_X_train, y_train)
        pred = m.predict(sub_X_test)
        prob = m.predict_proba(sub_X_test)[:, 1]

        acc = float(round(accuracy_score(y_test, pred), 4))
        prec = float(round(precision_score(y_test, pred, zero_division=0), 4))
        rec = float(round(recall_score(y_test, pred, zero_division=0), 4))
        f1 = float(round(f1_score(y_test, pred, zero_division=0), 4))
        auc = float(round(roc_auc_score(y_test, prob), 4))

        ablation_experiments.append({
            "experiment": f"Remove {feat}",
            "removed_feature": feat,
            "feature_count": len(sub_features),
            "features_used": sub_features,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "accuracy_delta": float(round(acc - baseline_metrics["accuracy"], 4)),
            "f1_delta": float(round(f1 - baseline_metrics["f1"], 4))
        })

    # 3. Feature Combination Analysis
    combinations = [
        ("Glucose Only", ["Glucose"]),
        ("Glucose + BMI", ["Glucose", "BMI"]),
        ("Glucose + BMI + Age", ["Glucose", "BMI", "Age"]),
        ("Glucose + BMI + Age + Insulin", ["Glucose", "BMI", "Age", "Insulin"]),
        ("All 8 Features", FEATURES)
    ]

    combination_results = []
    for name, feats in combinations:
        c_X_train = X_train[feats]
        c_X_test = X_test[feats]

        m = create_model()
        m.fit(c_X_train, y_train)
        pred = m.predict(c_X_test)
        prob = m.predict_proba(c_X_test)[:, 1]

        acc = float(round(accuracy_score(y_test, pred), 4))
        prec = float(round(precision_score(y_test, pred, zero_division=0), 4))
        rec = float(round(recall_score(y_test, pred, zero_division=0), 4))
        f1 = float(round(f1_score(y_test, pred, zero_division=0), 4))
        auc = float(round(roc_auc_score(y_test, prob), 4))

        combination_results.append({
            "combination_name": name,
            "feature_count": len(feats),
            "features": feats,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc
        })

    ablation_payload = {
        "description": "Ablation analysis evaluates how XGBoost model performance changes when selected features are removed or incrementally combined.",
        "baseline": baseline_metrics,
        "single_feature_ablation": ablation_experiments,
        "combination_ablation": combination_results
    }

    with open(OUTPUT_DIR / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(ablation_payload, f, indent=2)

    # Save models and metadata
    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "zero_as_missing": ZERO_AS_MISSING,
        "imputation_medians": medians,
        "threshold": 0.50,
        "feature_importances": feat_imp,
        "metrics": baseline_metrics,
        "hyperparameters": baseline_model.get_params()
    }
    joblib.dump(baseline_model, MODEL_DIR / "xgboost_diabetes.joblib")
    joblib.dump(metadata, MODEL_DIR / "model_metadata.joblib")

    # Also keep xgboost_diabetes_model.pkl for compatibility
    joblib.dump(baseline_model, MODEL_DIR / "xgboost_diabetes_model.pkl")

    print("\nTraining and Ablation pipeline successfully completed!")
    print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"Baseline Precision: {baseline_metrics['precision']:.4f}")
    print(f"Baseline Recall: {baseline_metrics['recall']:.4f}")
    print(f"Baseline F1 Score: {baseline_metrics['f1']:.4f}")
    print(f"Baseline ROC-AUC: {baseline_metrics['roc_auc']:.4f}")
    print(f"Saved artifacts to {MODEL_DIR} and {OUTPUT_DIR}")

if __name__ == "__main__":
    train_and_evaluate()
