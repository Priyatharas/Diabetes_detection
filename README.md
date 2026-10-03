# XGBoost Diabetes Prediction + Digital Twin

This is a complete runnable implementation based on the project described in the paper:
- Pima Indians Diabetes Dataset
- zero-as-missing preprocessing
- XGBoost classifier
- accuracy, precision, recall, F1, ROC-AUC
- confusion matrix
- 5-fold cross-validation
- ROC curve
- feature importance
- Flask web application
- 12-month Digital Twin scenario simulation

## 1. Dataset

Put the Pima Indians Diabetes CSV file in this folder and name it:

    diabetes.csv

Expected columns:

    Pregnancies, Glucose, BloodPressure, SkinThickness,
    Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome

The standard Pima dataset contains 768 rows.

## 2. Install

Windows:

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

Linux/macOS:

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## 3. Train and evaluate

    python train_model.py

This creates:
- models/xgboost_diabetes.joblib
- models/model_metadata.joblib
- outputs/metrics.json
- outputs/classification_report.txt
- outputs/confusion_matrix.png
- outputs/roc_curve.png
- outputs/feature_importance.png
- outputs/cv_scores.png

## 4. Start the web application

    python app.py

Open:

    http://127.0.0.1:5000

## Important research note

The Digital Twin module is a reproducible simulation, not a clinically validated disease-progression model.
The simulation changes selected input variables according to explicit project assumptions and re-runs the trained XGBoost model each month. These assumptions must be reported in the paper as simulation assumptions, not as clinical evidence.

The model's output probability is called "predicted probability" rather than calibrated "confidence".


## Using the uploaded XGBoost model

This version already contains `models/xgboost_diabetes_model.pkl`, the model supplied for this project.
The model was inspected and exposes the expected eight feature names:
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
DiabetesPedigreeFunction, Age.

Because the `.pkl` contains the trained classifier but not the original test
labels/data or preprocessing medians, this version does NOT invent accuracy,
precision, recall, F1 or ROC-AUC values. Those metrics require the original
test data (or the original training/evaluation script).

Run:

    pip install -r requirements.txt
    python app.py

Then open:

    http://127.0.0.1:5000
