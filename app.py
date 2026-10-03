from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "xgboost_diabetes_model.pkl"

app = Flask(__name__)

# This is the model supplied by the project owner.
model = joblib.load(MODEL_PATH)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# These are the preprocessing assumptions documented in the paper.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# We do not invent training-set medians here. For a trained model, the exact
# medians used during training should be reused if the original training
# preprocessing used median imputation. If no medians were saved, the web
# form asks for non-zero values for these fields.
THRESHOLD = 0.50


def prepare_input(values):
    row = pd.DataFrame([values], columns=FEATURES)

    # The paper says zero values in these columns were treated as invalid.
    # Therefore the web application rejects zero for these fields rather than
    # silently inventing a different median.
    for col in ZERO_AS_MISSING:
        if float(row.loc[0, col]) == 0:
            raise ValueError(f"{col} must be greater than 0.")

    return row


def predict_row(values):
    X = prepare_input(values)
    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(probability >= THRESHOLD)
    return prediction, probability


def digital_twin(values, months=12):
    """
    Reproducible project simulation.

    These parameter changes are project assumptions for demonstrating the
    Digital Twin workflow. They are NOT clinical treatment effects.
    """
    base = np.array([float(v) for v in values], dtype=float)
    idx = {name: i for i, name in enumerate(FEATURES)}

    results = {
        "Current condition": [],
        "No lifestyle change": [],
        "Improved lifestyle": [],
        "Improved medication adherence": []
    }

    for month in range(months + 1):
        progress = month / months

        scenarios = {
            "Current condition": base.copy(),
            "No lifestyle change": base.copy(),
            "Improved lifestyle": base.copy(),
            "Improved medication adherence": base.copy()
        }

        no_change = scenarios["No lifestyle change"]
        no_change[idx["Glucose"]] *= (1 + 0.10 * progress)
        no_change[idx["BMI"]] *= (1 + 0.04 * progress)
        no_change[idx["BloodPressure"]] *= (1 + 0.03 * progress)

        lifestyle = scenarios["Improved lifestyle"]
        lifestyle[idx["Glucose"]] *= (1 - 0.12 * progress)
        lifestyle[idx["BMI"]] *= (1 - 0.08 * progress)
        lifestyle[idx["BloodPressure"]] *= (1 - 0.05 * progress)

        medication = scenarios["Improved medication adherence"]
        medication[idx["Glucose"]] *= (1 - 0.10 * progress)
        medication[idx["BMI"]] *= (1 - 0.04 * progress)
        medication[idx["BloodPressure"]] *= (1 - 0.07 * progress)

        for name, state in scenarios.items():
            state[idx["Glucose"]] = np.clip(state[idx["Glucose"]], 40, 250)
            state[idx["BloodPressure"]] = np.clip(state[idx["BloodPressure"]], 20, 140)
            state[idx["BMI"]] = np.clip(state[idx["BMI"]], 10, 70)

            _, prob = predict_row(state.tolist())
            results[name].append(round(prob * 100, 2))

    return results


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    twin = None
    form = {f: "" for f in FEATURES}

    if request.method == "POST":
        try:
            for feature in FEATURES:
                form[feature] = request.form.get(feature, "").strip()

            values = [float(form[f]) for f in FEATURES]

            if any(v < 0 for v in values):
                raise ValueError("Values cannot be negative.")

            prediction, probability = predict_row(values)

            result = {
                "prediction": prediction,
                "probability": round(probability * 100, 2),
                "label": "High Risk of Diabetes" if prediction == 1
                         else "Low Risk of Diabetes"
            }

            twin = digital_twin(values, months=12)

        except ValueError as exc:
            error = str(exc)

    return render_template(
        "index.html",
        features=FEATURES,
        form=form,
        result=result,
        error=error,
        twin=twin
    )


if __name__ == "__main__":
    app.run(debug=True)
