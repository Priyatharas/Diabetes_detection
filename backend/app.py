import json
import uuid
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

from database import get_db_connection, init_db

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"
MODEL_PATH = BASE_DIR / "models" / "xgboost_diabetes.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.joblib"
METRICS_PATH = BASE_DIR / "outputs" / "metrics.json"
ABLATION_PATH = BASE_DIR / "outputs" / "ablation_results.json"

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIST) if FRONTEND_DIST.exists() else None,
    static_url_path=""
)
app.secret_key = "diabetes_xgboost_clinical_decision_support_secret_key"
CORS(app, supports_credentials=True)

# Ensure DB initialized
init_db()

# Load model and metadata
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")

model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH) if METADATA_PATH.exists() else {}
medians = metadata.get("imputation_medians", {
    "Glucose": 117.0, "BloodPressure": 72.0, "SkinThickness": 29.0, "Insulin": 125.0, "BMI": 32.3
})

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Normal clinical reference ranges (standard clinical guidelines)
CLINICAL_REFERENCES = {
    "Glucose": {"normal_min": 70, "normal_max": 99, "unit": "mg/dL", "label": "Fasting Blood Glucose", "desc": "Normal: 70-99 | Pre-diabetic: 100-125 | Diabetic: >= 126"},
    "BloodPressure": {"normal_min": 60, "normal_max": 80, "unit": "mmHg", "label": "Diastolic Blood Pressure", "desc": "Normal: < 80 | Elevated: 80-89 | High Stage 1: >= 90"},
    "SkinThickness": {"normal_min": 10, "normal_max": 30, "unit": "mm", "label": "Triceps Skin Fold Thickness", "desc": "Normal subcutaneous fat estimate: 10-30 mm"},
    "Insulin": {"normal_min": 15, "normal_max": 166, "unit": "mIU/L", "label": "2-Hour Serum Insulin", "desc": "Normal fasting: 15-166 mIU/L | Hyperinsulinemia: > 166"},
    "BMI": {"normal_min": 18.5, "normal_max": 24.9, "unit": "kg/m²", "label": "Body Mass Index", "desc": "Normal: 18.5-24.9 | Overweight: 25-29.9 | Obese: >= 30"},
    "DiabetesPedigreeFunction": {"normal_min": 0.08, "normal_max": 0.50, "unit": "score", "label": "Diabetes Pedigree Score", "desc": "Genetic predisposition score based on family history"},
    "Age": {"normal_min": 21, "normal_max": 80, "unit": "years", "label": "Patient Age", "desc": "Adult patient population range"},
    "Pregnancies": {"normal_min": 0, "normal_max": 17, "unit": "count", "label": "Gravidity (Pregnancies)", "desc": "Number of times pregnant"}
}

# In-memory active user sessions
ACTIVE_SESSIONS = {}

def get_current_user():
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if token and token in ACTIVE_SESSIONS:
        return ACTIVE_SESSIONS[token]
    user_id = session.get("user_id")
    if user_id:
        conn = get_db_connection()
        user = conn.execute("SELECT id, name, email, role, specialty FROM users WHERE id = ?", (user_id,)).fetchone()
        conn.close()
        if user:
            return dict(user)
    return None

# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    specialty = data.get("specialty", "Endocrinology & Diabetology").strip() or "Endocrinology & Diabetology"
    role = data.get("role", "Doctor").strip() or "Doctor"

    if not name or not email or not password:
        return jsonify({"error": "Full Name, Email and Password are required."}), 400

    if len(password) < 4:
        return jsonify({"error": "Password must be at least 4 characters long."}), 400

    conn = get_db_connection()
    existing = conn.execute("SELECT id FROM users WHERE LOWER(email) = ?", (email,)).fetchone()
    if existing:
        conn.close()
        return jsonify({"error": "An account with this email already exists. Please log in or use another email."}), 409

    pwd_hash = generate_password_hash(password)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO users (name, email, password_hash, role, specialty)
        VALUES (?, ?, ?, ?, ?)
    """, (name, email, pwd_hash, role, specialty))
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()

    token = str(uuid.uuid4())
    user_data = {
        "id": user_id,
        "name": name,
        "email": email,
        "role": role,
        "specialty": specialty
    }
    ACTIVE_SESSIONS[token] = user_data
    session["user_id"] = user_id

    return jsonify({
        "message": "Account created successfully",
        "token": token,
        "user": user_data
    }), 201

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE LOWER(email) = ?", (email,)).fetchone()
    conn.close()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid clinical credentials. Please check your email and password."}), 401

    token = str(uuid.uuid4())
    user_data = {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "specialty": user["specialty"]
    }
    ACTIVE_SESSIONS[token] = user_data
    session["user_id"] = user["id"]

    return jsonify({
        "message": "Authentication successful",
        "token": token,
        "user": user_data
    })

@app.route("/api/auth/logout", methods=["POST"])
def logout():
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if token in ACTIVE_SESSIONS:
        del ACTIVE_SESSIONS[token]
    session.pop("user_id", None)
    return jsonify({"message": "Successfully logged out"})

@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    user = get_current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"user": user})

# =============================================================================
# DASHBOARD OVERVIEW & STATS
# =============================================================================

@app.route("/api/dashboard/stats", methods=["GET"])
def dashboard_stats():
    conn = get_db_connection()
    total_patients = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    total_predictions = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    high_risk_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 1").fetchone()[0]
    low_risk_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 0").fetchone()[0]
    avg_prob = conn.execute("SELECT AVG(probability) FROM predictions").fetchone()[0] or 0.0

    recent = conn.execute("""
        SELECT id, patient_identifier, patient_name, age, glucose, bmi, blood_pressure,
               prediction, probability, risk_level, created_at
        FROM predictions
        ORDER BY id DESC LIMIT 10
    """).fetchall()

    conn.close()

    return jsonify({
        "total_patients": total_patients,
        "total_predictions": total_predictions,
        "high_risk_count": high_risk_count,
        "low_risk_count": low_risk_count,
        "average_probability": round(float(avg_prob), 1),
        "recent_predictions": [dict(r) for r in recent]
    })

# =============================================================================
# PREDICTION ENGINE (With Real SHAP Explanations & Reference Ranges)
# =============================================================================

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}

    # Validation
    parsed_values = {}
    validation_errors = []

    for feat in FEATURES:
        val = data.get(feat)
        if val is None or val == "":
            validation_errors.append(f"{feat} is required.")
            continue
        try:
            val_float = float(val)
            if val_float < 0:
                validation_errors.append(f"{feat} cannot be negative.")
            parsed_values[feat] = val_float
        except ValueError:
            validation_errors.append(f"{feat} must be a valid number.")

    if validation_errors:
        return jsonify({"error": "Validation failed", "details": validation_errors}), 400

    # Impute clinically invalid zero values with median as defined in research paper
    row_dict = {}
    imputed_fields = []
    for feat in FEATURES:
        val = parsed_values[feat]
        if feat in ZERO_AS_MISSING and val == 0:
            row_dict[feat] = medians[feat]
            imputed_fields.append(feat)
        else:
            row_dict[feat] = val

    df_input = pd.DataFrame([row_dict], columns=FEATURES)

    # Actual XGBoost prediction
    prob_diabetes = float(model.predict_proba(df_input)[0, 1])
    prob_no_diabetes = float(1.0 - prob_diabetes)
    prediction_class = int(prob_diabetes >= 0.50)

    risk_percentage = round(prob_diabetes * 100, 1)
    if risk_percentage >= 70:
        risk_level = "High Risk"
        risk_badge = "destructive"
    elif risk_percentage >= 40:
        risk_level = "Moderate Risk"
        risk_badge = "warning"
    else:
        risk_level = "Low Risk"
        risk_badge = "success"

    # SHAP local feature contributions
    shap_vals = explainer(df_input)
    local_shap = shap_vals.values[0]
    base_val = float(shap_vals.base_values[0])

    feature_contributions = []
    for feat, s_val in zip(FEATURES, local_shap):
        feature_contributions.append({
            "feature": feat,
            "patient_value": parsed_values[feat],
            "imputed_value": row_dict[feat],
            "shap_value": float(round(s_val, 4)),
            "impact_direction": "Increases Risk" if s_val > 0 else "Decreases Risk",
            "abs_impact": float(round(abs(s_val), 4))
        })
    feature_contributions.sort(key=lambda x: x["abs_impact"], reverse=True)

    # Clinical parameter profile with reference evaluation
    parameter_profile = []
    for feat in FEATURES:
        ref = CLINICAL_REFERENCES.get(feat, {})
        val = parsed_values[feat]
        status = "Normal"
        if "normal_max" in ref and val > ref["normal_max"]:
            status = "Elevated"
        elif "normal_min" in ref and val < ref["normal_min"] and val > 0:
            status = "Low"

        parameter_profile.append({
            "feature": feat,
            "label": ref.get("label", feat),
            "value": val,
            "unit": ref.get("unit", ""),
            "normal_range": f"{ref.get('normal_min', 0)} - {ref.get('normal_max', 0)} {ref.get('unit', '')}",
            "status": status,
            "description": ref.get("desc", "")
        })

    # Evidence-based clinical recommendations
    recommendations = []
    if risk_percentage >= 50:
        recommendations.append("Immediate Glycated Hemoglobin (HbA1c) and fasting plasma glucose laboratory evaluation.")
        if parsed_values.get("Glucose", 0) > 125:
            recommendations.append("Target fasting blood glucose control below 100 mg/dL with clinical dietary intervention.")
        if parsed_values.get("BMI", 0) >= 30:
            recommendations.append("Structured medically-supervised weight management program targeting 5-10% BMI reduction.")
        if parsed_values.get("BloodPressure", 0) >= 85:
            recommendations.append("Regular blood pressure monitoring; aim for systolic < 120 mmHg and diastolic < 80 mmHg.")
        recommendations.append("Referral to an Endocrinologist / Diabetologist for comprehensive clinical assessment.")
    else:
        recommendations.append("Maintain an active lifestyle with at least 150 minutes of moderate aerobic activity weekly.")
        recommendations.append("Follow a balanced, low-glycemic Mediterranean dietary pattern rich in dietary fiber.")
        recommendations.append("Annual routine health checkup and periodic glucose wellness screening.")

    # Save to SQLite Database
    conn = get_db_connection()
    patient_identifier = data.get("patient_identifier", f"PT-{np.random.randint(1000, 9999)}")
    patient_name = data.get("patient_name", "Anonymous Patient")

    # Ensure patient in patients table
    p_row = conn.execute("SELECT id FROM patients WHERE patient_identifier = ?", (patient_identifier,)).fetchone()
    if p_row:
        p_id = p_row["id"]
    else:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO patients (patient_identifier, name, age, gender)
            VALUES (?, ?, ?, ?)
        """, (patient_identifier, patient_name, int(parsed_values.get("Age", 30)), data.get("gender", "Female")))
        p_id = cursor.lastrowid
        conn.commit()

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            patient_id, patient_identifier, patient_name, pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi, diabetes_pedigree, age, prediction, probability,
            risk_level, feature_contributions, recommendations
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        p_id, patient_identifier, patient_name,
        parsed_values["Pregnancies"], parsed_values["Glucose"], parsed_values["BloodPressure"],
        parsed_values["SkinThickness"], parsed_values["Insulin"], parsed_values["BMI"],
        parsed_values["DiabetesPedigreeFunction"], parsed_values["Age"],
        prediction_class, risk_percentage, risk_level,
        json.dumps(feature_contributions), json.dumps(recommendations)
    ))
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return jsonify({
        "prediction_id": prediction_id,
        "patient_identifier": patient_identifier,
        "patient_name": patient_name,
        "prediction": prediction_class,
        "probability_diabetes": risk_percentage,
        "probability_no_diabetes": round(prob_no_diabetes * 100, 1),
        "risk_level": risk_level,
        "risk_badge": risk_badge,
        "base_value": base_val,
        "feature_contributions": feature_contributions,
        "parameter_profile": parameter_profile,
        "recommendations": recommendations,
        "imputed_fields": imputed_fields,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "medical_disclaimer": "This prediction is generated by an XGBoost machine-learning predictive model for research and decision-support purposes. It is not an official medical diagnosis."
    })

# =============================================================================
# DIGITAL TWIN 12-MONTH SIMULATION (Research Paper Section VI.G & VI.H)
# =============================================================================

@app.route("/api/simulate", methods=["POST"])
def simulate():
    data = request.get_json() or {}
    base_values = {f: float(data.get(f, medians.get(f, 0))) for f in FEATURES}
    months = int(data.get("months", 12))

    monthly_trajectory = []
    scenarios = ["Current Condition", "No Lifestyle Change", "Improved Lifestyle", "Improved Medication Adherence"]

    for month in range(0, months + 1):
        progress = month / max(1, months)
        month_record = {"month": month}

        for scenario in scenarios:
            state = base_values.copy()

            if scenario == "Current Condition":
                # Static baseline
                pass
            elif scenario == "No Lifestyle Change":
                # Gradual disease progression without intervention
                state["Glucose"] = min(260.0, state["Glucose"] * (1.0 + 0.12 * progress))
                state["BMI"] = min(60.0, state["BMI"] * (1.0 + 0.05 * progress))
                state["BloodPressure"] = min(140.0, state["BloodPressure"] * (1.0 + 0.04 * progress))
            elif scenario == "Improved Lifestyle":
                # Regular aerobic activity and dietary improvements
                state["Glucose"] = max(75.0, state["Glucose"] * (1.0 - 0.14 * progress))
                state["BMI"] = max(19.0, state["BMI"] * (1.0 - 0.09 * progress))
                state["BloodPressure"] = max(65.0, state["BloodPressure"] * (1.0 - 0.06 * progress))
            elif scenario == "Improved Medication Adherence":
                # Clinical pharmacotherapy adherence and routine monitoring
                state["Glucose"] = max(75.0, state["Glucose"] * (1.0 - 0.18 * progress))
                state["BMI"] = max(19.0, state["BMI"] * (1.0 - 0.04 * progress))
                state["BloodPressure"] = max(65.0, state["BloodPressure"] * (1.0 - 0.08 * progress))

            # Impute zeros if any
            for col in ZERO_AS_MISSING:
                if state[col] == 0:
                    state[col] = medians[col]

            df_state = pd.DataFrame([state], columns=FEATURES)
            prob = float(model.predict_proba(df_state)[0, 1])
            month_record[scenario] = round(prob * 100, 1)

        monthly_trajectory.append(month_record)

    # Optional DB storage if prediction_id provided
    prediction_id = data.get("prediction_id")
    if prediction_id:
        try:
            conn = get_db_connection()
            for rec in monthly_trajectory:
                m = rec["month"]
                for sc in scenarios:
                    conn.execute("""
                        INSERT INTO simulation_results (prediction_id, scenario, month, risk)
                        VALUES (?, ?, ?, ?)
                    """, (prediction_id, sc, m, rec[sc]))
            conn.commit()
            conn.close()
        except Exception:
            pass

    return jsonify({
        "months": months,
        "scenarios": scenarios,
        "trajectory": monthly_trajectory,
        "title": "Illustrative Model-Based Risk Scenario Simulation (Digital Twin)",
        "disclaimer": "The Digital Twin illustrates scenario-based risk trajectories under varying clinical and lifestyle conditions. It is not an absolute longitudinal disease-onset forecast."
    })

# =============================================================================
# SCIENTIFIC SENSITIVITY CURVES (Graphs 8, 9, 10: Glucose, BMI & 2D Grid)
# =============================================================================

@app.route("/api/sensitivity", methods=["POST"])
def sensitivity():
    data = request.get_json() or {}
    base_values = {f: float(data.get(f, medians.get(f, 0))) for f in FEATURES}
    for col in ZERO_AS_MISSING:
        if base_values[col] == 0:
            base_values[col] = medians[col]

    # 1. Glucose Variation: 60 to 240 mg/dL (Graph 8)
    glucose_points = []
    for g in range(60, 241, 10):
        temp = base_values.copy()
        temp["Glucose"] = float(g)
        df_temp = pd.DataFrame([temp], columns=FEATURES)
        p = float(model.predict_proba(df_temp)[0, 1])
        glucose_points.append({
            "glucose": g,
            "risk_probability": round(p * 100, 1),
            "is_patient_value": abs(g - base_values["Glucose"]) < 5
        })

    # 2. BMI Variation: 16 to 55 kg/m² (Graph 9)
    bmi_points = []
    for b in range(16, 56, 2):
        temp = base_values.copy()
        temp["BMI"] = float(b)
        df_temp = pd.DataFrame([temp], columns=FEATURES)
        p = float(model.predict_proba(df_temp)[0, 1])
        bmi_points.append({
            "bmi": b,
            "risk_probability": round(p * 100, 1),
            "is_patient_value": abs(b - base_values["BMI"]) < 1
        })

    # 3. Two-Parameter Model Response (Glucose vs BMI Grid) (Graph 10)
    grid_data = []
    g_steps = [80, 100, 120, 140, 160, 180, 200]
    b_steps = [20, 25, 30, 35, 40, 45]
    for b in b_steps:
        row = {"bmi": b}
        for g in g_steps:
            temp = base_values.copy()
            temp["BMI"] = float(b)
            temp["Glucose"] = float(g)
            df_temp = pd.DataFrame([temp], columns=FEATURES)
            prob = float(model.predict_proba(df_temp)[0, 1])
            row[f"g_{g}"] = round(prob * 100, 1)
        grid_data.append(row)

    return jsonify({
        "glucose_curve": glucose_points,
        "bmi_curve": bmi_points,
        "two_parameter_grid": {
            "glucose_steps": g_steps,
            "bmi_steps": b_steps,
            "data": grid_data
        }
    })

# =============================================================================
# MODEL METRICS & ABLATION ANALYSIS
# =============================================================================

@app.route("/api/model-metrics", methods=["GET"])
def model_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        return jsonify(metrics)
    return jsonify({"error": "Metrics not generated. Run train_model.py first."}), 404

@app.route("/api/feature-importance", methods=["GET"])
def feature_importance():
    importances = model.feature_importances_
    feat_imp = [
        {
            "feature": feat,
            "importance": float(round(imp, 5)),
            "percentage": float(round(imp * 100, 2)),
            "label": CLINICAL_REFERENCES.get(feat, {}).get("label", feat)
        }
        for feat, imp in zip(FEATURES, importances)
    ]
    feat_imp.sort(key=lambda x: x["importance"], reverse=True)
    return jsonify({"features": feat_imp})

@app.route("/api/ablation", methods=["GET"])
def ablation():
    if ABLATION_PATH.exists():
        with open(ABLATION_PATH, "r", encoding="utf-8") as f:
            ablation_data = json.load(f)
        return jsonify(ablation_data)
    return jsonify({"error": "Ablation results not generated. Run train_model.py first."}), 404

# =============================================================================
# PATIENT RECORDS & HISTORY
# =============================================================================

@app.route("/api/history", methods=["GET"])
def history():
    conn = get_db_connection()
    search = request.args.get("search", "").strip()
    risk_filter = request.args.get("risk", "").strip()

    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if search:
        query += " AND (patient_identifier LIKE ? OR patient_name LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])

    if risk_filter:
        query += " AND risk_level = ?"
        params.append(risk_filter)

    query += " ORDER BY id DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()

    result = []
    for r in rows:
        item = dict(r)
        if item.get("feature_contributions"):
            try:
                item["feature_contributions"] = json.loads(item["feature_contributions"])
            except Exception:
                pass
        if item.get("recommendations"):
            try:
                item["recommendations"] = json.loads(item["recommendations"])
            except Exception:
                pass
        result.append(item)

    return jsonify({"predictions": result})

@app.route("/api/history/<int:pred_id>", methods=["GET"])
def history_item(pred_id):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Record not found"}), 404

    item = dict(row)
    if item.get("feature_contributions"):
        try:
            item["feature_contributions"] = json.loads(item["feature_contributions"])
        except Exception:
            pass
    if item.get("recommendations"):
        try:
            item["recommendations"] = json.loads(item["recommendations"])
        except Exception:
            pass

    return jsonify({"prediction": item})

@app.route("/api/patients", methods=["GET", "POST"])
def patients():
    conn = get_db_connection()
    if request.method == "POST":
        data = request.get_json() or {}
        p_id = data.get("patient_identifier", f"PT-{np.random.randint(1000, 9999)}")
        name = data.get("name", "New Patient")
        age = int(data.get("age", 30))
        gender = data.get("gender", "Female")
        contact = data.get("contact", "")

        conn.execute("""
            INSERT INTO patients (patient_identifier, name, age, gender, contact)
            VALUES (?, ?, ?, ?, ?)
        """, (p_id, name, age, gender, contact))
        conn.commit()
        conn.close()
        return jsonify({"message": "Patient created successfully", "patient_identifier": p_id})

    rows = conn.execute("SELECT * FROM patients ORDER BY id DESC").fetchall()
    conn.close()
    return jsonify({"patients": [dict(r) for r in rows]})

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path.startswith("api"):
        return jsonify({"error": f"API endpoint /{path} not found"}), 404

    if FRONTEND_DIST.exists():
        target = FRONTEND_DIST / path
        if path and target.exists() and target.is_file():
            return send_from_directory(str(FRONTEND_DIST), path)
        return send_from_directory(str(FRONTEND_DIST), "index.html")

    return jsonify({
        "message": "Diabetes XGBoost Clinical Decision Support API is running.",
        "status": "online",
        "api_endpoints": "/api/...",
        "frontend": "Run 'npm run dev' in the frontend directory or 'npm run build' to bundle the web UI."
    })

if __name__ == "__main__":
    print("Starting Diabetes XGBoost Clinical Decision Support Application...")
    print("Web Application URL: http://localhost:5000 (or http://127.0.0.1:5000)")
    app.run(host="0.0.0.0", port=5000, debug=False)
