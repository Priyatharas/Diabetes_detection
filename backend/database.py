import sqlite3
import json
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent / "diabetes_clinical.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'Doctor',
        specialty TEXT DEFAULT 'Endocrinology & Diabetology',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Patients table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_identifier TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT DEFAULT 'Female',
        contact TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Predictions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        patient_identifier TEXT,
        patient_name TEXT,
        pregnancies REAL NOT NULL,
        glucose REAL NOT NULL,
        blood_pressure REAL NOT NULL,
        skin_thickness REAL NOT NULL,
        insulin REAL NOT NULL,
        bmi REAL NOT NULL,
        diabetes_pedigree REAL NOT NULL,
        age REAL NOT NULL,
        prediction INTEGER NOT NULL,
        probability REAL NOT NULL,
        risk_level TEXT NOT NULL,
        feature_contributions TEXT,
        recommendations TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES patients(id)
    )
    """)

    # Simulation results table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS simulation_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id INTEGER,
        patient_identifier TEXT,
        scenario TEXT NOT NULL,
        month INTEGER NOT NULL,
        risk REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (prediction_id) REFERENCES predictions(id)
    )
    """)

    conn.commit()

    # Seed initial Doctor user if not exists
    cursor.execute("SELECT id FROM users WHERE email = ?", ("doctor@hospital.org",))
    if not cursor.fetchone():
        pwd_hash = generate_password_hash("doctor123")
        cursor.execute("""
        INSERT INTO users (name, email, password_hash, role, specialty)
        VALUES (?, ?, ?, ?, ?)
        """, (
            "Dr. Priyathara P, MD",
            "doctor@hospital.org",
            pwd_hash,
            "Chief Diabetologist",
            "Endocrinology & Clinical AI Research"
        ))
        conn.commit()
        print("Seeded default doctor account: doctor@hospital.org / doctor123")

    # Seed sample clinical patient records if empty
    cursor.execute("SELECT COUNT(*) FROM patients")
    if cursor.fetchone()[0] == 0:
        sample_patients = [
            ("PT-1001", "Eleanor Vance", 50, "Female", "+1-555-0101"),
            ("PT-1002", "Sarah Jenkins", 31, "Female", "+1-555-0102"),
            ("PT-1003", "Maria Rodriguez", 32, "Female", "+1-555-0103"),
            ("PT-1004", "Clara Oswald", 21, "Female", "+1-555-0104"),
            ("PT-1005", "Grace Hopper", 33, "Female", "+1-555-0105"),
            ("PT-1006", "Ada Lovelace", 26, "Female", "+1-555-0106"),
            ("PT-1007", "Hannah Abbott", 45, "Female", "+1-555-0107"),
            ("PT-1008", "Rachel Green", 29, "Female", "+1-555-0108"),
        ]
        cursor.executemany("""
        INSERT INTO patients (patient_identifier, name, age, gender, contact)
        VALUES (?, ?, ?, ?, ?)
        """, sample_patients)
        conn.commit()

        # Seed realistic initial predictions corresponding to real clinical rows
        # Row 1: High risk (Glucose 148, BMI 33.6, Age 50)
        # Row 2: Low risk (Glucose 85, BMI 26.6, Age 31)
        # Row 3: High risk (Glucose 183, BMI 23.3, Age 32)
        # Row 4: Low risk (Glucose 89, BMI 28.1, Age 21)
        # Row 5: High risk (Glucose 137, BMI 43.1, Age 33)
        sample_preds = [
            (
                1, "PT-1001", "Eleanor Vance", 6, 148, 72, 35, 125, 33.6, 0.627, 50,
                1, 78.4, "High Risk",
                json.dumps([{"feature": "Glucose", "value": 148, "impact": "+24.2%"}, {"feature": "BMI", "value": 33.6, "impact": "+11.5%"}]),
                json.dumps(["Immediate HbA1c screening", "Dietary carbohydrate restriction", "Target fasting blood glucose < 100 mg/dL", "Endocrinologist consultation"]),
                "2026-09-24 10:15:00"
            ),
            (
                2, "PT-1002", "Sarah Jenkins", 1, 85, 66, 29, 125, 26.6, 0.351, 31,
                0, 14.8, "Low Risk",
                json.dumps([{"feature": "Glucose", "value": 85, "impact": "-22.1%"}, {"feature": "BMI", "value": 26.6, "impact": "-5.4%"}]),
                json.dumps(["Annual glucose wellness screening", "Maintain current active physical lifestyle", "Balanced Mediterranean nutrition pattern"]),
                "2026-09-24 11:30:00"
            ),
            (
                3, "PT-1003", "Maria Rodriguez", 8, 183, 64, 29, 125, 23.3, 0.672, 32,
                1, 86.5, "High Risk",
                json.dumps([{"feature": "Glucose", "value": 183, "impact": "+35.1%"}, {"feature": "Pregnancies", "value": 8, "impact": "+8.2%"}]),
                json.dumps(["Urgent diagnostic glucose tolerance test", "Clinical evaluation for gestational/type 2 onset", "Lifestyle modification protocol"]),
                "2026-09-25 09:20:00"
            ),
            (
                4, "PT-1004", "Clara Oswald", 1, 89, 66, 23, 94, 28.1, 0.167, 21,
                0, 11.2, "Low Risk",
                json.dumps([{"feature": "Glucose", "value": 89, "impact": "-19.8%"}, {"feature": "Age", "value": 21, "impact": "-8.1%"}]),
                json.dumps(["Routine health maintenance", "Balanced caloric intake", "Physical fitness maintenance"]),
                "2026-09-25 14:45:00"
            ),
            (
                5, "PT-1005", "Grace Hopper", 0, 137, 40, 35, 168, 43.1, 2.288, 33,
                1, 82.1, "High Risk",
                json.dumps([{"feature": "BMI", "value": 43.1, "impact": "+18.9%"}, {"feature": "Pedigree", "value": 2.288, "impact": "+14.3%"}]),
                json.dumps(["Comprehensive metabolic panel", "Bariatric lifestyle counseling", "Cardiometabolic risk evaluation"]),
                "2026-09-26 08:30:00"
            ),
            (
                6, "PT-1006", "Ada Lovelace", 2, 92, 70, 24, 88, 22.4, 0.245, 26,
                0, 16.3, "Low Risk",
                json.dumps([{"feature": "Glucose", "value": 92, "impact": "-18.5%"}, {"feature": "BMI", "value": 22.4, "impact": "-12.0%"}]),
                json.dumps(["Standard preventative monitoring", "Regular hydration and exercise"]),
                "2026-09-26 12:10:00"
            )
        ]

        cursor.executemany("""
        INSERT INTO predictions (
            patient_id, patient_identifier, patient_name, pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi, diabetes_pedigree, age, prediction, probability,
            risk_level, feature_contributions, recommendations, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_preds)
        conn.commit()
        print("Seeded sample clinical prediction history.")

    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DB_PATH)
