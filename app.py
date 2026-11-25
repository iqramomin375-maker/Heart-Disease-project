import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

model = joblib.load(os.path.join(BASE_DIR, "model", "heart_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def get_risk_level(probability):
    if probability >= 0.66:
        return "High"
    if probability >= 0.33:
        return "Moderate"
    return "Low"


def build_recommendations(data, probability):
    risk_level = get_risk_level(probability)

    diet, exercise, lifestyle, followup = [], [], [], []

    # ---- Diet ----
    if data["chol"] >= 240:
        diet.append(
            "Cholesterol is high. Reduce fried food, red meat "
            "and full-fat dairy. Prefer whole grains, nuts and "
            "heart-friendly foods."
        )
    else:
        diet.append(
            "Cholesterol is in a healthier range. Continue "
            "choosing whole foods and a balanced diet."
        )

    if data["trestbps"] >= 130:
        diet.append(
            "Resting blood pressure is elevated. Consider "
            "reducing sodium intake and including potassium-rich "
            "foods such as leafy greens and bananas."
        )

    if data["fbs"] == 1:
        diet.append(
            "Fasting blood sugar is elevated. Limit refined "
            "sugars and highly processed carbohydrates."
        )

    # ---- Exercise ----
    if data["thalach"] < 120:
        exercise.append(
            "Peak heart rate during exertion is relatively low. "
            "Start physical activity gradually and seek medical "
            "guidance when appropriate."
        )
    else:
        exercise.append(
            "Maintain regular physical activity appropriate "
            "for your health and fitness level."
        )

    exercise.append(
        "Aim for consistent moderate activity and avoid "
        "sudden excessive exertion."
    )

    # ---- Lifestyle ----
    if data["age"] >= 55:
        lifestyle.append(
            "Given the age group, keeping regular cardiac "
            "health check-ups on the calendar can be useful."
        )

    lifestyle.append("Avoid tobacco in every form.")
    lifestyle.append(
        "Aim for around 7–8 hours of sleep and manage stress "
        "through healthy daily routines."
    )

    if data["oldpeak"] >= 2:
        lifestyle.append(
            "The ST depression value is relatively high and "
            "should be discussed with a qualified healthcare "
            "professional."
        )

    # ---- Follow-up ----
    if risk_level == "High":
        followup.append(
            "The model predicts a high likelihood of heart "
            "disease. Consider discussing the result with a "
            "qualified healthcare professional."
        )
    elif risk_level == "Moderate":
        followup.append(
            "The model predicts a moderate likelihood. Consider "
            "discussing your risk factors with a healthcare "
            "professional."
        )
    else:
        followup.append(
            "The model predicts a lower likelihood based on "
            "the entered values. Continue healthy habits and "
            "regular health monitoring."
        )

    followup.append(
        "This prediction is for educational purposes and "
        "should not be treated as a medical diagnosis."
    )

    return {
        "risk_level": risk_level,
        "probability_pct": round(probability * 100, 1),
        "diet": diet,
        "exercise": exercise,
        "lifestyle": lifestyle,
        "followup": followup,
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        patient = {
            "age": float(data["age"]),
            "sex": int(data["sex"]),
            "cp": int(data["cp"]),
            "trestbps": float(data["trestbps"]),
            "chol": float(data["chol"]),
            "fbs": int(data["fbs"]),
            "restecg": int(data["restecg"]),
            "thalach": float(data["thalach"]),
            "exang": int(data["exang"]),
            "oldpeak": float(data["oldpeak"]),
            "slope": int(data["slope"]),
            "ca": int(data["ca"]),
            "thal": int(data["thal"]),
        }

        # numpy array instead of pandas keeps the deployment much smaller
        patient_array = np.array([[patient[c] for c in FEATURE_COLUMNS]])
        patient_scaled = scaler.transform(patient_array)

        prediction = int(model.predict(patient_scaled)[0])
        probability = float(model.predict_proba(patient_scaled)[0][1])

        recommendation = build_recommendations(patient, probability)

        return jsonify({
            "success": True,
            "prediction": prediction,
            "prediction_text": (
                "Heart Disease Likely" if prediction == 1
                else "Heart Disease Unlikely"
            ),
            "probability": round(probability * 100, 1),
            "risk_level": recommendation["risk_level"],
            "recommendations": {
                "diet": recommendation["diet"],
                "exercise": recommendation["exercise"],
                "lifestyle": recommendation["lifestyle"],
                "followup": recommendation["followup"],
            },
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)