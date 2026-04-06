from flask import Flask, render_template, request, send_file
import json
import pandas as pd
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load dataset
data = load_breast_cancer()
all_features = list(data.feature_names)

# Only these 6 inputs will be taken from user
selected_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness"
]

# Create dataframe
df = pd.DataFrame(data.data, columns=all_features)
df["target"] = data.target

X = df.drop("target", axis=1)
y = df["target"]

# Save mean values
feature_means = X.mean()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)


def get_risk_level(malignant_percentage):
    if malignant_percentage < 30:
        return "Low Risk"
    elif malignant_percentage < 70:
        return "Moderate Risk"
    else:
        return "High Risk"


def get_color_theme(risk_percentage):
    if risk_percentage < 30:
        return {
            "primary": "#16a34a",
            "secondary": "#dcfce7",
            "accent": "#22c55e",
            "text": "#14532d",
            "shadow": "rgba(34, 197, 94, 0.25)"
        }
    elif risk_percentage < 70:
        return {
            "primary": "#f59e0b",
            "secondary": "#fef3c7",
            "accent": "#fbbf24",
            "text": "#92400e",
            "shadow": "rgba(245, 158, 11, 0.25)"
        }
    else:
        return {
            "primary": "#dc2626",
            "secondary": "#fee2e2",
            "accent": "#ef4444",
            "text": "#7f1d1d",
            "shadow": "rgba(239, 68, 68, 0.25)"
        }


def generate_ai_explanation(values, malignant_percentage):
    reasons = []

    if values["mean radius"] > 17:
        reasons.append("Mean radius is higher than the normal average range")
    if values["mean texture"] > 21:
        reasons.append("Mean texture value is elevated")
    if values["mean perimeter"] > 110:
        reasons.append("Mean perimeter is significantly high")
    if values["mean area"] > 700:
        reasons.append("Mean area is larger than typical benign cases")
    if values["mean smoothness"] > 0.11:
        reasons.append("Mean smoothness is on the higher side")
    if values["mean compactness"] > 0.16:
        reasons.append("Mean compactness indicates denser cell pattern")

    if not reasons:
        if malignant_percentage < 30:
            return "Most entered values are close to safer average ranges, so the model is showing a low estimated malignant risk."
        return "The prediction is influenced by a combination of entered values, even though none of the main six inputs are extremely high by themselves."

    return "Risk appears influenced by these factors: " + "; ".join(reasons) + "."


def get_doctor_recommendation(risk_percentage):
    if risk_percentage < 30:
        return "Routine monitoring is suggested."
    elif risk_percentage < 70:
        return "Consult a doctor."
    return "Immediate consultation required."


# ================= PDF FUNCTION =================
def generate_pdf(result):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Background
    pdf.setFillColor(colors.HexColor("#f1f5f9"))
    pdf.rect(0, 0, width, height, fill=1)

    # Header
    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.rect(0, height - 80, width, 80, fill=1)

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(50, height - 50, "Breast Cancer Report")

    pdf.setFont("Helvetica", 10)
    pdf.drawRightString(width - 50, height - 50,
        datetime.now().strftime("%d-%m-%Y %H:%M"))

    # Color logic
    if result["risk_percentage"] < 30:
        color = colors.green
    elif result["risk_percentage"] < 70:
        color = colors.orange
    else:
        color = colors.red

    # Prediction Card
    pdf.setFillColor(colors.white)
    pdf.roundRect(40, height - 180, width - 80, 80, 10, fill=1)

    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(60, height - 120, "Prediction")

    pdf.setFont("Helvetica", 12)
    pdf.drawString(60, height - 140, f"Result: {result['prediction']}")
    pdf.drawString(60, height - 160, f"Risk Level: {result['risk_level']}")

    # Risk Card
    pdf.setFillColor(colors.white)
    pdf.roundRect(40, height - 300, width - 80, 100, 10, fill=1)

    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(60, height - 220, "Risk Analysis")

    pdf.setFillColor(color)
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(60, height - 260,
                   f"{result['risk_percentage']:.2f}%")

    # Progress bar
    bar_width = 300
    pdf.setFillColor(colors.grey)
    pdf.rect(60, height - 290, bar_width, 10, fill=1)

    pdf.setFillColor(color)
    pdf.rect(60, height - 290,
             bar_width * result["risk_percentage"] / 100,
             10, fill=1)

    # Probabilities
    pdf.setFillColor(colors.white)
    pdf.roundRect(40, height - 450, width - 80, 120, 10, fill=1)

    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(60, height - 350, "Probabilities")

    pdf.setFont("Helvetica", 12)
    pdf.drawString(60, height - 380,
                   f"Benign: {result['benign_percentage']}%")
    pdf.drawString(60, height - 410,
                   f"Malignant: {result['malignant_percentage']}%")

    # Recommendation
    pdf.setFillColor(colors.white)
    pdf.roundRect(40, height - 600, width - 80, 120, 10, fill=1)

    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(60, height - 500, "Doctor Recommendation")

    pdf.setFont("Helvetica", 11)
    pdf.drawString(60, height - 530,
                   result["doctor_recommendation"])

    # Footer
    pdf.setFont("Helvetica", 9)
    pdf.setFillColor(colors.grey)
    pdf.drawCentredString(width / 2, 30,
        "AI Generated Report - Not a Medical Diagnosis")

    pdf.save()
    buffer.seek(0)
    return buffer


# ================= ROUTES =================
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    chart_data_json = None
    entered_values = {}

    if request.method == "POST":
        try:
            full_input = feature_means.copy()

            for feature in selected_features:
                value = request.form.get(feature)
                entered_values[feature] = value
                full_input[feature] = float(value)

            input_df = pd.DataFrame([full_input], columns=all_features)
            input_scaled = scaler.transform(input_df)

            prediction = model.predict(input_scaled)[0]
            probs = model.predict_proba(input_scaled)[0]

            malignant_prob = probs[0] * 100
            benign_prob = probs[1] * 100

            risk_percentage = malignant_prob
            risk_level = get_risk_level(risk_percentage)
            theme = get_color_theme(risk_percentage)
            ai_explanation = generate_ai_explanation(full_input, risk_percentage)
            doctor_recommendation = get_doctor_recommendation(risk_percentage)

            result = {
                "prediction": "Malignant" if prediction == 0 else "Benign",
                "benign_percentage": round(benign_prob, 2),
                "malignant_percentage": round(malignant_prob, 2),
                "risk_percentage": round(risk_percentage, 2),
                "risk_level": risk_level,
                "theme": theme,
                "ai_explanation": ai_explanation,
                "doctor_recommendation": doctor_recommendation
            }

            chart_data_json = json.dumps({
                "benign": round(benign_prob, 2),
                "malignant": round(malignant_prob, 2),
                "risk": round(risk_percentage, 2)
            })

        except Exception as e:
            result = {
                "error": f"Invalid input: {str(e)}",
                "theme": {
                    "primary": "#dc2626",
                    "secondary": "#fee2e2",
                    "accent": "#ef4444",
                    "text": "#7f1d1d",
                    "shadow": "rgba(239, 68, 68, 0.25)"
                }
            }

    return render_template(
        "index.html",
        features=selected_features,
        result=result,
        entered_values=entered_values,
        chart_data_json=chart_data_json
    )


# 🔥 DOWNLOAD PDF ROUTE
@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    result = json.loads(request.form.get("result"))
    pdf = generate_pdf(result)

    return send_file(
        pdf,
        as_attachment=True,
        download_name="report.pdf",
        mimetype="application/pdf"
    )


if __name__ == "__main__":
    app.run(debug=True)