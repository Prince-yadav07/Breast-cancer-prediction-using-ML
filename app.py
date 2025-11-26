from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model + scaler
saved = pickle.load(open("model.pkl", "rb"))
model = saved["model"]
scaler = saved["scaler"]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            vals = [
                float(request.form["mean_radius"]),
                float(request.form["mean_texture"]),
                float(request.form["mean_smoothness"]),
                float(request.form["mean_compactness"]),
                float(request.form["mean_concavity"])
            ]

            features_scaled = scaler.transform([vals])
            result = model.predict(features_scaled)[0]

            prediction = (
                "🟢 Non-Cancerous (Benign)"
                if result == 0 else
                "🔴 Cancerous (Malignant)"
            )

        except Exception as e:
            error = "Invalid input. Please enter valid numbers."

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
