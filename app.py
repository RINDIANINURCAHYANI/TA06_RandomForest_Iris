from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# --- Pastikan model ada ---
MODEL_PATH = "model/random_forest_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"File model tidak ditemukan di: {MODEL_PATH}")

# --- Load model ---
model = joblib.load(MODEL_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil input dari form
        sepal_length = float(request.form.get("sepal_length", 0))
        sepal_width = float(request.form.get("sepal_width", 0))
        petal_length = float(request.form.get("petal_length", 0))
        petal_width = float(request.form.get("petal_width", 0))

        # Buat input array
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Prediksi
        prediction = model.predict(input_data)[0]

        # Kirim hasil + gambar visualisasi ke halaman hasil
        return render_template(
            "result.html",
            species=prediction,
            confusion_img="static/img/confusion_matrix.png",
            feature_img="static/img/feature_importance.png",
            roc_img="static/img/roc_curve.png",
            tree_img="static/img/decision_tree.png"
        )

    except Exception as e:
        return render_template("result.html", species=f"Terjadi error: {e}")

if __name__ == "__main__":
    # Gunakan host='0.0.0.0' agar bisa diakses Railway
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
