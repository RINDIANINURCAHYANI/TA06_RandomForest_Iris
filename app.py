from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# --- Load model ---
model = joblib.load("model/random_forest_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil input dari form
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        # Prediksi
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)[0]

        # Kirim hasil ke result.html
        return render_template("result.html", species=prediction)

    except Exception as e:
        return render_template("result.html", species=f"Terjadi error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
