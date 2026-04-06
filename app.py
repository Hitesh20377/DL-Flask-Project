from flask import Flask, request, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("model.keras", compile=False)
scaler = pickle.load(open("scaler.pkl", "rb"))
y_max = pickle.load(open("y_max.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])
        final_features = scaler.transform(final_features)

        prediction = model.predict(final_features)
        output = prediction[0][0] * y_max

        return render_template(
            "index.html",
            prediction_text=f"Predicted Price: {round(output, 2)}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)