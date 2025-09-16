from flask import Flask, request, jsonify
import joblib, os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "iris_classifier_model.pkl"

# Train or load model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Loaded saved model")
else:
    iris = load_iris()
    X, y = iris.data, iris.target
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print("✅ Trained and saved new model")

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Iris Classifier API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data["features"]).reshape(1, -1)
        pred = int(model.predict(features)[0])
        return jsonify({"prediction": pred})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Codespaces/Gitpod provides a port
    app.run(host="0.0.0.0", port=port)
