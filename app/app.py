from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head><title>Electric Grid Stability Predictor</title>
<style>
body { font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px; }
h1 { color: #333; }
input { width: 100%; padding: 8px; margin: 5px 0; box-sizing: border-box; }
button { background: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; width: 100%; margin-top: 10px; }
#result { margin-top: 20px; padding: 15px; border-radius: 5px; font-size: 1.2em; font-weight: bold; }
.stable { background: #d4edda; color: #155724; }
.unstable { background: #f8d7da; color: #721c24; }
</style>
</head>
<body>
<h1>⚡ Electric Grid Stability Predictor</h1>
<p>Enter the 12 grid features to predict stability:</p>
{% for f in features %}
<label>{{ f }}</label>
<input type="number" step="any" id="{{ f }}" placeholder="{{ f }}">
{% endfor %}
<button onclick="predict()">Predict</button>
<div id="result"></div>
<script>
function predict() {
    const features = {{ features|tojson }};
    const values = features.map(f => parseFloat(document.getElementById(f).value));
    fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({features: values})
    })
    .then(r => r.json())
    .then(data => {
        const el = document.getElementById('result');
        el.textContent = 'Prediction: ' + data.prediction;
        el.className = data.prediction === 'Stable' ? 'stable' : 'unstable';
    });
}
</script>
</body>
</html>
"""

FEATURES = ["tau1","tau2","tau3","tau4","p1","p2","p3","p4","g1","g2","g3","g4"]

@app.route("/")
def home():
    return render_template_string(HTML, features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    scaled = scaler.transform(features)
    pred = model.predict(scaled)[0]
    label = "Stable" if pred == 1 else "Unstable"
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)