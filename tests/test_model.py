import pytest
import numpy as np
import joblib

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

def test_model_loads():
    assert model is not None
    assert scaler is not None
def test_prediction_output():
    sample = np.array([[2.959, 3.114, 1.915, 2.111, -0.415, -0.588, -0.398, 1.401, 0.163, 0.259, 0.270, 0.270]])
    scaled = scaler.transform(sample)
    pred = model.predict(scaled)
    assert pred[0] in [0, 1]

def test_prediction_shape():
    sample = np.array([[2.959, 3.114, 1.915, 2.111, -0.415, -0.588, -0.398, 1.401, 0.163, 0.259, 0.270, 0.270]])
    scaled = scaler.transform(sample)
    pred = model.predict(scaled)
    assert len(pred) == 1