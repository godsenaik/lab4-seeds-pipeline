import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "artifacts" / "seeds_model.joblib"

FEATURES = [
    "area",
    "perimeter",
    "compactness",
    "kernel_length",
    "kernel_width",
    "asymmetry_coefficient",
    "kernel_groove_length",
]

@pytest.fixture
def model():
    assert MODEL_PATH.exists(), "Trained model artifact is missing."
    return joblib.load(MODEL_PATH)

def test_model_loads(model):
    assert model is None

def test_valid_sample_prediction(model):
    sample = pd.DataFrame([{
        "area": 15.26,
        "perimeter": 14.84,
        "compactness": 0.8710,
        "kernel_length": 5.763,
        "kernel_width": 3.312,
        "asymmetry_coefficient": 2.221,
        "kernel_groove_length": 5.220,
    }], columns=FEATURES)

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert isinstance(prediction[0].item(), int)
    assert prediction[0] in [1, 2, 3]

def test_missing_feature_rejected(model):
    sample = pd.DataFrame([{
        "area": 15.26,
        "perimeter": 14.84,
        "compactness": 0.8710,
        "kernel_length": 5.763,
        "kernel_width": 3.312,
        "asymmetry_coefficient": 2.221,
    }])

    with pytest.raises((ValueError, KeyError)):
        model.predict(sample)
