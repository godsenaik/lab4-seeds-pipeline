from pathlib import Path
import json
import pandas as pd
import joblib

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from validate_data import validate_data

RANDOM_STATE = 42
IMPROVEMENT_MARGIN = 0.05
METRIC_NAME = "macro_f1"

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

def train_and_evaluate():
    df = validate_data()
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    baseline_score = f1_score(
        y_test, baseline_predictions, average="macro"
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", DummyClassifier(strategy="most_frequent")),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    model_score = f1_score(y_test, predictions, average="macro")

    required_score = baseline_score + IMPROVEMENT_MARGIN
    passed = model_score >= required_score

    metrics = {
        "dataset": "UCI Seeds",
        "task": "Multiclass classification",
        "metric": METRIC_NAME,
        "baseline_score": round(float(baseline_score), 4),
        "model_score": round(float(model_score), 4),
        "improvement_margin": IMPROVEMENT_MARGIN,
        "required_score": round(float(required_score), 4),
        "quality_gate": "PASS" if passed else "FAIL",
        "random_state": RANDOM_STATE,
        "test_size": 0.25,
    }

    with open(ARTIFACT_DIR / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nMODEL EVALUATION")
    print(json.dumps(metrics, indent=4))

    if not passed:
        raise SystemExit(
            "QUALITY GATE FAILED: Candidate did not improve "
            "over baseline by the required margin."
        )

    joblib.dump(model, ARTIFACT_DIR / "seeds_model.joblib")
    print("\nQUALITY GATE PASSED")
    print("Trained model saved to artifacts/seeds_model.joblib")

if __name__ == "__main__":
    train_and_evaluate()

