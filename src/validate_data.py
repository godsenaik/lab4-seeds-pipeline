from pathlib import Path
import pandas as pd
import requests

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "seeds_dataset.txt"

COLUMNS = [
    "area",
    "perimeter",
    "compactness",
    "kernel_length",
    "kernel_width",
    "asymmetry_coefficient",
    "kernel_groove_length",
    "target",
]

def validate_data():
    DATA_DIR.mkdir(exist_ok=True)

    if not DATA_FILE.exists():
        print("Downloading the Seeds dataset...")
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        DATA_FILE.write_bytes(response.content)

    if DATA_FILE.stat().st_size == 0:
        raise ValueError("Dataset file is empty.")

    df = pd.read_csv(
        DATA_FILE,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    if df.empty:
        raise ValueError("Dataset contains no rows.")

    if df.shape[1] != 8:
        raise ValueError(f"Expected 8 columns, found {df.shape[1]}.")

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    if not df.apply(lambda column: pd.to_numeric(column, errors="coerce").notna().all()).all():
        raise ValueError("Dataset contains non-numeric values.")

    if df["target"].nunique() < 2:
        raise ValueError("Target must contain at least two classes.")

    print("DATA VALIDATION PASSED")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(COLUMNS) - 1}")
    print(f"Target classes: {sorted(df['target'].unique().tolist())}")
    print(f"Dataset size: {DATA_FILE.stat().st_size} bytes")
    print("Missing values: 0")
    print("Dataset preview:")
    print(df.head())
    return df

if __name__ == "__main__":
    validate_data()
