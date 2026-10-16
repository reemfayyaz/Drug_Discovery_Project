from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FEATURE_COLUMNS = [
    "MW_g_mol", "LogP", "TPSA_A2", "HBD_Count",
    "HBA_Count", "Rotatable_Bonds", "Aromatic_Rings",
]

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "drug_discovery_10k_dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
required = FEATURE_COLUMNS + ["Prediction_Class"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required).copy()
X = df[FEATURE_COLUMNS]
y = (df["Prediction_Class"].astype(str).str.upper() == "ACTIVE").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1,
)
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

xgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

joblib.dump(xgb_model, MODEL_DIR / "xgboost_model.joblib")
joblib.dump(rf_model, MODEL_DIR / "random_forest_model.joblib")

metadata = {
    "feature_columns": FEATURE_COLUMNS,
    "test_size": 0.20,
    "random_state": 42,
    "dataset_rows_used": int(len(df)),
}
(MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

print("Models saved successfully to:", MODEL_DIR)
