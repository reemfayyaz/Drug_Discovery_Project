from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

FEATURE_COLUMNS = [
    "MW_g_mol",
    "LogP",
    "TPSA_A2",
    "HBD_Count",
    "HBA_Count",
    "Rotatable_Bonds",
    "Aromatic_Rings",
]

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "drug_discovery_10k_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"


st.set_page_config(
    page_title="Drug Discovery AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        .metric-card {
            padding: 1rem;
            border: 1px solid rgba(128,128,128,.25);
            border-radius: 12px;
            background: rgba(128,128,128,.06);
        }
        .app-note {
            padding: .9rem 1rem;
            border-left: 4px solid #4f8bf9;
            background: rgba(79,139,249,.08);
            border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_resource(show_spinner="Loading or training machine-learning models...")
def load_models(dataset_path: str):
    """Load saved models when available; otherwise train them from the project dataset."""
    xgb_path = MODEL_DIR / "xgboost_model.joblib"
    rf_path = MODEL_DIR / "random_forest_model.joblib"

    if xgb_path.exists() and rf_path.exists():
        return joblib.load(xgb_path), joblib.load(rf_path), "saved"

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    df = load_dataset(dataset_path).dropna(subset=FEATURE_COLUMNS + ["Prediction_Class"]).copy()
    X = df[FEATURE_COLUMNS]
    y = (df["Prediction_Class"].astype(str).str.upper() == "ACTIVE").astype(int)

    if y.nunique() < 2:
        raise ValueError("Prediction_Class must contain both ACTIVE and INACTIVE classes.")

    X_train, _, y_train, _ = train_test_split(
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
    return xgb_model, rf_model, "trained"


def extract_features(smiles: str):
    """Parse a SMILES string and calculate the seven model features."""
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is not installed. Install dependencies from requirements.txt.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    features = {
        "MW_g_mol": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA_A2": Descriptors.TPSA(mol),
        "HBD_Count": Descriptors.NumHDonors(mol),
        "HBA_Count": Descriptors.NumHAcceptors(mol),
        "Rotatable_Bonds": Descriptors.NumRotatableBonds(mol),
        "Aromatic_Rings": Descriptors.NumAromaticRings(mol),
    }
    return mol, features


def lipinski_rule_of_five(features: dict) -> tuple[int, str]:
    violations = sum(
        [
            features["MW_g_mol"] > 500,
            features["LogP"] > 5,
            features["HBD_Count"] > 5,
            features["HBA_Count"] > 10,
        ]
    )
    status = (
        "PASS"
        if violations <= 1
        else "FAIL"
    )
    return violations, status


def predict(features: dict, model):
    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    label = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0, 1])
    return ("ACTIVE" if label == 1 else "INACTIVE"), probability


def batch_screen(df: pd.DataFrame, model) -> pd.DataFrame:
    records = []
    for smiles in df["SMILES"].astype(str):
        mol, features = extract_features(smiles)
        if mol is None:
            records.append(
                {
                    "SMILES": smiles,
                    "Prediction": "INVALID",
                    "Probability_%": None,
                    "Lipinski_Violations": None,
                    "Lipinski_Status": "N/A",
                }
            )
            continue

        label, probability = predict(features, model)
        violations, status = lipinski_rule_of_five(features)
        records.append(
            {
                "SMILES": smiles,
                "Prediction": label,
                "Probability_%": round(probability * 100, 2),
                "Lipinski_Violations": violations,
                "Lipinski_Status": status,
                **{k: round(v, 4) for k, v in features.items()},
            }
        )
    return pd.DataFrame(records)


def render_sidebar() -> str:
    st.sidebar.title("🧬 Drug Discovery AI")
    st.sidebar.caption("Virtual screening & molecular profiling")
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Single Molecule", "Batch Screening", "Model Performance", "Dataset"],
    )
    st.sidebar.divider()
    st.sidebar.caption("Model features")
    st.sidebar.code("\n".join(FEATURE_COLUMNS))
    return page


def main():
    inject_css()
    page = render_sidebar()

    if not RDKIT_AVAILABLE:
        st.error("RDKit is unavailable. Install the project dependencies before using molecular functions.")
        st.stop()

    if not DATA_PATH.exists():
        st.warning(
            f"Dataset not found at `{DATA_PATH}`. "
            "Place `drug_discovery_10k_dataset.csv` inside the project's `data/` folder."
        )
        st.info("You can still review the application code, but model inference requires the training dataset.")
        st.stop()

    try:
        data = load_dataset(str(DATA_PATH))
        xgb_model, rf_model, model_source = load_models(str(DATA_PATH))
    except Exception as exc:
        st.error(f"Application initialization failed: {exc}")
        st.stop()

    if page == "Overview":
        st.title("Drug Discovery AI Platform")
        st.markdown(
            "A production-oriented Streamlit interface for molecular parsing, "
            "physicochemical profiling, Lipinski Rule-of-Five screening, and "
            "bioactivity classification."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dataset Rows", f"{len(data):,}")
        c2.metric("Features", len(FEATURE_COLUMNS))
        c3.metric("Model Engines", "2")
        c4.metric("Model Status", "Saved" if model_source == "saved" else "Runtime trained")

        st.subheader("Workflow")
        st.markdown(
            """
            1. **SMILES parsing** with RDKit  
            2. **Seven molecular descriptors** calculated from the structure  
            3. **XGBoost or Random Forest** classification  
            4. **Hit probability** reported from model probabilities  
            5. **Lipinski Rule of Five** drug-likeness filter  
            6. **Batch CSV screening** with downloadable results
            """
        )

        st.info(
            "Important: predictions are computational screening outputs, not clinical or "
            "regulatory conclusions. The PyTorch GNN shown in the original notebook is "
            "not used here because it was initialized but not trained on labeled data."
        )

    elif page == "Single Molecule":
        st.title("Single Molecule Analysis")
        st.caption("Enter a valid SMILES string to calculate descriptors and predict bioactivity.")

        left, right = st.columns([1.15, 0.85])
        with left:
            smiles = st.text_input(
                "SMILES",
                value="CCc1ccncc1",
                help="Example: CCc1ccncc1",
            )
            model_name = st.selectbox(
                "Inference model",
                ["XGBoost", "Random Forest"],
            )
            run = st.button("Run Analysis", type="primary", use_container_width=True)

        with right:
            st.markdown(
                '<div class="app-note"><b>Recommended workflow:</b> validate the SMILES, '
                'review descriptors, then interpret the model probability together with '
                'the Rule-of-Five result.</div>',
                unsafe_allow_html=True,
            )

        if run:
            mol, features = extract_features(smiles.strip())
            if mol is None:
                st.error("Invalid SMILES. Please enter a chemically valid SMILES string.")
                return

            model = xgb_model if model_name == "XGBoost" else rf_model
            prediction, probability = predict(features, model)
            violations, ro5_status = lipinski_rule_of_five(features)

            st.divider()
            col1, col2 = st.columns([0.9, 1.1])
            with col1:
                st.subheader("2D Structure")
                image = Draw.MolToImage(mol, size=(420, 320))
                st.image(image, caption=smiles)

            with col2:
                st.subheader("Prediction")
                p1, p2 = st.columns(2)
                p1.metric("Predicted Class", prediction)
                p2.metric("Active Probability", f"{probability:.1%}")

                if prediction == "ACTIVE":
                    st.success("Model classified the molecule as ACTIVE.")
                else:
                    st.warning("Model classified the molecule as INACTIVE.")

                if ro5_status == "PASS":
                    st.success(f"Lipinski Rule of Five: PASS ({violations} violation(s))")
                else:
                    st.warning(f"Lipinski Rule of Five: FAIL ({violations} violations)")

            st.subheader("Physicochemical Descriptors")
            descriptor_df = pd.DataFrame(
                {
                    "Descriptor": [
                        "Molecular Weight (g/mol)",
                        "LogP",
                        "TPSA (Å²)",
                        "H-bond Donors",
                        "H-bond Acceptors",
                        "Rotatable Bonds",
                        "Aromatic Rings",
                    ],
                    "Value": [
                        features["MW_g_mol"],
                        features["LogP"],
                        features["TPSA_A2"],
                        features["HBD_Count"],
                        features["HBA_Count"],
                        features["Rotatable_Bonds"],
                        features["Aromatic_Rings"],
                    ],
                }
            )
            st.dataframe(descriptor_df, use_container_width=True, hide_index=True)

    elif page == "Batch Screening":
        st.title("Batch Virtual Screening")
        st.caption("Upload a CSV containing a required `SMILES` column.")

        uploaded = st.file_uploader("Upload compound CSV", type=["csv"])
        model_name = st.selectbox("Inference model", ["XGBoost", "Random Forest"], key="batch_model")

        if uploaded:
            batch_df = pd.read_csv(uploaded)
            st.write("Input preview")
            st.dataframe(batch_df.head(10), use_container_width=True, hide_index=True)

            if "SMILES" not in batch_df.columns:
                st.error("The uploaded CSV must contain a column named `SMILES`.")
            elif st.button("Run Batch Screening", type="primary"):
                model = xgb_model if model_name == "XGBoost" else rf_model
                with st.spinner("Screening molecules..."):
                    results = batch_screen(batch_df[["SMILES"]].copy(), model)

                active_count = int((results["Prediction"] == "ACTIVE").sum())
                invalid_count = int((results["Prediction"] == "INVALID").sum())

                a, b, c = st.columns(3)
                a.metric("Compounds Processed", f"{len(results):,}")
                b.metric("Predicted Active", f"{active_count:,}")
                c.metric("Invalid SMILES", f"{invalid_count:,}")

                st.dataframe(results, use_container_width=True, hide_index=True)

                st.download_button(
                    "Download Screening Results",
                    results.to_csv(index=False).encode("utf-8"),
                    file_name="predicted_hits.csv",
                    mime="text/csv",
                    type="primary",
                )

    elif page == "Model Performance":
        st.title("Model Performance")
        st.caption("Evaluation is performed on a held-out stratified test split from the project dataset.")

        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split

        clean = data.dropna(subset=FEATURE_COLUMNS + ["Prediction_Class"]).copy()
        X = clean[FEATURE_COLUMNS]
        y = (clean["Prediction_Class"].astype(str).str.upper() == "ACTIVE").astype(int)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        rows = []
        for name, model in [("XGBoost", xgb_model), ("Random Forest", rf_model)]:
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]
            rows.append(
                {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision": precision_score(y_test, pred, zero_division=0),
                    "Recall": recall_score(y_test, pred, zero_division=0),
                    "F1": f1_score(y_test, pred, zero_division=0),
                    "ROC-AUC": roc_auc_score(y_test, prob),
                }
            )

        metrics_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        metrics_df.plot(kind="bar", ax=ax, ylim=(0, 1))
        ax.set_ylabel("Score")
        ax.set_xlabel("")
        ax.set_title("Model Comparison")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    elif page == "Dataset":
        st.title("Dataset Explorer")
        st.caption("Read-only view of the training dataset used by the application.")

        st.dataframe(data.head(100), use_container_width=True, hide_index=True)

        if "Prediction_Class" in data.columns:
            counts = data["Prediction_Class"].value_counts()
            st.subheader("Class Distribution")
            st.bar_chart(counts)

        st.subheader("Missing Values")
        missing = data.isna().sum().sort_values(ascending=False).to_frame("Missing Values")
        st.dataframe(missing, use_container_width=True)


if __name__ == "__main__":
    main()
