from __future__ import annotations

import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import torch
except Exception:
    torch = None

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False

try:
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except Exception:
    TORCH_GEOMETRIC_AVAILABLE = False

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
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

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
        .metric-card {padding: 1rem; border-radius: 10px; background: rgba(128,128,128,.06);}
        .app-note {padding: .9rem 1rem; border-left: 4px solid #4f8bf9; background: rgba(79,139,249,.08); border-radius: 6px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_models():
    models = {}

    candidates = [
        ("XGBoost", [MODEL_DIR / "xgb_model.pkl", MODEL_DIR / "xgboost_model.joblib"]),
        ("Random Forest", [MODEL_DIR / "rf_model.pkl", MODEL_DIR / "random_forest_model.joblib"]),
        ("GNN", [MODEL_DIR / "gnn_model.pt", MODEL_DIR / "gnn_model.pth"]),
    ]

    for name, paths in candidates:
        for path in paths:
            if not path.exists():
                continue
            try:
                if name == "GNN":
                    if torch is None:
                        st.warning("PyTorch is not installed; GNN model cannot be loaded.")
                        break
                    model = torch.load(path, map_location="cpu")
                    if isinstance(model, dict) and not hasattr(model, "forward"):
                        st.warning(f"{path.name} looks like a state_dict, not a full model. Reconstruct the model class before loading.")
                        continue
                    if hasattr(model, "eval"):
                        model.eval()
                    models[name] = model
                    break
                else:
                    with open(path, "rb") as f:
                        models[name] = pickle.load(f)
                    break
            except Exception as exc:
                st.warning(f"Failed to load {name} from {path.name}: {exc}")

    return models


def build_feature_row_from_smiles(smiles: str) -> dict | None:
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "MW_g_mol": float(Descriptors.MolWt(mol)),
        "LogP": float(Descriptors.MolLogP(mol)),
        "TPSA_A2": float(Descriptors.TPSA(mol)),
        "HBD_Count": int(Descriptors.NumHDonors(mol)),
        "HBA_Count": int(Descriptors.NumHAcceptors(mol)),
        "Rotatable_Bonds": int(Descriptors.NumRotatableBonds(mol)),
        "Aromatic_Rings": int(Descriptors.NumAromaticRings(mol)),
    }


def build_graph_from_smiles(smiles: str):
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required for graph-based GNN input.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetTotalNumHs()),
            float(atom.GetFormalCharge()),
            1.0 if atom.GetIsAromatic() else 0.0,
        ])

    x = np.asarray(node_features, dtype=np.float32)
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))

    edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)

    if TORCH_GEOMETRIC_AVAILABLE and torch is not None:
        return Data(x=torch.tensor(x), edge_index=torch.tensor(edge_index, dtype=torch.long))

    return x, edge_index


def predict_tabular(model, X_df: pd.DataFrame):
    if model is None:
        return None, None
    try:
        pred = model.predict(X_df)
    except Exception:
        pred = model.predict(X_df.to_numpy())

    try:
        prob = model.predict_proba(X_df)
    except Exception:
        try:
            prob = model.predict_proba(X_df.to_numpy())
        except Exception:
            prob = None

    return pred, prob


def predict_gnn(model, input_mode: str, features_df: pd.DataFrame | None = None, smiles: str | None = None):
    if torch is None:
        return None, None

    if input_mode == "SMILES":
        if not smiles:
            return None, None

        desc = build_feature_row_from_smiles(smiles)
        if desc is not None:
            X = pd.DataFrame([desc], columns=FEATURE_COLUMNS)
            try:
                with torch.no_grad():
                    tensor = torch.tensor(X.to_numpy(dtype=np.float32), dtype=torch.float32)
                    out = model(tensor)
                out_np = out.detach().cpu().numpy()
                if out_np.ndim == 2 and out_np.shape[1] > 1:
                    probs = out_np[0]
                    label = int(np.argmax(probs))
                    return np.array([label]), np.asarray([probs], dtype=np.float32)
                logits = np.asarray(out_np).reshape(-1)
                probs = 1.0 / (1.0 + np.exp(-logits))
                label = int(probs[0] > 0.5)
                return np.array([label]), np.asarray([[1.0 - probs[0], probs[0]]], dtype=np.float32)
            except Exception:
                pass

        # Fallback to graph-based route
        try:
            graph = build_graph_from_smiles(smiles)
            with torch.no_grad():
                try:
                    if TORCH_GEOMETRIC_AVAILABLE and hasattr(graph, "x"):
                        out = model(graph)
                    else:
                        x, edge_index = graph
                        try:
                            out = model(torch.tensor(x, dtype=torch.float32), torch.tensor(edge_index, dtype=torch.long))
                        except Exception:
                            out = model(torch.tensor(x, dtype=torch.float32))
                out_np = out.detach().cpu().numpy()
            if out_np.ndim == 2 and out_np.shape[1] > 1:
                probs = out_np[0]
                label = int(np.argmax(probs))
                return np.array([label]), np.asarray([probs], dtype=np.float32)
            logits = np.asarray(out_np).reshape(-1)
            probs = 1.0 / (1.0 + np.exp(-logits))
            label = int(probs[0] > 0.5)
            return np.array([label]), np.asarray([[1.0 - probs[0], probs[0]]], dtype=np.float32)
        except Exception as exc:
            st.warning(f"GNN graph input failed: {exc}")
            return None, None

    # descriptor input
    if features_df is not None:
        X = features_df[FEATURE_COLUMNS]
        try:
            with torch.no_grad():
                tensor = torch.tensor(X.to_numpy(dtype=np.float32), dtype=torch.float32)
                out = model(tensor)
            out_np = out.detach().cpu().numpy()
            if out_np.ndim == 2 and out_np.shape[1] > 1:
                probs = out_np[0]
                label = int(np.argmax(probs))
                return np.array([label]), np.asarray([probs], dtype=np.float32)
            logits = np.asarray(out_np).reshape(-1)
            probs = 1.0 / (1.0 + np.exp(-logits))
            label = int(probs[0] > 0.5)
            return np.array([label]), np.asarray([[1.0 - probs[0], probs[0]]], dtype=np.float32)
        except Exception as exc:
            st.warning(f"GNN descriptor input failed: {exc}")
            return None, None

    return None, None


def make_feature_df_from_sliders():
    data = {
        "MW_g_mol": st.sidebar.slider("Molecular Weight (g/mol)", 50.0, 900.0, 350.0, 1.0),
        "LogP": st.sidebar.slider("LogP", -3.0, 8.0, 2.5, 0.1),
        "TPSA_A2": st.sidebar.slider("TPSA (Å²)", 0.0, 250.0, 60.0, 1.0),
        "HBD_Count": st.sidebar.slider("H-bond Donors", 0, 10, 2, 1),
        "HBA_Count": st.sidebar.slider("H-bond Acceptors", 0, 20, 5, 1),
        "Rotatable_Bonds": st.sidebar.slider("Rotatable Bonds", 0, 15, 4, 1),
        "Aromatic_Rings": st.sidebar.slider("Aromatic Rings", 0, 6, 2, 1),
    }
    return pd.DataFrame([data], columns=FEATURE_COLUMNS)


def main():
    inject_css()
    st.title("Drug Discovery AI")
    st.caption("Run XGBoost, Random Forest, or GNN predictions using either SMILES or descriptor values.")

    feature_df = make_feature_df_from_sliders()
    st.sidebar.subheader("SMILES input")
    smiles_input = st.sidebar.text_input("SMILES string", value="CCc1ccncc1")

    model_name = st.selectbox("Model", ["XGBoost", "Random Forest", "GNN"])
    input_mode = st.radio("Input mode", ["SMILES", "Manual descriptors"], index=0 if model_name == "GNN" else 1)

    if model_name == "GNN":
        st.info("For GNN, the app will try SMILES first and then fall back to descriptor values automatically.")

    if st.button("Run prediction", type="primary"):
        models = load_models()
        model = models.get(model_name)

        if model is None:
            st.error(f"{model_name} model is not available in the models/ directory.")
            st.stop()

        if model_name == "GNN":
            pred, prob = predict_gnn(model, input_mode, feature_df, smiles_input if input_mode == "SMILES" else None)
            if pred is None or prob is None:
                st.error("GNN inference failed. Check whether the model was saved as a full PyTorch module and whether SMILES/RDKit are available.")
                st.stop()
            label = int(pred[0])
            conf = float(prob[0][label] * 100.0)
            st.success(f"Predicted class: {label}")
            st.metric("Confidence", f"{conf:.1f}%")
            st.dataframe(pd.DataFrame({"Class": [0, 1], "Probability": prob[0]}), use_container_width=True)
            return

        pred, prob = predict_tabular(model, feature_df)
        if pred is None:
            st.error("Prediction failed.")
            st.stop()

        label = int(pred[0])
        if prob is not None:
            conf = float(prob[0][label] * 100.0)
        else:
            conf = None

        st.success(f"Predicted label: {label}")
        if conf is not None:
            st.metric("Confidence", f"{conf:.1f}%")
        if prob is not None:
            st.dataframe(pd.DataFrame({"Class": [0, 1], "Probability": prob[0]}), use_container_width=True)

    st.subheader("Selected descriptor values")
    st.dataframe(feature_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
