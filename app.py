import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
from pathlib import Path

# Optional imports
try:
    import joblib
except Exception:
    joblib = None

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False

# Try import torch_geometric (optional)
try:
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except Exception:
    TORCH_GEOMETRIC_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="Molecular Property & Bioactivity Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #004080;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.title("🧬 Molecular Property & Bioactivity Predictor")
st.markdown("Predict molecular bioactivity and properties using pre-trained models (XGBoost, Random Forest, optional GNN). Adjust descriptors in the sidebar and select a model to run inference.")

# Sidebar - Feature Inputs
st.sidebar.header("🧪 Molecular Descriptors")
st.sidebar.markdown("Provide the chemical and topological parameters below:")

def user_input_features():
    mw = st.sidebar.slider("Molecular Weight (MW g/mol)", 50.0, 900.0, 350.0, 1.0, help="Molecular weight in grams per mole.")
    logp = st.sidebar.slider("LogP (Lipophilicity)", -3.0, 8.0, 2.5, 0.1, help="Octanol-water partition coefficient.")
    tpsa = st.sidebar.slider("TPSA (Å²)", 0.0, 250.0, 60.0, 1.0, help="Topological Polar Surface Area.")
    hbd = st.sidebar.slider("H-Bond Donors (HBD)", 0, 10, 2, 1, help="Number of hydrogen bond donors.")
    hba = st.sidebar.slider("H-Bond Acceptors (HBA)", 0, 20, 5, 1, help="Number of hydrogen bond acceptors.")
    rot_bonds = st.sidebar.slider("Rotatable Bonds", 0, 15, 4, 1, help="Number of rotatable single bonds.")
    aromatic_rings = st.sidebar.slider("Aromatic Rings", 0, 6, 2, 1, help="Number of aromatic rings.")
    data = {
        'MW_g_mol': mw,
        'LogP': logp,
        'TPSA_A2': tpsa,
        'HBD_Count': hbd,
        'HBA_Count': hba,
        'Rotatable_Bonds': rot_bonds,
        'Aromatic_Rings': aromatic_rings
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# SMILES input for graph-based GNN inference
st.sidebar.header("SMILES input (optional for GNN)")
smiles_input = st.sidebar.text_input("SMILES (for GNN graph input)", value="CCc1ccncc1", help="Provide a SMILES string to run graph-based GNN inference.")

# Models directory
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"

@st.cache_resource
def load_models():
    models = {}
    # XGBoost (pickle or joblib)
    xgb_path_pkl = MODEL_DIR / "xgb_model.pkl"
    xgb_path_joblib = MODEL_DIR / "xgboost_model.joblib"
    if xgb_path_pkl.exists():
        try:
            with open(xgb_path_pkl, "rb") as f:
                models['XGBoost'] = pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load XGBoost from {xgb_path_pkl}: {e}")
    elif xgb_path_joblib.exists() and joblib is not None:
        try:
            models['XGBoost'] = joblib.load(xgb_path_joblib)
        except Exception as e:
            st.warning(f"Failed to load XGBoost from {xgb_path_joblib}: {e}")

    # Random Forest (pickle or joblib)
    rf_path_pkl = MODEL_DIR / "rf_model.pkl"
    rf_path_joblib = MODEL_DIR / "random_forest_model.joblib"
    if rf_path_pkl.exists():
        try:
            with open(rf_path_pkl, "rb") as f:
                models['Random Forest'] = pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load Random Forest from {rf_path_pkl}: {e}")
    elif rf_path_joblib.exists() and joblib is not None:
        try:
            models['Random Forest'] = joblib.load(rf_path_joblib)
        except Exception as e:
            st.warning(f"Failed to load Random Forest from {rf_path_joblib}: {e}")

    # GNN (PyTorch)
    gnn_path_pt = MODEL_DIR / "gnn_model.pt"
    gnn_path_pth = MODEL_DIR / "gnn_model.pth"
    if (gnn_path_pt.exists() or gnn_path_pth.exists()) and torch is not None:
        p = gnn_path_pt if gnn_path_pt.exists() else gnn_path_pth
        try:
            gnn = torch.load(p, map_location='cpu')
            # if a state_dict is loaded (dict but not nn.Module) we cannot use it directly
            if isinstance(gnn, dict) and not isinstance(gnn, torch.nn.Module):
                st.warning(f"Loaded {p.name} appears to be a state_dict; reconstructing model is required to use it.")
            else:
                gnn.eval()
                models['GNN'] = gnn
        except Exception as e:
            st.warning(f"Failed to load GNN model from {p}: {e}")
    else:
        if (gnn_path_pt.exists() or gnn_path_pth.exists()) and torch is None:
            st.info("A GNN model file exists but PyTorch is not installed in the environment.")

    return models

models = load_models()
available_names = list(models.keys())
if not available_names:
    available_names = ["XGBoost", "Random Forest", "GNN"]  # show options even if not present

# Main Panel Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Current Molecular Profile")
    st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

# Model selection UI
with col2:
    st.subheader("🎯 Prediction & Analysis")
    selected_model = st.selectbox("Select model", options=available_names)

    # GNN input mode selection
    gnn_input_mode = st.radio("GNN input mode", options=["Descriptors (7 features)", "Graph (SMILES)"], index=0) if 'GNN' in available_names or 'GNN' in available_names else "Descriptors (7 features)"

    def smiles_to_graph(smiles: str):
        """Convert SMILES to a simple graph representation usable by many GNNs.
        Returns either a torch_geometric.data.Data if PyG is available, or a tuple (x, edge_index).
        x: [num_nodes, num_node_features]
        edge_index: [2, num_edges]
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required to convert SMILES to graph. Install rdkit via conda-forge.")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES provided.")

        # Node features: atomic number, degree, total_num_H, implicit valence, aromatic
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append([
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetTotalNumHs(),
                atom.GetFormalCharge(),
                1 if atom.GetIsAromatic() else 0,
            ])
        x = np.array(atom_features, dtype=np.float32)

        # Edges
        edge_list = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            edge_list.append((a1, a2))
            edge_list.append((a2, a1))
        if len(edge_list) == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        else:
            edge_index = np.array(edge_list, dtype=np.int64).T

        if TORCH_GEOMETRIC_AVAILABLE and torch is not None:
            x_t = torch.tensor(x, dtype=torch.float)
            edge_t = torch.tensor(edge_index, dtype=torch.long)
            data = Data(x=x_t, edge_index=edge_t)
            return data

        # fallback: return numpy arrays
        return x, edge_index

    def model_predict(model_name, X_df, gnn_mode='Descriptors', smiles=None):
        model_obj = models.get(model_name)
        # If model not present, create a lightweight fallback (for demo) only for XGBoost and RF
        if model_obj is None:
            if model_name == 'XGBoost':
                clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
                X_dummy = np.array([[350.0, 2.5, 60.0, 2, 5, 4, 2], [200.0, 1.2, 30.0, 1, 2, 1, 1], [500.0, 4.5, 120.0, 4, 8, 8, 3]])
                y_dummy = np.array([1, 0, 1])
                clf.fit(X_dummy, y_dummy)
                model_obj = clf
            elif model_name == 'Random Forest':
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                X_dummy = np.array([[350.0, 2.5, 60.0, 2, 5, 4, 2], [200.0, 1.2, 30.0, 1, 2, 1, 1], [500.0, 4.5, 120.0, 4, 8, 8, 3]])
                y_dummy = np.array([1, 0, 1])
                clf.fit(X_dummy, y_dummy)
                model_obj = clf
            else:
                return None, None

        # GNN inference
        if model_name == 'GNN':
            if torch is None or model_obj is None:
                return None, None

            # Descriptor-based input (7 features)
            if gnn_mode == 'Descriptors':
                try:
                    t = torch.tensor(X_df.values, dtype=torch.float32)
                    with torch.no_grad():
                        out = model_obj(t)
                    out_np = out.detach().cpu().numpy()
                    if out_np.ndim == 2 and out_np.shape[1] == 2:
                        probs = out_np
                        labels = probs.argmax(axis=1)
                    else:
                        logits = out_np.ravel()
                        probs_pos = 1 / (1 + np.exp(-logits))
                        probs = np.vstack([1 - probs_pos, probs_pos]).T
                        labels = (probs_pos > 0.5).astype(int)
                    return labels, probs
                except Exception as e:
                    st.warning(f"GNN inference failed with descriptor input: {e}")
                    return None, None

            # Graph-based input (SMILES -> graph)
            else:
                if smiles is None:
                    st.warning("No SMILES provided for graph-based GNN input.")
                    return None, None
                try:
                    graph = smiles_to_graph(smiles)
                except Exception as e:
                    st.warning(f"Failed to convert SMILES to graph: {e}")
                    return None, None

                try:
                    # If we have a torch_geometric Data object, pass directly
                    if TORCH_GEOMETRIC_AVAILABLE and isinstance(graph, Data):
                        data = graph
                        data = data.to(next(model_obj.parameters()).device) if hasattr(model_obj, 'parameters') else data
                        with torch.no_grad():
                            out = model_obj(data)
                        out_np = out.detach().cpu().numpy()
                    else:
                        # fallback: graph is (x, edge_index) numpy arrays
                        x_np, edge_index = graph
                        x_t = torch.tensor(x_np, dtype=torch.float32)
                        edge_t = torch.tensor(edge_index, dtype=torch.long)
                        # Try common call signatures
                        with torch.no_grad():
                            try:
                                out = model_obj(x_t, edge_t)
                            except Exception:
                                out = model_obj(x_t)
                        out_np = out.detach().cpu().numpy()

                    if out_np.ndim == 2 and out_np.shape[1] == 2:
                        probs = out_np
                        labels = probs.argmax(axis=1)
                    else:
                        logits = out_np.ravel()
                        probs_pos = 1 / (1 + np.exp(-logits))
                        probs = np.vstack([1 - probs_pos, probs_pos]).T
                        labels = (probs_pos > 0.5).astype(int)

                    return labels, probs
                except Exception as e:
                    st.warning(f"GNN inference with graph input failed: {e}")
                    return None, None

        # Standard sklearn/xgboost API
        try:
            labels = model_obj.predict(X_df.values)
        except Exception:
            labels = model_obj.predict(X_df)
        proba = None
        try:
            proba = model_obj.predict_proba(X_df.values)
        except Exception:
            try:
                proba = model_obj.predict_proba(X_df)
            except Exception:
                proba = None
        return labels, proba

    if st.button("Run Bioactivity Prediction"):
        labels, proba = model_predict(selected_model, input_df, gnn_mode=('Descriptors' if gnn_input_mode.startswith('Descriptors') else 'Graph'), smiles=smiles_input if gnn_input_mode.startswith('Graph') else None)
        if labels is None:
            st.error("Model is unavailable or cannot accept the provided input format.")
        else:
            prediction = int(labels[0])
            if proba is not None:
                # Determine confidence
                if proba.ndim == 2 and proba.shape[1] == 2:
                    confidence = proba[0, prediction] * 100
                else:
                    confidence = proba.ravel()[0] * 100
            else:
                confidence = None

            st.markdown("---")
            if prediction == 1:
                st.success(f"### Result: **Active / Positive Bioactivity**")
            else:
                st.warning(f"### Result: **Inactive / Negative Bioactivity**")
            if confidence is not None:
                st.metric(label="Prediction Confidence", value=f"{confidence:.2f}%")

            if proba is not None:
                st.markdown("#### Probability Breakdown")
                if proba.ndim == 2 and proba.shape[1] == 2:
                    prob_df = pd.DataFrame({
                        "Class": ["Inactive (0)", "Active (1)"],
                        "Probability": proba[0]
                    })
                    st.bar_chart(prob_df.set_index("Class"))
                else:
                    p = proba.ravel()[0]
                    prob_df = pd.DataFrame({"Class": ["Inactive (0)", "Active (1)"], "Probability": [1 - p, p]})
                    st.bar_chart(prob_df.set_index("Class"))
    else:
        st.info("👈 Adjust parameters in the sidebar and click **Run Bioactivity Prediction** to see results.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #6c757d;'>Built with Streamlit • Supports XGBoost, Random Forest, and optional GNN models</div>", unsafe_allow_html=True)
