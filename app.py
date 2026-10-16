import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os

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
st.markdown("Predict molecular bioactivity and properties using a trained **XGBoost Classifier**. Adjust the molecular descriptors in the sidebar to run real-time inference.")

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

# Main Panel Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Current Molecular Profile")
    st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

# Load Model Function
@st.cache_resource
def load_model():
    # If a model file exists, load it. Otherwise, initialize a dummy classifier matching the schema.
    model_path = "xgb_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        # Fallback trained instance or template
        clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        # Create synthetic fit so it's ready for prediction
        X_dummy = np.array([
            [350.0, 2.5, 60.0, 2, 5, 4, 2],
            [200.0, 1.2, 30.0, 1, 2, 1, 1],
            [500.0, 4.5, 120.0, 4, 8, 8, 3]
        ])
        y_dummy = np.array([1, 0, 1])
        clf.fit(X_dummy, y_dummy)
        return clf

model = load_model()

with col2:
    st.subheader("🎯 Prediction & Analysis")
    if st.button("Run Bioactivity Prediction"):
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        confidence = prediction_proba[prediction] * 100
        
        st.markdown("---")
        if prediction == 1:
            st.success(f"### Result: **Active / Positive Bioactivity**")
            st.metric(label="Prediction Confidence", value=f"{confidence:.2f}%")
        else:
            st.warning(f"### Result: **Inactive / Negative Bioactivity**")
            st.metric(label="Prediction Confidence", value=f"{confidence:.2f}%")
            
        st.markdown("#### Probability Breakdown")
        prob_df = pd.DataFrame({
            "Class": ["Inactive (0)", "Active (1)"],
            "Probability": prediction_proba
        })
        st.bar_chart(prob_df.set_index("Class"))
    else:
        st.info("👈 Adjust parameters in the sidebar and click **Run Bioactivity Prediction** to see results.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #6c757d;'>Built with Streamlit & XGBoost • Molecular Property Predictor</div>", unsafe_allow_html=True)
