import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import inspect

# Config setup
st.set_page_config(
    page_title="Virtual Screening & Drug Discovery Pipeline",
    page_icon="🧪",
    layout="wide",
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #2E86C1;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #5D6D7E;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🧪 Virtual Screening & Drug Discovery Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Exploratory Data Analysis & Bioactivity Predictive Modeling</div>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("📑 Navigation")
options = st.sidebar.radio(
    "Select Section:",
    ["Environment & Data Ingestion", "Exploratory Data Analysis (EDA)", "Physicochemical Properties", "Bioactivity Modeling"]
)

# Sample Synthetic Data Generator for demonstration
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n_samples = 250
    data = {
        'Molecule_ID': [f'CHEMBL_{i}' for i in range(1000, 1000 + n_samples)],
        'MW': np.random.uniform(150, 600, n_samples),
        'LogP': np.random.uniform(-1, 6, n_samples),
        'NumHDonors': np.random.randint(0, 10, n_samples),
        'NumHAcceptors': np.random.randint(1, 12, n_samples),
        'pIC50': np.random.uniform(3, 9, n_samples)
    }
    df = pd.DataFrame(data)
    # Define Lipinski Rule Compliance
    df['Lipinski_Pass'] = (df['MW'] <= 500) & (df['LogP'] <= 5) & (df['NumHDonors'] <= 5) & (df['NumHAcceptors'] <= 10)
    # Classification based on pIC50 (example)
    df['Bioactivity_Class'] = pd.cut(df['pIC50'], bins=[0, 5, 6, 10], labels=['inactive', 'intermediate', 'active'])
    return df

# Models directory
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"

# Model detection & wrappers
class ModelWrapper:
    def __init__(self, name, model, kind="joblib", requires_graph=False):
        self.name = name
        self.model = model
        self.kind = kind
        self.requires_graph = requires_graph

    def predict(self, X: pd.DataFrame):
        if self.requires_graph:
            raise RuntimeError("Model requires graph inputs and cannot accept descriptor table.")
        if self.kind == "torch":
            # Attempt to forward tabular input through torch model if possible
            import torch
            self.model.eval()
            with torch.no_grad():
                t = torch.tensor(X.values, dtype=torch.float32)
                out = self.model(t)
                # assume binary classification — logits or probabilities
                out_np = out.detach().cpu().numpy()
                if out_np.ndim == 2 and out_np.shape[1] > 1:
                    labels = out_np.argmax(axis=1)
                else:
                    labels = (out_np.ravel() > 0.5).astype(int)
                return labels
        else:
            return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        if self.requires_graph:
            return None
        if self.kind == "torch":
            import torch
            self.model.eval()
            with torch.no_grad():
                t = torch.tensor(X.values, dtype=torch.float32)
                out = self.model(t)
                out_np = out.detach().cpu().numpy()
                # if multiclass
                if out_np.ndim == 2 and out_np.shape[1] > 1:
                    # softmax
                    exp = np.exp(out_np - out_np.max(axis=1, keepdims=True))
                    probs = exp / exp.sum(axis=1, keepdims=True)
                    return probs
                else:
                    # sigmoid
                    probs = 1 / (1 + np.exp(-out_np.ravel()))
                    return np.vstack([1 - probs, probs]).T
        else:
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)
            # some xgboost/sklearn models may have decision_function
            if hasattr(self.model, "decision_function"):
                try:
                    df = self.model.decision_function(X)
                    # convert to probability-like via sigmoid
                    probs = 1 / (1 + np.exp(-df))
                    return np.vstack([1 - probs, probs]).T
                except Exception:
                    return None
            return None


def detect_models():
    wrappers = {}
    if not MODEL_DIR.exists():
        return wrappers

    # joblib / sklearn / xgboost pickles
    for p in MODEL_DIR.glob("*.joblib"):
        try:
            m = joblib.load(p)
            name = p.stem
            wrappers[name] = ModelWrapper(name, m, kind="joblib")
        except Exception as exc:
            st.warning(f"Failed loading {p.name}: {exc}")

    for p in MODEL_DIR.glob("*.pkl"):
        try:
            m = joblib.load(p)
            name = p.stem
            wrappers[name] = ModelWrapper(name, m, kind="joblib")
        except Exception as exc:
            st.warning(f"Failed loading {p.name}: {exc}")

    # torch models
    try:
        import torch
        for p in list(MODEL_DIR.glob("*.pt")) + list(MODEL_DIR.glob("*.pth")):
            try:
                m = torch.load(p, map_location="cpu")
                name = p.stem
                # determine if the loaded object is a state_dict or nn.Module
                requires_graph = False
                if isinstance(m, dict) and not isinstance(m, torch.nn.Module):
                    # state_dict — cannot safely load without model class -> mark as requires_graph to be safe
                    wrappers[name] = ModelWrapper(name, m, kind="torch", requires_graph=True)
                else:
                    wrappers[name] = ModelWrapper(name, m, kind="torch")
            except Exception as exc:
                st.warning(f"Failed loading {p.name} as torch model: {exc}")
    except Exception:
        # torch not installed — skip
        pass

    return wrappers

# Data Upload or Load Sample
st.sidebar.subheader("Dataset Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_sample_data()
    st.sidebar.info("Using synthetic dataset. Upload a CSV to use your own data.")

# Detect models present
available_models = detect_models()

# Section 1: Ingestion & Overview
if options == "Environment & Data Ingestion":
    st.header("📋 Dataset Overview & Properties")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Compounds", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    col3.metric("Lipinski Compliant", f"{df['Lipinski_Pass'].sum() if 'Lipinski_Pass' in df.columns else 'N/A'}")

    st.subheader("Raw Dataset Head")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe().T, use_container_width=True)

# Section 2: EDA
elif options == "Exploratory Data Analysis (EDA)":
    st.header("📊 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Bioactivity Class Distribution")
        if 'Bioactivity_Class' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x='Bioactivity_Class', palette='Set2', ax=ax)
            ax.set_title("Distribution of Bioactivity Classes")
            st.pyplot(fig)
        else:
            st.warning("Column 'Bioactivity_Class' not found in dataset.")

    with col2:
        st.subheader("pIC50 Value Distribution")
        if 'pIC50' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df['pIC50'], kde=True, color='teal', ax=ax)
            ax.set_title("Distribution of pIC50 Values")
            st.pyplot(fig)
        else:
            st.warning("Column 'pIC50' not found in dataset.")

    st.subheader("Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

# Section 3: Physicochemical Properties
elif options == "Physicochemical Properties":
    st.header("🧪 Physicochemical Property Visualizations")
    
    st.write("Examine chemical property spaces relevant to drug-likeness (Lipinski's Rule of 5).")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Molecular Weight vs LogP")
        if 'MW' in df.columns and 'LogP' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            hue_col = 'Bioactivity_Class' if 'Bioactivity_Class' in df.columns else None
            sns.scatterplot(data=df, x='MW', y='LogP', hue=hue_col, palette='viridis', alpha=0.8, ax=ax)
            ax.axvline(500, color='red', linestyle='--', label='MW Limit (500)')
            ax.axhline(5, color='orange', linestyle='--', label='LogP Limit (5)')
            ax.legend()
            st.pyplot(fig)

    with col2:
        st.subheader("Hydrogen Bond Donors vs Acceptors")
        if 'NumHDonors' in df.columns and 'NumHAcceptors' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x='NumHDonors', y='NumHAcceptors', palette='Blues', ax=ax)
            st.pyplot(fig)

# Section 4: Bioactivity Modeling
elif options == "Bioactivity Modeling":
    st.header("🤖 Bioactivity Predictive Modeling")
    st.write("Train a machine learning model to predict bioactivity (pIC50) or compare existing models.")

    features = st.multiselect("Select Feature Columns for Modeling:", 
                                options=[c for c in df.columns if c not in ['Molecule_ID', 'pIC50', 'Bioactivity_Class', 'Lipinski_Pass']],
                                default=[c for c in df.columns if c in ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors']])

    target = 'pIC50'

    # Show detected models
    st.subheader("Detected models in models/ directory")
    if available_models:
        for nm, w in available_models.items():
            st.write(f"- {nm} (kind={w.kind}, requires_graph={w.requires_graph})")
    else:
        st.info("No pre-trained models detected in models/. You can train a new model below or add model files to models/.")

    # If classification target available
    is_classification = 'Bioactivity_Class' in df.columns

    # Model comparison UI
    if is_classification and available_models:
        st.subheader("Model comparison (classification)")
        selected = st.multiselect("Select models to compare:", options=list(available_models.keys()), default=list(available_models.keys())[:2])
        if st.button("Run model comparison"):
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report

            clean = df.dropna(subset=features + ['Bioactivity_Class']).copy()
            X = clean[features]
            # convert classes to binary ACTIVE/INACTIVE if possible
            y = clean['Bioactivity_Class'].astype(str)
            # Try mapping: active -> ACTIVE, inactive -> INACTIVE; anything else -> intermediate -> treat as inactive
            y_bin = (y.str.lower().str.contains('active')).astype(int)

            X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42, stratify=y_bin)

            comparison_rows = []
            roc_data = {}
            pr_data = {}
            cm_data = {}

            for name in selected:
                wrapper = available_models.get(name)
                if wrapper is None:
                    st.warning(f"Model {name} not found at runtime")
                    continue
                if wrapper.requires_graph:
                    st.warning(f"Skipping {name}: model requires graph inputs (cannot use descriptor table).")
                    continue

                try:
                    preds = wrapper.predict(X_test)
                except Exception as exc:
                    st.warning(f"Failed predict() for {name}: {exc}")
                    continue

                proba = None
                try:
                    proba = wrapper.predict_proba(X_test)
                except Exception:
                    proba = None

                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, zero_division=0)
                rec = recall_score(y_test, preds, zero_division=0)
                f1 = f1_score(y_test, preds, zero_division=0)

                if proba is not None:
                    # handle multiclass probabilities
                    if proba.ndim == 2 and proba.shape[1] == 2:
                        prob_pos = proba[:, 1]
                        try:
                            roc_auc = roc_auc_score(y_test, prob_pos)
                        except Exception:
                            roc_auc = None
                    else:
                        prob_pos = proba.ravel()
                        try:
                            roc_auc = roc_auc_score(y_test, prob_pos)
                        except Exception:
                            roc_auc = None
                else:
                    prob_pos = None
                    roc_auc = None

                comparison_rows.append({
                    'Model': name,
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1': f1,
                    'ROC-AUC': roc_auc,
                })

                if prob_pos is not None:
                    fpr, tpr, _ = roc_curve(y_test, prob_pos)
                    precision, recall, _ = precision_recall_curve(y_test, prob_pos)
                    roc_data[name] = (fpr, tpr)
                    pr_data[name] = (precision, recall)

                cm = confusion_matrix(y_test, preds)
                cm_data[name] = cm

            if comparison_rows:
                metrics_df = pd.DataFrame(comparison_rows).set_index('Model')
                st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)

            # Plot ROC/PR
            if roc_data or pr_data:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                plotted = False
                for name, (fpr, tpr) in roc_data.items():
                    axes[0].plot(fpr, tpr, label=name)
                    plotted = True
                if plotted:
                    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
                    axes[0].set_title('ROC Curves')
                    axes[0].set_xlabel('False Positive Rate')
                    axes[0].set_ylabel('True Positive Rate')
                    axes[0].legend()

                plotted = False
                for name, (precision, recall) in pr_data.items():
                    axes[1].plot(recall, precision, label=name)
                    plotted = True
                if plotted:
                    axes[1].set_title('Precision-Recall Curves')
                    axes[1].set_xlabel('Recall')
                    axes[1].set_ylabel('Precision')
                    axes[1].legend()

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # Confusion matrices
            if cm_data:
                fig_cm, axs = plt.subplots(1, len(cm_data), figsize=(6 * len(cm_data), 4))
                if len(cm_data) == 1:
                    axs = [axs]
                for ax, (name, cm) in zip(axs, cm_data.items()):
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                    ax.set_title(f'Confusion Matrix — {name}')
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close(fig_cm)

            # Classification reports
            for name in selected:
                wrapper = available_models.get(name)
                if wrapper is None or wrapper.requires_graph:
                    continue
                try:
                    preds = wrapper.predict(X_test)
                    from sklearn.metrics import classification_report
                    report = classification_report(y_test, preds, target_names=['INACTIVE', 'ACTIVE'], zero_division=0, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.subheader(f"Classification report — {name}")
                    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
                except Exception as exc:
                    st.warning(f"Failed classification report for {name}: {exc}")

    # Training new model (example: RandomForest) - regression pipeline as before
    if st.button("Run Bioactivity Prediction Pipeline"):
        if not features or target not in df.columns:
            st.error("Please select valid features and ensure the target column (pIC50) is available.")
        else:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, r2_score

            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            col1, col2 = st.columns(2)
            col1.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
            col2.metric("R² Score", f"{r2:.4f}")

            # Plot Actual vs Predicted
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(y_test, y_pred, alpha=0.7, color='purple')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel("Actual pIC50")
            ax.set_ylabel("Predicted pIC50")
            ax.set_title("Actual vs Predicted Bioactivity")
            st.pyplot(fig)
