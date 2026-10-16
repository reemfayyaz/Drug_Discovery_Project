import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Config setup
st.set_page_config(
    page_title="Virtual Screening & Drug Discovery Pipeline",
    page_icon="🧪",
    layout="wide"
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
    # Classification based on pIC50
    df['Bioactivity_Class'] = pd.cut(df['pIC50'], bins=[0, 5, 6, 10], labels=['inactive', 'intermediate', 'active'])
    return df

# Data Upload or Load Sample
st.sidebar.subheader("Dataset Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_sample_data()
    st.sidebar.info("Using synthetic dataset. Upload a CSV to use your own data.")

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
    st.write("Train a machine learning model to predict bioactivity (pIC50).")

    features = st.multiselect("Select Feature Columns for Modeling:", 
                                options=[c for c in df.columns if c not in ['Molecule_ID', 'pIC50', 'Bioactivity_Class', 'Lipinski_Pass']],
                                default=['MW', 'LogP', 'NumHDonors', 'NumHAcceptors'])

    target = 'pIC50'

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
