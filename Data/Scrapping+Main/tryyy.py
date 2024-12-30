import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA

# Page configuration
st.set_page_config(
    page_title="Advanced Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    div.stButton > button:first-child {
        background-color: #0099ff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📊 Analysis Controls")
    st.markdown("---")
    
    # File upload section
    st.subheader("1. Data Upload")
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    
    if uploaded_file is not None:
        # Analysis options
        st.subheader("2. Analysis Options")
        show_eda = st.checkbox("Show EDA Plots", True)
        show_correlation = st.checkbox("Show Correlation Analysis", True)
        show_outliers = st.checkbox("Show Outlier Analysis", True)
        show_modeling = st.checkbox("Show Model Evaluation", True)

# Main content
st.title("🔍 Advanced Data Analysis Dashboard")
st.markdown("---")

if uploaded_file is not None:
    # Load and display data
    with st.spinner('Loading data...'):
        data = pd.read_csv(uploaded_file)
        data = data.apply(pd.to_numeric, errors='coerce')
    # Display dataset information
    col1, col2, col3 = st.columns(3)
    with col1:
         st.metric("Total Records", len(data))
    with col2:
        st.metric("Features", data.shape[1])
    with col3:
        missing_values = data.isna().sum().sum()  # Correct missing values count
        st.metric("Missing Values", missing_values)

    # Data preview in expander
    with st.expander("📋 Data Preview", expanded=True):
        st.dataframe(data.head(), use_container_width=True)
        
        # Basic statistics
        st.markdown("### 📊 Basic Statistics")
        st.dataframe(data.describe(), use_container_width=True)

    if show_eda:
        st.markdown("## 📈 Exploratory Data Analysis")
        
        # Distribution plots in tabs
        tab1, tab2, tab3 = st.tabs(["Distribution Plots", "Correlation Analysis", "Pairplot"])
        
        with tab1:
            cols = st.columns(3)
            for i, column in enumerate(data.columns):
                with cols[i % 3]:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.histplot(data[column].dropna(), kde=True)
                    plt.title(f'Distribution of {column}')
                    st.pyplot(fig)
                    plt.close()

        with tab2:
            if show_correlation:
                fig = plt.figure(figsize=(10, 6))
                sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
                plt.title('Correlation Heatmap')
                st.pyplot(fig)
                plt.close()

        with tab3:
            st.warning("⚠️ This might take a while for large datasets")
            if st.button("Generate Pairplot"):
                with st.spinner('Generating pairplot...'):
                    sns.pairplot(data)
                    st.pyplot()
                    plt.close()

    if show_outliers:
        st.markdown("## 🔍 Outlier Analysis")
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Outliers per Feature")
            st.dataframe(pd.DataFrame({
                'Feature': outliers.index,
                'Outlier Count': outliers.values
            }))
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            outliers.plot(kind='bar')
            plt.title('Outliers by Feature')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()

    if show_modeling:
        st.markdown("## 🤖 Model Evaluation")
        
        # Preprocessing
        with st.spinner('Preprocessing data...'):
            # Handle missing values
            data_filled = data.fillna(data.mean())
            
            # Feature scaling
            scaler = StandardScaler()
            data_scaled = pd.DataFrame(scaler.fit_transform(data_filled), columns=data.columns)
            
            # Feature Engineering
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_features = poly.fit_transform(data_scaled)
            poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(data.columns))
            data_final = pd.concat([data_scaled, poly_df], axis=1)
            
            # PCA
            pca = PCA(n_components=0.95)
            data_reduced = pca.fit_transform(data_final)

        # Target selection
        target_vars = {
            'Visibility': data_scaled['VV'],
            'Temperature': data_scaled['T'],
            'Humidity': data_scaled['H']
        }

        # Model training and evaluation
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Stacking": StackingRegressor(
                estimators=[
                    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
                ],
                final_estimator=LinearRegression()
            )
        }

        # Tabs for each target variable
        target_tabs = st.tabs(list(target_vars.keys()))
        
        for target_tab, (target_name, y) in zip(target_tabs, target_vars.items()):
            with target_tab:
                X_train, X_test, y_train, y_test = train_test_split(
                    data_reduced, y, test_size=0.2, random_state=42
                )
                
                results = []
                
                for model_name, model in models.items():
                    with st.spinner(f'Training {model_name} for {target_name}...'):
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        results.append({
                            'Model': model_name,
                            'MSE': mse,
                            'R²': r2
                        })
                
                # Display results
                results_df = pd.DataFrame(results)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(results_df, use_container_width=True)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    results_df.plot(x='Model', y=['MSE', 'R²'], kind='bar', ax=ax)
                    plt.title(f'{target_name} Model Performance')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close()

else:
    # Welcome message when no file is uploaded
    st.info("👋 Welcome! Please upload a CSV file to begin the analysis.")
    
    # Example data format
    st.markdown("""
    ### Expected Data Format
    Your CSV file should contain the following columns:
    - VV (Visibility)
    - T (Temperature)
    - H (Humidity)
    - Additional features as needed
    
    ### Features
    - 📊 Comprehensive Exploratory Data Analysis
    - 🔍 Outlier Detection
    - 📈 Correlation Analysis
    - 🤖 Advanced Model Evaluation
    - 📱 Interactive Visualizations
    """)