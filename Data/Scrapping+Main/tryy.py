import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA

# Streamlit UI Components
st.title("Data Analysis and Model Evaluation")

# Upload the CSV file
uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)
    
    # Convert all columns to numeric, forcing errors to NaN (e.g., 'nan' text will be converted to actual NaN)
    data = data.apply(pd.to_numeric, errors='coerce')
    
    st.write("### Data Preview", data.head())
    
    # EDA - Plotting distribution of each feature
    st.write("### Feature Distribution")
    fig, ax = plt.subplots(3, 3, figsize=(14, 10))
    for i, column in enumerate(data.columns):
        sns.histplot(data[column].dropna(), kde=True, ax=ax[i//3, i%3])
        ax[i//3, i%3].set_title(f'Distribution of {column}')
    plt.tight_layout()
    st.pyplot(fig)

    # Correlation heatmap
    st.write("### Correlation Heatmap")
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    st.pyplot(fig)

    # Pairplot
    st.write("### Pairplot of Features")
    sns.pairplot(data)
    st.pyplot()

    # Outlier detection using IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
    st.write("### Outliers Detected Per Feature")
    st.write(outliers)

    # Preprocessing (as before)
    # Handle missing values - Fill with mean
    data_filled = data.fillna(data.mean())

    # Feature scaling (standardizing numerical features)
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data_filled), columns=data.columns)

    # Feature Engineering - Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(data_scaled)
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(data.columns))
    data_final = pd.concat([data_scaled, poly_df], axis=1)

    # Dimensionality Reduction (PCA)
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    data_reduced = pca.fit_transform(data_final)

    # Select target variables
    y_visibility = data_scaled['VV']  # Target variable: Visibility (VV)
    y_temperature = data_scaled['T']  # Target variable: Temperature (T)
    y_humidity = data_scaled['H']  # Target variable: Humidity (H)

    # Split the data into training and testing sets
    X_train, X_test, y_train_visibility, y_test_visibility = train_test_split(data_reduced, y_visibility, test_size=0.2, random_state=42)
    X_train, X_test, y_train_temperature, y_test_temperature = train_test_split(data_reduced, y_temperature, test_size=0.2, random_state=42)
    X_train, X_test, y_train_humidity, y_test_humidity = train_test_split(data_reduced, y_humidity, test_size=0.2, random_state=42)

    # Models
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
    stacking_model = StackingRegressor(estimators=[('rf', rf_reg), ('gb', gb_reg)], final_estimator=LinearRegression())

    models = [rf_reg, gb_reg, stacking_model]
    model_names = ["Random Forest", "Gradient Boosting", "Stacking"]

    # Train models and get predictions
    predictions = {'Visibility': [], 'Temperature': [], 'Humidity': []}
    
    for model, model_name in zip(models, model_names):
        model.fit(X_train, y_train_visibility)
        predictions['Visibility'].append(model.predict(X_test))

        model.fit(X_train, y_train_temperature)
        predictions['Temperature'].append(model.predict(X_test))

        model.fit(X_train, y_train_humidity)
        predictions['Humidity'].append(model.predict(X_test))

    # Model Evaluation
    def evaluate_model(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, r2

    # Show evaluation results
    def display_results(target, predictions, y_test_target):
        st.write(f"### {target} Model Evaluation")
        mse_values, r2_values = [], []
        for i, pred in enumerate(predictions):  # Iterate over predictions for each model
            mse, r2 = evaluate_model(y_test_target, pred)  # Use y_test_visibility, y_test_temperature, or y_test_humidity depending on the target
            mse_values.append(mse)
            r2_values.append(r2)
            st.write(f"{model_names[i]} - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        return mse_values, r2_values

    # Now call the display_results with the correct input
    mse_vals, r2_vals = display_results("Visibility", predictions['Visibility'], y_test_visibility)
    mse_vals_temp, r2_vals_temp = display_results("Temperature", predictions['Temperature'], y_test_temperature)
    mse_vals_hum, r2_vals_hum = display_results("Humidity", predictions['Humidity'], y_test_humidity)

    # Visualize results
    def plot_results(models, mse_values, r2_values, target):
        st.write(f"### {target} Model Comparison")
        df = pd.DataFrame({
            'Model': models,
            'MSE': mse_values,
            'R²': r2_values
        })

        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        df.set_index('Model')[['MSE', 'R²']].plot(kind='bar', ax=ax[0], color=['salmon', 'lightblue'], width=0.8)
        ax[0].set_title(f'{target} Prediction Model Comparison')
        ax[0].set_ylabel('Score')

        sns.heatmap(df.set_index('Model').T, annot=True, fmt='.4f', cmap='Blues', cbar=False, ax=ax[1])
        ax[1].set_title(f'{target} Prediction Model Performance Comparison')

        st.pyplot(fig)

    plot_results(model_names, mse_vals, r2_vals, "Visibility")
    plot_results(model_names, mse_vals_temp, r2_vals_temp, "Temperature")
    plot_results(model_names, mse_vals_hum, r2_vals_hum, "Humidity")
