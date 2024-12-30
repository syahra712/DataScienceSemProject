import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

# Load the data
data = pd.read_csv("/Users/admin/Desktop/Data/Real-Data/Real_Combine.csv")

# Convert all columns to numeric, forcing errors to NaN (e.g., 'nan' text will be converted to actual NaN)
data = data.apply(pd.to_numeric, errors='coerce')

# EDA - Plotting distribution of each feature
plt.figure(figsize=(14, 10))

# Plot histogram for each feature
for i, column in enumerate(data.columns):
    plt.subplot(3, 3, i+1)
    sns.histplot(data[column].dropna(), kde=True)
    plt.title(f'Distribution of {column}')

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot to check for relationships between features
sns.pairplot(data)
plt.show()

# Outlier detection using IQR
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
print("Outliers detected per feature:")
print(outliers)

# Preprocessing
# Handle missing values - Fill with mean
data_filled = data.fillna(data.mean())

# Feature scaling (standardizing numerical features)
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data_filled), columns=data.columns)

# Feature Engineering - Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(data_scaled)

# Add polynomial features to the dataframe
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(data.columns))
data_final = pd.concat([data_scaled, poly_df], axis=1)

# Dimensionality Reduction (PCA)
pca = PCA(n_components=0.95)  # Keep 95% of variance
data_reduced = pca.fit_transform(data_final)

# Model Selection
# Select relevant features related to pollutants and meteorological variables
X = data_reduced  # Including all features after PCA reduction
y_visibility = data_scaled['VV']  # Target variable: Visibility (VV)
y_temperature = data_scaled['T']  # Target variable: Temperature (T)
y_humidity = data_scaled['H']  # Target variable: Humidity (H)

# Split the data into training and testing sets
X_train, X_test, y_train_visibility, y_test_visibility = train_test_split(X, y_visibility, test_size=0.2, random_state=42)
X_train, X_test, y_train_temperature, y_test_temperature = train_test_split(X, y_temperature, test_size=0.2, random_state=42)
X_train, X_test, y_train_humidity, y_test_humidity = train_test_split(X, y_humidity, test_size=0.2, random_state=42)

# Models
# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train_visibility)
y_pred_visibility_rf = rf_reg.predict(X_test)

rf_reg.fit(X_train, y_train_temperature)
y_pred_temperature_rf = rf_reg.predict(X_test)

rf_reg.fit(X_train, y_train_humidity)
y_pred_humidity_rf = rf_reg.predict(X_test)

# Gradient Boosting Regressor
gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_reg.fit(X_train, y_train_visibility)
y_pred_visibility_gb = gb_reg.predict(X_test)

gb_reg.fit(X_train, y_train_temperature)
y_pred_temperature_gb = gb_reg.predict(X_test)

gb_reg.fit(X_train, y_train_humidity)
y_pred_humidity_gb = gb_reg.predict(X_test)

# Stacking Model (Combining multiple models)
stacking_model = StackingRegressor(estimators=[('rf', rf_reg), ('gb', gb_reg)], final_estimator=LinearRegression())
stacking_model.fit(X_train, y_train_visibility)
y_pred_visibility_stack = stacking_model.predict(X_test)

stacking_model.fit(X_train, y_train_temperature)
y_pred_temperature_stack = stacking_model.predict(X_test)

stacking_model.fit(X_train, y_train_humidity)
y_pred_humidity_stack = stacking_model.predict(X_test)

# Model Evaluation Function (Modified for improved output)
def evaluate_model(y_true, y_pred, model_name, target_variable):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} - {target_variable} Prediction Performance")
    print(f"-------------------------------")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    return mse, r2

# Visualization of Evaluation Results (Updated with Visualizations)
def visualize_comparison(models, mse_values, r2_values, target_variable):
    # Create a figure for model comparison
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Create a DataFrame for better visualization
    model_comparison_df = pd.DataFrame({
        'Model': models,
        'MSE': mse_values,
        'R²': r2_values
    })

    # Plotting MSE and R² as bar charts
    model_comparison_df.set_index('Model')[['MSE', 'R²']].plot(kind='bar', ax=ax[0], color=['salmon', 'lightblue'], width=0.8)
    ax[0].set_title(f'{target_variable} Prediction Model Comparison: MSE and R²', fontsize=16)
    ax[0].set_ylabel('Score')
    ax[0].set_xlabel('Models')

    # Add text annotations to bars
    for p in ax[0].patches:
        ax[0].annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                       xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=10)

    # Model Evaluation Table (Heatmap)
    sns.set(style="whitegrid")
    table = sns.heatmap(model_comparison_df.set_index('Model').T, annot=True, fmt='.4f', cmap='Blues', cbar=False, ax=ax[1])
    table.set_title(f'{target_variable} Prediction Model Performance Comparison', fontsize=16)

    plt.tight_layout()
    plt.show()

# Perform evaluation for Visibility, Temperature, and Humidity and plot results
models = ['Random Forest', 'Gradient Boosting', 'Stacking']
predictions_visibility = [y_pred_visibility_rf, y_pred_visibility_gb, y_pred_visibility_stack]
predictions_temperature = [y_pred_temperature_rf, y_pred_temperature_gb, y_pred_temperature_stack]
predictions_humidity = [y_pred_humidity_rf, y_pred_humidity_gb, y_pred_humidity_stack]

# Evaluate and visualize for each target variable (Visibility, Temperature, Humidity)
mse_values_visibility = []
r2_values_visibility = []
for i, model_name in enumerate(models):
    mse, r2 = evaluate_model(y_test_visibility, predictions_visibility[i], f"{model_name}", "Visibility")
    mse_values_visibility.append(mse)
    r2_values_visibility.append(r2)

mse_values_temperature = []
r2_values_temperature = []
for i, model_name in enumerate(models):
    mse, r2 = evaluate_model(y_test_temperature, predictions_temperature[i], f"{model_name}", "Temperature")
    mse_values_temperature.append(mse)
    r2_values_temperature.append(r2)

mse_values_humidity = []
r2_values_humidity = []
for i, model_name in enumerate(models):
    mse, r2 = evaluate_model(y_test_humidity, predictions_humidity[i], f"{model_name}", "Humidity")
    mse_values_humidity.append(mse)
    r2_values_humidity.append(r2)

# Visualize the comparison for all target variables
visualize_comparison(models, mse_values_visibility, r2_values_visibility, "Visibility")
visualize_comparison(models, mse_values_temperature, r2_values_temperature, "Temperature")
visualize_comparison(models, mse_values_humidity, r2_values_humidity, "Humidity")
