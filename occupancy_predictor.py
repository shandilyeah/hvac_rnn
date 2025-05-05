import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from datetime import datetime
import xgboost as xgb
from collections import Counter
import joblib
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load the data
df = pd.read_csv('data.csv')

# Convert timestamp to datetime and extract features
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Handle missing values
# For Fake Count, since most values are missing, we'll drop this column
df = df.drop('Fake Count', axis=1)

# For other columns, we'll use forward fill for missing values
df = df.fillna(method='ffill')

# Prepare features and target for traditional models
features = ['People Count', 'RGB', 'Pico']
X = df[features]
y = df['Target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'models/scaler.joblib')

# Traditional ML models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)

# Train traditional models
rf_model.fit(X_train_scaled, y_train)
gb_model.fit(X_train_scaled, y_train)
xgb_model.fit(X_train_scaled, y_train)

# Create ensemble
ensemble = VotingRegressor([
    ('rf', rf_model),
    ('gb', gb_model),
    ('xgb', xgb_model)
])
ensemble.fit(X_train_scaled, y_train)

# Save all models
joblib.dump(rf_model, 'models/random_forest.joblib')
joblib.dump(gb_model, 'models/gradient_boosting.joblib')
joblib.dump(xgb_model, 'models/xgboost.joblib')
joblib.dump(ensemble, 'models/ensemble.joblib')

# Make predictions and round to nearest integer
y_pred_rf = np.round(rf_model.predict(X_test_scaled)).astype(int)
y_pred_gb = np.round(gb_model.predict(X_test_scaled)).astype(int)
y_pred_xgb = np.round(xgb_model.predict(X_test_scaled)).astype(int)
y_pred_ensemble = np.round(ensemble.predict(X_test_scaled)).astype(int)

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{model_name} Performance:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared Score: {r2:.2f}")
    print(f"Accuracy (exact match): {accuracy:.2%}")
    
    # Print confusion matrix
    confusion_matrix = pd.crosstab(y_true, y_pred, 
                                 rownames=['Actual'], 
                                 colnames=['Predicted'],
                                 margins=True)
    print(f"\nConfusion Matrix:")
    print(confusion_matrix)

# Evaluate all models
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_gb, "Gradient Boosting")
evaluate_model(y_test, y_pred_xgb, "XGBoost")
evaluate_model(y_test, y_pred_ensemble, "Ensemble")

# Feature importance for all tree-based models
print("\nFeature Importance:")
print("\nRandom Forest:")
feature_importance_rf = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance_rf)

print("\nGradient Boosting:")
feature_importance_gb = pd.DataFrame({
    'feature': features,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance_gb)

print("\nXGBoost:")
feature_importance_xgb = pd.DataFrame({
    'feature': features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance_xgb)

# Function to create scatter plot with size based on frequency
def create_scatter_plot(ax, y_true, y_pred, title):
    # Count frequency of each (actual, predicted) pair
    counts = Counter(zip(y_true, y_pred))
    
    # Create arrays for plotting
    x = []
    y = []
    sizes = []
    for (actual, predicted), count in counts.items():
        x.append(actual)
        y.append(predicted)
        sizes.append(count * 100)  # Scale the size for better visibility
    
    # Create scatter plot
    scatter = ax.scatter(x, y, s=sizes, alpha=0.5)
    
    # Add diagonal line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Set labels and title
    ax.set_xlabel('Actual Occupancy')
    ax.set_ylabel('Predicted Occupancy')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Frequency')

# Plot actual vs predicted for all models
plt.figure(figsize=(20, 10))

# Create subplots
ax1 = plt.subplot(2, 2, 1)
create_scatter_plot(ax1, y_test, y_pred_rf, 'Random Forest: Actual vs Predicted')

ax2 = plt.subplot(2, 2, 2)
create_scatter_plot(ax2, y_test, y_pred_gb, 'Gradient Boosting: Actual vs Predicted')

ax3 = plt.subplot(2, 2, 3)
create_scatter_plot(ax3, y_test, y_pred_xgb, 'XGBoost: Actual vs Predicted')

ax4 = plt.subplot(2, 2, 4)
create_scatter_plot(ax4, y_test, y_pred_ensemble, 'Ensemble: Actual vs Predicted')

plt.tight_layout()
plt.savefig('occupancy_prediction_comparison.png', dpi=300, bbox_inches='tight')
plt.close() 