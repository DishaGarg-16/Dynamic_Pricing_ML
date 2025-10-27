import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Starting model training...")

# Load data
try:
    df = pd.read_csv('synthetic_ecommerce_data.csv')
except FileNotFoundError:
    print("Error: 'synthetic_ecommerce_data.csv' not found.")
    print("Please run 'python 1_generate_data.py' first.")
    exit()

# Define features (X) and target (y)
target = 'units_sold'
# 'our_price' is the most important feature!
numerical_features = ['day_of_week', 'is_holiday', 'is_promotion', 'inventory_level', 'competitor_price', 'our_price']
categorical_features = ['product_id']

X = df[numerical_features + categorical_features]
y = df[target]

# Split the data (chronological split is better, but random split is simpler here)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Create a preprocessing pipeline ---
# This is a robust way to handle encoding and scaling
# 1. OneHotEncoder for categorical features
# 2. 'passthrough' for numerical features (XGBoost doesn't require scaling)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ],
    remainder='passthrough'
)

# --- Create the full model pipeline ---
# We bundle the preprocessor with the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    ))
])

# --- Train the model ---
print("Training the XGBoost model...")
model.fit(X_train, y_train)

# --- Evaluate the model ---
print("Evaluating model performance...")
y_pred = model.predict(X_test)

# Clip predictions at 0 (can't sell negative units)
y_pred = np.maximum(0, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"R-squared (R²): {r2:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print("------------------------\n")

# --- Save the model and preprocessor ---
# We save the *entire* pipeline, which includes the preprocessor
joblib.dump(model, 'demand_model_pipeline.pkl')

print(f"Successfully trained and saved model to 'demand_model_pipeline.pkl'")