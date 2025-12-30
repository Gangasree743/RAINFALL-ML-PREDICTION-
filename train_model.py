"""
Train and save the rainfall prediction models
Run this script once to generate RF_model.pkl and scaler.pkl
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

print("🌧️ Starting Model Training...\n")

# Load Dataset
print("📂 Loading dataset...")
df = pd.read_csv("weatherAUS.csv")
print(f"✓ Dataset loaded: {df.shape}")

# Data Preprocessing
print("\n🔧 Preprocessing data...")
df = df.loc[:, ~df.columns.str.contains("Unnamed")]
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode Categorical Columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

print("✓ Data preprocessing complete")

# Feature & Target Separation
print("\n📊 Separating features and target...")
X = df.drop("Rainfall", axis=1)
y = df["Rainfall"]
print(f"✓ Features shape: {X.shape}, Target shape: {y.shape}")

# Feature Selection
print("\n🎯 Selecting top 5 features...")
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()
print(f"✓ Selected features: {selected_features}")

# Feature Scaling
print("\n📐 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
print("✓ Features scaled")

# Train-Test Split
print("\n✂️ Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"✓ Training set: {X_train.shape}, Test set: {X_test.shape}")

# Train Random Forest Model
print("\n🌲 Training Random Forest Model...")
rf_model = RandomForestRegressor(n_estimators=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Evaluate Model
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = round(r2 * 100, 2)

print(f"\n📈 Model Performance:")
print(f"   • MAE: {mae:.4f}")
print(f"   • MSE: {mse:.4f}")
print(f"   • R² Score: {r2:.4f}")
print(f"   • Accuracy: {accuracy}%")

# Save Models
print("\n💾 Saving models...")
joblib.dump(rf_model, 'RF_model.pkl')
print("✓ RF_model.pkl saved")

joblib.dump(scaler, 'scaler.pkl')
print("✓ scaler.pkl saved")

print("\n✅ Training Complete! Models are ready for deployment.")
print("You can now run: streamlit run app.py")
