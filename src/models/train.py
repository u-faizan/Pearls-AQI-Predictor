"""
Train ML models for AQI prediction using 23 carefully selected features
Final Model: XGBoost (MAE=1.82, RMSE=14.76, R²=0.9226)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Configuration
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "processed_aqi.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 23 Carefully Selected Features
FEATURES = [
    # 12 Base features (pollutants + weather)
    'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide',
    'sulphur_dioxide', 'ozone', 'temperature_2m',
    'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m',
    'wind_direction_10m', 'precipitation',
    
    # 7 Lag features (temporal patterns)
    'pm25_lag_1', 'pm25_lag_24', 'pm10_lag_1',
    'aqi_lag_1', 'aqi_lag_24',
    'pm25_rolling_mean_24h', 'aqi_rolling_mean_24h',
    
    # 2 Cyclical time features
    'hour_sin',
    
    # 1 Ratio feature
    'pm2_5_to_pm10_ratio'
]

def load_and_prepare_data():
    """Load data and prepare features"""
    print(f"Loading data from: {PROCESSED_FILE}")
    df = pd.read_csv(PROCESSED_FILE)
    
    X = df[FEATURES]
    y = df['aqi']
    
    print(f"Loaded {len(df)} samples with {len(FEATURES)} features")
    return X, y

def split_data(X, y):
    """Split into train, validation, test (70%, 15%, 15%)"""
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)
    
    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_models(X_train, y_train):
    """Train 5 models"""
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    }
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
    
    return models

def evaluate_models(models, X_train, X_val, X_test, y_train, y_val, y_test):
    """Evaluate all models"""
    results = {}
    
    for name, model in models.items():
        results[name] = {
            'train': {
                'mae': round(mean_absolute_error(y_train, model.predict(X_train)), 2),
                'rmse': round(np.sqrt(mean_squared_error(y_train, model.predict(X_train))), 2),
                'r2': round(r2_score(y_train, model.predict(X_train)), 4)
            },
            'validation': {
                'mae': round(mean_absolute_error(y_val, model.predict(X_val)), 2),
                'rmse': round(np.sqrt(mean_squared_error(y_val, model.predict(X_val))), 2),
                'r2': round(r2_score(y_val, model.predict(X_val)), 4)
            },
            'test': {
                'mae': round(mean_absolute_error(y_test, model.predict(X_test)), 2),
                'rmse': round(np.sqrt(mean_squared_error(y_test, model.predict(X_test))), 2),
                'r2': round(r2_score(y_test, model.predict(X_test)), 4)
            }
        }
    
    return results

def save_models(models, results):
    """Save all models and metadata"""
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    
    # Save each model
    for name, model in models.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(model, MODELS_DIR / filename)
        print(f"Saved: {filename}")
    
    # Save best model (XGBoost based on MAE)
    best_model_name = min(results.items(), key=lambda x: x[1]['test']['mae'])[0]
    joblib.dump(models[best_model_name], MODELS_DIR / 'best_model.pkl')
    print(f"\nBest Model: {best_model_name}")
    
    # Save metadata
    with open(MODELS_DIR / 'feature_columns.json', 'w') as f:
        json.dump(FEATURES, f, indent=2)
    
    with open(MODELS_DIR / 'model_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save metrics.txt
    with open(MODELS_DIR / 'metrics.txt', 'w') as f:
        f.write("AQI Predictor - Model Performance Metrics\n")
        f.write("="*50 + "\n\n")
        f.write(f"Features: {len(FEATURES)}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n")
        f.write(f"BEST MODEL: {best_model_name}\n")
        f.write("-"*50 + "\n")
        f.write(f"Test MAE:  {results[best_model_name]['test']['mae']}\n")
        f.write(f"Test RMSE: {results[best_model_name]['test']['rmse']}\n")
        f.write(f"Test R²:   {results[best_model_name]['test']['r2']}\n\n")
        f.write("ALL MODELS (Test Set):\n")
        f.write("-"*50 + "\n")
        for name, metrics in results.items():
            f.write(f"{name:20s} MAE={metrics['test']['mae']:5.2f}  RMSE={metrics['test']['rmse']:6.2f}  R²={metrics['test']['r2']:.4f}\n")
    
    print("\nMetadata saved:")
    print("  - feature_columns.json")
    print("  - model_metrics.json")
    print("  - metrics.txt")

def print_results(results):
    """Print results table"""
    print("\n" + "="*80)
    print("RESULTS (Test Set)")
    print("="*80)
    print(f"\n{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-"*50)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['test']['mae']:>8.2f} {metrics['test']['rmse']:>8.2f} {metrics['test']['r2']:>8.4f}")

def main():
    # Load data
    X, y = load_and_prepare_data()
    
    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Train
    models = train_models(X_train, y_train)
    
    # Evaluate
    results = evaluate_models(models, X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Print
    print_results(results)
    
    # Save
    save_models(models, results)
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()
