"""
Train 5 ML models for AQI prediction
Compare Linear, Ridge, Random Forest, XGBoost, and LightGBM
Save all models and log metadata to MongoDB
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "processed_aqi.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def connect_to_mongodb():
    """Connect to MongoDB Atlas"""
    try:
        mongo_uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
        collection_name = os.getenv("MONGODB_COLLECTION", "model_metadata")
        
        if not mongo_uri:
            print("Warning: MONGODB_URI not found in .env")
            return None
        
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        # Test connection
        client.server_info()
        print(f"Connected to MongoDB: {db_name}.{collection_name}")
        
        return collection
    except Exception as e:
        print(f"Warning: Could not connect to MongoDB: {e}")
        return None

def load_data():
    """Load processed data"""
    print(f"Loading data from: {PROCESSED_FILE}")
    df = pd.read_csv(PROCESSED_FILE)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def prepare_features(df):
    """Prepare features and target"""
    target = 'aqi'
    
    # STRICT FEATURE SELECTION FOR FORECASTING (No future data leakage)
    exclude_cols = [
        'timestamp', 'aqi', 'aqi_pm25', 'aqi_pm10', 'city', 'latitude', 'longitude',
        'pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone',
        'dust', 'uv_index', 'ammonia', 'uv_index_clear_sky',
        # Exclude pollutant-specific lags because we can't forecast them recursively easily
        'pm25_lag_1', 'pm10_lag_1', 'pm25_lag_24', 'pm25_rolling_mean_24h'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target]
    
    print(f"\nFeatures selected ({len(feature_cols)}):")
    print(feature_cols)
    
    return X, y, feature_cols

def split_data(X, y):
    """Split data into train, validation, and test sets"""
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Second split: 75% train, 25% val (of the 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_and_tune_models(X_train, y_train):
    """Train 5 models with basic hyperparameter tuning"""
    
    models = {}
    
    print("\n" + "="*30)
    print("TRAINING 5 MODELS")
    print("="*30)

    # 1. Linear Regression (Baseline)
    print("\n1. Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = lr
    
    # 2. Ridge Regression
    print("2. Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    models['Ridge Regression'] = ridge
    
    # 3. Random Forest (Manual Tuning)
    print("3. Training Random Forest (n_estimators=200, max_depth=20)...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # 4. XGBoost
    print("4. Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    
    # 5. LightGBM
    print("5. Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    models['LightGBM'] = lgb_model
    
    return models

def evaluate_models(models, X, y, dataset_name):
    """Evaluate all models"""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        results[name] = {
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'r2': round(r2, 4)
        }
    return results

def save_all_models(models, best_model_name, scaler, feature_cols, all_metrics):
    """Save ALL models locally and return paths"""
    
    saved_paths = {}
    
    # Save Scaler and Features first
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    with open(MODELS_DIR / "feature_columns.json", 'w') as f:
        json.dump(feature_cols, f, indent=2)
    with open(MODELS_DIR / "model_metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)
        
    print("\n" + "="*30)
    print("SAVING MODELS")
    print("="*30)
    
    for name, model in models.items():
        # Create safe filename
        safe_name = name.lower().replace(" ", "_")
        filename = f"{safe_name}.pkl"
        path = MODELS_DIR / filename
        
        joblib.dump(model, path)
        saved_paths[name] = str(path)
        print(f"Saved: {filename}")
        
    # Create valid symlink/copy for 'best_model.pkl'
    best_safe_name = best_model_name.lower().replace(" ", "_")
    best_path = MODELS_DIR / f"{best_safe_name}.pkl"
    joblib.dump(models[best_model_name], MODELS_DIR / "best_model.pkl")
    print(f"Saved Best Model ({best_model_name}) as: best_model.pkl")
    
    return saved_paths

def save_metadata_to_mongodb(collection, all_metrics, saved_paths, feature_cols):
    """Save metadata for ALL models to MongoDB"""
    if collection is None:
        return
    
    experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Find best model
    best_model = max(all_metrics.items(), key=lambda x: x[1]['test']['r2'])[0]
    
    metadata = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'best_model': best_model,
        'num_features': len(feature_cols),
        'feature_names': feature_cols,
        'models': []
    }
    
    for name, metrics in all_metrics.items():
        model_meta = {
            'name': name,
            'metrics': metrics,
            'path': saved_paths.get(name)
        }
        metadata['models'].append(model_meta)
        
    try:
        result = collection.insert_one(metadata)
        print(f"\nSaved Experiment Metadata to MongoDB (ID: {result.inserted_id})")
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")

def main():
    try:
        mongo_collection = connect_to_mongodb()
        df = load_data()
        X, y, feature_cols = prepare_features(df)
        
        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Train 5 Models
        models = train_and_tune_models(X_train_scaled, y_train)
        
        # Evaluate
        print("\n" + "="*30)
        print("EVALUATION RESULTS (TEST SET)")
        print("="*30)
        
        all_metrics = {}
        for name, model in models.items():
            train_metrics = evaluate_models({name: model}, X_train_scaled, y_train, 'train')[name]
            val_metrics = evaluate_models({name: model}, X_val_scaled, y_val, 'validation')[name]
            test_metrics = evaluate_models({name: model}, X_test_scaled, y_test, 'test')[name]
            
            all_metrics[name] = {
                'train': train_metrics,
                'validation': val_metrics,
                'test': test_metrics
            }
            
            print(f"\n{name}:")
            print(f"  RMSE: {test_metrics['rmse']}")
            print(f"  R2:   {test_metrics['r2']}")
            print(f"  MAE:  {test_metrics['mae']}")

        # Pick Best
        best_model_name = max(all_metrics.items(), key=lambda x: x[1]['test']['r2'])[0]
        print(f"\n🏆 WINNER: {best_model_name}")
        
        # Save All
        saved_paths = save_all_models(models, best_model_name, scaler, feature_cols, all_metrics)
        
        # Log to DB
        save_metadata_to_mongodb(mongo_collection, all_metrics, saved_paths, feature_cols)
        
        print("\nDone! 5 Models Trained & Compared.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        raise

if __name__ == "__main__":
    main()
