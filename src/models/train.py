"""
Train ML models for AQI prediction
Save best model locally and store metadata in MongoDB
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
            raise ValueError("MONGODB_URI not found in .env file")
        
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        # Test connection
        client.server_info()
        print(f"Connected to MongoDB: {db_name}.{collection_name}")
        
        return collection
    except Exception as e:
        print(f"Warning: Could not connect to MongoDB: {e}")
        print("Model will be saved locally only")
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
    
    # Features to exclude
    exclude_cols = ['timestamp', 'aqi', 'aqi_pm25', 'aqi_pm10', 'city', 'latitude', 'longitude']
    
    # Select feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target]
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Target: {target}")
    
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

def scale_features(X_train, X_val, X_test):
    """Scale features using StandardScaler"""
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nFeatures scaled")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def train_models(X_train, y_train):
    """Train multiple models"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
    }
    
    trained_models = {}
    
    print("\nTraining models...")
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    print("All models trained")
    
    return trained_models

def evaluate_model(model, X, y, dataset_name):
    """Evaluate a single model"""
    y_pred = model.predict(X)
    
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    return {
        'dataset': dataset_name,
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'r2': round(r2, 4)
    }

def evaluate_all_models(models, X_train, X_val, X_test, y_train, y_val, y_test):
    """Evaluate all models on train, val, and test sets"""
    results = {}
    
    print("\nEvaluating models...")
    for name, model in models.items():
        print(f"\n{name}:")
        
        train_metrics = evaluate_model(model, X_train, y_train, 'train')
        val_metrics = evaluate_model(model, X_val, y_val, 'validation')
        test_metrics = evaluate_model(model, X_test, y_test, 'test')
        
        results[name] = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
        
        print(f"  Train - MAE: {train_metrics['mae']}, RMSE: {train_metrics['rmse']}, R2: {train_metrics['r2']}")
        print(f"  Val   - MAE: {val_metrics['mae']}, RMSE: {val_metrics['rmse']}, R2: {val_metrics['r2']}")
        print(f"  Test  - MAE: {test_metrics['mae']}, RMSE: {test_metrics['rmse']}, R2: {test_metrics['r2']}")
    
    return results

def select_best_model(results):
    """Select best model based on validation RMSE"""
    best_model_name = None
    best_rmse = float('inf')
    
    for name, metrics in results.items():
        val_rmse = metrics['validation']['rmse']
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} (Validation RMSE: {best_rmse})")
    return best_model_name

def save_model_locally(model, scaler, feature_cols, results, best_model_name):
    """Save model, scaler, and results locally"""
    # Save best model
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nSaved model to: {model_path}")
    
    # Save scaler
    scaler_path = MODELS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to: {scaler_path}")
    
    # Save feature columns
    features_path = MODELS_DIR / "feature_columns.json"
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Saved feature columns to: {features_path}")
    
    # Save results
    results_path = MODELS_DIR / "model_metrics.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to: {results_path}")
    
    return {
        'model_path': str(model_path),
        'scaler_path': str(scaler_path),
        'features_path': str(features_path),
        'results_path': str(results_path)
    }

def save_metadata_to_mongodb(collection, best_model_name, results, feature_cols, file_paths):
    """Save model metadata to MongoDB"""
    if collection is None:
        print("\nSkipping MongoDB save (not connected)")
        return None
    
    try:
        metadata = {
            'model_name': best_model_name,
            'model_version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'training_date': datetime.now().isoformat(),
            'metrics': results[best_model_name],
            'num_features': len(feature_cols),
            'feature_names': feature_cols,
            'file_paths': file_paths,
            'status': 'production'
        }
        
        # Insert into MongoDB
        result = collection.insert_one(metadata)
        print(f"\nSaved metadata to MongoDB (ID: {result.inserted_id})")
        
        return result.inserted_id
    except Exception as e:
        print(f"\nError saving to MongoDB: {e}")
        return None

def main():
    """Main training pipeline"""
    print("=" * 70)
    print("AQI MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    try:
        # Connect to MongoDB
        mongo_collection = connect_to_mongodb()
        
        # Load data
        df = load_data()
        
        # Prepare features
        X, y, feature_cols = prepare_features(df)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            X_train, X_val, X_test
        )
        
        # Train models
        models = train_models(X_train_scaled, y_train)
        
        # Evaluate models
        results = evaluate_all_models(
            models, 
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test
        )
        
        # Select best model
        best_model_name = select_best_model(results)
        best_model = models[best_model_name]
        
        # Save model locally
        file_paths = save_model_locally(
            best_model, scaler, feature_cols, results, best_model_name
        )
        
        # Save metadata to MongoDB
        save_metadata_to_mongodb(
            mongo_collection, best_model_name, results, feature_cols, file_paths
        )
        
        print("\n" + "=" * 70)
        print("SUCCESS! Model training complete")
        print("=" * 70)
        print(f"\nBest Model: {best_model_name}")
        print(f"Test RMSE: {results[best_model_name]['test']['rmse']}")
        print(f"Test R2: {results[best_model_name]['test']['r2']}")
        print(f"Test MAE: {results[best_model_name]['test']['mae']}")
        print("\nModel saved locally in models/ directory")
        print("Metadata saved to MongoDB (if connected)")
        print("\nNext step: Create prediction API or dashboard")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        raise

if __name__ == "__main__":
    main()
