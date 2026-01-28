"""
Upload trained model to MongoDB Model Registry
Run this once to migrate best model to MongoDB
"""

import joblib
import json
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(parent_dir))

from src.database.model_registry import ModelRegistry

def main():
    print("="*80)
    print("UPLOADING MODEL TO MONGODB")
    print("="*80)
    
    # Paths
    model_path = parent_dir / "models" / "best_model.pkl"
    metrics_path = parent_dir / "models" / "model_metrics.json"
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = joblib.load(model_path)
    print("✅ Model loaded")
    
    # Load metrics
    print(f"\nLoading metrics from: {metrics_path}")
    with open(metrics_path, 'r') as f:
        all_metrics = json.load(f)
    
    # Get XGBoost metrics (our best model)
    xgb_metrics = all_metrics['XGBoost']['test']
    print(f"Test MAE:  {xgb_metrics['mae']}")
    print(f"Test RMSE: {xgb_metrics['rmse']}")
    print(f"Test R²:   {xgb_metrics['r2']}")
    
    # Initialize model registry
    registry = ModelRegistry()
    
    # Save model
    print("\nUploading to MongoDB...")
    model_id = registry.save_model(
        model=model,
        model_name="XGBoost",
        metrics=xgb_metrics,
        version="v1.0_production"
    )
    
    # Set as active
    print("\nSetting as active model...")
    registry.set_active_model(model_id)
    
    print("\n" + "="*80)
    print("UPLOAD COMPLETE!")
    print(f"Model ID: {model_id}")
    print("="*80)

if __name__ == "__main__":
    main()
