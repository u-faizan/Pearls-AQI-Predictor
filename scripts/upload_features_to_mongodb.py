"""
Upload existing processed features to MongoDB
Run this once to migrate from CSV to MongoDB
"""

import pandas as pd
import os
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(parent_dir))

from src.database.feature_store import FeatureStore

def main():
    print("="*80)
    print("UPLOADING FEATURES TO MONGODB")
    print("="*80)
    
    # Load processed features
    csv_path = parent_dir / "data" / "processed" / "processed_aqi.csv"
    print(f"\nLoading features from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Initialize feature store
    feature_store = FeatureStore()
    
    # Save to MongoDB
    print("\nUploading to MongoDB...")
    feature_store.save_features(df)
    
    # Verify
    count = feature_store.count_records()
    print(f"\n✅ Total records in MongoDB: {count}")
    
    print("\n" + "="*80)
    print("UPLOAD COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
