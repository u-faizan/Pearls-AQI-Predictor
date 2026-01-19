"""
Upload processed AQI features to Hopsworks Feature Store
"""

import pandas as pd
import hopsworks
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "processed_aqi.csv"

def connect_to_hopsworks():
    """Connect to Hopsworks using API key from .env"""
    print("Connecting to Hopsworks...")
    
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    
    if not project_name or not api_key:
        raise ValueError("HOPSWORKS_PROJECT_NAME and HOPSWORKS_API_KEY must be set in .env file")
    
    # Login to Hopsworks
    project = hopsworks.login(
        project=project_name,
        api_key_value=api_key
    )
    
    print(f"Connected to project: {project.name}")
    return project

def create_feature_group(project, df):
    """Create and upload feature group"""
    print("\nCreating feature group...")
    
    # Get feature store
    fs = project.get_feature_store()
    
    # Define feature group
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        description="AQI and weather features for Islamabad",
        primary_key=["timestamp"],
        event_time="timestamp",
        online_enabled=False
    )
    
    # Insert data
    print(f"Uploading {len(df)} rows...")
    aqi_fg.insert(df, write_options={"wait_for_job": True})
    
    print("Feature group created successfully!")
    return aqi_fg

def main():
    """Main pipeline"""
    print("=" * 70)
    print("HOPSWORKS FEATURE UPLOAD PIPELINE")
    print("=" * 70)
    
    try:
        # Load processed data
        print(f"\nLoading data from: {PROCESSED_FILE}")
        df = pd.read_csv(PROCESSED_FILE)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Connect to Hopsworks
        project = connect_to_hopsworks()
        
        # Create feature group and upload
        feature_group = create_feature_group(project, df)
        
        print("\n" + "=" * 70)
        print("SUCCESS! Features uploaded to Hopsworks")
        print("=" * 70)
        print(f"\nFeature Group: aqi_features (version 1)")
        print(f"Rows: {len(df)}")
        print(f"Features: {len(df.columns)}")
        print("\nNext step: Create feature view and train model")
        
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("Please make sure HOPSWORKS_PROJECT_NAME and HOPSWORKS_API_KEY are set in .env")
    except Exception as e:
        print(f"\nERROR: {e}")
        raise

if __name__ == "__main__":
    main()
