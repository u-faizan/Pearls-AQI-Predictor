"""
Feature Engineering for AQI Prediction
Calculates EPA AQI and creates ML-ready features from raw data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob

# CONFIGURATION 
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# EPA AQI BREAKPOINTS
# Source: https://www.airnow.gov/aqi/aqi-calculator/
AQI_BREAKPOINTS = {
    "pm2_5": [
        (0.0, 12.0, 0, 50),       # Good
        (12.1, 35.4, 51, 100),    # Moderate
        (35.5, 55.4, 101, 150),   # Unhealthy for Sensitive Groups
        (55.5, 150.4, 151, 200),  # Unhealthy
        (150.5, 250.4, 201, 300), # Very Unhealthy
        (250.5, 500.4, 301, 500), # Hazardous
    ],
    "pm10": [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500),
    ],
}


def calculate_aqi_component(concentration, pollutant):
    """
    Calculate AQI for a single pollutant using EPA formula.
    
    Formula: AQI = [(I_high - I_low) / (C_high - C_low)] * (C - C_low) + I_low
    
    Args:
        concentration (float): Pollutant concentration
        pollutant (str): Pollutant name ('pm2_5' or 'pm10')
    
    Returns:
        float: AQI value (0-500)
    """
    if pd.isna(concentration) or pollutant not in AQI_BREAKPOINTS:
        return np.nan
    
    breakpoints = AQI_BREAKPOINTS[pollutant]
    
    for (c_low, c_high, i_low, i_high) in breakpoints:
        if c_low <= concentration <= c_high:
            # EPA AQI formula
            aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return round(aqi, 2)
    
    # If concentration exceeds all breakpoints, return max AQI
    return 500.0


def load_latest_raw_data():
    """Load the most recent CSV file from data/raw/"""
    csv_files = list(RAW_DIR.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")
    
    # Get the most recently modified file
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {latest_file.name}")
    
    return pd.read_csv(latest_file)


def engineer_features(df):
    """
    Create all features for ML model.
    
    Steps:
    1. Calculate AQI from pollutants
    2. Extract time-based features
    3. Create lag features
    4. Create rolling statistics
    5. Drop rows with NaN values
    
    Args:
        df (pd.DataFrame): Raw data
    
    Returns:
        pd.DataFrame: Processed data with features
    """
    print("\nStarting Feature Engineering...")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # ==================== 1. CALCULATE AQI ====================
    print("   Calculating EPA AQI...")
    df['aqi_pm25'] = df['pm2_5'].apply(lambda x: calculate_aqi_component(x, 'pm2_5'))
    df['aqi_pm10'] = df['pm10'].apply(lambda x: calculate_aqi_component(x, 'pm10'))
    
    # Overall AQI is the maximum of all component AQIs
    df['aqi'] = df[['aqi_pm25', 'aqi_pm10']].max(axis=1)
    
    # ==================== 2. TIME FEATURES ====================
    print("   Extracting time features...")
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df['timestamp'].dt.month
    df['day_of_month'] = df['timestamp'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # ==================== 3. LAG FEATURES ====================
    print("   Creating lag features...")
    # Previous hour's values
    df['aqi_lag_1'] = df['aqi'].shift(1)
    df['pm25_lag_1'] = df['pm2_5'].shift(1)
    df['pm10_lag_1'] = df['pm10'].shift(1)
    
    # Previous 3 hours
    df['aqi_lag_3'] = df['aqi'].shift(3)
    
    # Previous 24 hours (yesterday same time)
    df['aqi_lag_24'] = df['aqi'].shift(24)
    df['pm25_lag_24'] = df['pm2_5'].shift(24)
    
    # ==================== 4. ROLLING STATISTICS ====================
    print("   Creating rolling features...")
    # 24-hour rolling mean (smooths daily patterns)
    df['aqi_rolling_mean_24h'] = df['aqi'].rolling(window=24, min_periods=1).mean()
    df['pm25_rolling_mean_24h'] = df['pm2_5'].rolling(window=24, min_periods=1).mean()
    
    # 24-hour rolling std (measures volatility)
    df['aqi_rolling_std_24h'] = df['aqi'].rolling(window=24, min_periods=1).std()
    
    # 7-day (168 hours) rolling mean (weekly trend)
    df['aqi_rolling_mean_7d'] = df['aqi'].rolling(window=168, min_periods=1).mean()
    
    # ==================== 5. CLEAN DATA ====================
    print("   Cleaning data...")
    # Drop rows with NaN values (created by lag/rolling features)
    df_clean = df.dropna()
    
    print(f"\nFeature Engineering Complete!")
    print(f"   Original rows: {len(df)}")
    print(f"   Processed rows: {len(df_clean)}")
    print(f"   Features created: {len(df_clean.columns)}")
    
    return df_clean


def save_processed_data(df):
    """Save processed data to CSV"""
    output_file = PROCESSED_DIR / "processed_aqi.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\nSaved to: {output_file}")
    print(f"   Columns: {list(df.columns)}")
    
    return output_file


def main():
    """Main pipeline"""
    print("=" * 70)
    print("AQI FEATURE ENGINEERING PIPELINE")
    print("=" * 70)
    
    try:
        # Load raw data
        df = load_latest_raw_data()
        
        # Engineer features
        df_processed = engineer_features(df)
        
        # Save processed data
        output_file = save_processed_data(df_processed)
        
        # Show summary statistics
        print("\nAQI Summary Statistics:")
        print(df_processed['aqi'].describe())
        
        print("\n" + "=" * 70)
        print("SUCCESS! Ready for model training.")
        print("   Next step: python src/models/train.py")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    main()
