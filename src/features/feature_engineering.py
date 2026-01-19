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
    
    # ==================== 2. TIME FEATURES (CYCLICAL) ====================
    print("   Extracting time features (Cyclical)...")
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day_of_month'] = df['timestamp'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Cyclical Encoding (Sin/Cos)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

    # ==================== 3. WEATHER FEATURES (VECTORS) ====================
    print("   Engineering weather features...")
    # Wind Vector Decomposition
    # Convert wind_direction (degrees) to radians
    # wind_direction_10m is standard meteorology (0=North, 90=East)
    wd_rad = df['wind_direction_10m'] * np.pi / 180
    df['wind_x'] = df['wind_speed_10m'] * np.sin(wd_rad)
    df['wind_y'] = df['wind_speed_10m'] * np.cos(wd_rad)

    # Interaction Terms
    # Humidity * Temperature (Dew Point proxy)
    df['temp_humidity_interaction'] = df['temperature_2m'] * df['relative_humidity_2m']
    
    # ==================== 4. LAG FEATURES ====================
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
    
    # Extra Lags for Forecasting Stability
    df['aqi_lag_48'] = df['aqi'].shift(48)
    
    # ==================== 5. ROLLING STATISTICS ====================
    print("   Creating rolling features...")
    # 24-hour rolling mean (smooths daily patterns)
    df['aqi_rolling_mean_24h'] = df['aqi'].rolling(window=24, min_periods=1).mean()
    df['pm25_rolling_mean_24h'] = df['pm2_5'].rolling(window=24, min_periods=1).mean()
    
    # 24-hour rolling std (measures volatility)
    df['aqi_rolling_std_24h'] = df['aqi'].rolling(window=24, min_periods=1).std()
    
    # 7-day (168 hours) rolling mean (weekly trend)
    df['aqi_rolling_mean_7d'] = df['aqi'].rolling(window=168, min_periods=1).mean()
    
    # ==================== 6. ADDITIONAL POLLUTANT LAGS ====================
    print("   Creating additional pollutant lag features...")
    # Nitrogen dioxide showed 0.42 correlation - add lags
    df['nitrogen_dioxide_lag_1'] = df['nitrogen_dioxide'].shift(1)
    df['nitrogen_dioxide_lag_24'] = df['nitrogen_dioxide'].shift(24)
    
    # Ozone (inverse correlation -0.27, still useful)
    df['ozone_lag_1'] = df['ozone'].shift(1)
    df['ozone_lag_24'] = df['ozone'].shift(24)
    
    # Carbon monoxide
    df['carbon_monoxide_lag_1'] = df['carbon_monoxide'].shift(1)
    
    # ==================== 7. SHORT-TERM ROLLING WINDOWS ====================
    print("   Creating short-term rolling windows...")
    # 6-hour and 12-hour windows for recent trends
    df['pm2_5_rolling_mean_6h'] = df['pm2_5'].rolling(window=6, min_periods=1).mean()
    df['pm2_5_rolling_mean_12h'] = df['pm2_5'].rolling(window=12, min_periods=1).mean()
    df['aqi_rolling_std_6h'] = df['aqi'].rolling(window=6, min_periods=1).std()
    
    # ==================== 8. POLYNOMIAL FEATURES ====================
    print("   Creating polynomial features...")
    # Squared terms for non-linear relationships
    df['pm2_5_squared'] = df['pm2_5'] ** 2
    df['pm10_squared'] = df['pm10'] ** 2
    df['temperature_squared'] = df['temperature_2m'] ** 2
    
    # ==================== 9. INTERACTION TERMS ====================
    print("   Creating interaction features...")
    # PM2.5 interactions (highest correlated pollutant)
    df['pm2_5_temp_interaction'] = df['pm2_5'] * df['temperature_2m']
    df['pm2_5_humidity_interaction'] = df['pm2_5'] * df['relative_humidity_2m']
    df['pm2_5_wind_interaction'] = df['pm2_5'] * df['wind_speed_10m']
    df['pm2_5_pressure_interaction'] = df['pm2_5'] * df['surface_pressure']
    
    # Ozone-temperature (ozone forms in heat)
    df['ozone_temp_interaction'] = df['ozone'] * df['temperature_2m']
    
    # ==================== 10. POLLUTANT RATIOS ====================
    print("   Creating pollutant ratios...")
    # Fine to coarse particle ratio
    df['pm2_5_to_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 0.1)  # +0.1 to avoid division by zero
    
    # Traffic indicator
    df['no2_to_co_ratio'] = df['nitrogen_dioxide'] / (df['carbon_monoxide'] + 0.1)
    
    # ==================== 11. DOMAIN-SPECIFIC FEATURES ====================
    print("   Creating domain-specific features...")
    # Winter months (Nov-Feb) - pollution spike in Islamabad
    df['is_winter'] = df['month'].isin([11, 12, 1, 2]).astype(int)
    
    # Rush hour (7-9 AM, 5-7 PM) - traffic pollution
    df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) | 
                          ((df['hour'] >= 17) & (df['hour'] <= 19))).astype(int)
    
    # ==================== 12. CLEAN DATA ====================
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
