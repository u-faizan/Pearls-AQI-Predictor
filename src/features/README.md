# Feature Engineering Experiments

This document tracks all feature engineering experiments for the AQI Predictor project.

## Experiment Log

### Experiment 1: Baseline Features

**Status**: ✅ Completed

**Features Created**:
- EPA AQI calculation (PM2.5, PM10)
- Time features: hour, day_of_week, month, day_of_month, is_weekend
- Lag features: aqi_lag_1, aqi_lag_3, aqi_lag_24, pm25_lag_1, pm25_lag_24, pm10_lag_1
- Rolling statistics: 24h mean/std, 7-day mean for AQI and PM2.5

**Results**:
- Total features: 34
- Data points: 7,416 (from 7,440 raw)
- Best model: LightGBM
- Test R²: 0.4963
- Test RMSE: 37.30
- Test MAE: 13.79

**Notes**:
Baseline feature set provides ~50% variance explanation. Tree-based models (Random Forest, XGBoost, LightGBM) significantly outperform linear models. LightGBM shows best generalization with least overfitting.

---

### Experiment 2: Advanced Features

**Status**: ✅ Completed

**Features Added** (20 new):
- Additional pollutant lags: `nitrogen_dioxide_lag_1`, `nitrogen_dioxide_lag_24`, `ozone_lag_1`, `ozone_lag_24`, `carbon_monoxide_lag_1`
- Short-term rolling windows: `pm2_5_rolling_mean_6h`, `pm2_5_rolling_mean_12h`, `aqi_rolling_std_6h`
- Polynomial features: `pm2_5_squared`, `pm10_squared`, `temperature_squared`
- Interaction terms: `pm2_5_temp_interaction`, `pm2_5_humidity_interaction`, `pm2_5_wind_interaction`, `pm2_5_pressure_interaction`, `ozone_temp_interaction`
- Pollutant ratios: `pm2_5_to_pm10_ratio`, `no2_to_co_ratio`
- Domain features: `is_winter`, `is_rush_hour`

**Total Features**: 64 (up from 44)
**Data Points**: 7,392

**Model Performance**:

| Model | Split | MAE | RMSE | R² |
|-------|-------|-----|------|----|
| **Linear Regression** | Train | 11.45 | 34.21 | 0.6161 |
| | Validation | 10.90 | 31.78 | 0.6120 |
| | Test | 12.03 | 34.36 | 0.5725 |
| **Ridge Regression** | Train | 11.43 | 34.21 | 0.6161 |
| | Validation | 10.88 | 31.77 | 0.6122 |
| | Test | 12.00 | 34.36 | 0.5725 |
| **Random Forest** | Train | 1.28 | 9.35 | 0.9713 |
| | Validation | 2.71 | 19.56 | 0.8530 |
| | Test | 3.24 | 20.68 | 0.8452 |
| **XGBoost** | Train | 0.16 | 0.26 | 1.0000 |
| | Validation | 0.92 | 10.32 | 0.9591 |
| | Test | 0.99 | 11.68 | 0.9506 |
| **LightGBM** | Train | 1.26 | 5.74 | 0.9892 |
| | Validation | 2.35 | 14.03 | 0.9244 |
| | Test | 2.09 | 9.42 | 0.9679 |

**Comparison with Baseline**:

| Model | Baseline Test R² | Exp 2 Test R² | Change |
|-------|------------------|---------------|--------|
| Linear Regression | 0.3741 | 0.5725 | +0.1984 |
| Ridge Regression | 0.3741 | 0.5725 | +0.1984 |
| Random Forest | 0.4621 | 0.8452 | +0.3831 |
| XGBoost | 0.4627 | 0.9506 | +0.4879 |
| LightGBM | 0.4963 | 0.9679 | +0.4716 |

---

## Feature Engineering Guidelines

### Adding New Features
1. Create new version in `feature_engineering.py`
2. Save processed data with version suffix (e.g., `processed_aqi_v2.csv`)
3. Train models and compare metrics
4. Update this README with results
5. Keep best version for production

### Evaluation Criteria
- R² score improvement
- RMSE/MAE reduction
- Model generalization (train vs test gap)
- Feature importance analysis

### File Naming Convention
- `processed_aqi.csv` - Current production features
- `processed_aqi_v2.csv` - Experiment 2 features
- `processed_aqi_v3.csv` - Experiment 3 features

---
