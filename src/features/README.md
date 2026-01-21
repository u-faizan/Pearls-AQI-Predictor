# Feature Engineering

Final optimized feature set for AQI prediction.

## Final Model: 23 Features

**Status**: ✅ Production Ready

### Feature Categories:

**12 Base Features** (Pollutants + Weather):
- pm10, pm2_5, carbon_monoxide, nitrogen_dioxide, sulphur_dioxide, ozone
- temperature_2m, relative_humidity_2m, surface_pressure
- wind_speed_10m, wind_direction_10m, precipitation

**7 Lag Features** (Temporal Patterns):
- pm25_lag_1, pm25_lag_24, pm10_lag_1
- aqi_lag_1, aqi_lag_24
- pm25_rolling_mean_24h, aqi_rolling_mean_24h

**4 Engineered Features**:
- hour_sin (cyclical time)
- pm2_5_to_pm10_ratio (pollutant ratio)

### Performance (Test Set):

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 10.93 | 38.31 | 0.4788 |
| Ridge Regression | 10.93 | 38.31 | 0.4788 |
| Random Forest | 3.38 | 21.93 | 0.8292 |
| **XGBoost** ✅ | **1.82** | **14.76** | **0.9226** |
| LightGBM | 3.67 | 19.55 | 0.8643 |

**Winner**: XGBoost
- Excellent MAE (1.82 AQI points average error)
- Strong R² (92% variance explained)
- Balanced performance across all metrics

## Why 23 Features?

**Simplicity**: Fewer features = faster training, easier deployment
**Performance**: Achieves 92% R² with minimal complexity
**Interpretability**: Easy to understand which factors drive AQI
**No Overfitting**: Clean train-test performance

## Feature Importance

Top contributors to AQI prediction:
1. PM2.5 and PM10 (current + lags)
2. AQI historical values (lag_1, lag_24)
3. Weather conditions (temperature, humidity, pressure)
4. Time of day (hour_sin)

---
