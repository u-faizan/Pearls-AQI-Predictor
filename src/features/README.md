# Feature Engineering Experiments

This document tracks all feature engineering experiments for the AQI Predictor project.

## Experiment Log

### Experiment 1: Baseline Features (Jan 18, 2026)

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

### Experiment 2: [Planned]

**Goal**: Improve R² from 0.49 to 0.60+

**Proposed Features**:
- Polynomial features (pm2_5², temperature²)
- Interaction terms (pm2_5 × temperature, pm2_5 × humidity)
- Additional lag features (lag_6, lag_12, lag_48)
- Extended rolling windows (3h, 6h, 12h)
- Domain features (rush_hour, season, weekend)

**Status**: 🔄 Pending

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

*Last Updated: January 19, 2026*
