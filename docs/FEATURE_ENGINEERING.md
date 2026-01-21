# Feature Engineering

This document explains the features used for AQI prediction.

## Features (23 Total)

### 1. Base Features (12)
Raw pollutant and weather measurements:
- **Pollutants**: pm10, pm2_5, carbon_monoxide, nitrogen_dioxide, sulphur_dioxide, ozone
- **Weather**: temperature_2m, relative_humidity_2m, surface_pressure, wind_speed_10m, wind_direction_10m, precipitation

### 2. Lag Features (7)
Historical values to capture temporal patterns:
- `pm25_lag_1` - PM2.5 from 1 hour ago
- `pm25_lag_24` - PM2.5 from 24 hours ago
- `pm10_lag_1` - PM10 from 1 hour ago
- `aqi_lag_1` - AQI from 1 hour ago
- `aqi_lag_24` - AQI from 24 hours ago
- `pm25_rolling_mean_24h` - 24-hour average PM2.5
- `aqi_rolling_mean_24h` - 24-hour average AQI

### 3. Engineered Features (4)
- `hour_sin` - Cyclical time encoding (captures daily patterns)
- `pm2_5_to_pm10_ratio` - Ratio of fine to coarse particles

## Why These Features?

**Lag features** capture pollution persistence - air quality doesn't change instantly.

**Rolling averages** smooth out noise and show trends.

**Time features** capture daily patterns (rush hours, etc.).

**Ratios** provide relative pollution information.

## AQI Calculation

AQI (Air Quality Index) is calculated using EPA formula:

```
AQI = [(I_high - I_low) / (C_high - C_low)] × (C - C_low) + I_low
```

Where C is pollutant concentration and I values are from EPA breakpoints.

**AQI Scale**:
- 0-50: Good
- 51-100: Moderate
- 101-150: Unhealthy for Sensitive Groups
- 151-200: Unhealthy
- 201-300: Very Unhealthy
- 301-500: Hazardous

## Feature Importance

Based on model analysis:
1. **PM2.5 and PM10** (current + lags) - 40%
2. **AQI historical values** - 30%
3. **Weather conditions** - 20%
4. **Time of day** - 10%

## Data Processing

| Stage | Rows | Features |
|-------|------|----------|
| Raw data | 7,440 | 16 |
| After feature engineering | 7,416 | 23 |
| Final (cleaned) | 7,392 | 23 |

*24 rows lost due to lag features creating NaN values for initial hours*

---

**Reference**: [EPA AQI Calculator](https://www.airnow.gov/aqi/aqi-calculator/)
