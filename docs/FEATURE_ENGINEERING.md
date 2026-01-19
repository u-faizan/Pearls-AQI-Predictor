# Feature Engineering for AQI Prediction

**Purpose**: Transform raw air quality data into meaningful features that machine learning models can use to predict future AQI values.

---

## 📊 What is Feature Engineering?

Feature engineering is the process of creating new variables (features) from raw data that help ML models make better predictions. Think of it as giving the model the right "clues" to solve the puzzle.

**Example**: Instead of just giving temperature values, we also tell the model "it's 3 PM" and "it's a weekday" - these extra clues help predict pollution patterns.

---

## 🎯 Features We Created

### 1. **AQI Calculation** (Target Variable)

**What**: Air Quality Index - a single number (0-500) that represents air pollution level.

**Why**: 
- Converts multiple pollutant concentrations into one easy-to-understand number
- Standardized by EPA (Environmental Protection Agency)
- Our ML model will predict this value

**How it works**:
```
PM2.5 = 35 µg/m³  →  AQI = 100 (Moderate)
PM2.5 = 150 µg/m³ →  AQI = 200 (Unhealthy)
```

**EPA AQI Scale**:
| AQI Range | Category | Health Impact |
|-----------|----------|---------------|
| 0-50 | Good | No health risk |
| 51-100 | Moderate | Sensitive people may be affected |
| 101-150 | Unhealthy for Sensitive Groups | Children, elderly at risk |
| 151-200 | Unhealthy | Everyone may experience effects |
| 201-300 | Very Unhealthy | Health alert |
| 301-500 | Hazardous | Emergency conditions |

**Features created**:
- `aqi` - Overall AQI (max of all pollutants)
- `aqi_pm25` - AQI from PM2.5
- `aqi_pm10` - AQI from PM10

---

### 2. **Time-Based Features**

**What**: Features extracted from the timestamp (hour, day, month, etc.)

**Why**: Air pollution has strong temporal patterns:
- **Hourly**: Rush hour (7-9 AM, 5-7 PM) has higher pollution
- **Daily**: Weekends often have less traffic pollution
- **Monthly**: Winter months typically have worse air quality

**Features created**:
- `hour` (0-23) - Hour of day
- `day_of_week` (0-6) - Monday=0, Sunday=6
- `month` (1-12) - Month of year
- `day_of_month` (1-31) - Day of month
- `is_weekend` (0 or 1) - 1 if Saturday/Sunday

**Example use case**:
```
If hour=8 and day_of_week=1 (Tuesday)
→ Model learns: "Morning rush hour on weekday = higher pollution"
```

---

### 3. **Lag Features** (Historical Values)

**What**: Past values of pollution levels (1 hour ago, 3 hours ago, 24 hours ago)

**Why**: 
- **Air pollution is persistent** - if it's polluted now, it's likely to stay polluted
- **Yesterday's pollution affects today** - pollution doesn't disappear instantly
- Helps model understand trends and momentum

**Features created**:
- `aqi_lag_1` - AQI 1 hour ago
- `pm25_lag_1` - PM2.5 1 hour ago
- `pm10_lag_1` - PM10 1 hour ago
- `aqi_lag_3` - AQI 3 hours ago
- `aqi_lag_24` - AQI 24 hours ago (yesterday same time)
- `pm25_lag_24` - PM2.5 24 hours ago

**Example**:
```
Current time: 3 PM
aqi_lag_1 = AQI at 2 PM
aqi_lag_24 = AQI at 3 PM yesterday

If both are high → Model predicts: "Pollution will likely stay high"
```

---

### 4. **Rolling Statistics** (Moving Averages)

**What**: Average values over a sliding time window (24 hours, 7 days)

**Why**:
- **Smooths out noise** - removes random fluctuations
- **Shows trends** - is pollution getting better or worse?
- **Captures longer-term patterns** - weekly cycles, seasonal changes

**Features created**:
- `aqi_rolling_mean_24h` - Average AQI over last 24 hours
- `pm25_rolling_mean_24h` - Average PM2.5 over last 24 hours
- `aqi_rolling_std_24h` - Standard deviation (volatility) over 24 hours
- `aqi_rolling_mean_7d` - Average AQI over last 7 days (168 hours)

**Example**:
```
Last 24 hours AQI: [100, 105, 110, 108, 112, ...]
Rolling mean = 107 (smoothed value)

If rolling mean is increasing → Model learns: "Pollution trend is worsening"
```

---

## 🔍 Why These Features Matter

### **Without Feature Engineering**:
```
Input: pm2_5=35, pm10=50, temperature=25
Model: "I see some numbers, but I don't know what they mean"
Prediction: Random guess
```

### **With Feature Engineering**:
```
Input: 
- AQI=100 (Moderate pollution)
- hour=8 (morning rush hour)
- is_weekend=0 (weekday)
- aqi_lag_1=95 (pollution increasing)
- aqi_rolling_mean_24h=90 (upward trend)

Model: "Morning rush hour on weekday + pollution increasing + upward trend"
Prediction: AQI will be ~110 in next hour (Unhealthy for Sensitive Groups)
```

---

## 📈 Feature Importance Ranking

Based on typical AQI prediction models:

1. **Lag features** (40%) - Past pollution is best predictor of future pollution
2. **Rolling averages** (30%) - Trends and momentum matter
3. **Time features** (20%) - Daily and weekly patterns
4. **Raw pollutants** (10%) - Current concentrations

---

## 🧮 Mathematical Formulas

### EPA AQI Formula:
```
AQI = [(I_high - I_low) / (C_high - C_low)] × (C - C_low) + I_low

Where:
- C = Pollutant concentration
- C_low, C_high = Breakpoint concentrations
- I_low, I_high = Breakpoint AQI values
```

### Rolling Mean (24-hour):
```
rolling_mean_24h[t] = (AQI[t] + AQI[t-1] + ... + AQI[t-23]) / 24
```

### Rolling Standard Deviation:
```
rolling_std_24h[t] = sqrt(variance of last 24 hours)
```

---

## 📊 Data Transformation Summary

| Stage | Rows | Columns | Description |
|-------|------|---------|-------------|
| **Raw Data** | 7,440 | 16 | Original API data |
| **After AQI Calculation** | 7,440 | 19 | Added AQI columns |
| **After Time Features** | 7,440 | 24 | Added hour, day, month, etc. |
| **After Lag Features** | 7,440 | 30 | Added historical values |
| **After Rolling Features** | 7,440 | 34 | Added moving averages |
| **Final (after cleaning)** | 7,416 | 34 | Removed NaN rows |

**Why 24 rows lost?** 
- Lag features create NaN for first few rows (no history available)
- Example: `aqi_lag_24` is NaN for first 24 hours

---

## 🎓 Key Takeaways

1. **AQI is our target** - What we're trying to predict
2. **Time features capture patterns** - Rush hours, weekends, seasons
3. **Lag features capture persistence** - Pollution doesn't change instantly
4. **Rolling features capture trends** - Is it getting better or worse?

**Together, these features give the ML model a complete picture of:**
- What's happening now (current pollutants)
- What happened before (lag features)
- What's the trend (rolling averages)
- When is it happening (time features)

This is why feature engineering is often called **"the secret sauce"** of machine learning!

---

## 📚 Further Reading

- [EPA AQI Calculator](https://www.airnow.gov/aqi/aqi-calculator/)
- [EPA AQI Basics](https://www.airnow.gov/aqi/aqi-basics/)
- [Time Series Feature Engineering](https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/)

---

*Created for AQI Prediction Project - Islamabad, Pakistan*
