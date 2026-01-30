# AQI Predictor - Project Documentation

**Air Quality Index Prediction System for Islamabad, Pakistan**

---

## 📋 Project Overview

| Property | Value |
|----------|-------|
| **Project Name** | AQI Predictor |
| **Objective** | Predict Air Quality Index using Machine Learning |
| **Location** | Islamabad, Pakistan (33.6996°N, 73.0362°E) |
| **Timeline** | January 2026 - February 2026 |
| **Status** | Phase 2 - Feature Engineering |

---

## 🎯 Project Goals

1. Collect historical air quality and weather data
2. Perform exploratory data analysis
3. Engineer meaningful features for prediction
4. Train and evaluate machine learning models
5. Deploy a prediction system with automated updates

---

## 📊 Data Collection Journey

### Initial Attempts

#### 1. AQICN API (aqicn.org)
- **Attempt Date**: Early January 2026
- **Issue**: DNS blocking - couldn't access the API
- **Solution**: Used VPN to bypass blocking
- **Limitation**: Only provides **2 months** of historical data
- **Result**: ❌ Insufficient data for robust model training

#### 2. OpenWeather API (openweathermap.org)
- **Attempt Date**: Mid January 2026
- **Issue**: Similar DNS blocking issues
- **Solution**: Used VPN
- **Limitation**: Also limited to **2 months** of historical data
- **Result**: ❌ Not enough historical data

#### 3. OpenMeteo API (open-meteo.com) ✅
- **Selected Date**: January 2026
- **Advantages**:
  - ✅ **No API key required** (free and open)
  - ✅ **No DNS blocking** - accessible without VPN
  - ✅ Provides **60+ days** of historical data
  - ✅ Comprehensive air quality and weather data
  - ✅ Reliable and well-documented API
- **Result**: ✅ **Selected as primary data source**

### Why OpenMeteo?

After testing multiple APIs, OpenMeteo emerged as the best choice because:

1. **Accessibility**: No VPN required, no blocking issues
2. **Data Volume**: Sufficient historical data for training
3. **Cost**: Completely free, no API key needed
4. **Reliability**: Stable API with good uptime
5. **Data Quality**: Comprehensive pollutant and weather measurements

---

## 🌐 Data Sources

### OpenMeteo Air Quality API

**Endpoint**: `https://air-quality.open-meteo.com/v1/air-quality`

**Parameters**:
```python
{
    "latitude": 33.6996,
    "longitude": 73.0362,
    "hourly": [
        "pm10", "pm2_5", "carbon_monoxide",
        "nitrogen_dioxide", "sulphur_dioxide", "ozone"
    ],
    "timezone": "Asia/Karachi",
    "past_days": 60
}
```

**Pollutants Collected**:
- PM10 - Particulate Matter 10 micrometers (µg/m³)
- PM2.5 - Particulate Matter 2.5 micrometers (µg/m³)
- CO - Carbon Monoxide (µg/m³)
- NO₂ - Nitrogen Dioxide (µg/m³)
- SO₂ - Sulphur Dioxide (µg/m³)
- O₃ - Ozone (µg/m³)

### OpenMeteo Weather API

**Endpoint**: `https://api.open-meteo.com/v1/forecast`

**Parameters**:
```python
{
    "latitude": 33.6996,
    "longitude": 73.0362,
    "hourly": [
        "temperature_2m", "relative_humidity_2m",
        "surface_pressure", "wind_speed_10m",
        "wind_direction_10m", "precipitation"
    ],
    "timezone": "Asia/Karachi",
    "past_days": 60
}
```

**Weather Variables**:
- Temperature at 2m (°C)
- Relative Humidity at 2m (%)
- Surface Pressure (hPa)
- Wind Speed at 10m (km/h)
- Wind Direction at 10m (°)
- Precipitation (mm)

---

## 📁 Project Structure

```
AQI_Predictor/
├── src/
│   ├── data/
│   │   └── data_collector.py      # Data collection from OpenMeteo
│   └── features/
│       └── calculate_aqi.py       # AQI calculation using EPA standards
├── data/
│   ├── raw/                       # Raw data from API
│   │   └── raw_data_islamabad_*.csv
│   └── processed/                 # Processed data with AQI
│       └── aqi_data.csv
├── notebooks/
│   └── eda/                       # Exploratory Data Analysis
│       ├── 01_data_exploration.ipynb
│       └── 02_aqi_calculation.ipynb
├── docs/
│   └── DOCUMENTATION.md           # This file
├── .env                           # Environment variables
├── requirements.txt               # Python dependencies
└── readme.md                      # Project README
```

---

## 🛠️ Technology Stack

### Core Technologies
- **Python**: 3.10+
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Visualization**: matplotlib, seaborn, plotly

### APIs
- **OpenMeteo**: Air quality and weather data

### Future Technologies (Planned)
- **Database**: MongoDB Atlas (feature store & model registry)
- **Web Framework**: Streamlit or FastAPI
- **CI/CD**: GitHub Actions
- **Deployment**: Cloud platform (TBD)

---

## 📈 Development Phases

### ✅ Phase 1: Data Collection & Exploration (Completed)
- [x] Research and test different APIs (AQICN, OpenWeather, OpenMeteo)
- [x] Resolve DNS blocking issues
- [x] Select OpenMeteo as primary data source
- [x] Implement data collection script
- [x] Collect 1 year of historical data (8,784 hours)
- [x] Perform exploratory data analysis
- [x] Calculate AQI using EPA standards

### 🔄 Phase 2: Feature Engineering (Current)
- [x] Calculate AQI from pollutant concentrations
- [ ] Create time-based features
- [ ] Engineer weather interaction features
- [ ] Feature selection and importance analysis
- [ ] Document feature engineering decisions

### 🔄 Phase 3: Model Development (Upcoming)
- [ ] Train baseline models
- [ ] Implement advanced models (XGBoost, LightGBM)
- [ ] Hyperparameter tuning
- [ ] Model evaluation and comparison
- [ ] Select best performing model

### 🔄 Phase 4: Production Deployment (Planned)
- [ ] Set up MongoDB for feature store
- [ ] Implement model registry
- [ ] Create automated data collection pipeline
- [ ] Build prediction API
- [ ] Develop web dashboard
- [ ] Set up CI/CD pipeline

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- Internet connection (for API access)

### Installation

```bash
# Clone repository
git clone https://github.com/u-faizan/Pearls-AQI-Predictor.git
cd Pearls-AQI-Predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file:

```env
# City Configuration
CITY_NAME=Islamabad
CITY_LATITUDE=33.6996
CITY_LONGITUDE=73.0362

# MongoDB (for later phases)
MONGODB_URI=your_mongodb_uri
MONGODB_DATABASE=aqi_predictor
```

### Collect Data

```bash
# Run data collection script
python src/data/data_collector.py
```

---

## 📝 Lessons Learned

### API Selection Process

1. **Don't assume availability**: Popular APIs may be blocked in certain regions
2. **Test before committing**: Always test API access and data availability
3. **Consider data volume**: Ensure the API provides sufficient historical data
4. **Free isn't always limited**: OpenMeteo proves free APIs can be excellent
5. **Documentation matters**: Well-documented APIs save development time

### Technical Challenges

1. **DNS Blocking**: 
   - Problem: AQICN and OpenWeather were blocked
   - Solution: VPN access, but found better alternative (OpenMeteo)
   
2. **Data Limitations**:
   - Problem: 2-month limit on commercial APIs
   - Solution: OpenMeteo provides 60+ days without restrictions

---

## 📊 Current Status

**Last Updated**: January 30, 2026

**Progress**:
- ✅ Project setup complete
- ✅ Data source selected (OpenMeteo)
- ✅ Data collection script implemented
- ✅ Collected 1 year of historical data (8,784 hourly records)
- ✅ Exploratory Data Analysis completed
- ✅ AQI calculation implemented
- 🔄 Feature engineering in progress

**Data Summary**:
- **Total Records**: 8,784 hourly observations
- **Date Range**: Dec 24, 2024 - Dec 24, 2025
- **Mean AQI**: 180.9 (Unhealthy)
- **Dominant Pollutants**: Ozone (50.9%), PM2.5 (46.8%)
- **AQI Categories**: Only 16 hours were "Good" quality

---

## 🤝 Contributing

This is an internship project. Feedback and suggestions are welcome!

---

## 📄 License

MIT License

---

*Documentation maintained as part of the AQI Predictor project*
*Internship Project - January 2026 to February 2026*
