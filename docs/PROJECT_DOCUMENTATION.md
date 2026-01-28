# AQI Predictor - Project Documentation
**Air Quality Index Forecasting System for Islamabad, Pakistan**

---

## 📋 Project Information

| **Property** | **Value** |
|--------------|-----------|
| **Project Name** | AQI Predictor |
| **Objective** | Predict Air Quality Index for next 3 days using Machine Learning |
| **City** | Islamabad, Pakistan |
| **Timeline** | December 2025 - February 2026 (60 days) |
| **Developer** | Umar Faizan |
| **Type** | Data Science Internship Project |

---

## 🎯 Project Objectives

The main objectives of this project are:

1. **Data Collection**: Fetch real-time and historical air quality data from APIs
2. **Feature Engineering**: Extract meaningful features from raw pollutant data
3. **Model Training**: Train and compare multiple ML models for AQI prediction
4. **Automation**: Implement CI/CD pipelines for automated data collection and model retraining
5. **Visualization**: Build an interactive Streamlit dashboard for predictions
6. **Documentation**: Maintain comprehensive documentation for reproducibility

---

## 🏗️ System Architecture

```
┌─────────────────┐
│  OpenMeteo API  │ (Free - No Key Required)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Data Collection Pipeline          │
│   (src/data/data_collector.py)     │
│   • Fetch air quality data          │
│   • Fetch weather data              │
│   • Merge datasets                  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Feature Engineering               │
│   (src/features/feature_engineering)│
│   • Calculate EPA AQI               │
│   • Create 23 optimized features    │
│   • Time + Lag + Engineered         │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   MongoDB Atlas                     │
│   (Feature Store & Model Registry)  │
│   • Collection: aqi_features        │
│   • Collection: model_registry      │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Model Training                    │
│   (src/models/train.py)             │
│   • XGBoost (Best: MAE=1.82)        │
│   • LightGBM, Random Forest         │
│   • Linear, Ridge Regression        │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Model Registry (MongoDB)          │
│   • Versioning                      │
│   • Active model tracking           │
│   • Performance metrics             │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Prediction API (Planned)          │
│   • Load model from MongoDB         │
│   • Real-time predictions           │
│   • Streamlit Dashboard             │
└─────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### **Programming Languages**

- Python 3.10+

### **Data Processing**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **pyarrow** - Efficient data storage (Parquet format)

### **Machine Learning**
- **scikit-learn** - ML algorithms and tools
  - Linear Regression
  - Ridge Regression
  - Random Forest Regressor
- **XGBoost** - Gradient boosting (Best model: MAE=1.82)
- **LightGBM** - Fast gradient boosting
- **joblib** - Model serialization

### **Database**
- **MongoDB Atlas** - Cloud NoSQL database
  - **Feature Store**: Stores processed AQI features (7,392+ records)
  - **Model Registry**: Stores trained models with versioning
  - **pymongo** - Python MongoDB driver

### **Web Framework**
- **Streamlit** - Interactive dashboard
- **plotly** - Interactive visualizations
- **matplotlib** - Static plots

### **APIs**
- **OpenMeteo API** - Free air quality and weather data
  - Air Quality API: `https://air-quality.open-meteo.com/v1/air-quality`
  - Weather API: `https://api.open-meteo.com/v1/forecast`

### **Automation**
- **GitHub Actions** - CI/CD pipelines
- **APScheduler** - Task scheduling

### **Development Tools**
- **python-dotenv** - Environment variable management
- **requests** - HTTP requests
- **pytest** - Testing framework

---

## 📊 Data Sources

### **OpenMeteo Air Quality API**

**Endpoint**: `https://air-quality.open-meteo.com/v1/air-quality`

**Parameters**:
- `latitude`: 33.6996 (Islamabad)
- `longitude`: 73.0362 (Islamabad)
- `hourly`: Pollutant measurements
- `timezone`: Asia/Karachi
- `past_days`: 60 (for historical data)

**Pollutants Collected**:
1. **PM2.5** - Particulate Matter 2.5 micrometers (µg/m³)
2. **PM10** - Particulate Matter 10 micrometers (µg/m³)
3. **O₃** - Ozone (µg/m³)
4. **NO₂** - Nitrogen Dioxide (µg/m³)
5. **SO₂** - Sulphur Dioxide (µg/m³)
6. **CO** - Carbon Monoxide (mg/m³)

### **OpenMeteo Weather API**

**Endpoint**: `https://api.open-meteo.com/v1/forecast`

**Meteorological Data**:
1. **Temperature** - 2m above ground (°C)
2. **Relative Humidity** - 2m above ground (%)
3. **Surface Pressure** - Atmospheric pressure (hPa)
4. **Wind Speed** - 10m above ground (km/h)
5. **Wind Direction** - 10m above ground (°)
6. **Precipitation** - Rainfall (mm)

---

## 📁 Project Structure

```
AQI_Predictor/
│
├── data/
│   ├── raw/                    # Raw API data (CSV)
│   └── processed/              # Processed features (CSV)
│
├── models/                     # Trained ML models (.pkl)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── model_metrics.json
│
├── src/
│   ├── data/
│   │   ├── data_collector.py  # OpenMeteo data fetcher
│   │   └── aqicn_collector.py # AQICN alternate collector
│   │
│   ├── features/
│   │   └── feature_engineering.py  # AQI calculation & features
│   │
│   ├── models/
│   │   ├── train.py           # Model training pipeline
│   │   └── predict.py         # Prediction functions
│   │
│   └── utils/
│       └── config.py          # Configuration utilities
│
├── app/
│   ├── streamlit_app.py       # Main dashboard
│   └── components/            # Dashboard components
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── tests/                     # Unit tests
│   ├── test_data_collector.py
│   └── test_models.py
│
├── .github/
│   └── workflows/            # GitHub Actions
│       ├── feature_pipeline.yml
│       └── training_pipeline.yml
│
├── docs/                     # Documentation
│   └── PROJECT_DOCUMENTATION.md  # This file
│
├── .env                      # Environment variables (not committed)
├── .env.example              # Environment template
├── .gitignore               # Git ignore rules
├── requirements.txt         # Python dependencies
└── readme.md               # Project overview
```

---

## 🔧 Installation & Setup

### **Prerequisites**
- Python 3.10 or higher
- Git
- Internet connection (for API access)

### **Step 1: Clone Repository**
```bash
git clone <repository-url>
cd AQI_Predictor
```

### **Step 2: Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Configure Environment**
```bash
# Copy environment template
copy .env.example .env   # Windows
cp .env.example .env     # Linux/Mac

# Edit .env file with your city coordinates
# For Islamabad (default):
CITY_NAME=Islamabad
CITY_LATITUDE=33.6996
CITY_LONGITUDE=73.0362
```

### **Step 5: Verify Installation**
```bash
python --version  # Should be 3.10+
pip list          # Should show all dependencies
```

---

## 🚀 Usage Guide

### **Phase 1: Data Collection**

**Fetch 60 days of historical data:**
```bash
python src/data/data_collector.py
```

**Expected Output:**
- CSV file saved to `data/raw/openmeteo_combined_YYYYMMDD_HHMMSS.csv`
- ~1440 hourly records (60 days × 24 hours)
- 13 columns (time + 6 pollutants + 6 weather features)

**Verify:**
```bash
# Check data folder
ls data/raw/

# View first few rows
import pandas as pd
df = pd.read_csv('data/raw/openmeteo_combined_*.csv')
print(df.head())
```

---

### **Phase 2: Feature Engineering**

**Process raw data and calculate AQI:**
```bash
python src/features/feature_engineering.py
```

**What it does:**
1. Loads raw data from `data/raw/`
2. Calculates AQI using EPA formula (PM2.5, PM10)
3. Creates time-based features (hour, day_of_week, month, is_weekend)
4. Creates lag features (1h, 3h, 24h previous values)
5. Creates rolling statistics (24h, 7-day averages)
6. Saves to `data/processed/processed_aqi.csv`

**Features Created:**

| Category | Features | Purpose |
|----------|----------|---------|
| **AQI Calculation** | `aqi`, `aqi_pm25`, `aqi_pm10` | Target variable (0-500 scale) |
| **Time Features** | `hour`, `day_of_week`, `month`, `day_of_month`, `is_weekend` | Capture daily/weekly patterns |
| **Lag Features** | `aqi_lag_1`, `aqi_lag_3`, `aqi_lag_24`, `pm25_lag_1`, `pm25_lag_24`, `pm10_lag_1` | Use past pollution to predict future |
| **Rolling Stats** | `aqi_rolling_mean_24h`, `pm25_rolling_mean_24h`, `aqi_rolling_std_24h`, `aqi_rolling_mean_7d` | Smooth noise, show trends |

**Results (Islamabad Data):**
- **Input**: 7,440 raw rows
- **Output**: 7,416 processed rows (24 lost due to lag features)
- **Total Features**: 34 columns
- **Average AQI**: 102.6 (Unhealthy for Sensitive Groups)
- **AQI Range**: 13.3 - 500

**EPA AQI Scale:**
- 0-50: Good
- 51-100: Moderate
- 101-150: Unhealthy for Sensitive Groups ← **Islamabad average**
- 151-200: Unhealthy
- 201-300: Very Unhealthy
- 301-500: Hazardous

---

### **Phase 3: Model Training**

**Train and compare models:**
```bash
python src/models/train.py
```

**Models Trained:**
1. **Linear Regression** - Baseline linear model
2. **Ridge Regression** - Regularized linear model
3. **Random Forest** - Ensemble tree-based model
4. **XGBoost** - Gradient boosting model
5. **LightGBM** - Light gradient boosting model

**Evaluation Metrics:**
- **RMSE** - Root Mean Square Error (lower is better)
- **MAE** - Mean Absolute Error (lower is better)
- **R² Score** - Coefficient of determination (higher is better)

---

## 🧪 Feature Engineering Experiments

### **Experiment 1: Baseline Features** (Jan 18, 2026)

**Features Created** (34 total):
- EPA AQI calculation from PM2.5 and PM10
- Time features: hour, day_of_week, month, is_weekend
- Lag features: 1h, 3h, 24h for AQI, PM2.5, PM10
- Rolling statistics: 24h and 7-day mean/std

**Model Performance (Test Set):**

| Model | MAE | RMSE | R² | Notes |
|-------|-----|------|----|-------|
| Linear Regression | 18.12 | 41.57 | 0.3741 | Consistent, no overfitting |
| Ridge Regression | 18.12 | 41.57 | 0.3741 | Same as linear |
| Random Forest | 12.53 | 38.54 | 0.4621 | Overfitting (train R²=0.92) |
| XGBoost | 12.70 | 38.52 | 0.4627 | Overfitting (train R²=0.96) |
| **LightGBM** ✅ | **13.79** | **37.30** | **0.4963** | **Best generalization** |

**Winner**: LightGBM
- Explains ~50% of AQI variance
- Lowest RMSE on test set
- Best balance between performance and generalization

**Key Findings**:
- Tree-based models significantly outperform linear models
- Random Forest and XGBoost show severe overfitting
- LightGBM provides best test performance with reasonable training metrics
- Current R²=0.50 is solid baseline, target is 0.60+ for production

**Artifacts**:
- All models saved: `models/*.pkl`
- Metrics: `models/model_metrics.json`
- Best model: `models/best_model.pkl` (LightGBM)
- Feature list: `models/feature_columns.json`

**Next Steps**:
- Experiment with polynomial and interaction features
- Add more lag features and rolling windows
- Try domain-specific features (rush hour, season)
- Target: R² > 0.60

---

---

### **Final Model: 23 Features**

**Optimized feature set** for production deployment.

**Features** (23 total):
- 12 Base: Pollutants (PM10, PM2.5, CO, NO2, SO2, O3) + Weather (temp, humidity, pressure, wind, precipitation)
- 7 Lag: pm25_lag_1, pm25_lag_24, pm10_lag_1, aqi_lag_1, aqi_lag_24, pm25_rolling_mean_24h, aqi_rolling_mean_24h
- 4 Engineered: hour_sin, pm2_5_to_pm10_ratio

**Model Performance (Test Set)**:

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 10.93 | 38.31 | 0.4788 |
| Ridge Regression | 10.93 | 38.31 | 0.4788 |
| Random Forest | 3.38 | 21.93 | 0.8292 |
| **XGBoost** ✅ | **1.82** | **14.76** | **0.9226** |
| LightGBM | 3.67 | 19.55 | 0.8643 |

**Winner**: XGBoost
- MAE: 1.82 (±1.82 AQI points average error)
- R²: 0.9226 (92% variance explained)
- Balanced performance, minimal overfitting

**Why 23 Features?**
- Simpler than 64-feature alternatives
- Faster training and inference
- Easier to interpret and maintain
- Excellent performance with minimal complexity

**Artifacts**:
- Models: `models/*.pkl`
- Metrics: `models/model_metrics.json`, `models/metrics.txt`
- Best model: `models/best_model.pkl` (XGBoost)
- Features: `models/feature_columns.json`

---

### **Phase 4: Dashboard**

**Launch Streamlit app:**
```bash
streamlit run app/streamlit_app.py
```

**Dashboard Features:**
- 🏠 **Home**: Current AQI and 3-day forecast
- 📈 **Historical Data**: Trends and pollutant breakdown
- 🤖 **Model Performance**: Comparison of all models
- ℹ️ **About**: Project information

**Access:**
- Open browser to `http://localhost:8501`

---

---

## 📈 Implementation Progress

### ✅ **Phase 1: Model Development** (Completed)

#### Data Collection
- [x] OpenMeteo API integration
- [x] Collected 7,440 hourly records for Islamabad
- [x] Data quality validation

#### Feature Engineering
- [x] EPA AQI calculation
- [x] Created 23 optimized features (12 base + 7 lag + 4 engineered)
- [x] Feature importance analysis

#### Model Training
- [x] Trained 5 models (Linear, Ridge, Random Forest, XGBoost, LightGBM)
- [x] Selected XGBoost as best model (MAE=1.82, R²=0.9226)
- [x] Saved models and metrics

#### Documentation
- [x] Complete project documentation
- [x] Feature engineering guide
- [x] Model performance metrics

---

### � **Phase 2: Production Deployment** (In Progress)

#### MongoDB Integration
- [ ] Set up MongoDB Atlas cluster
- [ ] Create feature store schema
- [ ] Implement model registry
- [ ] Test CRUD operations

#### Automated Data Collection
- [ ] Hourly data collection script
- [ ] GitHub Actions workflow (hourly)
- [ ] Feature calculation with historical lags

#### Model Registry & Retraining
- [ ] Model versioning system
- [ ] Daily retraining script
- [ ] GitHub Actions workflow (daily)
- [ ] Active model tracking

#### CI/CD Pipeline
- [ ] Automated testing
- [ ] Deployment workflows
- [ ] Monitoring and alerts

---

### 📅 **Next Steps**

**Week 1**: MongoDB setup and feature store implementation
**Week 2**: Automated hourly data collection
**Week 3**: Model registry and daily retraining
**Week 4**: CI/CD pipeline and testing

---
- [ ] Set up GitHub Actions

#### **Week 4** (Feb 2-10)
- [ ] Testing and bug fixes
- [ ] Documentation completion
- [ ] Final submission

---

## 📊 Expected Results

### **Model Performance Targets**
- **RMSE**: < 25
- **MAE**: < 20
- **R² Score**: > 0.65

### **Data Requirements**
- **Minimum**: 30 days of hourly data (~720 samples)
- **Target**: 60 days of hourly data (~1440 samples)
- **Features**: 20+ engineered features

---

## 🔍 Technical Concepts Learned

### **1. API Integration**
- Making HTTP GET requests
- Parsing JSON responses
- Error handling and timeouts

### **2. Data Processing**
- Pandas DataFrame operations
- Data cleaning and merging
- Handling missing values

### **3. Feature Engineering**
- Time-based feature extraction
- Lag features for time series
- Rolling window statistics
- Domain-specific features (AQI calculation)

### **4. Machine Learning**
- Supervised learning
- Train/test split
- Feature scaling
- Model comparison
- Cross-validation
- Hyperparameter tuning

### **5. Web Development**
- Streamlit framework
- Interactive visualizations
- Dashboard design

### **6. DevOps**
- Version control with Git
- CI/CD with GitHub Actions
- Environment management
- Automated scheduling

---

## 📚 References

### **AQI Calculation**
- [EPA AQI Basics](https://www.airnow.gov/aqi/aqi-basics/)
- [EPA AQI Calculator](https://www.airnow.gov/aqi/aqi-calculator/)

### **APIs**
- [OpenMeteo Documentation](https://open-meteo.com/en/docs)
- [OpenMeteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api)

### **Machine Learning**
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Random Forest Guide](https://scikit-learn.org/stable/modules/ensemble.html#forest)

### **Web Framework**
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 🐛 Troubleshooting

### **Issue: API Request Fails**
**Solution:**
- Check internet connection
- Verify coordinates are correct
- Check API endpoint is accessible

### **Issue: Missing Data**
**Solution:**
- Some pollutants may not be available for all locations
- Handle missing values in feature engineering
- Use data imputation techniques

### **Issue: Model Poor Performance**
**Solution:**
- Collect more data (increase past_days)
- Try different features
- Tune hyperparameters
- Check for data quality issues

---

## 📝 Future Enhancements

1. **Multiple Cities**: Support prediction for multiple cities
2. **Advanced Models**: Add XGBoost, LightGBM, Neural Networks
3. **Real-time Alerts**: Email/SMS notifications for high AQI
4. **Mobile App**: Flutter/React Native mobile application
5. **Weather Integration**: Include weather forecasts for better predictions
6. **SHAP Analysis**: Model explainability and feature importance

---

## 👤 Author

**Umar Faizan**
- University: NUML Islamabad
- CGPA: 3.6
- Domain: Data Science
- Project Type: Internship Project
- Duration: December 2025 - Febrauary 2026

---

## 📄 License

MIT License - Feel free to use this project for learning and development.

---

*Last Updated: January 19, 2026*
*Status: Model Training & Experimentation Complete ✅*
