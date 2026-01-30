# 🌍 Pearls AQI Predictor

Air Quality Index (AQI) prediction system for Islamabad, Pakistan using machine learning.

## 🎯 Project Overview

| Property | Value |
|----------|-------|
| **Project Name** | AQI Predictor |
| **Objective** | Predict Air Quality Index using Machine Learning |
| **Location** | Islamabad, Pakistan (33.6996°N, 73.0362°E) |
| **Timeline** | January 2026 - February 2026 |
| **Status** | Phase 2 - Feature Engineering |

## 📊 Current Progress

### Completed ✅
- [x] Data collection from OpenMeteo API (1 year of hourly data)
- [x] Exploratory Data Analysis (EDA)
- [x] AQI calculation using EPA standards
- [x] Data processing pipeline

### In Progress 🔄
- [ ] Feature engineering
- [ ] Feature selection

### Upcoming ⏳
- [ ] Model training and evaluation
- [ ] Model deployment

## 📈 Data Summary

- **Total Records**: 8,784 hourly observations
- **Date Range**: Dec 24, 2024 - Dec 24, 2025
- **Pollutants**: PM2.5, PM10, O₃, NO₂, SO₂, CO
- **Weather Variables**: Temperature, Humidity, Pressure, Wind, Precipitation, Cloud Cover
- **Mean AQI**: 180.9 (Unhealthy)
- **Dominant Pollutants**: Ozone (50.9%), PM2.5 (46.8%)

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Data**: OpenMeteo API (Weather & Air Quality)
- **Analysis**: pandas, numpy, matplotlib, seaborn
- **ML**: scikit-learn, XGBoost, LightGBM (planned)

## 📁 Project Structure

```
AQI_Predictor/
├── src/
│   ├── data/
│   │   └── data_collector.py      # Data collection from API
│   └── features/
│       └── calculate_aqi.py       # AQI calculation
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
│   └── DOCUMENTATION.md           # Project documentation
└── readme.md                      # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Internet connection (for API access)

### Installation

```bash
# Clone repository
git clone https://github.com/u-faizan/Pearls-AQI-Predictor.git
cd Pearls-AQI-Predictor

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

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
```

### Usage

```bash
# 1. Collect data from API
python src/data/data_collector.py

# 2. Calculate AQI
python src/features/calculate_aqi.py

# 3. Explore data in notebooks
jupyter notebook notebooks/eda/
```

## 📝 Documentation

- [Full Documentation](docs/DOCUMENTATION.md) - Complete project documentation
- [EDA Notebooks](notebooks/eda/) - Data exploration and analysis

## 🤝 Contributing

This is an internship project. Suggestions and feedback are welcome!

## 📄 License

MIT License

---

*Developed as part of a Data Science internship program (Jan 2026 - Feb 2026)*
