# 🌍 Pearls AQI Predictor

Air Quality Index (AQI) prediction system for Islamabad, Pakistan using machine learning.

## 🎯 Project Overview

This project aims to build an end-to-end ML pipeline for predicting Air Quality Index using real-time weather and air quality data.

**Status:** 🚧 In Development (Phase 1: Data Collection & Exploration)

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Data**: OpenMeteo API (Weather & Air Quality)
- **ML**: Scikit-learn, XGBoost, LightGBM
- **Database**: MongoDB Atlas
- **Deployment**: Streamlit/FastAPI (planned)

## 📁 Project Structure

```
AQI_Predictor/
├── src/
│   ├── data/              # Data collection scripts
│   ├── features/          # Feature engineering
│   ├── models/            # Model training
│   └── database/          # MongoDB integration
├── data/
│   ├── raw/              # Raw data from API
│   └── processed/        # Processed features
├── models/               # Trained models
├── notebooks/            # Jupyter notebooks for analysis
└── docs/                 # Documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- MongoDB Atlas account (free tier)

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

# Set up environment variables
cp .env.example .env
# Add your API keys to .env
```

### Environment Variables

Create a `.env` file with:

```env
# OpenWeather API (optional - using OpenMeteo instead)
OPENWEATHER_API_KEY=your_key_here

# MongoDB (for feature store and model registry)
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DATABASE=aqi_predictor

# City Configuration
CITY_NAME=Islamabad
CITY_LATITUDE=33.6996
CITY_LONGITUDE=73.0362
```

## 📊 Current Progress

### Phase 1: Data Collection & Exploration ✅
- [x] Set up project structure
- [x] Configure OpenMeteo API
- [x] Implement data collection pipeline
- [ ] Exploratory Data Analysis
- [ ] Feature engineering
- [ ] Feature selection

### Phase 2: Model Development (Upcoming)
- [ ] Train baseline models
- [ ] Feature importance analysis
- [ ] Model optimization
- [ ] Model evaluation

### Phase 3: Production Deployment (Planned)
- [ ] MongoDB integration
- [ ] Automated data collection
- [ ] Model registry
- [ ] CI/CD pipeline
- [ ] Web dashboard

## 📝 Documentation

- [Project Documentation](docs/PROJECT_DOCUMENTATION.md)
- [Feature Engineering](docs/FEATURE_ENGINEERING.md)
- [Implementation Plan](implementation_plan.md)

## 🤝 Contributing

This is an internship project. Suggestions and feedback are welcome!

## 📄 License

MIT License

---

*Developed as part of a Data Science internship program (Jan 2026 - Feb 2026)*
