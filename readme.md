# 🌍 Pearls AQI Predictor

End-to-end ML pipeline for Air Quality Index (AQI) prediction with automated data collection, MongoDB storage, and real-time predictions.

## 🎯 Overview

This project predicts AQI using real-time weather and pollutant data with machine learning models stored in MongoDB.

**Key Features:**
- ✅ **23-feature XGBoost model** (MAE: 1.82, R²: 0.9226)
- ✅ **MongoDB integration** for features and models
- ✅ Automated data collection pipeline
- ✅ Model registry with versioning
- 🚧 CI/CD with GitHub Actions (in progress)
- 🚧 Interactive web dashboard (planned)

## 🛠️ Tech Stack

- **ML/Data**: Python, Scikit-learn, XGBoost, LightGBM, Pandas
- **Database**: MongoDB Atlas (Feature Store & Model Registry)
- **APIs**: OpenMeteo (Weather & Air Quality)
- **CI/CD**: GitHub Actions (planned)
- **Web**: Streamlit/FastAPI (planned)

## 📊 Model Performance

| Model | Test MAE | Test RMSE | Test R² |
|-------|----------|-----------|---------|
| Linear Regression | 10.93 | 38.31 | 0.4788 |
| Ridge Regression | 10.93 | 38.31 | 0.4788 |
| Random Forest | 3.38 | 21.93 | 0.8292 |
| **XGBoost** ✅ | **1.82** | **14.76** | **0.9226** |
| LightGBM | 3.67 | 19.55 | 0.8643 |

## 🚀 Quick Start

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
# Add your MongoDB URI and API keys to .env
```

### MongoDB Setup

1. Create free MongoDB Atlas cluster at [mongodb.com/cloud/atlas](https://mongodb.com/cloud/atlas)
2. Get connection string
3. Add to `.env`:
```env
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DATABASE=aqi_predictor
```

### Upload Data to MongoDB

```bash
# Upload features
python scripts/upload_features_to_mongodb.py

# Upload trained model
python scripts/upload_model_to_mongodb.py

# Test connection
python scripts/test_mongodb_connection.py
```

## 📁 Project Structure

```
AQI_Predictor/
├── src/
│   ├── data/              # Data collection scripts
│   ├── features/          # Feature engineering
│   ├── models/            # Model training
│   └── database/          # MongoDB integration
│       ├── mongodb_client.py
│       ├── feature_store.py
│       └── model_registry.py
├── scripts/               # Utility scripts
├── data/
│   ├── raw/              # Raw data (gitignored)
│   └── processed/        # Processed features (gitignored)
├── models/               # Trained models (gitignored, stored in MongoDB)
├── docs/                 # Documentation
└── notebooks/            # Jupyter notebooks for EDA
```

## 📊 Project Status

✅ **Phase 1: Model Development** (Completed)
- Data collection
- Feature engineering (23 features)
- Model training (5 models)
- Best model selection (XGBoost)

🚧 **Phase 2: Production Deployment** (In Progress)
- MongoDB integration ✅
- Hourly data collection (planned)
- Daily model retraining (planned)
- CI/CD pipeline (planned)

## 📝 Documentation

- [Project Documentation](docs/PROJECT_DOCUMENTATION.md)
- [Feature Engineering](docs/FEATURE_ENGINEERING.md)
- [Implementation Plan](implementation_plan.md)

## 🤝 Contributing

This is an internship project. Contributions and suggestions are welcome!

## 📄 License

MIT License

---

*Developed as part of a Data Science internship program (Jan 2026 - Feb 2026)*
