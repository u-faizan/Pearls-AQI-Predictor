# 🌍 Pearls AQI Predictor

End-to-end ML pipeline for 3-day Air Quality Index (AQI) forecasting with automated data collection, feature engineering, and real-time predictions.

## 🎯 Overview

This project predicts AQI for the next 3 days using real-time weather and pollutant data from open meteo and Hopsworks APIs.

**Key Features:**
- ✅ Automated hourly data collection
- ✅ ML-powered 3-day AQI forecasts
- ✅ Interactive web dashboard
- ✅ Hazardous AQI alerts
- ✅ Model explainability with SHAP

## 🛠️ Tech Stack

- **ML/Data**: Python, Scikit-learn, Pandas
- **Feature Store**: Hopsworks
- **Web**: Streamlit, FastAPI
- **CI/CD**: GitHub Actions
- **APIs**: open meteo, Hopsworks

## 🚀 Quick Start

```bash
# Clone repository
git clone <repo-url>
cd pearls-aqi-predictor

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your API keys to .env

# Run dashboard
streamlit run app/streamlit_app.py
```

## 📤 Push to GitHub

```bash
# Stage all changes
git add .

# Commit with a message
git commit -m "Update data collector and documentation"

# Push to the remote repository (replace <branch> with your branch, e.g., main)
git push origin <branch>
```


## 📊 Project Status

🚧 **In Development** - Internship Project (Jan 2026 - Feb 2026)

## 📝 Documentation

- [Implementation Plan](implementation_plan.md)
- [Task Breakdown](task.md)

## 📄 License

MIT License

---

*Developed as part of a Data Science internship program.*
