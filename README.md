# 🌊 Flash Flood Prediction System for Hilly Regions

**Smart India Hackathon Project — Mountain Hydrology & Disaster Early Warning**  
**Focus Area:** Himachal Pradesh, India (Shimla, Beas & Sutlej Basins)

---

## 📌 Project Overview

Hilly and mountainous regions like Himachal Pradesh face recurring catastrophic flash floods and cloudbursts during the monsoon season. Due to steep slopes, narrow valleys, and fragile geology, high-intensity precipitation combined with antecedent soil moisture saturation can trigger devastating flash floods and debris flows within hours.

This project implements an **end-to-end Machine Learning Flash Flood Early Warning System**. It ingests multi-source meteorological and hydrological data, performs physics-informed feature engineering, trains calibrated predictive models on a strict chronological partition, and serves interactive predictions via an intuitive Streamlit dashboard.

---

## 🏗️ System Architecture

```
NASA POWER / MERRA-2 Weather Data (2015–2025)
                     ↓
        src/data/data_loader.py
   (CSV Ingestion & Header Processing)
                     ↓
      src/features/feature_engineering.py
   (Rolling Rainfall, Lag Features & API Index)
                     ↓
      src/data/create_labeled_dataset.py
   (Historical HPSDMA/IMD Disasters + CWC Hydrological Thresholds)
                     ↓
         data/processed/himachal_flood_labeled.csv
                     ↓
          src/ml/train.py
   (Chronological Train: 2015-2022 | Test: 2023-2025)
   (Logistic Regression vs. Random Forest vs. HistGradientBoosting)
                     ↓
      models/flash_flood_model.joblib
   (Serialized Best Model + Evaluation Metrics + Feature Importances)
                     ↓
          src/ml/predict.py
   (Inference Engine: Probability, Risk Tiers & Factor Attribution)
                     ↓
         dashboard/app.py
   (Interactive Streamlit Dashboard & Real-Time Weather Simulator)
```

---

## 📂 Repository Structure

```
flash-flood-ai/
├── README.md                           # Complete project documentation
├── requirements.txt                    # Project dependencies (pandas, scikit-learn, joblib, streamlit, plotly)
├── dashboard/
│   └── app.py                          # Streamlit web dashboard with ML prediction & simulation
├── data/
│   ├── raw/
│   │   └── himachal_weather.csv        # 11 years (2015–2025) NASA POWER daily weather observations
│   └── processed/
│       ├── himachal_features.csv       # 17 engineered hydrometeorological features
│       └── himachal_flood_labeled.csv  # Verified ground-truth labeled training dataset
├── models/
│   └── flash_flood_model.joblib        # Serialized best trained model artifact
├── src/
│   ├── data/
│   │   ├── data_loader.py              # Ingestion class for NASA POWER CSV format
│   │   └── create_labeled_dataset.py   # Ground-truth labeling pipeline
│   ├── features/
│   │   └── feature_engineering.py      # Antecedent Precipitation Index, rolling sums, interactions
│   ├── ml/
│   │   ├── train.py                    # Multi-model training, chronological split & evaluation
│   │   └── predict.py                  # Production inference & factor attribution class
│   └── utils/
│       └── config.py                   # Centralized paths, thresholds, and configurations
└── tests/
    └── test_pipeline.py                # Automated unit tests covering all modules
```

---

## 🔬 Ground-Truth Labeling Methodology

Rather than inventing arbitrary labels, the dataset combines two scientifically rigorous criteria:
1. **Confirmed Historical Disaster Chronicles (2015–2025):**
   Mapped against official disaster records from the Himachal Pradesh State Disaster Management Authority (**HPSDMA**), **NDMA**, **SANDRP**, and **IMD**. Examples include:
   - July 8–11, 2023: Catastrophic multi-district cloudbursts (302 mm 3-day rainfall)
   - August 13–15, 2023: Shimla Shiv temple cloudburst and flash flood disaster
   - July 30–August 2, 2024: Samej (Rampur) and Mandi flash floods
   - July 30, 2022: Kullu & Shimla flash floods
   - August 17–19, 2019: Yamuna/Sutlej basin cloudburst disaster
2. **Hydrological Flash Flood Guidance (IMD / CWC):**
   - **Heavy Rainfall:** Daily precipitation $\ge 64.5\text{ mm}$ (IMD Heavy Rain standard)
   - **Soil Saturation Trigger:** Daily rain $\ge 35\text{ mm}$ with 3-day antecedent rainfall $\ge 75\text{ mm}$
   - **Prolonged Runoff:** Daily rain $\ge 25\text{ mm}$ with 7-day antecedent rainfall $\ge 120\text{ mm}$

**Result:** An authentic positive disaster class frequency of **1.64%** (66 flood days across 4,018 daily records), reflecting real-world hydrometeorological disaster rarity.

---

## 🚀 Setup & Execution Guide

### 1. Environment Setup

```bash
# Clone repository and enter folder
cd flash-flood-ai

# Activate your virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Feature Engineering & Dataset Labeling

Generate rolling rainfall statistics, lag terms, and the Antecedent Precipitation Index (API):

```bash
python src/features/feature_engineering.py
```

Create the ground-truth labeled disaster dataset:

```bash
python src/data/create_labeled_dataset.py
```

### 3. Model Training & Evaluation

Train the benchmark models using a strict **chronological time-series split**:
- **Train period:** 2015–2022 (2,922 days)
- **Test period:** 2023–2025 (1,096 days — contains the extreme 2023 & 2024 monsoon seasons)

```bash
python src/ml/train.py
```

#### Evaluation Benchmark:
| Model | Accuracy | Precision | Recall (Disasters) | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Logistic Regression** | 89.32% | 0.2083 | **90.91%** | 0.3390 | 0.9789 | 0.7683 |
| **Random Forest Classifier** | 98.18% | 0.7241 | 63.64% | 0.6774 | 0.9770 | 0.7543 |
| **HistGradientBoosting (Best)** | **98.36%** | **0.7778** | **63.64%** | **0.7000** | **0.9550** | **0.7886** |

The best model is automatically saved to `models/flash_flood_model.joblib`.

### 4. Running Inference Programmatically

```bash
python src/ml/predict.py
```

Example output:
```python
from src.ml.predict import FloodPredictor

predictor = FloodPredictor()
result = predictor.predict_single({
    "PRECTOTCORR": 126.85,
    "T2M": 21.0,
    "RH2M": 92.5,
    "WS2M": 3.5,
    "RAIN_3DAY": 302.0,
    "RAIN_7DAY": 355.0,
})

print(result["risk_level"])      # 'SEVERE'
print(result["probability_pct"]) # 99.9%
```

### 5. Launching the Interactive Web Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard includes:
- **Historical Weather Observatory:** Year & Month filtering without multi-year chart overlap.
- **Trained ML Prediction Engine:** Date-based ground-truth comparison, calibrated risk gauges, and feature importance drivers.
- **Real-Time Live Weather:** 1-click fetch from Open-Meteo for Shimla, Kullu, Mandi, Kangra, Chamba, Rampur, Solan, and Kinnaur.
- **What-If Weather Simulator:** Sliders to test custom rainfall and catchment conditions in real-time.
- **Himachal District GIS Map & NDMA Alert:** Geospatial risk grid with NDMA standard emergency operating protocols.
- **Original Heuristic Baseline:** Kept and clearly labeled as an empirical benchmark.

### 6. Launching the FastAPI REST Microservice

```bash
uvicorn src.api.main:app --reload --port 8000
```
- Interactive Swagger UI documentation: `http://127.0.0.1:8000/docs`
- Health check: `GET http://127.0.0.1:8000/health`
- Live district prediction: `GET http://127.0.0.1:8000/predict/live/Shimla`

### 7. Interactive Jupyter Demo Notebook

```bash
jupyter notebook notebooks/flash_flood_eda_and_modeling.ipynb
```

### 8. Running Automated Test Suite

```bash
python -m unittest discover -s tests
```


---

## 👥 Authors & Acknowledgments
- Developed for the **Smart India Hackathon**
- Meteorological telemetry sourced from **NASA POWER / MERRA-2 Project**
- Hydrological guidelines referenced from **India Meteorological Department (IMD)** & **Central Water Commission (CWC)**
