from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ml.predict import FloodPredictor
from src.data.live_weather import fetch_live_weather, HIMACHAL_DISTRICTS

app = FastAPI(
    title="🌊 Flash Flood Prediction & Early Warning API",
    description="Production REST API for Mountain Hydrology & Flash Flood Risk Inference in Himachal Pradesh",
    version="1.0.0"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor singleton
try:
    predictor = FloodPredictor()
except Exception as e:
    predictor = None


# --- Pydantic Data Models ---

class WeatherInput(BaseModel):
    PRECTOTCORR: float = Field(..., description="Daily Precipitation in mm (e.g. 45.0)", ge=0.0)
    T2M: float = Field(..., description="2-Meter Temperature in °C (e.g. 18.5)")
    RH2M: float = Field(..., description="Relative Humidity in % (e.g. 85.0)", ge=0.0, le=100.0)
    WS2M: float = Field(..., description="Wind Speed in m/s (e.g. 2.5)", ge=0.0)
    RAIN_3DAY: Optional[float] = Field(None, description="Prior 3-Day Cumulative Rainfall in mm", ge=0.0)
    RAIN_7DAY: Optional[float] = Field(None, description="Prior 7-Day Cumulative Rainfall in mm", ge=0.0)
    district: Optional[str] = Field("Shimla", description="District or Catchment Name")

    class Config:
        json_schema_extra = {
            "example": {
                "PRECTOTCORR": 68.5,
                "T2M": 19.2,
                "RH2M": 88.0,
                "WS2M": 3.1,
                "RAIN_3DAY": 125.0,
                "RAIN_7DAY": 180.0,
                "district": "Shimla (Sutlej Basin)"
            }
        }


class FactorDetail(BaseModel):
    feature: str
    value: float
    importance: float


class PredictionResponse(BaseModel):
    district: str
    prediction: int
    flood_probability: float
    probability_pct: float
    risk_level: str
    alert_code: str
    advisory_message: str
    top_contributing_factors: List[FactorDetail]
    model_used: str
    timestamp: str


# --- Helper Functions ---

def get_alert_code_and_message(risk_level: str, prob_pct: float, district: str):
    if risk_level == "SEVERE":
        return "RED", f"RED EVACUATION ALERT: Catastrophic flash flood and cloudburst risk ({prob_pct}%) in {district}. Immediate downstream evacuation required."
    elif risk_level == "HIGH":
        return "ORANGE", f"ORANGE WARNING: High risk of stream swelling and flash floods ({prob_pct}%) in {district}. Deploy NDRF/SDRF teams to standby."
    elif risk_level == "MODERATE":
        return "YELLOW", f"YELLOW WATCH: Moderate soil moisture saturation and flash flood potential ({prob_pct}%) in {district}. Monitor culverts and nullahs."
    else:
        return "GREEN", f"GREEN NORMAL: Normal meteorological state ({prob_pct}%) in {district}. No immediate flood threat."


# --- Endpoints ---

@app.get("/")
def root():
    return {
        "system": "Flash Flood Prediction System - Himachal Pradesh",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health",
        "model_info": "/model-info",
        "supported_districts": list(HIMACHAL_DISTRICTS.keys())
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if predictor is not None else "degraded",
        "model_loaded": predictor is not None,
        "model_name": predictor.model_name if predictor else None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model-info")
def model_info():
    if not predictor:
        raise HTTPException(status_code=503, detail="ML model is not loaded.")
    return {
        "model_name": predictor.model_name,
        "metrics": predictor.metrics,
        "training_features": predictor.feature_names,
        "feature_importances": predictor.feature_importances,
    }


@app.get("/districts")
def list_districts():
    return {
        "districts": HIMACHAL_DISTRICTS
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_flood_risk(input_data: WeatherInput):
    if not predictor:
        raise HTTPException(status_code=503, detail="ML model is not loaded.")

    data_dict = input_data.model_dump()
    district = data_dict.pop("district", "Himachal Pradesh")

    res = predictor.predict_single(data_dict)
    alert_code, advisory = get_alert_code_and_message(
        res["risk_level"], res["probability_pct"], district
    )

    return PredictionResponse(
        district=district,
        prediction=res["prediction"],
        flood_probability=res["probability"],
        probability_pct=res["probability_pct"],
        risk_level=res["risk_level"],
        alert_code=alert_code,
        advisory_message=advisory,
        top_contributing_factors=[
            FactorDetail(**f) for f in res["top_contributing_factors"]
        ],
        model_used=res["model_used"],
        timestamp=datetime.now().isoformat()
    )


@app.get("/predict/live/{district_name}", response_model=PredictionResponse)
def predict_live_district(district_name: str):
    if not predictor:
        raise HTTPException(status_code=503, detail="ML model is not loaded.")

    # Match district name flexibly
    matched_name = None
    for d in HIMACHAL_DISTRICTS:
        if district_name.lower() in d.lower():
            matched_name = d
            break

    if not matched_name:
        matched_name = list(HIMACHAL_DISTRICTS.keys())[0]

    weather = fetch_live_weather(matched_name)
    res = predictor.predict_single(weather)

    alert_code, advisory = get_alert_code_and_message(
        res["risk_level"], res["probability_pct"], matched_name
    )

    return PredictionResponse(
        district=matched_name,
        prediction=res["prediction"],
        flood_probability=res["probability"],
        probability_pct=res["probability_pct"],
        risk_level=res["risk_level"],
        alert_code=alert_code,
        advisory_message=advisory,
        top_contributing_factors=[
            FactorDetail(**f) for f in res["top_contributing_factors"]
        ],
        model_used=res["model_used"],
        timestamp=datetime.now().isoformat()
    )

