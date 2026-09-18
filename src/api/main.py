from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ml.predict import FloodPredictor
from src.utils.config import (
    FORECAST_HORIZON_DAYS,
    MODE_HISTORICAL,
    MODE_REALTIME,
    MODE_FORECAST,
    MODE_UNAVAILABLE,
    LABEL_HISTORICAL,
    LABEL_REALTIME,
    LABEL_FORECAST_DISCLAIMER,
    MSG_BEYOND_HORIZON,
)
from src.data.live_weather import (
    fetch_live_weather,
    fetch_forecast_weather,
    fetch_weather_for_date_and_mode,
    route_prediction_date,
    HIMACHAL_DISTRICTS,
)

SYSTEM_TODAY = date(2026, 9, 19)

app = FastAPI(
    title="🌊 Flash Flood Prediction & Early Warning API",
    description="Production REST API for Mountain Hydrology & Flash Flood Risk Inference in Himachal Pradesh with 2026 Real-Time & Forecast Horizon",
    version="2.0.0"
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
    RAIN_14DAY: Optional[float] = Field(None, description="Prior 14-Day Cumulative Rainfall in mm", ge=0.0)
    district: Optional[str] = Field("Shimla (Sutlej Basin)", description="District or Catchment Name")

    class Config:
        json_schema_extra = {
            "example": {
                "PRECTOTCORR": 68.5,
                "T2M": 19.2,
                "RH2M": 88.0,
                "WS2M": 3.1,
                "RAIN_3DAY": 125.0,
                "RAIN_7DAY": 180.0,
                "RAIN_14DAY": 230.0,
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
    mode: Optional[str] = None
    target_date: Optional[str] = None
    disclaimer: Optional[str] = None
    forecast_days_ahead: Optional[int] = None


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
        "version": "2.0.0",
        "system_date": SYSTEM_TODAY.isoformat(),
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "docs_url": "/docs",
        "health_check": "/health",
        "model_info": "/model-info",
        "forecast_horizon_info": "/forecast-horizon",
        "supported_districts": list(HIMACHAL_DISTRICTS.keys()),
        "endpoints": {
            "predict_custom": "POST /predict",
            "predict_live": "GET /predict/live/{district_name}",
            "predict_forecast": "GET /predict/forecast/{district_name}?target_date=YYYY-MM-DD",
            "predict_routed_date": "GET /predict/date/{district_name}?target_date=YYYY-MM-DD"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if predictor is not None else "degraded",
        "model_loaded": predictor is not None,
        "model_name": predictor.model_name if predictor else None,
        "system_date": SYSTEM_TODAY.isoformat(),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/forecast-horizon")
def get_forecast_horizon():
    """Returns the maximum supported NWP weather forecast horizon and boundary constraints."""
    max_d = SYSTEM_TODAY + timedelta(days=FORECAST_HORIZON_DAYS)
    return {
        "system_reference_date": SYSTEM_TODAY.isoformat(),
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "supported_start_date": (SYSTEM_TODAY + timedelta(days=1)).isoformat(),
        "supported_end_date": max_d.isoformat(),
        "disclaimer": LABEL_FORECAST_DISCLAIMER,
        "beyond_horizon_message": MSG_BEYOND_HORIZON
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
        timestamp=datetime.now().isoformat(),
        mode="Custom Simulation",
        target_date=SYSTEM_TODAY.isoformat()
    )


@app.get("/predict/live/{district_name}", response_model=PredictionResponse)
def predict_live_district(district_name: str):
    """Mode 2: Real-Time Risk inference based on currently available weather observations."""
    if not predictor:
        raise HTTPException(status_code=503, detail="ML model is not loaded.")

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
        timestamp=datetime.now().isoformat(),
        mode=MODE_REALTIME,
        target_date=SYSTEM_TODAY.isoformat(),
        disclaimer=LABEL_REALTIME
    )


@app.get("/predict/forecast/{district_name}", response_model=PredictionResponse)
def predict_forecast_district(
    district_name: str,
    target_date: str = Query(..., description="Target forecast date in ISO format (YYYY-MM-DD)")
):
    """Mode 3: Short-Term Forecast Risk strictly restricted to 7-day NWP horizon."""
    if not predictor:
        raise HTTPException(status_code=503, detail="ML model is not loaded.")

    matched_name = None
    for d in HIMACHAL_DISTRICTS:
        if district_name.lower() in d.lower():
            matched_name = d
            break
    if not matched_name:
        matched_name = list(HIMACHAL_DISTRICTS.keys())[0]

    try:
        tgt_d = datetime.fromisoformat(target_date).date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")

    status_code, mode_name, meta = route_prediction_date(tgt_d, reference_date=SYSTEM_TODAY)

    if status_code == "UNAVAILABLE":
        raise HTTPException(status_code=400, detail=MSG_BEYOND_HORIZON)
    elif status_code == "HISTORICAL":
        raise HTTPException(
            status_code=400,
            detail=f"Target date {target_date} is in the past. Use /predict/date/{district_name} for historical analysis."
        )

    weather = fetch_forecast_weather(matched_name, tgt_d, reference_date=SYSTEM_TODAY)
    res = predictor.predict_single(weather)

    days_ahead = (tgt_d - SYSTEM_TODAY).days
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
        timestamp=datetime.now().isoformat(),
        mode=MODE_FORECAST,
        target_date=tgt_d.isoformat(),
        disclaimer=LABEL_FORECAST_DISCLAIMER,
        forecast_days_ahead=days_ahead
    )


@app.get("/predict/date/{district_name}", response_model=PredictionResponse)
def predict_routed_date(
    district_name: str,
    target_date: str = Query(..., description="Target date in ISO format (YYYY-MM-DD)")
):
    """
    Unified date router endpoint:
      - target_date < today -> Historical Analysis
      - target_date == today -> Real-Time Risk
      - today < target_date <= today + 7 -> Short-Term Forecast Risk
      - target_date > today + 7 -> Rejection (HTTP 400 with beyond-horizon notice)
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="ML model is not loaded.")

    matched_name = None
    for d in HIMACHAL_DISTRICTS:
        if district_name.lower() in d.lower():
            matched_name = d
            break
    if not matched_name:
        matched_name = list(HIMACHAL_DISTRICTS.keys())[0]

    try:
        tgt_d = datetime.fromisoformat(target_date).date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")

    weather = fetch_weather_for_date_and_mode(matched_name, tgt_d, reference_date=SYSTEM_TODAY)

    if not weather.get("is_allowed", True) or weather.get("status_code") == "UNAVAILABLE":
        raise HTTPException(status_code=400, detail=MSG_BEYOND_HORIZON)

    res = predictor.predict_single(weather)
    alert_code, advisory = get_alert_code_and_message(
        res["risk_level"], res["probability_pct"], matched_name
    )

    days_ahead = (tgt_d - SYSTEM_TODAY).days if tgt_d > SYSTEM_TODAY else 0

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
        timestamp=datetime.now().isoformat(),
        mode=weather.get("mode", "Automated Routed Prediction"),
        target_date=tgt_d.isoformat(),
        disclaimer=weather.get("disclaimer", weather.get("label", None)),
        forecast_days_ahead=days_ahead if days_ahead > 0 else None
    )

