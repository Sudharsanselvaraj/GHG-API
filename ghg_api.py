from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import requests
import random

app = FastAPI()

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Models and Feature Order ---
try:
    model_co2 = joblib.load("model_co2.pkl")
    model_no2 = joblib.load("model_no2.pkl")
    feature_order = joblib.load("feature_order.pkl")
except Exception as e:
    print("⚠️ Error loading models or data:", e)
    raise

# --- Input Schema ---
class LocationInput(BaseModel):
    lat: float
    lon: float

# --- Simulated Fire Feature Generator ---
def get_fires_near(lat, lon):
    return {
        "fire_count": random.randint(0, 10),
        "avg_frp": round(random.uniform(0, 80), 2),
        "max_frp": round(random.uniform(0, 120), 2),
        "avg_confidence": round(random.uniform(50, 95), 2),
        "avg_brightness": round(random.uniform(280, 330), 2)
    }

# --- Weather API Call ---
def fetch_weather(lat, lon):
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m"
            f"&current_weather=true&timezone=auto"
        )
        r = requests.get(url).json()
        current = r.get("current_weather", {})
        hourly = r.get("hourly", {})
        return {
            "temperature": current.get("temperature", 0),
            "wind_speed": current.get("windspeed", 0),
            "pressure": hourly.get("pressure_msl", [0])[0],
            "humidity": hourly.get("relative_humidity_2m", [0])[0]
        }
    except Exception:
        return {"temperature": 0, "wind_speed": 0, "pressure": 0, "humidity": 0}

# --- Root Health Endpoint ---
@app.get("/")
def home():
    return {"message": "🌍 GHG-FuseNet API is live!"}

# --- Predict Endpoint ---
@app.post("/predict/")
def predict(data: LocationInput):
    weather = fetch_weather(data.lat, data.lon)
    fire = get_fires_near(data.lat, data.lon)
    features = {**fire, **weather}
    df_input = pd.DataFrame([features])[feature_order]

    co2 = model_co2.predict(df_input)[0]
    no2 = model_no2.predict(df_input)[0]

    return {
        "location": {"lat": data.lat, "lon": data.lon},
        "weather": weather,
        "fire": fire,
        "co2": round(co2, 2),
        "no2": round(no2, 2),
        "alerts": {
            "co2": "⚠️ High" if co2 > 450 else "✅ Safe",
            "no2": "⚠️ Hazardous" if no2 > 80 else "✅ Acceptable"
        }
    }
