from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import requests

app = FastAPI()

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Models, Feature Order, and FIRMS CSV ---
try:
    model_co2 = joblib.load("model_co2.pkl")
    model_no2 = joblib.load("model_no2.pkl")
    feature_order = joblib.load("feature_order.pkl")

    df_fires = pd.read_csv("fire_archive_SV-C2_635121.csv")
    df_fires = df_fires.dropna(subset=['latitude', 'longitude'])
    df_fires['confidence'] = pd.to_numeric(df_fires['confidence'], errors='coerce')
    df_fires['confidence'] = df_fires['confidence'].fillna(60)

    # Filter region (e.g., South Asia)
    df_fires = df_fires[
        (df_fires['latitude'] >= 5) & (df_fires['latitude'] <= 40) &
        (df_fires['longitude'] >= 60) & (df_fires['longitude'] <= 100)
    ].reset_index(drop=True)

except Exception as e:
    print("❌ Error loading model or data:", e)
    raise

# --- Request Input Schema ---
class LocationInput(BaseModel):
    lat: float
    lon: float

# --- Get Nearby Fire Stats using Haversine Optimization ---
def get_fires_near(lat, lon, radius_km=50):
    df_local = df_fires[
        (df_fires['latitude'] >= lat - 1) & (df_fires['latitude'] <= lat + 1) &
        (df_fires['longitude'] >= lon - 1) & (df_fires['longitude'] <= lon + 1)
    ].copy()

    R = 6371
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(df_local['latitude'].values)
    lon2 = np.radians(df_local['longitude'].values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = R * c

    df_local['distance'] = distances
    df_nearby = df_local[df_local['distance'] <= radius_km]

    return {
        "fire_count": len(df_nearby),
        "avg_frp": df_nearby["frp"].mean() if not df_nearby.empty else 0,
        "max_frp": df_nearby["frp"].max() if not df_nearby.empty else 0,
        "avg_confidence": df_nearby["confidence"].mean() if not df_nearby.empty else 0,
        "avg_brightness": df_nearby["brightness"].mean() if not df_nearby.empty else 0,
    }

# --- Fetch Real-Time Weather ---
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

# --- Root Health Check ---
@app.get("/")
def home():
    return {"message": "🌍 GHG-FuseNet API is live!"}

# --- Main Prediction Endpoint ---
@app.post("/predict/")
def predict(data: LocationInput):
    weather = fetch_weather(data.lat, data.lon)
    fire = get_fires_near(data.lat, data.lon)
    features = {**fire, **weather}

    df_input = pd.DataFrame([features])[feature_order]

    co2 = model_co2.predict(df_input)[0]
    no2 = model_no2.predict(df_input)[0]

    # ✅ Initialize clean section safely
    ghg_causes = []
    ghg_effects = []
    precautions = []

    # ✅ Region-aware logic based on lat/lon and pollution values
    lat, lon = data.lat, data.lon
    wind_speed = weather.get("wind_speed", 0)
    humidity = weather.get("humidity", 0)
    fire_count = fire.get("fire_count", 0)

    if 8 <= lat <= 30 and fire_count > 300:
        ghg_causes.append("🔥 Crop burning and forest fires are active in your region.")
    if co2 > 450:
        ghg_causes.append("🚗 Vehicular and industrial emissions likely contributed to elevated CO₂.")
    if no2 > 70:
        ghg_causes.append("🏭 NO₂ spike likely from transportation or nearby thermal plants.")

    if no2 > 80:
        ghg_effects.append("😷 Risk of lung inflammation and asthma in children.")
    if co2 > 500 and wind_speed < 5:
        ghg_effects.append("🌫️ Low wind may cause heat stress and trap pollutants near ground level.")

    if fire_count > 500:
        precautions.append("🚫 Avoid areas near farmland or burning zones.")
    if wind_speed < 4 and humidity > 80:
        precautions.append("🧼 Use air purifiers or natural ventilation to improve indoor air.")
    if co2 > 450 or no2 > 60:
        precautions.append("😷 Limit outdoor activity during peak pollution hours.")

    precautions.append("🌳 Support afforestation and check updates on local air quality.")

    return {
        "location": {"lat": lat, "lon": lon},
        "weather": weather,
        "fire": fire,
        "co2": round(co2, 2),
        "no2": round(no2, 2),
        "alerts": {
            "co2": "⚠️ High" if co2 > 450 else "✅ Safe",
            "no2": "⚠️ Hazardous" if no2 > 80 else "✅ Acceptable"
        },
        "ghg_causes": ghg_causes,
        "ghg_effects": ghg_effects,
        "precautions": precautions
    }
