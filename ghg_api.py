from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64

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
        }, hourly
    except Exception:
        return {"temperature": 0, "wind_speed": 0, "pressure": 0, "humidity": 0}, {}

# --- Root Health Check ---
@app.get("/")
def home():
    return {"message": "🌍 GHG-FuseNet API is live!"}

# --- Main Prediction Endpoint ---
@app.post("/predict/")
def predict(data: LocationInput, hours: int = Query(24, ge=1, le=72)):
    weather, forecast_hourly = fetch_weather(data.lat, data.lon)
    fire = get_fires_near(data.lat, data.lon)
    features = {**fire, **weather}

    df_input = pd.DataFrame([features])[feature_order]
    co2 = model_co2.predict(df_input)[0]
    no2 = model_no2.predict(df_input)[0]

    # --- GHG Cause & Precaution Logic ---
    ghg_causes = []
    ghg_effects = []
    precautions = []

    lat, lon = data.lat, data.lon
    wind_speed = weather.get("wind_speed", 0)
    humidity = weather.get("humidity", 0)
    fire_count = fire.get("fire_count", 0)

    if 8 <= lat <= 30 and fire_count > 300:
        ghg_causes.append("🔥 Crop burning and forest fires are active in your region.")
    if co2 > 450:
        ghg_causes.append("🚗 Fossil fuel combustion and regional fire hotspots")
    if fire_count > 500:
        ghg_causes.append("🔥 Large-scale biomass burning detected nearby")

    if no2 > 80:
        ghg_effects.append("😷 High respiratory risk: asthma, lung inflammation")
    elif co2 > 450:
        ghg_effects.append("😓 Fatigue and reduced concentration in vulnerable groups")

    if co2 > 450:
        precautions.append("✅ Stay hydrated and ventilate indoor spaces")
    if fire_count > 500:
        precautions.append("🚫 Avoid any open waste or crop burning activities")

    precautions.append("🌳 Support afforestation and monitor alerts regularly")

    # --- Forecasting with Plot ---
    forecast = []
    base64_plot = None

    if forecast_hourly:
        times = forecast_hourly.get("time", [])
        temp = forecast_hourly.get("temperature_2m", [])
        wind = forecast_hourly.get("wind_speed_10m", [])
        hum = forecast_hourly.get("relative_humidity_2m", [])
        pres = forecast_hourly.get("pressure_msl", [])

        limit = min(hours, len(times))
        timestamps, co2_preds, no2_preds = [], [], []

        for i in range(limit):
            feature_forecast = {
                "temperature": temp[i],
                "wind_speed": wind[i],
                "pressure": pres[i],
                "humidity": hum[i],
                **fire
            }
            df_f = pd.DataFrame([feature_forecast])[feature_order]
            pred_co2 = model_co2.predict(df_f)[0]
            pred_no2 = model_no2.predict(df_f)[0]

            forecast.append({
                "timestamp": times[i],
                "co2": round(pred_co2, 2),
                "no2": round(pred_no2, 2)
            })
            timestamps.append(times[i])
            co2_preds.append(pred_co2)
            no2_preds.append(pred_no2)

        # 📊 Generate Base64 Plot
        plt.figure(figsize=(12, 5))
        plt.plot(timestamps, co2_preds, label='CO₂ (ppm)', color='green', marker='o')
        plt.plot(timestamps, no2_preds, label='NO₂ (ppb)', color='red', marker='x')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Concentration")
        plt.title("Forecasted CO₂ and NO₂ Levels")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        base64_plot = base64.b64encode(buf.read()).decode('utf-8')

    return {
        "location": {"lat": lat, "lon": lon},
        "weather": weather,
        "fire": fire,
        "co2": round(co2, 2),
        "no2": round(no2, 2),
        "alerts": {
            "co2": "⚠️ High" if co2 > 350 else "✅ Safe",
            "no2": "⚠️ Hazardous" if no2 > 80 else "✅ Acceptable"
        },
        "ghg_causes": ghg_causes,
        "ghg_effects": ghg_effects,
        "precautions": precautions,
        "forecast": forecast,
        "forecast_plot_base64": base64_plot
    }
