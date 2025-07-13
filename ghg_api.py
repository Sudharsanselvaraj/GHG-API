from fastapi import FastAPI, Query
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

    df_fires = df_fires[
        (df_fires['latitude'] >= 5) & (df_fires['latitude'] <= 40) &
        (df_fires['longitude'] >= 60) & (df_fires['longitude'] <= 100)
    ].reset_index(drop=True)

except Exception as e:
    print("❌ Error loading model or data:", e)
    raise

class LocationInput(BaseModel):
    lat: float
    lon: float

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

@app.get("/")
def home():
    return {"message": "🌍 GHG-FuseNet API is live!"}

@app.post("/predict/")
def predict(data: LocationInput, hours: int = Query(24, ge=1, le=72)):
    weather, forecast_hourly = fetch_weather(data.lat, data.lon)
    fire = get_fires_near(data.lat, data.lon)
    features = {**fire, **weather}

    df_input = pd.DataFrame([features])[feature_order]
    co2 = model_co2.predict(df_input)[0]
    no2 = model_no2.predict(df_input)[0]

    # Alerts and explanations
    disaster_risks = {}

    if fire["fire_count"] > 1000 or fire["avg_frp"] > 10:
        disaster_risks["fire_risk"] = "🔥 Fire activity is high due to elevated fire counts and energy release (FRP)."
    elif fire["fire_count"] > 300:
        disaster_risks["fire_risk"] = "⚠️ Moderate fire activity nearby. Stay cautious."

    if weather["temperature"] > 38:
        disaster_risks["heatwave"] = "🌡 Extremely high temperatures indicate a heatwave risk."

    if weather["wind_speed"] > 25 and weather["pressure"] < 1000:
        disaster_risks["storm_warning"] = "🌪 Strong winds and low pressure could signal storm conditions."

    if weather["humidity"] < 20 and fire["fire_count"] > 200:
        disaster_risks["drought_alert"] = "🚱 Low humidity and high fire activity suggest possible drought conditions."

    if co2 > 400 and no2 > 40:
        disaster_risks["smog_alert"] = "🌫 Dangerous air quality from high CO₂ and NO₂. Smog alert issued."

    # Logic based on CO2 and NO2 only
    ghg_causes = []
    if co2 > 300:
        ghg_causes.append("🚗 Elevated fossil fuel emissions likely in the area.")
    if no2 > 30:
        ghg_causes.append("🏭 Industrial activity or vehicle exhaust may be high.")

    ghg_effects = []
    if co2 > 300:
        ghg_effects.append("🌡 Potential for long-term climate warming.")
    if no2 > 30:
        ghg_effects.append("😷 Respiratory irritation and increased asthma risk.")

    precautions = []
    if co2 > 300:
        precautions.append("💨 Ensure proper indoor ventilation.")
    if no2 > 30:
        precautions.append("😷 Wear masks in polluted environments.")
    precautions.append("🌳 Support clean energy and afforestation efforts.")

    # Batch prediction for performance
    forecast = []
    if forecast_hourly:
        times = forecast_hourly.get("time", [])
        temp = forecast_hourly.get("temperature_2m", [])
        wind = forecast_hourly.get("wind_speed_10m", [])
        hum = forecast_hourly.get("relative_humidity_2m", [])
        pres = forecast_hourly.get("pressure_msl", [])

        limit = min(hours, len(times))
        batch_data = pd.DataFrame([{
            "temperature": temp[i],
            "wind_speed": wind[i],
            "pressure": pres[i],
            "humidity": hum[i],
            **fire
        } for i in range(limit)])[feature_order]

        pred_co2 = model_co2.predict(batch_data)
        pred_no2 = model_no2.predict(batch_data)

        forecast = [{
            "timestamp": times[i],
            "temperature": round(temp[i], 2),
            "co2": round(pred_co2[i], 2),
            "no2": round(pred_no2[i], 2)
        } for i in range(limit)]

    return {
        "location": {"lat": data.lat, "lon": data.lon},
        "weather": weather,
        "fire": fire,
        "co2": round(co2, 2),
        "no2": round(no2, 2),
        "alerts": {
            "co2": "⚠️ High" if co2 > 300 else "✅ Safe",
            "no2": "⚠️ Hazardous" if no2 > 30 else "✅ Acceptable"
        },
        "ghg_causes": ghg_causes,
        "ghg_effects": ghg_effects,
        "precautions": precautions,
        "forecast": forecast,
        "disaster_risks": disaster_risks
    }
