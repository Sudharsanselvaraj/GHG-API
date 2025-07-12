from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
from geopy.distance import geodesic

app = FastAPI()

# Load models and files
model_co2 = joblib.load("model_co2.pkl")
model_no2 = joblib.load("model_no2.pkl")
feature_order = joblib.load("feature_order.pkl")
df_fires = pd.read_csv("fire_archive_SV-C2_635121.csv")

df_fires['confidence'] = pd.to_numeric(df_fires['confidence'], errors='coerce')
df_fires['confidence'].fillna(60, inplace=True)

class LocationInput(BaseModel):
    lat: float
    lon: float

def fetch_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m&current_weather=true&timezone=auto"
    r = requests.get(url).json()
    current = r.get("current_weather", {})
    hourly = r.get("hourly", {})
    return {
        "temperature": current.get("temperature", 0),
        "wind_speed": current.get("windspeed", 0),
        "pressure": hourly.get("pressure_msl", [0])[0],
        "humidity": hourly.get("relative_humidity_2m", [0])[0]
    }

def get_fires_near(lat, lon, radius_km=50):
    point = (lat, lon)
    nearby = df_fires[df_fires.apply(
        lambda row: geodesic(point, (row['latitude'], row['longitude'])).km <= radius_km,
        axis=1
    )]
    return {
        "fire_count": len(nearby),
        "avg_frp": nearby["frp"].mean() if not nearby.empty else 0,
        "max_frp": nearby["frp"].max() if not nearby.empty else 0,
        "avg_confidence": nearby["confidence"].mean() if not nearby.empty else 0,
        "avg_brightness": nearby["brightness"].mean() if not nearby.empty else 0,
    }

@app.get("/")
def home():
    return {"message": "🌍 GHG-FuseNet API is live!"}

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
