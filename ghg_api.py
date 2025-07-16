from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import requests
import folium
from folium.plugins import HeatMap
from math import radians, cos, sin, sqrt, atan2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and fire data
try:
    model_co2 = joblib.load("model_co2.pkl")
    model_no2 = joblib.load("model_no2.pkl")
    feature_order = joblib.load("feature_order.pkl")

    df_fires = pd.read_csv("fire_archive_SV-C2_635121.csv")
    df_fires = df_fires.dropna(subset=['latitude', 'longitude'])
    df_fires['confidence'] = pd.to_numeric(df_fires['confidence'], errors='coerce').fillna(60)
    df_fires['frp'] = pd.to_numeric(df_fires['frp'], errors='coerce').fillna(0)
    df_fires['brightness'] = pd.to_numeric(df_fires['brightness'], errors='coerce').fillna(0)
    df_fires = df_fires[
        (df_fires['latitude'] >= 5) & (df_fires['latitude'] <= 40) &
        (df_fires['longitude'] >= 60) & (df_fires['longitude'] <= 100)
    ].reset_index(drop=True)
except Exception as e:
    print("\u274c Error loading model or data:", e)
    raise

class LocationInput(BaseModel):
    lat: float
    lon: float

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def get_fires_near(lat, lon, radius_km=50):
    df_fires['distance'] = df_fires.apply(
        lambda row: haversine(lat, lon, row['latitude'], row['longitude']), axis=1
    )
    df_nearby = df_fires[df_fires['distance'] <= radius_km]

    return {
        "fire_count": len(df_nearby),
        "avg_frp": df_nearby["frp"].mean() if not df_nearby.empty else 0,
        "max_frp": df_nearby["frp"].max() if not df_nearby.empty else 0,
        "avg_confidence": df_nearby["confidence"].mean() if not df_nearby.empty else 0,
        "avg_brightness": df_nearby["brightness"].mean() if not df_nearby.empty else 0,
    }, df_nearby

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

def get_nearby_places(lat, lon, radius_km=10, types=["school", "hospital"], max_results_per_type=5):
    overpass_url = "http://overpass-api.de/api/interpreter"
    radius_m = radius_km * 1000
    type_queries = "".join([
        f"""
        node(around:{radius_m},{lat},{lon})["amenity"="{t}"];
        way(around:{radius_m},{lat},{lon})["amenity"="{t}"];
        relation(around:{radius_m},{lat},{lon})["amenity"="{t}"];
        """ for t in types
    ])
    query = f"""
    [out:json];
    (
        {type_queries}
    );
    out center;
    """

    headers = {"User-Agent": "ghg-alert-system"}
    try:
        response = requests.post(overpass_url, data=query, headers=headers, timeout=25)
        data = response.json()
    except Exception as e:
        print("\u274c Overpass API error:", e)
        return []

    places_by_type = {t: [] for t in types}

    for element in data.get("elements", []):
        tags = element.get("tags", {})
        place_type = tags.get("amenity")
        if place_type not in types:
            continue

        name = tags.get("name", f"{place_type.title()}")
        elat = element.get("lat") or element.get("center", {}).get("lat")
        elon = element.get("lon") or element.get("center", {}).get("lon")
        if elat is None or elon is None:
            continue

        distance = np.sqrt((lat - elat)**2 + (lon - elon)**2) * 111
        places_by_type[place_type].append({
            "name": name,
            "type": place_type,
            "lat": elat,
            "lon": elon,
            "distance_km": round(distance, 2)
        })

    results = []
    for t in types:
        sorted_places = sorted(places_by_type[t], key=lambda x: x["distance_km"])
        results.extend(sorted_places[:max_results_per_type])

    return results

def generate_disaster_map(lat, lon, fire_points, nearby_places=[]):
    m = folium.Map(location=[lat, lon], zoom_start=8)
    folium.Marker([lat, lon], tooltip="User Location", icon=folium.Icon(color='blue')).add_to(m)
    heat_data = [[row['latitude'], row['longitude']] for _, row in fire_points.iterrows()]
    if heat_data:
        HeatMap(heat_data, radius=15, blur=20, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)
    for place in nearby_places:
        folium.Marker(
            [place['lat'], place['lon']],
            tooltip=f"{place['type'].title()}: {place['name']}",
            icon=folium.Icon(color="red" if place.get("affected") else "green", icon="info-sign")
        ).add_to(m)
    map_path = "ghg_alert_map.html"
    m.save(map_path)
    return map_path

def simulate_notifications(disaster_risks):
    print("\n\ud83d\udce2 Simulated Alert Notification:")
    for name, risk in disaster_risks.items():
        if risk['status'] != "Safe":
            print(f"Sending ALERT email/SMS for {name.upper()}: {risk['status']} - {risk['reason']}")
        else:
            print(f"{name.upper()} is safe. No alert sent.")

@app.get("/")
def home():
    return {"message": "\ud83c\udf0d GHG-FuseNet API is live!"}

@app.post("/predict/")
def predict(data: LocationInput, hours: int = Query(24, ge=1, le=72)):
    weather, forecast_hourly = fetch_weather(data.lat, data.lon)
    fire, fire_points = get_fires_near(data.lat, data.lon)
    features = {**fire, **weather}
    df_input = pd.DataFrame([features])[feature_order]
    co2 = model_co2.predict(df_input)[0]
    no2 = model_no2.predict(df_input)[0]

    disaster_risks = {
        "fire_risk": {"status": "Safe", "reason": "No active fire hazards nearby."},
        "heatwave": {"status": "Safe", "reason": "Temperature levels are within a normal range."},
        "storm_warning": {"status": "Safe", "reason": "Wind speed and atmospheric pressure are stable."},
        "drought_alert": {"status": "Safe", "reason": "No signs of drought; humidity is sufficient."},
        "smog_alert": {"status": "Safe", "reason": "Air quality is currently acceptable."},
    }

    if fire["fire_count"] > 1000 or fire["avg_frp"] > 10:
        disaster_risks["fire_risk"] = {
            "status": "Alert",
            "reason": "\ud83d\udd25 Fire activity is high due to elevated fire counts and energy release (FRP)."
        }
    elif fire["fire_count"] > 300:
        disaster_risks["fire_risk"] = {
            "status": "Caution",
            "reason": "\u26a0 Moderate fire activity nearby. Stay cautious."
        }

    if weather["temperature"] > 38:
        disaster_risks["heatwave"] = {
            "status": "Alert",
            "reason": "\ud83c\udf21 Extremely high temperatures indicate a heatwave risk."
        }
    if weather["wind_speed"] > 25 and weather["pressure"] < 1000:
        disaster_risks["storm_warning"] = {
            "status": "Alert",
            "reason": "\ud83c\udf2a Strong winds and low pressure could signal storm conditions."
        }
    if weather["humidity"] < 20 and fire["fire_count"] > 200:
        disaster_risks["drought_alert"] = {
            "status": "Alert",
            "reason": "\ud83d\udeb1 Low humidity and high fire activity suggest possible drought conditions."
        }
    if co2 > 400 and no2 > 40:
        disaster_risks["smog_alert"] = {
            "status": "Alert",
            "reason": "\ud83c\udf2b Dangerous air quality from high CO\u2082 and NO\u2082. Smog alert issued."
        }

    simulate_notifications(disaster_risks)

    nearby_places = get_nearby_places(data.lat, data.lon)
    affected_places = []
    for place in nearby_places:
        is_affected = any(alert["status"] == "Alert" for alert in disaster_risks.values())
        place["affected"] = is_affected
        affected_places.append(place)

    grouped_places = {}
    for place in affected_places:
        info = {
            "name": place["name"],
            "lat": place["lat"],
            "lon": place["lon"],
            "distance_km": place["distance_km"],
            "affected": place["affected"]
        }
        grouped_places.setdefault(place["type"], []).append(info)
    for t in grouped_places:
        grouped_places[t] = sorted(grouped_places[t], key=lambda x: x["distance_km"])

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

    ghg_causes, ghg_effects, precautions = [], [], []
    if co2 > 300:
        ghg_causes.append("\ud83d\ude97 Elevated fossil fuel emissions likely in the area.")
        ghg_effects.append("\ud83c\udf21 Potential for long-term climate warming.")
        precautions.append("\ud83d\udca8 Ensure proper indoor ventilation.")
    if no2 > 30:
        ghg_causes.append("\ud83c\udfe3 Industrial activity or vehicle exhaust may be high.")
        ghg_effects.append("\ud83d\ude37 Respiratory irritation and increased asthma risk.")
        precautions.append("\ud83d\ude37 Wear masks in polluted environments.")
    precautions.append("\ud83c\udf33 Support clean energy and afforestation efforts.")

    map_path = generate_disaster_map(data.lat, data.lon, fire_points, affected_places)

    return {
        "location": {"lat": data.lat, "lon": data.lon},
        "weather": weather,
        "fire": fire,
        "co2": round(co2, 2),
        "no2": round(no2, 2),
        "alerts": {
            "co2": "\u26a0 High" if co2 > 300 else "\u2705 Safe",
            "no2": "\u26a0 Hazardous" if no2 > 30 else "\u2705 Acceptable"
        },
        "ghg_causes": ghg_causes,
        "ghg_effects": ghg_effects,
        "precautions": precautions,
        "forecast": forecast,
        "disaster_risks": disaster_risks,
        "map_url": map_path,
        "affected_nearby_places": grouped_places
    }
