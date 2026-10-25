
from pathlib import Path
 
import requests
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
 
from step2_risk_model import RealDataRiskModel
from predict_flood_risk import FloodRiskPredictor
 
# ============================================================
# CONFIG
# ============================================================
 
BASE_DIR = Path(__file__).resolve().parent
WEATHER_CSV = BASE_DIR / "dharali_weather_2015_2025_clean.csv"
 
DHARALI_LAT = 31.04085
DHARALI_LON = 78.78153
 
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
 
st.set_page_config(
    page_title="Dharali PWD Control Room",
    page_icon="\U0001F30A",
    layout="wide",
)
 
# ============================================================
# STYLING
# ============================================================
 
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #1a1f2b;
        border: 1px solid #2a3040;
        border-radius: 10px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .risk-banner {
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 18px;
        color: white;
    }
    .risk-banner h2 { margin: 0; font-size: 1.1rem; opacity: 0.85; }
    .risk-banner h1 { margin: 4px 0 0 0; font-size: 2.4rem; }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 6px;
        border-left: 4px solid #1565C0;
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)
 
RISK_COLOR = {
    "Low": "#2E7D32",
    "Moderate": "#F9A825",
    "High": "#EF6C00",
    "Severe": "#C62828",
}
 
MONITORING_POINTS = [
    {"name": "Gangotri", "lat": 30.9947, "lon": 78.9398, "role": "Upstream reference"},
    {"name": "Harsil", "lat": 31.0398, "lon": 78.7394, "role": "Upstream sensor point"},
    {"name": "Dharali", "lat": DHARALI_LAT, "lon": DHARALI_LON, "role": "Flood monitoring site"},
    {"name": "Bhatwari", "lat": 30.8183, "lon": 78.6186, "role": "Downstream village"},
    {"name": "Uttarkashi", "lat": 30.7306, "lon": 78.4437, "role": "District HQ"},
]
 
# ============================================================
# LOAD MODELS (cached so this only runs once per session)
# ============================================================
 
@st.cache_resource
def load_models():
    percentile_model = RealDataRiskModel(str(WEATHER_CSV))
    try:
        xgb_predictor = FloodRiskPredictor(weather_csv=WEATHER_CSV)
        xgb_ok = True
    except FileNotFoundError:
        xgb_predictor = None
        xgb_ok = False
    return percentile_model, xgb_predictor, xgb_ok
 
 
@st.cache_data(ttl=1800)
def fetch_live_weather():
    """Live current + 7-day forecast weather from Open-Meteo. No API key needed."""
    params = {
        "latitude": DHARALI_LAT,
        "longitude": DHARALI_LON,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "daily": "precipitation_sum,soil_moisture_0_to_10cm_mean",
        "timezone": "Asia/Kolkata",
        "forecast_days": 7,
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()
 
 
def ensemble_score(pct_score, xgb_prob_pct):
    """Simple, transparent ensemble: average of the two model scores (0-100 scale)."""
    if xgb_prob_pct is None:
        combined = pct_score
    else:
        combined = (pct_score + xgb_prob_pct) / 2
    if combined >= 90:
        label = "Severe"
    elif combined >= 75:
        label = "High"
    elif combined >= 50:
        label = "Moderate"
    else:
        label = "Low"
    return combined, label
 
 
@st.cache_data
def build_combined_history(_percentile_model, _xgb_predictor, xgb_ok):
    history = _percentile_model.valid.copy()
    scores, labels = [], []
    for _, r in history.iterrows():
        s, l = _percentile_model.score(r["rolling_rainfall_sum"], r["rolling_soil_avg"])
        scores.append(s)
        labels.append(l)
    history["percentile_score"] = scores
    history["percentile_risk"] = labels
 
    if xgb_ok:
        xgb_preds = _xgb_predictor.top_risk_days(n=len(_xgb_predictor.featured_weather))
        xgb_preds = xgb_preds.rename(columns={
            "xgb_probability_percent": "xgboost_score",
            "xgb_risk": "xgboost_risk",
        })
        merged = history.merge(xgb_preds[["date", "xgboost_score", "xgboost_risk"]], on="date", how="left")
    else:
        merged = history.copy()
        merged["xgboost_score"] = np.nan
        merged["xgboost_risk"] = "N/A"
 
    ens_scores, ens_labels = [], []
    for _, r in merged.iterrows():
        xgb_val = r["xgboost_score"] if xgb_ok and pd.notna(r["xgboost_score"]) else None
        s, l = ensemble_score(r["percentile_score"], xgb_val)
        ens_scores.append(s)
        ens_labels.append(l)
    merged["ensemble_score"] = ens_scores
    merged["ensemble_risk"] = ens_labels
 
    return merged[[
        "date", "rainfall_mm", "soil_moisture",
        "percentile_score", "percentile_risk",
        "xgboost_score", "xgboost_risk",
        "ensemble_score", "ensemble_risk",
    ]].sort_values("date", ascending=False).reset_index(drop=True)
 
 
# ============================================================
# LOAD EVERYTHING
# ============================================================
 
percentile_model, xgb_predictor, xgb_ok = load_models()
combined = build_combined_history(percentile_model, xgb_predictor, xgb_ok)
 
# ============================================================
# HEADER
# ============================================================
 
st.title("\U0001F30A Dharali Flood Monitoring")
st.caption("PWD / Control Room Dashboard \u2014 Team Doon ka Dum, SIH26192 \u00b7 Bhagirathi / Kheer Ganga stretch")
 
if not xgb_ok:
    st.info(
        "XGBoost model not found. Run train_xgboost.py first to enable the ensemble score. "
        "Showing percentile-model-only results for now."
    )
 
# ============================================================
# LIVE WEATHER + CURRENT RISK
# ============================================================
 
try:
    live = fetch_live_weather()
    live_ok = True
except Exception:
    live = None
    live_ok = False
 
pct_score = pct_label = xgb_prob = final_score = final_label = None
 
if live_ok:
    current = live["current"]
    daily = live["daily"]
 
    recent_rain = daily["precipitation_sum"][:3]
    recent_soil = [s if s is not None else 0.5 for s in daily["soil_moisture_0_to_10cm_mean"][:3]]
 
    pct_score, pct_label = percentile_model.score(sum(recent_rain), np.mean(recent_soil))
 
    if xgb_ok:
        try:
            rain7 = (daily["precipitation_sum"] + [0] * 7)[:7]
            soil7 = [s if s is not None else 0.5 for s in (daily["soil_moisture_0_to_10cm_mean"] + [0.5] * 7)[:7]]
            result = xgb_predictor.predict_manual(rainfall_last_7_days=rain7, soil_moisture_last_7_days=soil7)
            xgb_prob = result["probability_percent"]
        except Exception:
            xgb_prob = None
 
    final_score, final_label = ensemble_score(pct_score, xgb_prob)
 
banner_color = RISK_COLOR.get(final_label, "#455A64") if final_label else "#455A64"
banner_text = f"{final_label} \u2014 {final_score:.0f}/100" if final_label else "Live weather unavailable"
st.markdown(
    f'<div class="risk-banner" style="background-color:{banner_color};">'
    f'<h2>CURRENT DHARALI RISK LEVEL</h2><h1>{banner_text}</h1></div>',
    unsafe_allow_html=True,
)
 
if not live_ok:
    st.warning(
        "Could not reach the live weather API (Open-Meteo) right now \u2014 check your internet "
        "connection. All other sections (map, historical data) still work normally."
    )
 
st.markdown('<div class="section-header">Model Breakdown</div>', unsafe_allow_html=True)
b1, b2, b3 = st.columns(3)
b1.metric("Percentile Model Score", f"{pct_score:.1f}" if pct_score is not None else "\u2014", pct_label or "")
b2.metric("XGBoost Probability", f"{xgb_prob:.1f}%" if xgb_prob is not None else "N/A")
b3.metric("Ensemble (Final)", f"{final_score:.1f}" if final_score is not None else "\u2014", final_label or "")
 
if live_ok:
    st.markdown('<div class="section-header">Live Current Weather</div>', unsafe_allow_html=True)
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Temperature", f"{current.get('temperature_2m', '\u2014')} \u00b0C")
    w2.metric("Humidity", f"{current.get('relative_humidity_2m', '\u2014')} %")
    w3.metric("Current Rain", f"{current.get('precipitation', '\u2014')} mm")
    w4.metric("Wind", f"{current.get('wind_speed_10m', '\u2014')} km/h")
    st.caption(f"Live data from Open-Meteo, last updated: {current.get('time', '\u2014')}")
 
st.markdown("---")
 
# ============================================================
# SATELLITE MAP
# ============================================================
 
st.markdown('<div class="section-header">\U0001F6F0 Satellite View \u2014 Monitored River Stretch</div>', unsafe_allow_html=True)
 
fmap = folium.Map(location=[30.95, 78.75], zoom_start=10, tiles="Esri.WorldImagery")
 
for point in MONITORING_POINTS:
    if point["name"] == "Dharali" and final_label:
        color = RISK_COLOR.get(final_label, "#3388ff")
        popup = (
            f"<b>Dharali</b><br>Current Risk: {final_label}<br>"
            f"Final Score: {final_score:.1f}<br>"
            f"XGBoost: {f'{xgb_prob:.1f}%' if xgb_prob is not None else 'N/A'}"
        )
    else:
        color = "#3388ff"
        popup = f"<b>{point['name']}</b><br>{point['role']}"
 
    folium.CircleMarker(
        location=[point["lat"], point["lon"]],
        radius=11,
        popup=folium.Popup(popup, max_width=280),
        tooltip=point["name"],
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        weight=2,
    ).add_to(fmap)
 
folium.PolyLine(
    locations=[[p["lat"], p["lon"]] for p in MONITORING_POINTS],
    color="#1565C0",
    weight=3,
    opacity=0.7,
    tooltip="River flow direction (upstream to downstream)",
).add_to(fmap)
 
st_folium(fmap, width=None, height=480, returned_objects=[])
st.caption("Satellite basemap: Esri World Imagery. Marker color reflects the live ensemble risk level at Dharali.")
 
st.markdown("---")
 
# ============================================================
# 7-DAY FORECAST
# ============================================================
 
st.markdown('<div class="section-header">\U0001F52E 7-Day Flood Risk Forecast</div>', unsafe_allow_html=True)
 
if live_ok:
    try:
        f_dates = daily["time"]
        f_rain = daily["precipitation_sum"]
        f_soil = [s if s is not None else 0.5 for s in daily["soil_moisture_0_to_10cm_mean"]]
 
        rows = []
        for i in range(len(f_dates)):
            window_rain = f_rain[max(0, i - 2):i + 1]
            window_soil = f_soil[max(0, i - 2):i + 1]
            if len(window_rain) < 1:
                continue
            p_score, p_label = percentile_model.score(sum(window_rain), np.mean(window_soil))
            rows.append({
                "Date": f_dates[i],
                "Rainfall (mm)": round(f_rain[i], 1),
                "Soil Moisture": round(f_soil[i], 3),
                "Percentile Score": round(p_score, 1),
                "Percentile Risk": p_label,
            })
        forecast_df = pd.DataFrame(rows)
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
 
        high_risk_days = forecast_df[forecast_df["Percentile Risk"].isin(["High", "Severe"])]
        if not high_risk_days.empty:
            worst = forecast_df.sort_values("Percentile Score", ascending=False).iloc[0]
            st.warning(f"\u26a0\ufe0f Highest forecast risk: {worst['Date']} \u2014 {worst['Percentile Risk']} ({worst['Percentile Score']:.1f})")
        else:
            st.success("No high-risk days in the 7-day forecast.")
    except Exception as e:
        st.error(f"Could not build forecast table: {e}")
else:
    st.info("Forecast requires a live connection to Open-Meteo.")
 
st.markdown("---")
 
# ============================================================
# HISTORICAL PREDICTIONS
# ============================================================
 
st.markdown('<div class="section-header">\U0001F4CA Historical Flood Risk \u2014 All Dates, Both Models</div>', unsafe_allow_html=True)
 
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    date_range = st.date_input(
        "Filter by date range",
        value=(combined["date"].min().date(), combined["date"].max().date()),
        min_value=combined["date"].min().date(),
        max_value=combined["date"].max().date(),
    )
with col2:
    risk_filter = st.multiselect("Filter by ensemble risk", options=["Low", "Moderate", "High", "Severe"], default=[])
with col3:
    st.write("")
    show_high_only = st.checkbox("Show only High/Severe rows")
 
filtered = combined.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[(filtered["date"].dt.date >= start) & (filtered["date"].dt.date <= end)]
if risk_filter:
    filtered = filtered[filtered["ensemble_risk"].isin(risk_filter)]
if show_high_only:
    filtered = filtered[filtered["ensemble_risk"].isin(["High", "Severe"])]
 
display_df = filtered.copy()
display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
for c in ["percentile_score", "xgboost_score", "ensemble_score"]:
    display_df[c] = display_df[c].round(1)
 
st.dataframe(
    display_df.rename(columns={
        "date": "Date", "rainfall_mm": "Rainfall (mm)", "soil_moisture": "Soil Moisture",
        "percentile_score": "Percentile Score", "percentile_risk": "Percentile Risk",
        "xgboost_score": "XGBoost Score (%)", "xgboost_risk": "XGBoost Risk",
        "ensemble_score": "Ensemble Score", "ensemble_risk": "Ensemble Risk",
    }),
    use_container_width=True, height=420, hide_index=True,
)
st.caption(f"Showing {len(filtered)} of {len(combined)} total days.")
 
csv_bytes = display_df.to_csv(index=False).encode("utf-8")
st.download_button("Download this table as CSV", data=csv_bytes, file_name="flood_risk_predictions.csv", mime="text/csv")
 
st.markdown("---")
st.caption(
    "Percentile model: calibrated on real 2015-2025 NASA POWER rainfall/soil-moisture data for "
    "Dharali. XGBoost model: trained on the same data plus documented historical flash-flood "
    "events for Uttarkashi district (illustrative \u2014 small number of real labeled events). "
    "Live weather: Open-Meteo, free API, updated every 30 minutes."
)