import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import joblib

from datetime import datetime
from pathlib import Path
from scipy import stats

# ============================================================
# DHARALI FLOOD RISK DASHBOARD
# ============================================================
# Single-file Streamlit application.
#
# It combines:
#   1. Historical NASA POWER-derived weather CSV
#   2. Existing percentile/statistical risk model
#   3. Optional trained XGBoost flood classifier
#   4. Live/recent Open-Meteo weather
#   5. 7-day weather forecast
#   6. Forecast flood-risk projection
#   7. Historical date analysis
#   8. Model performance (when XGBoost has been trained)
#
# Put this file in the same folder as:
#
#   app.py
#   dharali_weather_2015_2025_clean.csv
#   step2_risk_model.py
#   xgboost_flood_model.py
#   event_labels.csv                         (optional)
#   xgboost_model/xgboost_flood_model.pkl   (optional)
#
# ============================================================


# ============================================================
# OPTIONAL XGBOOST IMPORT
# ============================================================

try:
    from xgboost_flood_model import (
        FEATURES as XGB_FEATURES,
        load_weather as xgb_load_weather,
        add_features as xgb_add_features,
        historical_predictions as xgb_historical_predictions,
        probability_to_risk as xgb_probability_to_risk,
    )
    XGB_MODULE_AVAILABLE = True

except Exception:
    XGB_MODULE_AVAILABLE = False

    XGB_FEATURES = [
        "rain_1d",
        "rain_3d",
        "rain_7d",
        "rain_max_3d",
        "rain_max_7d",
        "rain_mean_3d",
        "rain_std_3d",
        "soil_1d",
        "soil_3d",
        "soil_7d",
        "soil_change_3d",
        "rain_change_3d",
        "rain_percentile",
        "soil_percentile",
    ]


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR /
    "dharali_weather_2015_2025_clean.csv"
)

XGB_MODEL_FILE = (
    BASE_DIR /
    "xgboost_model" /
    "xgboost_flood_model.pkl"
)

XGB_METADATA_FILE = (
    BASE_DIR /
    "xgboost_model" /
    "model_metadata.json"
)

EVENT_LABEL_FILE = (
    BASE_DIR /
    "event_labels.csv"
)

HISTORICAL_XGB_OUTPUT = (
    BASE_DIR /
    "historical_xgboost_predictions.csv"
)

TIMEZONE = "Asia/Kolkata"

DHARALI_LATITUDE = 31.04085
DHARALI_LONGITUDE = 78.78153

OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

LOOKBACK_DAYS = 3

FORECAST_DAYS = 7

REQUEST_TIMEOUT = 30

# FastAPI backend is the single source of truth for live/current/forecast risk.
BACKEND_URL = "http://127.0.0.1:8000"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dharali Flood Risk Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .risk-container {
        padding: 28px;
        border-radius: 18px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
        border: 1px solid #dddddd;
    }

    .risk-level {
        font-size: 40px;
        font-weight: 800;
    }

    .risk-score {
        font-size: 21px;
        margin-top: 8px;
    }

    .small-note {
        font-size: 13px;
        color: #666666;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌊 Dharali Flood Risk Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Historical + Live Weather + XGBoost + Forecast Flood-Risk System
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def risk_icon(risk):
    if risk == "Severe":
        return "🔴"

    if risk == "High":
        return "🟠"

    if risk == "Moderate":
        return "🟡"

    return "🟢"


def risk_background(risk):

    if risk == "Severe":
        return "#ffd6d6"

    if risk == "High":
        return "#ffe5cc"

    if risk == "Moderate":
        return "#fff4bf"

    return "#d9ffd9"


def risk_description(risk):

    if risk == "Severe":
        return (
            "Weather conditions are extremely elevated "
            "relative to the historical Dharali reference."
        )

    if risk == "High":
        return (
            "Weather conditions are significantly elevated "
            "relative to historical conditions."
        )

    if risk == "Moderate":
        return (
            "Conditions are above the normal historical range "
            "and should be monitored."
        )

    return (
        "Conditions are within a relatively normal "
        "historical range."
    )


def display_risk(score, risk, title="CURRENT FLOOD RISK"):

    icon = risk_icon(risk)

    background = risk_background(risk)

    st.markdown(
        f"""
        <div class="risk-container"
             style="background-color:{background};">

            <div class="risk-level">
                {icon} {risk}
            </div>

            <div class="risk-score">
                {title}
            </div>

            <div class="risk-score">
                Statistical Risk:
                <strong>{score:.2f}</strong> percentile
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        risk_description(risk)
    )


def percentile_to_risk(score):

    if score >= 95:
        return "Severe"

    if score >= 90:
        return "High"

    if score >= 75:
        return "Moderate"

    return "Low"


def combined_probability_risk(probability):

    if probability >= 0.75:
        return "Severe"

    if probability >= 0.50:
        return "High"

    if probability >= 0.25:
        return "Moderate"

    return "Low"


# ============================================================
# FASTAPI BACKEND
# ============================================================

def backend_get(endpoint, timeout=REQUEST_TIMEOUT):
    response = requests.get(
        f"{BACKEND_URL}{endpoint}",
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def get_backend_health():
    return backend_get("/api/health")


def get_backend_live():
    return backend_get("/api/live")


def get_backend_current_risk():
    return backend_get("/api/current-risk")


def get_backend_forecast():
    return backend_get("/api/forecast")


def get_backend_historical():
    return backend_get("/api/historical")


def get_backend_model_status():
    return backend_get("/api/model-status")


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

@st.cache_data
def load_historical_data():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Could not find:\n{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    required = {
        "date",
        "rainfall_mm",
        "soil_moisture",
    }

    missing = required - set(df.columns)

    if missing:

        raise ValueError(
            f"Historical CSV is missing: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    df["rainfall_mm"] = pd.to_numeric(
        df["rainfall_mm"],
        errors="coerce",
    )

    df["soil_moisture"] = pd.to_numeric(
        df["soil_moisture"],
        errors="coerce",
    )

    df = (
        df
        .dropna(
            subset=[
                "date",
                "rainfall_mm",
                "soil_moisture",
            ]
        )
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# EXISTING PERCENTILE MODEL
# ============================================================

@st.cache_resource
def load_percentile_model():

    from step2_risk_model import (
        RealDataRiskModel
    )

    return RealDataRiskModel(
        str(DATA_FILE),
        lookback_days=LOOKBACK_DAYS,
    )


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

@st.cache_resource
def load_xgb_model():

    if not XGB_MODEL_FILE.exists():
        return None

    try:
        return joblib.load(
            XGB_MODEL_FILE
        )

    except Exception:
        return None


@st.cache_data
def load_xgb_metadata():

    if not XGB_METADATA_FILE.exists():
        return {}

    try:

        with open(
            XGB_METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except Exception:

        return {}


# ============================================================
# LOAD MODELS
# ============================================================

try:

    historical_df = load_historical_data()

except Exception as error:

    st.error(
        "❌ Historical data could not be loaded."
    )

    st.code(
        str(error)
    )

    st.stop()


try:

    percentile_model = (
        load_percentile_model()
    )

except Exception as error:

    percentile_model = None

    st.warning(
        "⚠️ Existing percentile model could not be loaded."
    )

    st.code(
        str(error)
    )


xgb_model = load_xgb_model()

xgb_metadata = load_xgb_metadata()


# ============================================================
# HISTORICAL XGBOOST FEATURES
# ============================================================

@st.cache_data
def build_historical_xgb_features():

    if not XGB_MODULE_AVAILABLE:
        return None

    try:

        return xgb_add_features(
            xgb_load_weather(
                str(DATA_FILE)
            )
        )

    except Exception:

        return None


historical_xgb_features = (
    build_historical_xgb_features()
)


# ============================================================
# OPEN-METEO LIVE + FORECAST
# ============================================================

@st.cache_data(ttl=300)
def fetch_weather_data():

    params = {

        "latitude":
            DHARALI_LATITUDE,

        "longitude":
            DHARALI_LONGITUDE,

        "timezone":
            TIMEZONE,

        # Recent history used for current risk
        "past_days":
            7,

        # Future forecast
        "forecast_days":
            FORECAST_DAYS,

        "hourly":
            ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "rain",
                    "precipitation_probability",
                    "soil_moisture_0_to_7cm",
                    "wind_speed_10m",
                ]
            ),

        "daily":
            ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "rain_sum",
                    "precipitation_probability_max",
                ]
            ),

        "current":
            ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "rain",
                    "wind_speed_10m",
                    "weather_code",
                ]
            ),

        "temperature_unit":
            "celsius",

        "precipitation_unit":
            "mm",

        "wind_speed_unit":
            "kmh",
    }

    response = requests.get(
        OPEN_METEO_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError(
            "No hourly weather data returned."
        )

    hourly = data["hourly"]

    timestamps = hourly.get(
        "time",
        []
    )

    if not timestamps:
        raise ValueError(
            "No weather timestamps returned."
        )

    hourly_df = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    timestamps
                ),

            "temperature_c":
                hourly.get(
                    "temperature_2m",
                    []
                ),

            "humidity":
                hourly.get(
                    "relative_humidity_2m",
                    []
                ),

            "rainfall_mm":
                hourly.get(
                    "rain",
                    []
                ),

            "rain_probability":
                hourly.get(
                    "precipitation_probability",
                    []
                ),

            "soil_moisture":
                hourly.get(
                    "soil_moisture_0_to_7cm",
                    []
                ),

            "wind_kmh":
                hourly.get(
                    "wind_speed_10m",
                    []
                ),
        }
    )

    hourly_df["datetime"] = (
        pd.to_datetime(
            hourly_df["datetime"]
        )
    )

    hourly_df["date"] = (
        hourly_df["datetime"].dt.date
    )

    numeric_columns = [
        "temperature_c",
        "humidity",
        "rainfall_mm",
        "rain_probability",
        "soil_moisture",
        "wind_kmh",
    ]

    for column in numeric_columns:

        hourly_df[column] = pd.to_numeric(
            hourly_df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Daily aggregation
    # --------------------------------------------------------

    daily_df = (
        hourly_df
        .groupby("date")
        .agg(
            rainfall_mm=(
                "rainfall_mm",
                "sum",
            ),

            soil_moisture=(
                "soil_moisture",
                "mean",
            ),

            temperature_c=(
                "temperature_c",
                "mean",
            ),

            temperature_max_c=(
                "temperature_c",
                "max",
            ),

            temperature_min_c=(
                "temperature_c",
                "min",
            ),

            humidity=(
                "humidity",
                "mean",
            ),

            rain_probability=(
                "rain_probability",
                "max",
            ),

            wind_kmh=(
                "wind_kmh",
                "mean",
            ),
        )
        .reset_index()
    )

    daily_df["date"] = pd.to_datetime(
        daily_df["date"]
    )

    # Keep only complete/usable values
    daily_df = daily_df.dropna(
        subset=[
            "rainfall_mm",
            "soil_moisture",
        ]
    )

    return (
        data,
        hourly_df,
        daily_df,
    )


# ============================================================
# PERCENTILE CALCULATION
# ============================================================

def calculate_percentiles(
    rainfall_3day,
    soil_average,
):

    if percentile_model is None:

        return (
            np.nan,
            np.nan,
            np.nan,
            "Unknown",
        )

    rainfall_pct = stats.percentileofscore(
        percentile_model.ref_rainfall,
        rainfall_3day,
        kind="mean",
    )

    soil_pct = stats.percentileofscore(
        percentile_model.ref_soil,
        soil_average,
        kind="mean",
    )

    combined = (
        rainfall_pct +
        soil_pct
    ) / 2

    risk = percentile_to_risk(
        combined
    )

    return (
        float(rainfall_pct),
        float(soil_pct),
        float(combined),
        risk,
    )


# ============================================================
# BUILD XGBOOST FEATURE ROWS
# ============================================================

def add_xgb_forecast_features(
    combined_daily,
    historical_reference,
):

    df = combined_daily.copy()

    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    rainfall = df[
        "rainfall_mm"
    ]

    soil = df[
        "soil_moisture"
    ]

    # --------------------------------------------------------
    # Same features used during XGBoost training
    # --------------------------------------------------------

    df["rain_1d"] = rainfall

    df["rain_3d"] = (
        rainfall
        .rolling(3, min_periods=3)
        .sum()
    )

    df["rain_7d"] = (
        rainfall
        .rolling(7, min_periods=7)
        .sum()
    )

    df["rain_max_3d"] = (
        rainfall
        .rolling(3, min_periods=3)
        .max()
    )

    df["rain_max_7d"] = (
        rainfall
        .rolling(7, min_periods=7)
        .max()
    )

    df["rain_mean_3d"] = (
        rainfall
        .rolling(3, min_periods=3)
        .mean()
    )

    df["rain_std_3d"] = (
        rainfall
        .rolling(3, min_periods=3)
        .std()
        .fillna(0)
    )

    df["soil_1d"] = soil

    df["soil_3d"] = (
        soil
        .rolling(3, min_periods=3)
        .mean()
    )

    df["soil_7d"] = (
        soil
        .rolling(7, min_periods=7)
        .mean()
    )

    df["soil_change_3d"] = (
        soil -
        soil.shift(3)
    )

    df["rain_change_3d"] = (
        rainfall -
        rainfall.shift(3)
    )

    # --------------------------------------------------------
    # Percentiles MUST use historical reference only.
    # Forecast observations should not change their own
    # historical reference distribution.
    # --------------------------------------------------------

    historical_rain = (
        pd.to_numeric(
            historical_reference[
                "rainfall_mm"
            ],
            errors="coerce",
        )
        .dropna()
        .values
    )

    historical_soil = (
        pd.to_numeric(
            historical_reference[
                "soil_moisture"
            ],
            errors="coerce",
        )
        .dropna()
        .values
    )

    df["rain_percentile"] = (
        df["rain_1d"]
        .apply(
            lambda value:
                stats.percentileofscore(
                    historical_rain,
                    value,
                    kind="mean",
                )
            if pd.notna(value)
            else np.nan
        )
        / 100
    )

    df["soil_percentile"] = (
        df["soil_1d"]
        .apply(
            lambda value:
                stats.percentileofscore(
                    historical_soil,
                    value,
                    kind="mean",
                )
            if pd.notna(value)
            else np.nan
        )
        / 100
    )

    return df


# ============================================================
# XGBOOST FORECAST PREDICTIONS
# ============================================================

def calculate_forecast_xgb(
    daily_weather
):

    if xgb_model is None:
        return daily_weather.copy()

    if not XGB_MODULE_AVAILABLE:
        return daily_weather.copy()

    # Historical reference + forecast data.
    # Historical data provides the rolling context.
    hist = historical_df[
        [
            "date",
            "rainfall_mm",
            "soil_moisture",
        ]
    ].copy()

    forecast = daily_weather[
        [
            "date",
            "rainfall_mm",
            "soil_moisture",
        ]
    ].copy()

    # Remove overlap and combine.
    hist = hist[
        hist["date"] <
        forecast["date"].min()
    ]

    combined = pd.concat(
        [
            hist,
            forecast,
        ],
        ignore_index=True,
    )

    feature_df = (
        add_xgb_forecast_features(
            combined,
            historical_df,
        )
    )

    forecast_features = feature_df[
        feature_df["date"].isin(
            forecast["date"]
        )
    ].copy()

    usable = forecast_features.dropna(
        subset=XGB_FEATURES
    ).copy()

    if usable.empty:
        return daily_weather.copy()

    usable[
        "xgb_probability"
    ] = (
        xgb_model
        .predict_proba(
            usable[XGB_FEATURES]
        )[:, 1]
    )

    usable[
        "xgb_probability_percent"
    ] = (
        usable[
            "xgb_probability"
        ] * 100
    )

    usable[
        "xgb_risk"
    ] = usable[
        "xgb_probability"
    ].apply(
        combined_probability_risk
    )

    result = daily_weather.merge(
        usable[
            [
                "date",
                "xgb_probability",
                "xgb_probability_percent",
                "xgb_risk",
            ]
        ],
        on="date",
        how="left",
    )

    return result


# ============================================================
# BUILD FORECAST RISK
# ============================================================

def calculate_forecast_risk(
    daily_weather
):

    df = daily_weather.copy()

    # --------------------------------------------------------
    # Historical percentile risk for every forecast day.
    # --------------------------------------------------------

    hist = historical_df[
        [
            "date",
            "rainfall_mm",
            "soil_moisture",
        ]
    ].copy()

    combined = pd.concat(
        [
            hist,
            df[
                [
                    "date",
                    "rainfall_mm",
                    "soil_moisture",
                ]
            ],
        ],
        ignore_index=True,
    )

    combined = combined.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    combined["rain_3d"] = (
        combined[
            "rainfall_mm"
        ]
        .rolling(
            3,
            min_periods=3
        )
        .sum()
    )

    combined["soil_3d"] = (
        combined[
            "soil_moisture"
        ]
        .rolling(
            3,
            min_periods=3
        )
        .mean()
    )

    historical_rain_3d = (
        historical_df[
            "rainfall_mm"
        ]
        .rolling(3)
        .sum()
        .dropna()
        .values
    )

    historical_soil_3d = (
        historical_df[
            "soil_moisture"
        ]
        .rolling(3)
        .mean()
        .dropna()
        .values
    )

    forecast_dates = set(
        df["date"]
    )

    forecast_rows = combined[
        combined["date"].isin(
            forecast_dates
        )
    ].copy()

    rain_pcts = []
    soil_pcts = []
    combined_scores = []
    risks = []

    for _, row in forecast_rows.iterrows():

        if (
            pd.isna(row["rain_3d"])
            or
            pd.isna(row["soil_3d"])
        ):

            rain_pct = np.nan
            soil_pct = np.nan
            combined_score = np.nan
            risk = "Unknown"

        else:

            rain_pct = (
                stats.percentileofscore(
                    historical_rain_3d,
                    row["rain_3d"],
                    kind="mean",
                )
            )

            soil_pct = (
                stats.percentileofscore(
                    historical_soil_3d,
                    row["soil_3d"],
                    kind="mean",
                )
            )

            combined_score = (
                rain_pct +
                soil_pct
            ) / 2

            risk = percentile_to_risk(
                combined_score
            )

        rain_pcts.append(
            rain_pct
        )

        soil_pcts.append(
            soil_pct
        )

        combined_scores.append(
            combined_score
        )

        risks.append(
            risk
        )

    forecast_result = pd.DataFrame(
        {
            "date":
                forecast_rows["date"].values,

            "rain_3d":
                forecast_rows["rain_3d"].values,

            "soil_3d":
                forecast_rows["soil_3d"].values,

            "rain_percentile":
                rain_pcts,

            "soil_percentile":
                soil_pcts,

            "statistical_score":
                combined_scores,

            "statistical_risk":
                risks,
        }
    )

    df = df.merge(
        forecast_result,
        on="date",
        how="left",
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    df = calculate_forecast_xgb(
        df
    )

    # --------------------------------------------------------
    # Ensemble
    #
    # We only create a final ensemble score if XGBoost is
    # actually available. The weight is clearly shown and
    # should be recalibrated after proper historical validation.
    # --------------------------------------------------------

    def ensemble_row(row):

        statistical = (
            row["statistical_score"]
            if pd.notna(
                row["statistical_score"]
            )
            else np.nan
        )

        probability = (
            row["xgb_probability"] * 100
            if pd.notna(
                row.get(
                    "xgb_probability",
                    np.nan
                )
            )
            else np.nan
        )

        if pd.notna(
            statistical
        ) and pd.notna(
            probability
        ):

            # XGBoost receives 60%; statistical model 40%.
            final_score = (
                0.40 * statistical +
                0.60 * probability
            )

            return final_score

        if pd.notna(
            statistical
        ):

            return statistical

        return np.nan

    df[
        "final_risk_score"
    ] = df.apply(
        ensemble_row,
        axis=1,
    )

    df[
        "final_risk"
    ] = df[
        "final_risk_score"
    ].apply(
        lambda value:
            percentile_to_risk(value)
            if pd.notna(value)
            else "Unknown"
    )

    return df


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🌊 Dharali Flood Model"
)

st.sidebar.markdown(
    "---"
)

mode = st.sidebar.radio(
    "Analysis Mode",
    [
        "🔴 Live & Forecast",
        "📅 Historical Analysis",
        "🧪 Manual / Demo",
        "📊 Model Performance",
    ],
)

st.sidebar.markdown(
    "---"
)

st.sidebar.subheader(
    "System"
)

st.sidebar.write(
    "Historical data: 2015–2025"
)

st.sidebar.write(
    "Location: Dharali, Uttarakhand"
)

st.sidebar.write(
    f"Lookback: {LOOKBACK_DAYS} days"
)

st.sidebar.write(
    f"Forecast: {FORECAST_DAYS} days"
)

st.sidebar.write(
    "Live source: Open-Meteo"
)

st.sidebar.write(
    "Statistical model: Percentile"
)

if xgb_model is not None:

    st.sidebar.success(
        "🤖 XGBoost model: Loaded"
    )

else:

    st.sidebar.warning(
        "🤖 XGBoost model: Not trained"
    )


st.sidebar.markdown(
    "---"
)

st.sidebar.subheader(
    "Statistical Risk"
)

st.sidebar.write(
    "🟢 Low: < 75"
)

st.sidebar.write(
    "🟡 Moderate: 75–89.99"
)

st.sidebar.write(
    "🟠 High: 90–94.99"
)

st.sidebar.write(
    "🔴 Severe: ≥ 95"
)


# ============================================================
# DASHBOARD OVERVIEW
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Historical Records",
        f"{len(historical_df):,}",
    )

with col2:

    st.metric(
        "Historical Start",
        historical_df[
            "date"
        ]
        .min()
        .strftime("%d %b %Y"),
    )

with col3:

    st.metric(
        "Historical End",
        historical_df[
            "date"
        ]
        .max()
        .strftime("%d %b %Y"),
    )

with col4:

    st.metric(
        "Forecast Horizon",
        f"{FORECAST_DAYS} days",
    )


st.markdown(
    "---"
)


# ============================================================
# LIVE + FORECAST MODE
# ============================================================

if mode == "🔴 Live & Forecast":

    st.header(
        "🔴 Live Dharali Flood Monitoring"
    )

    st.write(
        """
        Live weather and all operational flood-risk predictions on
        this page come from the FastAPI backend. The Streamlit app
        does not independently recalculate the current or forecast
        ensemble risk.
        """
    )

    try:
        health = get_backend_health()

        if health.get("status") == "healthy":
            st.success("🟢 FastAPI backend connected")
        else:
            st.warning(
                "🟡 Backend connected, but model status is degraded."
            )

    except Exception as error:
        st.error("🔴 FastAPI backend is not running.")
        st.code(
            "uvicorn backend.server:app --reload --port 8000"
        )
        st.exception(error)
        st.stop()

    if st.button(
        "🔄 FETCH LIVE DATA + 7-DAY FORECAST",
        type="primary",
        use_container_width=True,
    ):

        try:
            with st.spinner(
                "Loading current risk and forecast from backend..."
            ):
                live_data = get_backend_live()
                risk_data = get_backend_current_risk()
                forecast_data = get_backend_forecast()

            current = live_data.get("current", {})
            prediction = risk_data.get("prediction", {})

            statistical = prediction.get("statistical", {})
            xgboost = prediction.get("xgboost", {})
            ensemble = prediction.get("ensemble", {})

            prediction_date = pd.to_datetime(
                prediction.get("date"),
                errors="coerce",
            )

            today = (
                pd.Timestamp.now(tz=TIMEZONE)
                .tz_localize(None)
                .normalize()
            )

            # Guard against the old backend bug where /api/current-risk
            # selected daily["date"].max() and therefore returned the last
            # forecast date instead of today.
            if (
                pd.notna(prediction_date)
                and prediction_date.normalize() != today
            ):
                st.error(
                    "⚠️ /api/current-risk is not returning today's date. "
                    f"Backend returned {prediction_date.date()}, while "
                    f"Dharali local date is {today.date()}. "
                    "Fix backend/server.py before treating this as current risk."
                )

            st.success(
                "✅ Backend data successfully retrieved."
            )

            # ----------------------------------------------------
            # CURRENT WEATHER
            # ----------------------------------------------------

            st.subheader(
                "🕐 Current Dharali Conditions"
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                st.metric(
                    "Temperature",
                    f"{current.get('temperature_c', '—')} °C",
                )

            with c2:
                st.metric(
                    "Rain",
                    f"{current.get('rain_mm', '—')} mm",
                )

            with c3:
                st.metric(
                    "Humidity",
                    f"{current.get('humidity', '—')} %",
                )

            with c4:
                st.metric(
                    "Wind",
                    f"{current.get('wind_kmh', '—')} km/h",
                )

            with c5:
                st.metric(
                    "Weather Code",
                    str(current.get("weather_code", "—")),
                )

            if current.get("time"):
                st.caption(
                    "Latest backend weather timestamp: "
                    f"{current['time']} ({TIMEZONE})"
                )

            # ----------------------------------------------------
            # FINAL CURRENT RISK
            # ----------------------------------------------------

            final_score = ensemble.get("score")
            final_risk = ensemble.get("risk", "Unknown")

            st.subheader(
                "🚨 FINAL FLOOD RISK"
            )

            if final_score is not None:
                display_risk(
                    float(final_score),
                    final_risk,
                    title="ENSEMBLE FLOOD RISK",
                )
            else:
                st.warning(
                    "Final ensemble score is unavailable."
                )

            # ----------------------------------------------------
            # MODEL BREAKDOWN
            # ----------------------------------------------------

            st.subheader(
                "🤖 Model Breakdown"
            )

            m1, m2, m3 = st.columns(3)

            statistical_score = statistical.get("score")
            xgb_percent = xgboost.get("probability_percent")
            ensemble_score = ensemble.get("score")

            with m1:
                st.metric(
                    "Statistical Score",
                    "—" if statistical_score is None
                    else f"{float(statistical_score):.1f}",
                )
                st.caption(
                    statistical.get("risk", "Unknown")
                )

            with m2:
                st.metric(
                    "XGBoost Classifier Output",
                    "—" if xgb_percent is None
                    else f"{float(xgb_percent):.1f}%",
                )
                st.caption(
                    xgboost.get("risk", "Unknown")
                )

            with m3:
                st.metric(
                    "Final Ensemble Score",
                    "—" if ensemble_score is None
                    else f"{float(ensemble_score):.1f}",
                )
                st.caption(
                    ensemble.get("risk", "Unknown")
                )

            st.caption(
                "Prototype ensemble weighting: "
                f"Statistical {ensemble.get('statistical_weight', 0.4) * 100:.0f}% "
                f"+ XGBoost {ensemble.get('xgboost_weight', 0.6) * 100:.0f}%."
            )

            # ----------------------------------------------------
            # LIVE DAILY DATA
            # ----------------------------------------------------

            daily_records = live_data.get("daily", [])
            daily_df_backend = pd.DataFrame(daily_records)

            if not daily_df_backend.empty and "date" in daily_df_backend:
                daily_df_backend["date"] = pd.to_datetime(
                    daily_df_backend["date"],
                    errors="coerce",
                )

                recent = (
                    daily_df_backend[
                        daily_df_backend["date"] <= today
                    ]
                    .sort_values("date")
                    .tail(7)
                    .copy()
                )

                if not recent.empty:
                    st.subheader(
                        "📋 Recent Synchronized Daily Data"
                    )
                    recent["date"] = recent["date"].dt.strftime(
                        "%d %b %Y"
                    )
                    st.dataframe(
                        recent,
                        use_container_width=True,
                        hide_index=True,
                    )

            # ----------------------------------------------------
            # FORECAST — FUTURE DAYS ONLY
            # ----------------------------------------------------

            st.markdown("---")
            st.header(
                "🔮 7-Day Forecast + Flood Risk"
            )

            forecast_df = pd.DataFrame(
                forecast_data.get("forecast", [])
            )

            if forecast_df.empty:
                st.warning(
                    "No forecast data available."
                )

            else:
                forecast_df["date"] = pd.to_datetime(
                    forecast_df["date"],
                    errors="coerce",
                )

                # Current Risk = today; Forecast = tomorrow onward.
                forecast_df = (
                    forecast_df[
                        forecast_df["date"] > today
                    ]
                    .sort_values("date")
                    .head(FORECAST_DAYS)
                    .copy()
                )

                if forecast_df.empty:
                    st.warning(
                        "Backend returned no future forecast days."
                    )

                else:
                    display_forecast = forecast_df.copy()

                    display_forecast["Date"] = (
                        display_forecast["date"]
                        .dt.strftime("%d %b %Y")
                    )

                    rename_map = {
                        "rainfall_mm": "Rainfall (mm)",
                        "soil_moisture": "Soil Moisture",
                        "statistical_score": "Statistical Score",
                        "statistical_risk": "Statistical Risk",
                        "xgb_probability_percent":
                            "XGBoost Classifier Output (%)",
                        "xgb_risk": "XGBoost Risk",
                        "final_risk_score": "Final Score",
                        "final_risk": "Final Risk",
                    }

                    display_forecast = display_forecast.rename(
                        columns=rename_map
                    )

                    wanted = [
                        "Date",
                        "Rainfall (mm)",
                        "Soil Moisture",
                        "Statistical Score",
                        "Statistical Risk",
                        "XGBoost Classifier Output (%)",
                        "XGBoost Risk",
                        "Final Score",
                        "Final Risk",
                    ]

                    available = [
                        c for c in wanted
                        if c in display_forecast.columns
                    ]

                    st.dataframe(
                        display_forecast[available],
                        use_container_width=True,
                        hide_index=True,
                    )

                    valid = forecast_df[
                        forecast_df[
                            "final_risk_score"
                        ].notna()
                    ] if "final_risk_score" in forecast_df else pd.DataFrame()

                    if not valid.empty:
                        highest = (
                            valid.sort_values(
                                "final_risk_score",
                                ascending=False,
                            )
                            .iloc[0]
                        )

                        st.subheader(
                            "⚠️ Highest Forecast Risk"
                        )

                        h1, h2, h3, h4 = st.columns(4)

                        with h1:
                            st.metric(
                                "Date",
                                highest["date"].strftime(
                                    "%d %b %Y"
                                ),
                            )

                        with h2:
                            st.metric(
                                "Forecast Rain",
                                f"{float(highest['rainfall_mm']):.2f} mm",
                            )

                        with h3:
                            st.metric(
                                "Final Score",
                                f"{float(highest['final_risk_score']):.1f}",
                            )

                        with h4:
                            risk_name = highest.get(
                                "final_risk",
                                "Unknown",
                            )
                            st.metric(
                                "Risk",
                                f"{risk_icon(risk_name)} {risk_name}",
                            )

                    if {
                        "date",
                        "rainfall_mm",
                    }.issubset(forecast_df.columns):
                        st.subheader(
                            "🌧️ Forecast Rainfall"
                        )
                        st.bar_chart(
                            forecast_df[
                                ["date", "rainfall_mm"]
                            ].set_index("date")
                        )

                    if {
                        "date",
                        "final_risk_score",
                    }.issubset(forecast_df.columns):
                        st.subheader(
                            "📈 Forecast Flood-Risk Score"
                        )
                        st.line_chart(
                            forecast_df[
                                ["date", "final_risk_score"]
                            ].set_index("date")
                        )

            # ----------------------------------------------------
            # HOURLY WEATHER
            # ----------------------------------------------------

            st.markdown("---")
            st.header(
                "⏱️ Hourly Weather Timeline"
            )

            hourly_df_backend = pd.DataFrame(
                live_data.get("hourly", [])
            )

            if not hourly_df_backend.empty:
                time_column = (
                    "datetime"
                    if "datetime" in hourly_df_backend.columns
                    else "time"
                    if "time" in hourly_df_backend.columns
                    else None
                )

                if time_column:
                    hourly_df_backend[time_column] = pd.to_datetime(
                        hourly_df_backend[time_column],
                        errors="coerce",
                    )

                    now_naive = (
                        pd.Timestamp.now(tz=TIMEZONE)
                        .tz_localize(None)
                    )

                    hourly_future = (
                        hourly_df_backend[
                            hourly_df_backend[time_column] >= now_naive
                        ]
                        .head(48)
                        .copy()
                    )

                    st.dataframe(
                        hourly_future,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.dataframe(
                        hourly_df_backend.head(48),
                        use_container_width=True,
                        hide_index=True,
                    )

            st.subheader(
                "🧠 Operational Model Logic"
            )

            st.markdown(
                """
                **Current Risk:** today's backend prediction.

                **Forecast:** tomorrow through the next seven available
                forecast days.

                **Final ensemble score:** 40% statistical historical-risk
                score + 60% XGBoost classifier output on a 0–100 scale.

                The XGBoost value is a classifier output and should not be
                presented as a calibrated real-world flood probability
                until calibration and historical validation are completed.
                """
            )

        except requests.exceptions.RequestException as error:
            st.error(
                "❌ Could not communicate with the FastAPI backend."
            )
            st.code(str(error))

        except Exception as error:
            st.error(
                "❌ Live/forecast analysis failed."
            )
            st.exception(error)

    else:
        st.info(
            "👆 Click **FETCH LIVE DATA + 7-DAY FORECAST** "
            "to load data from the FastAPI backend."
        )


# ============================================================
# HISTORICAL ANALYSIS
# ============================================================

elif mode == "📅 Historical Analysis":

    st.header(
        "📅 Historical Flood-Risk Analysis"
    )

    st.write(
        """
        Select any date from the historical weather dataset.
        The dashboard reconstructs the 3-day/7-day weather
        context and runs the statistical model. If a trained
        XGBoost model is available, the same model is also run
        on the historical feature vector.
        """
    )

    min_date = (
        historical_df[
            "date"
        ].min()
        .date()
    )

    max_date = (
        historical_df[
            "date"
        ].max()
        .date()
    )

    selected_date = st.date_input(
        "Select historical date",
        value=pd.Timestamp(
            "2025-08-05"
        ).date(),
        min_value=min_date,
        max_value=max_date,
    )

    selected_timestamp = pd.Timestamp(
        selected_date
    )

    row = historical_df[
        historical_df[
            "date"
        ] == selected_timestamp
    ]

    if row.empty:

        st.error(
            "Selected date is not available."
        )

    else:

        idx = row.index[0]

        # Need at least 7 rows for feature context
        history_context = (
            historical_df
            .loc[
                max(
                    0,
                    idx - 6
                ):
                idx
            ]
            .copy()
        )

        rainfall_3day = (
            history_context[
                "rainfall_mm"
            ]
            .tail(3)
            .sum()
        )

        soil_3day = (
            history_context[
                "soil_moisture"
            ]
            .tail(3)
            .mean()
        )

        (
            rainfall_pct,
            soil_pct,
            statistical_score,
            statistical_risk,
        ) = calculate_percentiles(
            rainfall_3day,
            soil_3day,
        )

        selected = row.iloc[0]

        st.subheader(
            f"📍 {selected_timestamp.strftime('%d %B %Y')}"
        )

        # ----------------------------------------------------
        # Weather values
        # ----------------------------------------------------

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        with c1:

            st.metric(
                "Daily Rainfall",
                f"{selected['rainfall_mm']:.2f} mm",
            )

        with c2:

            st.metric(
                "3-Day Rainfall",
                f"{rainfall_3day:.2f} mm",
            )

        with c3:

            st.metric(
                "Soil Moisture",
                f"{selected['soil_moisture']:.4f}",
            )

        with c4:

            st.metric(
                "3-Day Soil Avg",
                f"{soil_3day:.4f}",
            )

        # ----------------------------------------------------
        # Statistical risk
        # ----------------------------------------------------

        st.subheader(
            "📊 Statistical Model"
        )

        display_risk(
            statistical_score,
            statistical_risk,
            title="HISTORICAL STATISTICAL RISK",
        )

        # ----------------------------------------------------
        # XGBoost historical prediction
        # ----------------------------------------------------

        xgb_probability = None

        if (
            xgb_model is not None
            and
            historical_xgb_features is not None
        ):

            xgb_row = (
                historical_xgb_features[
                    historical_xgb_features[
                        "date"
                    ] == selected_timestamp
                ]
            )

            if not xgb_row.empty:

                usable = (
                    xgb_row
                    .dropna(
                        subset=XGB_FEATURES
                    )
                )

                if not usable.empty:

                    xgb_probability = float(
                        xgb_model
                        .predict_proba(
                            usable[
                                XGB_FEATURES
                            ]
                        )[0, 1]
                    )

                    xgb_risk = (
                        combined_probability_risk(
                            xgb_probability
                        )
                    )

                    st.subheader(
                        "🤖 XGBoost Historical Prediction"
                    )

                    a, b, c = (
                        st.columns(3)
                    )

                    with a:

                        st.metric(
                            "Flood Probability",
                            f"{xgb_probability * 100:.2f}%",
                        )

                    with b:

                        st.metric(
                            "XGBoost Risk",
                            f"{risk_icon(xgb_risk)} {xgb_risk}",
                        )

                    with c:

                        st.metric(
                            "Model Threshold",
                            "50%",
                        )

                    # Feature values
                    st.write(
                        "XGBoost feature vector:"
                    )

                    feature_display = (
                        usable[
                            XGB_FEATURES
                        ]
                        .T
                        .rename(
                            columns={
                                usable.index[0]:
                                    "Value"
                            }
                        )
                    )

                    st.dataframe(
                        feature_display,
                        use_container_width=True,
                    )

        else:

            st.info(
                """
                🤖 XGBoost is not currently available.

                The historical statistical model is still
                fully functional.

                Train XGBoost using documented flood-event
                labels to enable the ML historical prediction.
                """
            )

        # ----------------------------------------------------
        # Actual event label, if available
        # ----------------------------------------------------

        actual_flood = None

        if EVENT_LABEL_FILE.exists():

            try:

                labels = pd.read_csv(
                    EVENT_LABEL_FILE
                )

                labels["date"] = (
                    pd.to_datetime(
                        labels[
                            "date"
                        ].astype(str),
                        errors="coerce",
                    )
                )

                label_row = labels[
                    labels[
                        "date"
                    ] == selected_timestamp
                ]

                if not label_row.empty:

                    actual_flood = int(
                        label_row.iloc[0][
                            "flood"
                        ]
                    )

                    st.subheader(
                        "📝 Documented Event Label"
                    )

                    if actual_flood == 1:

                        st.error(
                            "🔴 Labeled as FLOOD / FLOOD EVENT"
                        )

                    else:

                        st.success(
                            "🟢 Labeled as NON-FLOOD"
                        )

            except Exception:

                pass

        # ----------------------------------------------------
        # 7-day context
        # ----------------------------------------------------

        st.subheader(
            "🌧️ Weather Context"
        )

        context_display = (
            history_context.copy()
        )

        context_display[
            "date"
        ] = context_display[
            "date"
        ].dt.strftime(
            "%d %b %Y"
        )

        context_display = (
            context_display.rename(
                columns={
                    "date":
                        "Date",

                    "rainfall_mm":
                        "Rainfall (mm)",

                    "soil_moisture":
                        "Soil Moisture",
                }
            )
        )

        st.dataframe(
            context_display,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "📈 Historical Rainfall"
        )

        chart = (
            history_context[
                [
                    "date",
                    "rainfall_mm",
                ]
            ]
            .set_index("date")
        )

        st.bar_chart(
            chart
        )

        # ----------------------------------------------------
        # Event detection comparison
        # ----------------------------------------------------

        if (
            actual_flood is not None
            and
            xgb_probability is not None
        ):

            prediction = (
                1
                if xgb_probability >= 0.50
                else 0
            )

            st.subheader(
                "🎯 Model vs Actual Event"
            )

            if prediction == actual_flood:

                st.success(
                    "✅ XGBoost classification matches "
                    "the documented event label."
                )

            else:

                st.warning(
                    "⚠️ XGBoost classification does not "
                    "match the documented event label."
                )


# ============================================================
# MANUAL / DEMO
# ============================================================

elif mode == "🧪 Manual / Demo":

    st.header(
        "🧪 Manual / Demo Risk Assessment"
    )

    st.write(
        """
        Enter hypothetical weather conditions to test the
        statistical risk model. XGBoost requires a complete
        feature vector and therefore is only evaluated when
        enough historical context is available.
        """
    )

    rainfall_values = historical_df[
        "rainfall_mm"
    ]

    soil_values = historical_df[
        "soil_moisture"
    ]

    default_rain = float(
        rainfall_values
        .median()
    )

    default_soil = float(
        soil_values
        .median()
    )

    c1, c2 = (
        st.columns(2)
    )

    with c1:

        rain_1 = st.number_input(
            "Day 1 rainfall (mm)",
            min_value=0.0,
            value=default_rain,
            step=1.0,
        )

        rain_2 = st.number_input(
            "Day 2 rainfall (mm)",
            min_value=0.0,
            value=default_rain,
            step=1.0,
        )

        rain_3 = st.number_input(
            "Day 3 rainfall (mm)",
            min_value=0.0,
            value=default_rain,
            step=1.0,
        )

    with c2:

        soil_1 = st.number_input(
            "Day 1 soil moisture",
            min_value=0.0,
            value=default_soil,
            step=0.01,
        )

        soil_2 = st.number_input(
            "Day 2 soil moisture",
            min_value=0.0,
            value=default_soil,
            step=0.01,
        )

        soil_3 = st.number_input(
            "Day 3 soil moisture",
            min_value=0.0,
            value=default_soil,
            step=0.01,
        )

    if st.button(
        "🚨 CALCULATE DEMO RISK",
        type="primary",
        use_container_width=True,
    ):

        rainfall_3day = (
            rain_1 +
            rain_2 +
            rain_3
        )

        soil_average = (
            soil_1 +
            soil_2 +
            soil_3
        ) / 3

        (
            rainfall_pct,
            soil_pct,
            score,
            risk,
        ) = calculate_percentiles(
            rainfall_3day,
            soil_average,
        )

        st.subheader(
            "📊 Statistical Result"
        )

        display_risk(
            score,
            risk,
            title="DEMO STATISTICAL RISK",
        )

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        with c1:

            st.metric(
                "3-Day Rainfall",
                f"{rainfall_3day:.2f} mm",
            )

        with c2:

            st.metric(
                "3-Day Soil",
                f"{soil_average:.4f}",
            )

        with c3:

            st.metric(
                "Rain Percentile",
                f"{rainfall_pct:.1f}%",
            )

        with c4:

            st.metric(
                "Soil Percentile",
                f"{soil_pct:.1f}%",
            )

        if xgb_model is not None:

            st.info(
                """
                XGBoost is loaded, but this demo screen
                intentionally does not manufacture missing
                7-day feature values. Use Live/Forecast or
                Historical Analysis for a real XGBoost vector.
                """
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif mode == "📊 Model Performance":

    st.header(
        "📊 Model Performance"
    )

    st.write(
        """
        This section reads the metrics saved by the XGBoost
        training pipeline. Metrics are shown only when a
        trained model and metadata file exist.
        """
    )

    if xgb_model is None:

        st.warning(
            """
            🤖 XGBoost model has not been trained yet.

            Add documented historical labels to
            `event_labels.csv` and run:

            `python train_xgboost.py`
            """
        )

    else:

        st.success(
            "🤖 XGBoost model loaded successfully."
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        if xgb_metadata:

            st.subheader(
                "Model Configuration"
            )

            st.json(
                {
                    key: value
                    for key, value
                    in xgb_metadata.items()
                    if key not in [
                        "validation_metrics",
                        "test_metrics",
                    ]
                }
            )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        validation_metrics = (
            xgb_metadata.get(
                "validation_metrics",
                {}
            )
        )

        test_metrics = (
            xgb_metadata.get(
                "test_metrics",
                {}
            )
        )

        st.subheader(
            "Validation — 2024"
        )

        if validation_metrics:

            c1, c2, c3, c4, c5 = (
                st.columns(5)
            )

            with c1:

                st.metric(
                    "Accuracy",
                    f"{validation_metrics.get('accuracy', 0) * 100:.1f}%",
                )

            with c2:

                st.metric(
                    "Precision",
                    f"{validation_metrics.get('precision', 0) * 100:.1f}%",
                )

            with c3:

                st.metric(
                    "Recall",
                    f"{validation_metrics.get('recall', 0) * 100:.1f}%",
                )

            with c4:

                st.metric(
                    "F1",
                    f"{validation_metrics.get('f1', 0) * 100:.1f}%",
                )

            with c5:

                auc = validation_metrics.get(
                    "roc_auc"
                )

                st.metric(
                    "ROC-AUC",
                    "N/A"
                    if auc is None
                    else f"{auc:.3f}",
                )

            st.write(
                "Confusion matrix:"
            )

            st.write(
                validation_metrics.get(
                    "confusion_matrix",
                    []
                )
            )

        else:

            st.info(
                "No validation metrics available."
            )

        st.subheader(
            "Final Test — 2025"
        )

        if test_metrics:

            c1, c2, c3, c4, c5 = (
                st.columns(5)
            )

            with c1:

                st.metric(
                    "Accuracy",
                    f"{test_metrics.get('accuracy', 0) * 100:.1f}%",
                )

            with c2:

                st.metric(
                    "Precision",
                    f"{test_metrics.get('precision', 0) * 100:.1f}%",
                )

            with c3:

                st.metric(
                    "Recall",
                    f"{test_metrics.get('recall', 0) * 100:.1f}%",
                )

            with c4:

                st.metric(
                    "F1",
                    f"{test_metrics.get('f1', 0) * 100:.1f}%",
                )

            with c5:

                auc = test_metrics.get(
                    "roc_auc"
                )

                st.metric(
                    "ROC-AUC",
                    "N/A"
                    if auc is None
                    else f"{auc:.3f}",
                )

            st.write(
                "Confusion matrix:"
            )

            st.write(
                test_metrics.get(
                    "confusion_matrix",
                    []
                )
            )

        else:

            st.info(
                "No final-test metrics available."
            )

        # ----------------------------------------------------
        # Feature importance
        # ----------------------------------------------------

        if hasattr(
            xgb_model,
            "feature_importances_"
        ):

            st.subheader(
                "🌳 XGBoost Feature Importance"
            )

            importance = pd.DataFrame(
                {
                    "Feature":
                        XGB_FEATURES,

                    "Importance":
                        xgb_model.feature_importances_,
                }
            )

            importance = (
                importance
                .sort_values(
                    "Importance",
                    ascending=False,
                )
                .reset_index(
                    drop=True
                )
            )

            importance[
                "Importance"
            ] = importance[
                "Importance"
            ].round(5)

            st.dataframe(
                importance,
                use_container_width=True,
                hide_index=True,
            )

            st.bar_chart(
                importance
                .set_index(
                    "Feature"
                )
            )


# ============================================================
# FOOTER / DATA NOTE
# ============================================================

st.markdown(
    "---"
)

st.caption(
    """
    Prototype note: the historical dataset and the live
    Open-Meteo soil-moisture variable may not represent exactly
    the same physical measurement/depth. The dashboard therefore
    treats the live layer as an operational prototype rather than
    a validated hydrological warning system.

    XGBoost probabilities should only be interpreted as model
    outputs after the classifier has been trained and evaluated
    using documented historical flood labels.
    """
)
