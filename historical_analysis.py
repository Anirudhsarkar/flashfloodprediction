from pathlib import Path
import pandas as pd

from xgboost_flood_model import (
    load_weather,
    add_features,
    historical_predictions,
    load_bundle,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


WEATHER_CSV = (
    BASE_DIR /
    "dharali_weather_2015_2025_clean.csv"
)


MODEL_PATH = (
    BASE_DIR /
    "xgboost_model" /
    "xgboost_flood_model.pkl"
)


OUTPUT_CSV = (
    BASE_DIR /
    "historical_xgboost_predictions.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 65)
print("DHARALI HISTORICAL FLOOD RISK ANALYSIS")
print("=" * 65)


print()
print(
    "Loading trained XGBoost model..."
)


model = load_bundle(
    MODEL_PATH
)


# ============================================================
# LOAD WEATHER
# ============================================================

print(
    "Loading historical weather data..."
)


weather = load_weather(
    WEATHER_CSV
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print(
    "Creating historical features..."
)


featured_weather = add_features(
    weather
)


# ============================================================
# XGBOOST PREDICTIONS
# ============================================================

print(
    "Calculating XGBoost flood probability..."
)


predictions = historical_predictions(
    model,
    featured_weather
)


# ============================================================
# SAVE FULL HISTORY
# ============================================================

columns_to_save = [

    "date",

    "rainfall_mm",

    "soil_moisture",

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

    "xgb_probability",

    "xgb_probability_percent",

    "xgb_risk",
]


predictions[
    columns_to_save
].to_csv(

    OUTPUT_CSV,

    index=False
)


print()
print(
    f"Historical predictions saved:\n"
    f"{OUTPUT_CSV}"
)


# ============================================================
# ANALYZE DHARALI FLOOD DATE
# ============================================================

TARGET_DATE = pd.Timestamp(
    "2025-08-05"
)


event = predictions[
    predictions["date"]
    ==
    TARGET_DATE
]


print()
print("=" * 65)
print("5 AUGUST 2025 DHARALI EVENT")
print("=" * 65)


if event.empty:

    print(
        "05 August 2025 could not "
        "be found in the dataset."
    )

else:

    row = event.iloc[0]

    print()

    print(
        f"Date:"
        f"                 "
        f"{row['date'].date()}"
    )

    print()

    print(
        f"Rainfall - 1 day:"
        f"     "
        f"{row['rain_1d']:.2f} mm"
    )

    print(
        f"Rainfall - 3 days:"
        f"    "
        f"{row['rain_3d']:.2f} mm"
    )

    print(
        f"Rainfall - 7 days:"
        f"    "
        f"{row['rain_7d']:.2f} mm"
    )

    print(
        f"Maximum rain - 3d:"
        f"    "
        f"{row['rain_max_3d']:.2f} mm"
    )

    print(
        f"Maximum rain - 7d:"
        f"    "
        f"{row['rain_max_7d']:.2f} mm"
    )

    print()

    print(
        f"Soil moisture:"
        f"          "
        f"{row['soil_1d']:.4f}"
    )

    print(
        f"Soil moisture 3d:"
        f"       "
        f"{row['soil_3d']:.4f}"
    )

    print(
        f"Soil moisture 7d:"
        f"       "
        f"{row['soil_7d']:.4f}"
    )

    print()

    print(
        f"Rain percentile:"
        f"        "
        f"{row['rain_percentile'] * 100:.2f}%"
    )

    print(
        f"Soil percentile:"
        f"        "
        f"{row['soil_percentile'] * 100:.2f}%"
    )

    print()

    print(
        "XGBoost Flood Probability:"
    )

    print(
        f"{row['xgb_probability_percent']:.2f}%"
    )

    print()

    print(
        f"XGBoost Risk:"
        f"           "
        f"{row['xgb_risk']}"
    )


# ============================================================
# HIGHEST HISTORICAL RISK DAYS
# ============================================================

print()
print("=" * 65)
print("TOP 20 HIGHEST XGBOOST RISK DATES")
print("=" * 65)
print()


top_dates = (

    predictions

    .sort_values(
        "xgb_probability",
        ascending=False
    )

    .head(20)

)


display_columns = [

    "date",

    "rain_1d",

    "rain_3d",

    "rain_7d",

    "soil_3d",

    "xgb_probability_percent",

    "xgb_risk",
]


print(
    top_dates[
        display_columns
    ].to_string(
        index=False
    )
)