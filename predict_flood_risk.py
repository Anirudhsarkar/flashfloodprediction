"""
predict_flood_risk.py
-----------------------
Standalone inference script. Does NOT train anything -- just loads the
already-trained model (from xgboost_model/xgboost_flood_model.pkl) and
scores it against real data.

Run this AFTER train_xgboost.py has produced xgboost_model/xgboost_flood_model.pkl.

Usage examples are in the __main__ block at the bottom.
"""

from pathlib import Path
import pandas as pd

from xgboost_flood_model import (
    load_weather,
    add_features,
    load_bundle,
    predict_row,
    probability_to_risk,
    historical_predictions,
    FEATURES,
)

BASE_DIR = Path(__file__).resolve().parent
WEATHER_CSV = BASE_DIR / "dharali_weather_2015_2025_clean.csv"
MODEL_PATH = BASE_DIR / "xgboost_model" / "xgboost_flood_model.pkl"


class FloodRiskPredictor:
    """Wraps the trained XGBoost model for easy reuse (e.g. from app.py)."""

    def __init__(self, model_path: Path = MODEL_PATH, weather_csv: Path = WEATHER_CSV):
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}.\n"
                f"Run train_xgboost.py first to create it."
            )
        self.model = load_bundle(model_path)

        weather = load_weather(weather_csv)
        self.featured_weather = add_features(weather)

    def predict_for_date(self, date_str: str) -> dict:
        """
        Score a specific historical date (format: YYYY-MM-DD or YYYYMMDD).
        Only works for dates present in the weather CSV.
        """
        target = pd.Timestamp(date_str)
        row = self.featured_weather[self.featured_weather["date"] == target]

        if row.empty:
            raise ValueError(f"Date {date_str} not found in {WEATHER_CSV.name}")

        if row[FEATURES].isna().any(axis=None):
            raise ValueError(
                f"Date {date_str} doesn't have a complete feature window yet "
                f"(needs at least 30 prior days of data)."
            )

        probability = predict_row(self.model, row.iloc[0])
        return {
            "date": str(target.date()),
            "probability": probability,
            "probability_percent": round(probability * 100, 2),
            "risk_label": probability_to_risk(probability),
        }

    def predict_latest(self) -> dict:
        """Score the most recent date available in the weather CSV."""
        latest_date = self.featured_weather["date"].max()
        return self.predict_for_date(str(latest_date.date()))

    def predict_manual(self, rainfall_last_7_days: list, soil_moisture_last_7_days: list) -> dict:
        """
        Score a hypothetical/live scenario by directly supplying the last
        7 days of rainfall (mm) and soil moisture (0-1), most recent last.
        Useful once you have a live sensor/forecast feed instead of the CSV.
        """
        if len(rainfall_last_7_days) < 7 or len(soil_moisture_last_7_days) < 7:
            raise ValueError("Need at least 7 days of rainfall and soil moisture.")

        temp_df = pd.DataFrame({
            "date": pd.date_range(end=pd.Timestamp.today(), periods=len(rainfall_last_7_days)),
            "rainfall_mm": rainfall_last_7_days,
            "soil_moisture": soil_moisture_last_7_days,
        })
        # prepend enough history so expanding-percentile features aren't NaN
        combined = pd.concat([self.featured_weather[["date", "rainfall_mm", "soil_moisture"]], temp_df])
        combined = combined.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
        featured = add_features(combined)

        last_row = featured.iloc[-1]
        if last_row[FEATURES].isna().any():
            raise ValueError("Not enough history to compute all features for this input.")

        probability = predict_row(self.model, last_row)
        return {
            "probability": probability,
            "probability_percent": round(probability * 100, 2),
            "risk_label": probability_to_risk(probability),
        }

    def top_risk_days(self, n: int = 10) -> pd.DataFrame:
        """Return the n highest-risk days across the entire historical dataset."""
        preds = historical_predictions(self.model, self.featured_weather)
        return preds.sort_values("xgb_probability", ascending=False).head(n)[
            ["date", "rainfall_mm", "soil_moisture", "xgb_probability_percent", "xgb_risk"]
        ]


if __name__ == "__main__":
    predictor = FloodRiskPredictor()

    print("=" * 60)
    print("SCORE THE ACTUAL DHARALI FLOOD DATE")
    print("=" * 60)
    result = predictor.predict_for_date("2025-08-05")
    print(result)

    print("\n" + "=" * 60)
    print("SCORE THE MOST RECENT DATE IN THE DATASET")
    print("=" * 60)
    result = predictor.predict_latest()
    print(result)

    print("\n" + "=" * 60)
    print("TOP 10 HIGHEST-RISK DAYS IN 10 YEARS OF DATA")
    print("=" * 60)
    print(predictor.top_risk_days(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("MANUAL/LIVE SCORING EXAMPLE (hypothetical new event)")
    print("=" * 60)
    result = predictor.predict_manual(
        rainfall_last_7_days=[5, 8, 12, 15, 20, 35, 45],
        soil_moisture_last_7_days=[0.55, 0.58, 0.60, 0.63, 0.68, 0.72, 0.76],
    )
    print(result)
