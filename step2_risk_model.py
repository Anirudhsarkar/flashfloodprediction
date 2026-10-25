
"""
step2_risk_model.py
-------------------
Step 2: turns the Step 1 percentile analysis into a reusable,
live risk-scoring model.

This model is calibrated entirely on real NASA POWER weather data
for Dharali's coordinates from 2015-2025.

How it works:
- Builds a reference distribution of 3-day rolling rainfall sums.
- Builds a reference distribution of 3-day rolling soil-moisture averages.
- For new conditions, calculates their percentile against the
  historical distribution.
- Combines rainfall and soil-moisture percentiles into one risk score.

Usage:

    model = RealDataRiskModel(
        "dharali_weather_2015_2025_clean.csv"
    )

    score, label = model.score(
        rainfall_3day_mm=62.6,
        soil_moisture_avg=0.723
    )
"""

import pandas as pd
import numpy as np
from scipy import stats


class RealDataRiskModel:

    def __init__(
        self,
        csv_path: str,
        lookback_days: int = 3
    ):

        self.lookback_days = lookback_days

        # ----------------------------------------------------
        # Load CSV
        # ----------------------------------------------------

        print(f"Loading weather data from: {csv_path}")

        df = pd.read_csv(
            csv_path,
            dtype={"date": str}
        )

        # Remove accidental spaces from column names
        df.columns = df.columns.str.strip()

        print("Columns found:")
        print(df.columns.tolist())

        # ----------------------------------------------------
        # Check required columns
        # ----------------------------------------------------

        required_columns = [
            "date",
            "rainfall_mm",
            "soil_moisture"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}\n"
                f"Available columns: {df.columns.tolist()}"
            )

        # ----------------------------------------------------
        # Convert data types
        # ----------------------------------------------------

        df["date"] = pd.to_datetime(
            df["date"],
            format="%Y%m%d"
        )

        df["rainfall_mm"] = pd.to_numeric(
            df["rainfall_mm"],
            errors="coerce"
        )

        df["soil_moisture"] = pd.to_numeric(
            df["soil_moisture"],
            errors="coerce"
        )

        # ----------------------------------------------------
        # Sort chronologically
        # ----------------------------------------------------

        df = (
            df
            .sort_values("date")
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Remove invalid rows
        # ----------------------------------------------------

        df = df.dropna(
            subset=[
                "date",
                "rainfall_mm",
                "soil_moisture"
            ]
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # Build rolling features
        # ----------------------------------------------------

        # 3-day rainfall total
        df["rolling_rainfall_sum"] = (
            df["rainfall_mm"]
            .rolling(self.lookback_days)
            .sum()
        )

        # 3-day average soil moisture
        df["rolling_soil_avg"] = (
            df["soil_moisture"]
            .rolling(self.lookback_days)
            .mean()
        )

        # ----------------------------------------------------
        # Store data
        # ----------------------------------------------------

        self.history = df

        self.valid = df.dropna(
            subset=[
                "rolling_rainfall_sum",
                "rolling_soil_avg"
            ]
        )

        # ----------------------------------------------------
        # Reference distributions
        # ----------------------------------------------------

        self.ref_rainfall = (
            self.valid["rolling_rainfall_sum"]
            .values
        )

        self.ref_soil = (
            self.valid["rolling_soil_avg"]
            .values
        )

        # ----------------------------------------------------
        # Calculate historical thresholds
        # ----------------------------------------------------

        self.thresholds = {

            "rainfall_75":
                float(
                    np.percentile(
                        self.ref_rainfall,
                        75
                    )
                ),

            "rainfall_90":
                float(
                    np.percentile(
                        self.ref_rainfall,
                        90
                    )
                ),

            "rainfall_95":
                float(
                    np.percentile(
                        self.ref_rainfall,
                        95
                    )
                ),

            "rainfall_99":
                float(
                    np.percentile(
                        self.ref_rainfall,
                        99
                    )
                ),

            "soil_75":
                float(
                    np.percentile(
                        self.ref_soil,
                        75
                    )
                ),

            "soil_90":
                float(
                    np.percentile(
                        self.ref_soil,
                        90
                    )
                ),

            "soil_95":
                float(
                    np.percentile(
                        self.ref_soil,
                        95
                    )
                ),

            "soil_99":
                float(
                    np.percentile(
                        self.ref_soil,
                        99
                    )
                ),
        }

        print(
            f"\nLoaded {len(self.history)} days."
        )

        print(
            f"Historical period: "
            f"{self.history['date'].min().date()} "
            f"to "
            f"{self.history['date'].max().date()}"
        )


    # ========================================================
    # SCORE NEW CONDITIONS
    # ========================================================

    def score(
        self,
        rainfall_3day_mm: float,
        soil_moisture_avg: float
    ):
        """
        Returns:

            (combined_percentile, risk_label)

        rainfall_3day_mm:
            Total rainfall over the last 3 days.

        soil_moisture_avg:
            Average soil moisture over the same 3 days.
        """

        # ----------------------------------------------------
        # Rainfall percentile
        # ----------------------------------------------------

        rainfall_pct = stats.percentileofscore(
            self.ref_rainfall,
            rainfall_3day_mm,
            kind="mean"
        )

        # ----------------------------------------------------
        # Soil moisture percentile
        # ----------------------------------------------------

        soil_pct = stats.percentileofscore(
            self.ref_soil,
            soil_moisture_avg,
            kind="mean"
        )

        # ----------------------------------------------------
        # Combined risk
        # ----------------------------------------------------

        combined = (
            rainfall_pct + soil_pct
        ) / 2

        return (
            float(combined),
            self._label(combined)
        )


    # ========================================================
    # SCORE FROM DAILY SERIES
    # ========================================================

    def score_from_daily_series(
        self,
        recent_rainfall: list,
        recent_soil_moisture: list
    ):
        """
        Pass the last 3 daily readings directly.

        Example:

            rainfall:
                [15.0, 20.0, 45.0]

            soil moisture:
                [0.68, 0.71, 0.75]
        """

        if len(recent_rainfall) != self.lookback_days:

            raise ValueError(
                f"Expected {self.lookback_days} rainfall "
                f"values, got {len(recent_rainfall)}"
            )

        if len(recent_soil_moisture) != self.lookback_days:

            raise ValueError(
                f"Expected {self.lookback_days} soil moisture "
                f"values, got {len(recent_soil_moisture)}"
            )

        rainfall_total = sum(
            recent_rainfall
        )

        soil_average = np.mean(
            recent_soil_moisture
        )

        return self.score(
            rainfall_total,
            soil_average
        )


    # ========================================================
    # SCORE HISTORICAL DATE
    # ========================================================

    def score_date(
        self,
        date_str: str
    ):
        """
        Look up and score a historical date.

        Date format:
            YYYYMMDD

        Example:
            20250805
        """

        target_date = pd.Timestamp(
            date_str
        )

        row = self.history[
            self.history["date"] == target_date
        ]

        if row.empty:

            raise ValueError(
                f"Date {date_str} was not found "
                f"in the historical dataset."
            )

        if (
            row[
                [
                    "rolling_rainfall_sum",
                    "rolling_soil_avg"
                ]
            ]
            .isna()
            .any(axis=None)
        ):

            raise ValueError(
                f"No complete "
                f"{self.lookback_days}-day window "
                f"ending on {date_str}"
            )

        r = row.iloc[0]

        return self.score(
            r["rolling_rainfall_sum"],
            r["rolling_soil_avg"]
        )


    # ========================================================
    # RISK LABEL
    # ========================================================

    @staticmethod
    def _label(
        percentile: float
    ) -> str:

        if percentile >= 95:
            return "Severe"

        if percentile >= 90:
            return "High"

        if percentile >= 75:
            return "Moderate"

        return "Low"


# ============================================================
# TEST / VALIDATION
# ============================================================

if __name__ == "__main__":

    # IMPORTANT:
    # Use the CLEAN CSV file.
    model = RealDataRiskModel(
        "dharali_weather_2015_2025_clean.csv"
    )


    # --------------------------------------------------------
    # Print thresholds
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "CALIBRATED REAL-DATA THRESHOLDS"
    )

    print(
        "========================================"
    )

    for key, value in model.thresholds.items():

        print(
            f"  {key}: {value:.3f}"
        )


    # --------------------------------------------------------
    # Validate actual flood date
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "VALIDATION: ACTUAL DHARALI FLOOD DATE"
    )

    print(
        "========================================"
    )

    score, label = model.score_date(
        "20250805"
    )

    print(
        f"5 Aug 2025 "
        f"(real flood date)"
    )

    print(
        f"Score: {score:.1f}"
    )

    print(
        f"Risk label: {label}"
    )


    # --------------------------------------------------------
    # Sanity check ordinary historical day
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "SANITY CHECK: ORDINARY HISTORICAL DAY"
    )

    print(
        "========================================"
    )

    score, label = model.score_date(
        "20180301"
    )

    print(
        f"1 Mar 2018"
    )

    print(
        f"Score: {score:.1f}"
    )

    print(
        f"Risk label: {label}"
    )


    # --------------------------------------------------------
    # Example live scoring
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "LIVE SCORING EXAMPLE"
    )

    print(
        "========================================"
    )

    score, label = model.score_from_daily_series(

        recent_rainfall=[
            15.0,
            20.0,
            45.0
        ],

        recent_soil_moisture=[
            0.68,
            0.71,
            0.75
        ]
    )

    print(
        f"Hypothetical new event"
    )

    print(
        f"Score: {score:.1f}"
    )

    print(
        f"Risk label: {label}"
    )

