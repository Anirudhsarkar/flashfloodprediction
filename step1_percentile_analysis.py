
"""
step1_percentile_analysis.py
------------------------------
Step 1 of the real-data Tier 1 approach.

Instead of training a classifier on one labeled event, we build a
statistical baseline from 10 years of real rainfall + soil moisture data,
then find where the known Dharali flood event (August 2025) sits in that
historical distribution.

This answers:
"How extreme were the conditions before Dharali flooded,
compared to every other day in the last 10 years?"

Required file:
    dharali_weather_2015_2025_clean.csv

Expected columns:
    date
    rainfall_mm
    soil_moisture
"""

import pandas as pd
import numpy as np
from scipy import stats


# ============================================================
# FILE SETTINGS
# ============================================================

# IMPORTANT:
# This is the CLEAN CSV file.
FULL_HISTORY_FILE = "dharali_weather_2015_2025_clean.csv"

# Kept for future use in later steps.
FLOOD_WINDOW_FILE = "dharali_flood_event_jul_aug_2025.csv"


# ============================================================
# FLOOD EVENT SETTINGS
# ============================================================

# Dharali flash flood date
FLOOD_DATE = "2025-08-05"

# Number of days to look back before the flood.
# 3 means:
#   August 3
#   August 4
#   August 5
LOOKBACK_DAYS = 3


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

def load_history(path: str) -> pd.DataFrame:
    """
    Load the historical weather CSV and prepare the date column.
    """

    print(f"Loading historical data from: {path}")

    df = pd.read_csv(path)

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

    print("Columns found:")
    print(df.columns.tolist())

    # Check that required columns exist
    required_columns = [
        "date",
        "rainfall_mm",
        "soil_moisture"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert date from YYYYMMDD format
    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y%m%d"
    )

    # Make sure rainfall and soil moisture are numeric
    df["rainfall_mm"] = pd.to_numeric(
        df["rainfall_mm"],
        errors="coerce"
    )

    df["soil_moisture"] = pd.to_numeric(
        df["soil_moisture"],
        errors="coerce"
    )

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # Remove rows where important values are missing
    df = df.dropna(
        subset=[
            "date",
            "rainfall_mm",
            "soil_moisture"
        ]
    ).reset_index(drop=True)

    return df


# ============================================================
# PERCENTILE FUNCTION
# ============================================================

def percentile_rank(
    value: float,
    distribution: np.ndarray
) -> float:
    """
    Calculate the percentage of historical values
    that were lower than the given value.
    """

    return float(
        stats.percentileofscore(
            distribution,
            value,
            kind="mean"
        )
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    history = load_history(FULL_HISTORY_FILE)

    print(
        f"\nLoaded {len(history)} days of history "
        f"({history.date.min().date()} to "
        f"{history.date.max().date()})\n"
    )


    # --------------------------------------------------------
    # 2. DEFINE FLOOD WINDOW
    # --------------------------------------------------------

    flood_date = pd.Timestamp(FLOOD_DATE)

    window_start = (
        flood_date
        - pd.Timedelta(days=LOOKBACK_DAYS - 1)
    )

    event_window = history[
        (history["date"] >= window_start)
        & (history["date"] <= flood_date)
    ].copy()


    # --------------------------------------------------------
    # 3. CHECK FLOOD WINDOW
    # --------------------------------------------------------

    if event_window.empty:

        raise ValueError(
            f"No rows found for "
            f"{window_start.date()} to "
            f"{flood_date.date()} "
            f"in {FULL_HISTORY_FILE}.\n\n"
            f"Check FLOOD_DATE and make sure the "
            f"historical dataset covers this period."
        )


    # --------------------------------------------------------
    # 4. PRINT FLOOD CONDITIONS
    # --------------------------------------------------------

    print(
        f"Conditions leading up to the flood "
        f"({window_start.date()} to {flood_date.date()}):\n"
    )

    print(
        event_window[
            [
                "date",
                "rainfall_mm",
                "soil_moisture"
            ]
        ].to_string(index=False)
    )


    # --------------------------------------------------------
    # 5. CALCULATE FLOOD-EVENT FEATURES
    # --------------------------------------------------------

    # Total rainfall during the lookback period
    antecedent_rainfall = (
        event_window["rainfall_mm"].sum()
    )

    # Maximum rainfall on any single day
    peak_daily_rainfall = (
        event_window["rainfall_mm"].max()
    )

    # Average soil moisture during the period
    avg_soil_moisture = (
        event_window["soil_moisture"].mean()
    )


    # --------------------------------------------------------
    # 6. BUILD HISTORICAL ROLLING FEATURES
    # --------------------------------------------------------

    # 3-day rolling rainfall total
    history["rolling_rainfall_sum"] = (
        history["rainfall_mm"]
        .rolling(LOOKBACK_DAYS)
        .sum()
    )

    # 3-day average soil moisture
    history["rolling_soil_avg"] = (
        history["soil_moisture"]
        .rolling(LOOKBACK_DAYS)
        .mean()
    )


    # Remove incomplete rolling windows
    valid = history.dropna(
        subset=[
            "rolling_rainfall_sum",
            "rolling_soil_avg"
        ]
    )


    # --------------------------------------------------------
    # 7. CALCULATE PERCENTILES
    # --------------------------------------------------------

    rainfall_pct = percentile_rank(
        antecedent_rainfall,
        valid["rolling_rainfall_sum"].values
    )

    soil_pct = percentile_rank(
        avg_soil_moisture,
        valid["rolling_soil_avg"].values
    )


    # --------------------------------------------------------
    # 8. PRINT RESULTS
    # --------------------------------------------------------

    print(
        "\n--- Where the flood-event conditions sit "
        "in 10 years of history ---"
    )

    print(
        f"\n3-day antecedent rainfall: "
        f"{antecedent_rainfall:.1f} mm"
    )

    print(
        f"Rainfall percentile: "
        f"{rainfall_pct:.1f}th percentile"
    )

    print(
        f"\nPeak single-day rainfall: "
        f"{peak_daily_rainfall:.1f} mm"
    )

    print(
        f"\n3-day average soil moisture: "
        f"{avg_soil_moisture:.3f}"
    )

    print(
        f"Soil moisture percentile: "
        f"{soil_pct:.1f}th percentile"
    )


    # --------------------------------------------------------
    # 9. COMBINED RISK PERCENTILE
    # --------------------------------------------------------

    combined_pct = (
        rainfall_pct + soil_pct
    ) / 2

    print(
        f"\nCombined risk percentile "
        f"(average of rainfall + soil moisture): "
        f"{combined_pct:.1f}"
    )


    # --------------------------------------------------------
    # 10. RISK VERDICT
    # --------------------------------------------------------

    if combined_pct >= 95:

        verdict = (
            "EXTREME — top 5% of conditions "
            "seen in the historical dataset"
        )

    elif combined_pct >= 90:

        verdict = (
            "SEVERE — top 10% of conditions "
            "seen in the historical dataset"
        )

    elif combined_pct >= 75:

        verdict = (
            "HIGH — top 25% of conditions "
            "seen in the historical dataset"
        )

    else:

        verdict = (
            "MODERATE or lower — not statistically "
            "extreme in this history"
        )


    print(f"\nVerdict: {verdict}")


    # --------------------------------------------------------
    # 11. CALCULATE HISTORICAL THRESHOLDS
    # --------------------------------------------------------

    thresholds = {

        "rainfall_95th_pct_mm":
            float(
                np.percentile(
                    valid["rolling_rainfall_sum"],
                    95
                )
            ),

        "rainfall_90th_pct_mm":
            float(
                np.percentile(
                    valid["rolling_rainfall_sum"],
                    90
                )
            ),

        "rainfall_75th_pct_mm":
            float(
                np.percentile(
                    valid["rolling_rainfall_sum"],
                    75
                )
            ),

        "soil_moisture_95th_pct":
            float(
                np.percentile(
                    valid["rolling_soil_avg"],
                    95
                )
            ),

        "soil_moisture_90th_pct":
            float(
                np.percentile(
                    valid["rolling_soil_avg"],
                    90
                )
            ),

        "soil_moisture_75th_pct":
            float(
                np.percentile(
                    valid["rolling_soil_avg"],
                    75
                )
            ),

        "flood_event_rainfall_percentile":
            rainfall_pct,

        "flood_event_soil_moisture_percentile":
            soil_pct,

        "flood_event_combined_percentile":
            combined_pct,

        "lookback_days":
            LOOKBACK_DAYS
    }


    # --------------------------------------------------------
    # 12. SAVE THRESHOLDS
    # --------------------------------------------------------

    output_file = "tier1_risk_thresholds.csv"

    pd.Series(thresholds).to_csv(
        output_file,
        header=["value"]
    )

    print(
        f"\nSaved calibrated thresholds to "
        f"{output_file}"
    )

    print(
        "\nStep 1 completed successfully."
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()

