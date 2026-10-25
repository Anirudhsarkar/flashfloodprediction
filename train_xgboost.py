from pathlib import Path
import json

from xgboost_flood_model import (
    make_labeled_dataset,
    chronological_split,
    train_model,
    evaluate,
    save_bundle,
    FEATURES,
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


EVENT_LABELS = (
    BASE_DIR /
    "event_labels.csv"
)


MODEL_OUTPUT_DIR = (
    BASE_DIR /
    "xgboost_model"
)


TRAINING_DATA_OUTPUT = (
    BASE_DIR /
    "xgboost_training_data.csv"
)


# ============================================================
# BUILD DATASET
# ============================================================

print()
print("=" * 60)
print("DHARALI XGBOOST FLOOD MODEL")
print("=" * 60)


print()
print(
    "Loading weather data and historical labels..."
)


dataset = make_labeled_dataset(
    WEATHER_CSV,
    EVENT_LABELS
)


dataset.to_csv(
    TRAINING_DATA_OUTPUT,
    index=False
)


print(
    f"Training dataset saved to:\n"
    f"{TRAINING_DATA_OUTPUT}"
)


# ============================================================
# SPLIT DATA
# ============================================================

train, validation, test = (
    chronological_split(dataset)
)


print()
print("=" * 60)
print("CHRONOLOGICAL DATA SPLIT")
print("=" * 60)


print(
    f"Total labeled rows: "
    f"{len(dataset)}"
)


print()
print(
    "TRAINING PERIOD"
)

print(
    "2015-2023"
)

print(
    f"Rows: {len(train)}"
)

print(
    f"Flood rows: "
    f"{int(train['flood'].sum())}"
)


print()
print(
    "VALIDATION PERIOD"
)

print(
    "2024"
)

print(
    f"Rows: {len(validation)}"
)

print(
    f"Flood rows: "
    f"{int(validation['flood'].sum())}"
)


print()
print(
    "TEST PERIOD"
)

print(
    "2025"
)

print(
    f"Rows: {len(test)}"
)

print(
    f"Flood rows: "
    f"{int(test['flood'].sum())}"
)


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)


model = train_model(
    train,
    validation
)


print()
print(
    "Training complete."
)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print()
print("=" * 60)
print("VALIDATION RESULTS - 2024")
print("=" * 60)


validation_metrics = evaluate(
    model,
    validation
)


if validation_metrics:

    print(
        json.dumps(
            validation_metrics,
            indent=4
        )
    )

else:

    print(
        "No validation data available."
    )


# ============================================================
# TEST EVALUATION
# ============================================================

print()
print("=" * 60)
print("FINAL TEST RESULTS - 2025")
print("=" * 60)


test_metrics = evaluate(
    model,
    test
)


if test_metrics:

    print(
        json.dumps(
            test_metrics,
            indent=4
        )
    )

else:

    print(
        "No test data available."
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)


importance_list = list(
    zip(
        FEATURES,
        model.feature_importances_
    )
)


importance_list = sorted(
    importance_list,
    key=lambda item: item[1],
    reverse=True
)


for feature, importance in importance_list:

    print(
        f"{feature:25s} "
        f"{importance:.5f}"
    )


# ============================================================
# MODEL METADATA
# ============================================================

metadata = {

    "model_type":
        "XGBoost Binary Classifier",

    "objective":
        "binary:logistic",

    "features":
        FEATURES,

    "training_period":
        "2015-2023",

    "validation_period":
        "2024",

    "testing_period":
        "2025",

    "decision_threshold":
        0.50,

    "validation_metrics":
        validation_metrics,

    "test_metrics":
        test_metrics,
}


# ============================================================
# SAVE MODEL
# ============================================================

save_bundle(
    model,
    MODEL_OUTPUT_DIR,
    metadata
)


print()
print("=" * 60)
print("DONE")
print("=" * 60)


print()
print(
    "Model:"
)

print(
    MODEL_OUTPUT_DIR /
    "xgboost_flood_model.pkl"
)


print()
print(
    "Next run:"
)

print(
    "python historical_analysis.py"
)