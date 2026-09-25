"""Run transparent station-aware baseline models for Paper 1.

Models are trained using 2015-2022 and evaluated only on the later 2023-2024
records.  This prevents the overly optimistic result produced by a random
train/test split.  Each groundwater parameter is modelled separately because
their units and value ranges are different.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
# Keep Matplotlib's small cache inside the project, where the script always
# has write permission on this Windows installation.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".matplotlib"))

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


INPUT = PROJECT / "data_processed" / "paper1_ml_annual_long_ready.csv"
OUTPUT = PROJECT / "outputs"

TRAIN_END_YEAR = 2022
TEST_START_YEAR = 2023
FEATURES_NUMERIC = ["Year", "AirTemp_C", "Rainfall_mm", "SPI_12"]
FEATURES_CATEGORICAL = ["City", "Stable_Station_ID"]
TARGET = "Annual_Median"


def metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    """Calculate prediction metrics in the original parameter unit."""
    return {
        "MAE": mean_absolute_error(y_true, prediction),
        "RMSE": mean_squared_error(y_true, prediction) ** 0.5,
        "R2": r2_score(y_true, prediction),
    }


def station_aware_forest() -> Pipeline:
    """Create the pre-processing and Random Forest baseline pipeline."""
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", FEATURES_NUMERIC),
            (
                "station_and_city",
                OneHotEncoder(handle_unknown="ignore"),
                FEATURES_CATEGORICAL,
            ),
        ]
    )
    forest = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        # One worker keeps the run portable on restricted Windows systems.
        n_jobs=1,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", forest)])


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            "ML-ready data is missing. Run: python src\\01_prepare_ml_data.py"
        )
    OUTPUT.mkdir(exist_ok=True)
    data = pd.read_csv(INPUT)
    required = {TARGET, "Parameter_Standard", *FEATURES_NUMERIC, *FEATURES_CATEGORICAL}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"ML-ready dataset is missing: {sorted(missing)}")

    # Retain missing observations in the master data, but a supervised model
    # cannot use a row without its measured target. The excluded count is
    # reported and the unmodified master remains the source of record.
    model_data = data.dropna(subset=[TARGET, *FEATURES_NUMERIC, *FEATURES_CATEGORICAL]).copy()
    excluded_missing_target = int(data[TARGET].isna().sum())
    train = model_data.loc[model_data["Year"] <= TRAIN_END_YEAR].copy()
    test = model_data.loc[model_data["Year"] >= TEST_START_YEAR].copy()
    if train.empty or test.empty:
        raise ValueError("The temporal train/test split is empty")

    records: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for parameter in sorted(data["Parameter_Standard"].unique()):
        train_part = train.loc[train["Parameter_Standard"].eq(parameter)].copy()
        test_part = test.loc[test["Parameter_Standard"].eq(parameter)].copy()
        x_train = train_part[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
        x_test = test_part[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
        y_train = train_part[TARGET]
        y_test = test_part[TARGET]

        train_stations = set(train_part["Stable_Station_ID"])
        unseen_stations = int((~test_part["Stable_Station_ID"].isin(train_stations)).sum())

        models = {
            "Dummy_median": DummyRegressor(strategy="median"),
            "Station_aware_random_forest": station_aware_forest(),
        }
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            prediction = np.asarray(model.predict(x_test), dtype=float)
            result = metrics(y_test, prediction)
            records.append(
                {
                    "Parameter_Standard": parameter,
                    "Model": model_name,
                    "Train_records": len(train_part),
                    "Test_records": len(test_part),
                    "Test_records_new_station": unseen_stations,
                    **result,
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "Model": model_name,
                        "Parameter_Standard": parameter,
                        "City": test_part["City"].to_numpy(),
                        "Stable_Station_ID": test_part["Stable_Station_ID"].to_numpy(),
                        "Year": test_part["Year"].to_numpy(),
                        "Observed_Annual_Median": y_test.to_numpy(),
                        "Predicted_Annual_Median": prediction,
                    }
                )
            )

    metrics_table = pd.DataFrame(records).sort_values(["Parameter_Standard", "Model"])
    prediction_table = pd.concat(predictions, ignore_index=True)
    metrics_table.to_csv(OUTPUT / "baseline_temporal_metrics.csv", index=False)
    prediction_table.to_csv(OUTPUT / "baseline_temporal_predictions.csv", index=False)

    plot_data = metrics_table.pivot(
        index="Parameter_Standard", columns="Model", values="MAE"
    )
    ax = plot_data.plot(kind="bar", figsize=(11, 5), rot=35)
    ax.set_title("Temporal test MAE: 2023-2024")
    ax.set_xlabel("Groundwater parameter")
    ax.set_ylabel("MAE (parameter's original unit)")
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(OUTPUT / "baseline_temporal_mae.png", dpi=200)
    plt.close()

    design = "\n".join(
        [
            "Paper 1 baseline modelling design",
            "================================",
            "Purpose: predict each annual groundwater parameter separately.",
            "Training period: 2015-2022.",
            "Held-out temporal test: 2023-2024.",
            "Features: City, Stable_Station_ID, Year, AirTemp_C, Rainfall_mm, SPI_12.",
            "Not used as a feature: Days_Measured (it describes observation effort,",
            "not a value known before a future sampling year).",
            "Models compared: median dummy baseline and station-aware Random Forest.",
            "Metrics: MAE, RMSE, R2, reported separately for each parameter.",
            f"Rows excluded only from model fitting because Annual_Median is blank: {excluded_missing_target}.",
            "This is the first baseline only; no hyperparameter tuning or final model",
            "selection has been performed.",
        ]
    )
    (OUTPUT / "baseline_model_design.txt").write_text(design + "\n", encoding="utf-8")

    print("Baseline modelling completed.")
    print(f"Training records (2015-{TRAIN_END_YEAR}): {len(train):,}")
    print(f"Test records ({TEST_START_YEAR}-2024): {len(test):,}")
    print(f"Rows retained in source data but excluded from fitting (blank target): {excluded_missing_target:,}")
    print("Created baseline metrics, predictions, figure, and model-design files in outputs/.")


if __name__ == "__main__":
    main()
