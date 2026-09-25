"""Select Paper 1 models without using the final 2023-2024 test period.

Candidate selection uses 2015-2021 for training and 2022 for validation.
The selected model for each parameter is then refitted on 2015-2022 and
evaluated once on untouched 2023-2024 records.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


PROJECT = Path(__file__).resolve().parents[1]
INPUT = PROJECT / "data_processed" / "paper1_ml_annual_long_ready.csv"
OUTPUT = PROJECT / "outputs"

NUMERIC = ["Year", "AirTemp_C", "Rainfall_mm", "SPI_12"]
CATEGORICAL = ["City", "Stable_Station_ID"]
TARGET = "Annual_Median"


def make_pipeline(regressor: object) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC),
            ("station_and_city", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", regressor)])


def candidates(parameter: str) -> dict[str, object]:
    """Return appropriate candidate models for one groundwater parameter."""
    random_forest = make_pipeline(
        RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, max_features=0.8,
            random_state=42, n_jobs=1,
        )
    )
    extra_trees = make_pipeline(
        ExtraTreesRegressor(
            n_estimators=100, min_samples_leaf=2, max_features=0.8,
            random_state=42, n_jobs=1,
        )
    )
    result: dict[str, object] = {
        "Random_forest_raw": random_forest,
        "Extra_trees_raw": extra_trees,
    }
    # pH is unitless and bounded, so it remains on its original scale. The
    # concentration parameters are non-negative and can benefit from log1p.
    if parameter != "pH":
        result["Extra_trees_log1p"] = TransformedTargetRegressor(
            regressor=make_pipeline(
                ExtraTreesRegressor(
                    n_estimators=100, min_samples_leaf=2, max_features=0.8,
                    random_state=42, n_jobs=1,
                )
            ),
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    return result


def score(observed: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(observed, predicted),
        "RMSE": mean_squared_error(observed, predicted) ** 0.5,
        "R2": r2_score(observed, predicted),
    }


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError("Run: python src\\01_prepare_ml_data.py")
    OUTPUT.mkdir(exist_ok=True)
    data = pd.read_csv(INPUT).dropna(subset=[TARGET, *NUMERIC, *CATEGORICAL]).copy()
    feature_columns = NUMERIC + CATEGORICAL

    selection_train = data.loc[data["Year"] <= 2021].copy()
    validation = data.loc[data["Year"].eq(2022)].copy()
    final_train = data.loc[data["Year"] <= 2022].copy()
    final_test = data.loc[data["Year"] >= 2023].copy()

    validation_records: list[dict[str, object]] = []
    final_records: list[dict[str, object]] = []
    final_predictions: list[pd.DataFrame] = []
    selected_records: list[dict[str, object]] = []

    for parameter in sorted(data["Parameter_Standard"].unique()):
        select_train_part = selection_train.loc[selection_train["Parameter_Standard"].eq(parameter)]
        validation_part = validation.loc[validation["Parameter_Standard"].eq(parameter)]
        if select_train_part.empty or validation_part.empty:
            raise ValueError(f"Missing selection data for {parameter}")

        candidate_results: list[dict[str, object]] = []
        for name, model in candidates(parameter).items():
            model.fit(select_train_part[feature_columns], select_train_part[TARGET])
            predicted = np.asarray(model.predict(validation_part[feature_columns]), dtype=float)
            row = {
                "Parameter_Standard": parameter,
                "Candidate_model": name,
                "Selection_train_records": len(select_train_part),
                "Validation_records_2022": len(validation_part),
                **score(validation_part[TARGET], predicted),
            }
            validation_records.append(row)
            candidate_results.append(row)

        selected = min(candidate_results, key=lambda row: row["MAE"])
        selected_name = str(selected["Candidate_model"])
        selected_records.append(
            {
                "Parameter_Standard": parameter,
                "Selected_model": selected_name,
                "Selection_metric": "lowest 2022 MAE",
                "Validation_MAE": selected["MAE"],
                "Validation_RMSE": selected["RMSE"],
                "Validation_R2": selected["R2"],
            }
        )

        train_part = final_train.loc[final_train["Parameter_Standard"].eq(parameter)]
        test_part = final_test.loc[final_test["Parameter_Standard"].eq(parameter)]
        final_models = {
            "Dummy_median": DummyRegressor(strategy="median"),
            selected_name: candidates(parameter)[selected_name],
        }
        for name, model in final_models.items():
            model.fit(train_part[feature_columns], train_part[TARGET])
            predicted = np.asarray(model.predict(test_part[feature_columns]), dtype=float)
            final_records.append(
                {
                    "Parameter_Standard": parameter,
                    "Model": name,
                    "Train_records_2015_2022": len(train_part),
                    "Final_test_records_2023_2024": len(test_part),
                    **score(test_part[TARGET], predicted),
                }
            )
            if name != "Dummy_median":
                final_predictions.append(
                    pd.DataFrame(
                        {
                            "Parameter_Standard": parameter,
                            "Selected_model": name,
                            "City": test_part["City"].to_numpy(),
                            "Stable_Station_ID": test_part["Stable_Station_ID"].to_numpy(),
                            "Year": test_part["Year"].to_numpy(),
                            "Observed_Annual_Median": test_part[TARGET].to_numpy(),
                            "Predicted_Annual_Median": predicted,
                        }
                    )
                )

    pd.DataFrame(validation_records).sort_values(["Parameter_Standard", "Candidate_model"]).to_csv(
        OUTPUT / "candidate_validation_metrics_2022.csv", index=False
    )
    pd.DataFrame(selected_records).sort_values("Parameter_Standard").to_csv(
        OUTPUT / "selected_models_by_parameter.csv", index=False
    )
    pd.DataFrame(final_records).sort_values(["Parameter_Standard", "Model"]).to_csv(
        OUTPUT / "final_temporal_metrics_2023_2024.csv", index=False
    )
    pd.concat(final_predictions, ignore_index=True).to_csv(
        OUTPUT / "final_temporal_predictions_2023_2024.csv", index=False
    )

    notes = "\n".join(
        [
            "Paper 1 model-selection protocol",
            "================================",
            "Candidate models are selected separately for each parameter using",
            "only 2015-2021 training data and 2022 validation MAE.",
            "The selected model is then retrained on 2015-2022 and evaluated once",
            "on the untouched 2023-2024 temporal test set.",
            "This prevents choosing a model because it happened to fit the final",
            "test period best. Station ID is a categorical feature, not numeric.",
        ]
    )
    (OUTPUT / "model_selection_protocol.txt").write_text(notes + "\n", encoding="utf-8")
    print("Time-based model selection completed.")
    print("Created validation results, selected models, and final 2023-2024 metrics in outputs/.")


if __name__ == "__main__":
    main()
