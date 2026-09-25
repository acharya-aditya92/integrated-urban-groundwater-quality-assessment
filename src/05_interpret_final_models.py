"""Calculate held-out permutation importance for the selected Paper 1 models."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".matplotlib"))

import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


INPUT = PROJECT / "data_processed" / "paper1_ml_annual_long_ready.csv"
OUTPUT = PROJECT / "outputs"
SELECTED = OUTPUT / "selected_models_by_parameter.csv"
NUMERIC = ["Year", "AirTemp_C", "Rainfall_mm", "SPI_12"]
CATEGORICAL = ["City", "Stable_Station_ID"]
TARGET = "Annual_Median"


def load_selection_module():
    spec = importlib.util.spec_from_file_location(
        "paper1_model_selection", PROJECT / "src" / "04_select_final_models.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the model-selection script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not INPUT.exists() or not SELECTED.exists():
        raise FileNotFoundError("Run: python src\\04_select_final_models.py")
    OUTPUT.mkdir(exist_ok=True)
    data = pd.read_csv(INPUT).dropna(subset=[TARGET, *NUMERIC, *CATEGORICAL]).copy()
    selected = pd.read_csv(SELECTED)
    selection_module = load_selection_module()
    features = NUMERIC + CATEGORICAL
    rows: list[dict[str, object]] = []

    for selected_row in selected.itertuples(index=False):
        parameter = selected_row.Parameter_Standard
        model_name = selected_row.Selected_model
        train = data.loc[
            data["Parameter_Standard"].eq(parameter) & data["Year"].le(2022)
        ]
        test = data.loc[
            data["Parameter_Standard"].eq(parameter) & data["Year"].ge(2023)
        ]
        model = selection_module.candidates(parameter)[model_name]
        model.fit(train[features], train[TARGET])
        importance = permutation_importance(
            model,
            test[features],
            test[TARGET],
            scoring="neg_mean_absolute_error",
            n_repeats=10,
            random_state=42,
            n_jobs=1,
        )
        for feature, mean, std in zip(
            features, importance.importances_mean, importance.importances_std
        ):
            rows.append(
                {
                    "Parameter_Standard": parameter,
                    "Selected_model": model_name,
                    "Feature": feature,
                    "MAE_increase_when_permuted": mean,
                    "Importance_standard_deviation": std,
                }
            )

    table = pd.DataFrame(rows).sort_values(
        ["Parameter_Standard", "MAE_increase_when_permuted"], ascending=[True, False]
    )
    table.to_csv(OUTPUT / "final_feature_permutation_importance.csv", index=False)

    # Raw MAE has a different unit for every parameter. This relative table
    # therefore allows comparison *within* each parameter, not across units.
    table["Positive_MAE_increase"] = table["MAE_increase_when_permuted"].clip(lower=0)
    denominator = table.groupby("Parameter_Standard")["Positive_MAE_increase"].transform("sum")
    table["Relative_importance_percent"] = (
        100 * table["Positive_MAE_increase"] / denominator.replace(0, float("nan"))
    ).fillna(0)
    table.to_csv(OUTPUT / "final_feature_relative_importance.csv", index=False)

    pivot = table.pivot(
        index="Feature", columns="Parameter_Standard", values="MAE_increase_when_permuted"
    )
    fig, axis = plt.subplots(figsize=(11, 4.5))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_title("Held-out permutation importance: MAE increase when feature is shuffled")
    fig.colorbar(image, ax=axis, label="Increase in MAE")
    fig.tight_layout()
    fig.savefig(OUTPUT / "final_feature_importance_raw_units.png", dpi=220)
    plt.close(fig)

    relative = table.pivot(
        index="Feature", columns="Parameter_Standard", values="Relative_importance_percent"
    )
    fig, axis = plt.subplots(figsize=(11, 4.5))
    image = axis.imshow(relative.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    axis.set_xticks(range(len(relative.columns)), relative.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(relative.index)), relative.index)
    axis.set_title("Relative held-out feature importance within each parameter")
    fig.colorbar(image, ax=axis, label="Share of positive importance (%)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "final_feature_relative_importance.png", dpi=220)
    plt.close(fig)

    notes = "\n".join(
        [
            "How to interpret feature importance",
            "===================================",
            "A larger positive value means that shuffling that feature made held-out",
            "2023-2024 predictions worse; the model relied more on that feature.",
            "Feature importance is an association within this model and dataset.",
            "It does not establish that climate or station identity causes a change",
            "in groundwater quality.",
            "Use final_feature_relative_importance.png to compare feature rankings",
            "within an individual parameter; do not compare raw MAE units across parameters.",
        ]
    )
    (OUTPUT / "feature_importance_notes.txt").write_text(notes + "\n", encoding="utf-8")
    print("Final-model feature importance completed.")
    print("Created a feature-importance table, heatmap, and interpretation notes in outputs/.")


if __name__ == "__main__":
    main()
