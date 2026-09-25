"""Evaluate WQI derived from the final parameter-model predictions.

This is deliberately a derived evaluation: the models predict the seven
parameters first, then exactly the same documented rule-based WQI formula is
applied.  WQI itself is never used as a machine-learning target.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "outputs"
CONFIG_FILE = PROJECT / "wqi_standards_trinkwv_2023.json"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".matplotlib"))
import matplotlib.pyplot as plt


PARAMETERS = ["EC", "pH", "Manganese", "Iron", "Nitrate", "Chloride", "Sulfate"]
KEYS = ["City", "Stable_Station_ID", "Year"]


def classify(wqi: float) -> str:
    if wqi <= 25:
        return "Excellent"
    if wqi <= 50:
        return "Good"
    if wqi <= 75:
        return "Poor"
    if wqi <= 100:
        return "Very Poor"
    return "Unsuitable"


def quality_rating(parameter: str, values: pd.Series, settings: dict[str, object]) -> pd.Series:
    if parameter != "pH":
        return 100 * values / float(settings["limit"])

    ideal = float(settings["ideal"])
    lower = float(settings["lower_limit"])
    upper = float(settings["upper_limit"])
    return np.where(
        values.ge(ideal),
        100 * (values - ideal).abs() / (upper - ideal),
        100 * (values - ideal).abs() / (ideal - lower),
    )


def calculate_wqi(frame: pd.DataFrame, prefix: str, config: dict[str, object]) -> pd.DataFrame:
    """Add WQI calculated from a set of parameter columns with a given prefix."""
    weights = {
        parameter: 1 / float(config["parameters"][parameter]["weight_basis"])
        for parameter in PARAMETERS
    }
    total_weight = sum(weights.values())
    contributions = []
    for parameter in PARAMETERS:
        column = f"{prefix}{parameter}"
        rating = quality_rating(parameter, frame[column], config["parameters"][parameter])
        contribution = rating * weights[parameter]
        output_column = f"Contribution_{prefix}{parameter}"
        frame[output_column] = contribution
        contributions.append(output_column)
    frame[f"{prefix}WQI"] = frame[contributions].sum(axis=1) / total_weight
    frame[f"{prefix}WQI_Class"] = frame[f"{prefix}WQI"].map(classify)
    return frame


def main() -> None:
    predictions_file = OUTPUT / "final_temporal_predictions_2023_2024.csv"
    observed_file = OUTPUT / "observed_wqi_records.csv"
    if not predictions_file.exists() or not observed_file.exists() or not CONFIG_FILE.exists():
        raise FileNotFoundError("Run scripts 04 and 06 before evaluating derived WQI")

    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_file)
    observed = pd.read_csv(observed_file)

    needed_prediction_columns = {*KEYS, "Parameter_Standard", "Predicted_Annual_Median"}
    missing = needed_prediction_columns.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")

    prediction_wide = predictions.pivot_table(
        index=KEYS,
        columns="Parameter_Standard",
        values="Predicted_Annual_Median",
        aggfunc="first",
    ).reset_index()
    missing_parameters = set(PARAMETERS).difference(prediction_wide.columns)
    if missing_parameters:
        raise ValueError(f"Predictions do not include: {sorted(missing_parameters)}")
    prediction_wide = prediction_wide.dropna(subset=PARAMETERS).copy()
    prediction_wide = prediction_wide.rename(columns={p: f"Predicted_{p}" for p in PARAMETERS})

    observed_needed = [*KEYS, *PARAMETERS, "WQI", "WQI_Class"]
    missing_observed = set(observed_needed).difference(observed.columns)
    if missing_observed:
        raise ValueError(f"Observed WQI file is missing columns: {sorted(missing_observed)}")
    observed = observed[observed_needed].rename(
        columns={**{p: f"Observed_{p}" for p in PARAMETERS}, "WQI": "Observed_WQI", "WQI_Class": "Observed_WQI_Class"}
    )

    comparison = prediction_wide.merge(observed, on=KEYS, how="inner", validate="one_to_one")
    if comparison.empty:
        raise ValueError("No held-out prediction rows have all seven observed WQI parameters")

    # Concentrations cannot be negative.  This safeguard is applied only at the
    # WQI transformation stage and is recorded so it cannot be hidden.
    constrained = []
    for parameter in PARAMETERS:
        col = f"Predicted_{parameter}"
        if parameter != "pH":
            count = int(comparison[col].lt(0).sum())
            if count:
                comparison.loc[comparison[col].lt(0), col] = 0.0
                constrained.append(f"{parameter}: {count}")

    comparison = calculate_wqi(comparison, "Predicted_", config)
    comparison["WQI_Absolute_Error"] = (comparison["Predicted_WQI"] - comparison["Observed_WQI"]).abs()
    comparison["WQI_Class_Match"] = comparison["Predicted_WQI_Class"].eq(comparison["Observed_WQI_Class"])

    mae = mean_absolute_error(comparison["Observed_WQI"], comparison["Predicted_WQI"])
    rmse = np.sqrt(mean_squared_error(comparison["Observed_WQI"], comparison["Predicted_WQI"]))
    r2 = r2_score(comparison["Observed_WQI"], comparison["Predicted_WQI"])
    class_accuracy = comparison["WQI_Class_Match"].mean()
    metrics = pd.DataFrame(
        [{
            "Evaluation_set": "2023-2024 held-out records with all seven predicted and observed parameters",
            "WQI_records": len(comparison),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Exact_WQI_class_agreement": class_accuracy,
        }]
    )
    city_metrics = (
        comparison.groupby("City", as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "WQI_records": len(group),
                    "MAE": mean_absolute_error(group["Observed_WQI"], group["Predicted_WQI"]),
                    "RMSE": np.sqrt(mean_squared_error(group["Observed_WQI"], group["Predicted_WQI"])),
                    "Exact_WQI_class_agreement": group["WQI_Class_Match"].mean(),
                }
            ),
            include_groups=False,
        )
        .sort_values("City")
    )
    class_comparison = (
        comparison.groupby(["Observed_WQI_Class", "Predicted_WQI_Class"], as_index=False)
        .size()
        .rename(columns={"size": "Records"})
    )

    OUTPUT.mkdir(exist_ok=True)
    comparison.to_csv(OUTPUT / "derived_wqi_heldout_predictions_2023_2024.csv", index=False)
    metrics.to_csv(OUTPUT / "derived_wqi_heldout_metrics_2023_2024.csv", index=False)
    city_metrics.to_csv(OUTPUT / "derived_wqi_heldout_metrics_by_city_2023_2024.csv", index=False)
    class_comparison.to_csv(OUTPUT / "derived_wqi_heldout_class_comparison_2023_2024.csv", index=False)

    low = min(comparison["Observed_WQI"].min(), comparison["Predicted_WQI"].min())
    high = max(comparison["Observed_WQI"].max(), comparison["Predicted_WQI"].max())
    low = max(low, 0.01)
    high = max(high, low * 10)
    fig, axis = plt.subplots(figsize=(7, 6))
    for city, group in comparison.groupby("City", sort=True):
        axis.scatter(group["Observed_WQI"], group["Predicted_WQI"], label=city, alpha=0.8)
    axis.plot([low, high], [low, high], "k--", linewidth=1, label="Perfect prediction")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel("Observed WQI")
    axis.set_ylabel("WQI derived from ML parameter predictions")
    axis.set_title("Held-out derived WQI: 2023-2024")
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "derived_wqi_observed_vs_predicted_2023_2024.png", dpi=220)
    plt.close(fig)

    report = "\n".join(
        [
            "Paper 1 derived WQI evaluation report",
            "====================================",
            "Purpose: evaluate WQI obtained by applying the documented rule to final parameter-model predictions.",
            "This is not a direct WQI machine-learning model.",
            "Evaluation period: 2023-2024, held out from all final-model fitting.",
            f"Complete WQI records available for comparison: {len(comparison):,}",
            f"WQI MAE: {mae:.4f}",
            f"WQI RMSE: {rmse:.4f}",
            f"WQI R2: {r2:.4f}",
            f"Exact WQI class agreement: {class_accuracy:.1%}",
            "Negative concentration predictions constrained to zero before applying WQI: " + (", ".join(constrained) if constrained else "none"),
            "Caution: inverse-limit weighting makes this WQI particularly sensitive to iron and manganese values.",
            "Use the WQI prediction only together with parameter-level results and the WQI sensitivity analysis planned next.",
            "",
            "Metrics by city:",
            city_metrics.to_string(index=False),
        ]
    )
    (OUTPUT / "derived_wqi_heldout_method_report.txt").write_text(report + "\n", encoding="utf-8")
    print("Derived held-out WQI evaluation completed.")
    print(f"Complete 2023-2024 WQI records evaluated: {len(comparison):,}")
    print(f"WQI MAE: {mae:.4f}; RMSE: {rmse:.4f}; R2: {r2:.4f}; class agreement: {class_accuracy:.1%}")
    print("Created held-out WQI predictions, metrics, class comparison, figure, and method report in outputs/.")


if __name__ == "__main__":
    main()
