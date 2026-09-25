"""Create transparent diagnostics for the Paper 1 baseline temporal test."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".matplotlib"))

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


OUTPUT = PROJECT / "outputs"
PREDICTIONS = OUTPUT / "baseline_temporal_predictions.csv"


def summarise(group: pd.DataFrame) -> pd.Series:
    observed = group["Observed_Annual_Median"]
    predicted = group["Predicted_Annual_Median"]
    return pd.Series(
        {
            "Test_records": len(group),
            "MAE": mean_absolute_error(observed, predicted),
            "RMSE": mean_squared_error(observed, predicted) ** 0.5,
            "R2": r2_score(observed, predicted) if len(group) >= 2 else np.nan,
        }
    )


def main() -> None:
    if not PREDICTIONS.exists():
        raise FileNotFoundError("Run: python src\\02_run_baseline_models.py")

    data = pd.read_csv(PREDICTIONS)
    forest = data.loc[data["Model"].eq("Station_aware_random_forest")].copy()
    if forest.empty:
        raise ValueError("Station-aware Random Forest predictions were not found")

    by_city = (
        forest.groupby(["Parameter_Standard", "City"], as_index=False)
        .apply(summarise, include_groups=False)
        .sort_values(["Parameter_Standard", "City"])
    )
    by_city.to_csv(OUTPUT / "baseline_temporal_metrics_by_city.csv", index=False)

    parameters = sorted(forest["Parameter_Standard"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for axis, parameter in zip(axes.flat, parameters):
        part = forest.loc[forest["Parameter_Standard"].eq(parameter)]
        for city, city_part in part.groupby("City"):
            axis.scatter(
                city_part["Observed_Annual_Median"],
                city_part["Predicted_Annual_Median"],
                alpha=0.7,
                s=28,
                label=city,
            )
        lower = min(part["Observed_Annual_Median"].min(), part["Predicted_Annual_Median"].min())
        upper = max(part["Observed_Annual_Median"].max(), part["Predicted_Annual_Median"].max())
        padding = (upper - lower) * 0.05 if upper > lower else 1
        axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], "k--", lw=1)
        axis.set_title(parameter)
        axis.set_xlabel("Observed annual median")
        axis.set_ylabel("Predicted annual median")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="City", loc="lower right")
    fig.suptitle("Station-aware Random Forest: 2023-2024 temporal test", y=0.98)
    fig.tight_layout()
    fig.savefig(OUTPUT / "baseline_observed_vs_predicted.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    notes = "\n".join(
        [
            "How to interpret baseline diagnostics",
            "====================================",
            "Each point is a held-out 2023 or 2024 annual parameter record.",
            "Points near the dashed line have accurate predictions.",
            "City-level metrics are descriptive: small city-parameter subsets have",
            "limited test records and should not be over-interpreted alone.",
            "The primary Paper 1 comparison remains the pooled temporal metrics by",
            "parameter in baseline_temporal_metrics.csv.",
        ]
    )
    (OUTPUT / "baseline_diagnostic_notes.txt").write_text(notes + "\n", encoding="utf-8")
    print("Baseline diagnostics completed.")
    print("Created city-level metrics, observed-vs-predicted chart, and notes in outputs/.")


if __name__ == "__main__":
    main()
