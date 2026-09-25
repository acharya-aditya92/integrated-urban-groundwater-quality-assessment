"""Sensitivity check for the documented observed WQI weighting choice.

The main WQI remains the project's inverse-limit weighted arithmetic index.
This script does not replace it.  It transparently compares it with an
equal-weight version using the same quality ratings and standards, showing how
much the ranking and class results depend on the weighting decision.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "outputs"
PARAMETERS = ["EC", "pH", "Manganese", "Iron", "Nitrate", "Chloride", "Sulfate"]

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".matplotlib"))
import matplotlib.pyplot as plt


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


def main() -> None:
    records_file = OUTPUT / "observed_wqi_records.csv"
    if not records_file.exists():
        raise FileNotFoundError("Run src\\06_calculate_observed_wqi.py first")

    frame = pd.read_csv(records_file)
    rating_columns = [f"QI_{parameter}" for parameter in PARAMETERS]
    missing = set(["WQI", "WQI_Class", *rating_columns]).difference(frame.columns)
    if missing:
        raise ValueError(f"Observed WQI records are missing: {sorted(missing)}")

    frame["Equal_Weight_WQI"] = frame[rating_columns].mean(axis=1)
    frame["Equal_Weight_WQI_Class"] = frame["Equal_Weight_WQI"].map(classify)
    frame["Same_WQI_Class"] = frame["WQI_Class"].eq(frame["Equal_Weight_WQI_Class"])
    rank_rho, rank_pvalue = spearmanr(frame["WQI"], frame["Equal_Weight_WQI"])

    summary = pd.DataFrame(
        [
            {
                "Records": len(frame),
                "Main_method": "Inverse-limit weighted arithmetic WQI",
                "Sensitivity_method": "Equal-weight arithmetic mean of the same QI values",
                "Spearman_rank_correlation": rank_rho,
                "Rank_correlation_p_value": rank_pvalue,
                "Exact_class_agreement": frame["Same_WQI_Class"].mean(),
                "Main_WQI_median": frame["WQI"].median(),
                "Equal_weight_WQI_median": frame["Equal_Weight_WQI"].median(),
            }
        ]
    )
    class_comparison = (
        frame.groupby(["WQI_Class", "Equal_Weight_WQI_Class"], as_index=False)
        .size()
        .rename(columns={"size": "Records"})
        .sort_values(["WQI_Class", "Equal_Weight_WQI_Class"])
    )
    city_summary = (
        frame.groupby("City", as_index=False)
        .agg(
            Records=("WQI", "size"),
            Main_WQI_median=("WQI", "median"),
            Equal_weight_WQI_median=("Equal_Weight_WQI", "median"),
            Exact_class_agreement=("Same_WQI_Class", "mean"),
        )
        .sort_values("City")
    )

    frame.to_csv(OUTPUT / "wqi_weight_sensitivity_records.csv", index=False)
    summary.to_csv(OUTPUT / "wqi_weight_sensitivity_summary.csv", index=False)
    city_summary.to_csv(OUTPUT / "wqi_weight_sensitivity_by_city.csv", index=False)
    class_comparison.to_csv(OUTPUT / "wqi_weight_sensitivity_class_comparison.csv", index=False)

    low = max(min(frame["WQI"].min(), frame["Equal_Weight_WQI"].min()), 0.01)
    high = max(frame["WQI"].max(), frame["Equal_Weight_WQI"].max())
    fig, axis = plt.subplots(figsize=(7, 6))
    for city, group in frame.groupby("City", sort=True):
        axis.scatter(group["WQI"], group["Equal_Weight_WQI"], label=city, alpha=0.75)
    axis.plot([low, high], [low, high], "k--", linewidth=1, label="Same score")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel("Main inverse-limit weighted WQI")
    axis.set_ylabel("Sensitivity equal-weight WQI")
    axis.set_title("WQI weighting sensitivity: same standards, alternative weights")
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "wqi_weight_sensitivity_comparison.png", dpi=220)
    plt.close(fig)

    report = "\n".join(
        [
            "Paper 1 WQI weighting sensitivity report",
            "========================================",
            "Main index: inverse-limit weighted arithmetic WQI.",
            "Sensitivity index: equal-weight arithmetic mean of the same seven quality ratings.",
            "This test does not replace the main WQI; it makes the weighting dependence transparent.",
            f"Records compared: {len(frame):,}",
            f"Spearman rank correlation: {rank_rho:.4f}",
            f"Exact class agreement: {frame['Same_WQI_Class'].mean():.1%}",
            "Interpretation: low agreement means that WQI classes are materially dependent on the weighting method.",
            "",
            "By city:",
            city_summary.to_string(index=False),
        ]
    )
    (OUTPUT / "wqi_weight_sensitivity_method_report.txt").write_text(report + "\n", encoding="utf-8")
    print("WQI weighting sensitivity calculation completed.")
    print(f"Records compared: {len(frame):,}")
    print(f"Rank correlation: {rank_rho:.4f}; exact class agreement: {frame['Same_WQI_Class'].mean():.1%}")
    print("Created sensitivity records, summaries, comparison, figure, and report in outputs/.")


if __name__ == "__main__":
    main()
