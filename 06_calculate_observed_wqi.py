"""Calculate a documented rule-based WQI for the 436 complete Paper 1 records."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data_raw"
OUTPUT = PROJECT / "outputs"
WQI_WORKBOOK = RAW / "Paper1_WQI_Master_Workbook.xlsx"
CONFIG_FILE = PROJECT / "wqi_standards_trinkwv_2023.json"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".matplotlib"))
import matplotlib.pyplot as plt


PARAMETERS = ["EC", "pH", "Manganese", "Iron", "Nitrate", "Chloride", "Sulfate"]
REQUIRED = ["City", "Station_ID", "Stable_Station_ID", "Year", *PARAMETERS]


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
    """Return a 0+ quality rating, where 100 equals the standard limit."""
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


def main() -> None:
    if not WQI_WORKBOOK.exists() or not CONFIG_FILE.exists():
        raise FileNotFoundError("The WQI master workbook or WQI standards configuration is missing")
    OUTPUT.mkdir(exist_ok=True)
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    frame = pd.read_excel(WQI_WORKBOOK, sheet_name="Paper1_WQI_Master")
    missing = set(REQUIRED).difference(frame.columns)
    if missing:
        raise ValueError(f"WQI master is missing columns: {sorted(missing)}")
    frame = frame[REQUIRED].copy()
    if len(frame) != 436:
        raise ValueError(f"Expected 436 WQI-complete records, found {len(frame)}")
    if frame[PARAMETERS].isna().any().any():
        raise ValueError("Observed WQI must use only complete records")
    if frame.duplicated(["Stable_Station_ID", "Year"]).any():
        raise ValueError("Duplicate station-year record in WQI master")

    weights: dict[str, float] = {}
    for parameter in PARAMETERS:
        settings = config["parameters"][parameter]
        weight = 1 / float(settings["weight_basis"])
        weights[parameter] = weight
        frame[f"QI_{parameter}"] = quality_rating(parameter, frame[parameter], settings)
        frame[f"Contribution_{parameter}"] = frame[f"QI_{parameter}"] * weight

    total_weight = sum(weights.values())
    contribution_columns = [f"Contribution_{parameter}" for parameter in PARAMETERS]
    frame["WQI"] = frame[contribution_columns].sum(axis=1) / total_weight
    frame["WQI_Class"] = frame["WQI"].map(classify)

    standards = pd.DataFrame(
        [
            {
                "Parameter": parameter,
                "Unit": config["parameters"][parameter]["unit"],
                "Limit_or_range": (
                    f"{config['parameters'][parameter]['lower_limit']}-{config['parameters'][parameter]['upper_limit']}"
                    if parameter == "pH"
                    else config["parameters"][parameter]["limit"]
                ),
                "Ideal_value": config["parameters"][parameter].get("ideal", np.nan),
                "Raw_weight": weights[parameter],
                "Normalized_weight": weights[parameter] / total_weight,
                "Note": config["parameters"][parameter].get("note", ""),
            }
            for parameter in PARAMETERS
        ]
    )
    city_summary = (
        frame.groupby("City", as_index=False)
        .agg(
            WQI_records=("WQI", "size"),
            WQI_mean=("WQI", "mean"),
            WQI_median=("WQI", "median"),
            WQI_min=("WQI", "min"),
            WQI_max=("WQI", "max"),
        )
        .sort_values("City")
    )
    class_summary = (
        frame.groupby(["City", "WQI_Class"], as_index=False)
        .size()
        .rename(columns={"size": "Records"})
        .sort_values(["City", "WQI_Class"])
    )
    contribution_long = frame.melt(
        id_vars=["City", "Stable_Station_ID", "Year", "WQI", "WQI_Class"],
        value_vars=[f"QI_{parameter}" for parameter in PARAMETERS],
        var_name="Quality_rating_parameter",
        value_name="Quality_rating",
    )
    contribution_long["Parameter"] = contribution_long["Quality_rating_parameter"].str.removeprefix("QI_")

    frame.to_csv(OUTPUT / "observed_wqi_records.csv", index=False)
    standards.to_csv(OUTPUT / "wqi_standards_and_weights.csv", index=False)
    city_summary.to_csv(OUTPUT / "observed_wqi_summary_by_city.csv", index=False)
    class_summary.to_csv(OUTPUT / "observed_wqi_classification_by_city.csv", index=False)
    contribution_long.to_csv(OUTPUT / "observed_wqi_quality_ratings.csv", index=False)

    order = ["Excellent", "Good", "Poor", "Very Poor", "Unsuitable"]
    fig, axis = plt.subplots(figsize=(10, 5))
    for city, city_data in frame.groupby("City", sort=True):
        axis.hist(city_data["WQI"], bins=30, alpha=0.5, label=city)
    axis.axvline(25, color="black", linestyle="--", linewidth=1)
    axis.axvline(50, color="black", linestyle="--", linewidth=1)
    axis.axvline(75, color="black", linestyle="--", linewidth=1)
    axis.axvline(100, color="black", linestyle="--", linewidth=1)
    axis.set_title("Observed WQI distribution by city")
    axis.set_xlabel("Weighted arithmetic WQI")
    axis.set_ylabel("Records")
    axis.legend(title="City")
    fig.tight_layout()
    fig.savefig(OUTPUT / "observed_wqi_distribution_by_city.png", dpi=220)
    plt.close(fig)

    report = "\n".join(
        [
            "Paper 1 observed WQI calculation report",
            "=======================================",
            f"Complete station-year records scored: {len(frame):,}",
            "Method: weighted arithmetic WQI with inverse-limit weights.",
            "Rule: WQI is calculated from observed parameter values; it is not an ML target.",
            f"Standard basis: {config['standard_source']}",
            "EC assumption: reported EC values are treated as comparable to the TrinkwV 25 °C reference.",
            "Scope: screening index aligned with drinking-water parameters; not a legal compliance decision.",
            "Classes: <=25 Excellent; <=50 Good; <=75 Poor; <=100 Very Poor; >100 Unsuitable.",
            "",
            "Records by city:",
            city_summary.to_string(index=False),
            "",
            "Records by class:",
            class_summary.to_string(index=False),
        ]
    )
    (OUTPUT / "observed_wqi_method_report.txt").write_text(report + "\n", encoding="utf-8")
    print("Observed WQI calculation completed.")
    print(f"WQI records scored: {len(frame):,}")
    print("Created observed WQI records, standards, summaries, figure, and method report in outputs/.")


if __name__ == "__main__":
    main()
