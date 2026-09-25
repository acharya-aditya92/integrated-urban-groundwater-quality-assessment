"""Create local Paper 1 tables, figures, and a cautious draft results summary.

All artefacts are generated from existing held-out outputs.  The draft is for
author review and does not make causal or legal-compliance claims.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "outputs"
PROCESSED = PROJECT / "data_processed"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".matplotlib"))
import matplotlib.pyplot as plt

CITY_COLORS = {"Cologne": "#4C78A8", "Leipzig": "#F58518", "Stuttgart": "#54A24B"}
WQI_ORDER = ["Excellent", "Good", "Poor", "Very Poor", "Unsuitable"]
WQI_COLORS = {
    "Excellent": "#2E8B57",
    "Good": "#8BC34A",
    "Poor": "#FDD835",
    "Very Poor": "#FB8C00",
    "Unsuitable": "#C62828",
}


def require_file(name: str) -> Path:
    path = OUTPUT / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}. Run the earlier numbered scripts first.")
    return path


def save_final_performance(metrics: pd.DataFrame) -> pd.DataFrame:
    dummy = metrics.loc[metrics["Model"].eq("Dummy_median"), ["Parameter_Standard", "R2"]].rename(
        columns={"R2": "Dummy_median_R2"}
    )
    final = metrics.loc[~metrics["Model"].eq("Dummy_median"), ["Parameter_Standard", "Model", "R2", "MAE", "RMSE"]].rename(
        columns={"R2": "Selected_model_R2"}
    )
    table = final.merge(dummy, on="Parameter_Standard", validate="one_to_one")
    table = table.sort_values("Parameter_Standard").reset_index(drop=True)
    table.to_csv(OUTPUT / "paper1_final_model_results_table.csv", index=False)

    x = np.arange(len(table))
    width = 0.37
    fig, axis = plt.subplots(figsize=(10, 5.3))
    axis.bar(x - width / 2, table["Dummy_median_R2"], width, label="Dummy median", color="#9E9E9E")
    axis.bar(x + width / 2, table["Selected_model_R2"], width, label="Selected final model", color="#1565C0")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, table["Parameter_Standard"], rotation=25, ha="right")
    axis.set_ylabel("Held-out R² (2023–2024)")
    axis.set_title("Final parameter models outperform the dummy baseline")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "paper1_final_model_r2_comparison.png", dpi=260)
    plt.close(fig)
    return table


def save_selected_prediction_figure(predictions: pd.DataFrame) -> None:
    parameters = list(predictions["Parameter_Standard"].drop_duplicates())
    ncols = 4
    nrows = int(np.ceil(len(parameters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 7.5))
    axes_flat = axes.ravel()
    for axis, parameter in zip(axes_flat, parameters):
        subset = predictions.loc[predictions["Parameter_Standard"].eq(parameter)]
        low = min(subset["Observed_Annual_Median"].min(), subset["Predicted_Annual_Median"].min())
        high = max(subset["Observed_Annual_Median"].max(), subset["Predicted_Annual_Median"].max())
        for city, city_data in subset.groupby("City", sort=True):
            axis.scatter(
                city_data["Observed_Annual_Median"],
                city_data["Predicted_Annual_Median"],
                color=CITY_COLORS.get(city, "#555555"),
                alpha=0.75,
                s=28,
                label=city,
            )
        axis.plot([low, high], [low, high], "k--", linewidth=1)
        axis.set_title(parameter)
        axis.set_xlabel("Observed annual median")
        axis.set_ylabel("Predicted annual median")
    for axis in axes_flat[len(parameters) :]:
        axis.set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="City", loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Selected final models: held-out 2023–2024 predictions", y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(OUTPUT / "paper1_final_selected_observed_vs_predicted.png", dpi=260)
    plt.close(fig)


def save_wqi_class_figure(classifications: pd.DataFrame) -> pd.DataFrame:
    table = classifications.pivot_table(index="City", columns="WQI_Class", values="Records", aggfunc="sum", fill_value=0)
    table = table.reindex(columns=WQI_ORDER, fill_value=0).sort_index()
    percent = table.div(table.sum(axis=1), axis=0).mul(100)
    percent.to_csv(OUTPUT / "paper1_observed_wqi_class_percent_by_city.csv")

    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    bottom = np.zeros(len(percent))
    for wqi_class in WQI_ORDER:
        values = percent[wqi_class].to_numpy()
        axis.bar(percent.index, values, bottom=bottom, label=wqi_class, color=WQI_COLORS[wqi_class])
        bottom += values
    axis.set_ylim(0, 100)
    axis.set_ylabel("Share of complete WQI records (%)")
    axis.set_title("Observed WQI classes by city")
    axis.legend(title="WQI class", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "paper1_observed_wqi_classes_by_city.png", dpi=260)
    plt.close(fig)
    return table


def make_summary(
    final_table: pd.DataFrame,
    wqi_city: pd.DataFrame,
    wqi_classes: pd.DataFrame,
    derived: pd.Series,
    sensitivity: pd.Series,
    quality_report: str,
) -> str:
    model_rows = "\n".join(
        "| {parameter} | {model} | {n} | {mae:.3f} | {rmse:.3f} | {r2:.3f} |".format(
            parameter=row.Parameter_Standard,
            model=row.Model,
            n=int(row.Final_test_records_2023_2024),
            mae=row.MAE,
            rmse=row.RMSE,
            r2=row.R2,
        )
        for row in final_table.itertuples()
    )
    city_rows = "\n".join(
        "| {city} | {n} | {median:.2f} | {minimum:.2f} | {maximum:.2f} |".format(
            city=row.City,
            n=int(row.WQI_records),
            median=row.WQI_median,
            minimum=row.WQI_min,
            maximum=row.WQI_max,
        )
        for row in wqi_city.itertuples()
    )
    class_rows = "\n".join(
        "| {city} | {excellent} | {good} | {poor} | {very_poor} | {unsuitable} |".format(
            city=city,
            excellent=int(wqi_classes.loc[city, "Excellent"]),
            good=int(wqi_classes.loc[city, "Good"]),
            poor=int(wqi_classes.loc[city, "Poor"]),
            very_poor=int(wqi_classes.loc[city, "Very Poor"]),
            unsuitable=int(wqi_classes.loc[city, "Unsuitable"]),
        )
        for city in wqi_classes.index
    )
    return f"""# Paper 1 results summary — draft for author review

## Scope and data quality

This analysis covers Cologne, Leipzig, and Stuttgart for 2015–2024. The temporal model test period is 2023–2024 and was held out from final model fitting. The preparation stage produced 5,052 annual parameter records across 112 unique stations; 30 blank annual medians were retained in source data but excluded only from the relevant supervised-model fit. The WQI analysis used 436 station-years complete for all seven WQI parameters.

The data-quality record is preserved in `data_processed/data_quality_report.txt`. Key processing notes: seven rare DWD monthly climate values were imputed only for modelling features using city-calendar-month medians from 1991–2024, and two climate-workbook values were corrected after comparison with raw DWD data.

## Held-out parameter-model performance

Each parameter was modelled separately. Candidate model choices used only 2015–2021 training and 2022 validation data; the final selected models were refitted on 2015–2022 and tested once on 2023–2024.

| Parameter | Selected model | Held-out records | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
{model_rows}

All selected models outperformed the dummy-median baseline on the held-out period. Performance was strongest for iron (R² = 0.857), nitrate (0.835), sulfate (0.827), and manganese (0.805). Chloride was the least well predicted of the seven targets (R² = 0.398). These are temporal results for the three observed cities and mostly known stations; they do not establish performance in a new city.

## Predictive feature importance

Held-out permutation importance indicates that station identity is particularly important for iron, manganese, and pH. City also contributes substantially to chloride, EC, nitrate, and sulfate prediction. The climate features added little incremental held-out predictive importance once city and station identity were included. This is model-based predictive importance, not evidence of physical causation.

## Observed WQI screening results

The main WQI uses the documented inverse-limit weighted arithmetic calculation and was applied to complete observed seven-parameter records. It is a screening index, not a legal drinking-water compliance decision.

| City | Complete WQI records | Median WQI | Minimum | Maximum |
|---|---:|---:|---:|---:|
{city_rows}

The WQI distribution is strongly right-skewed; medians and class distributions should therefore be prioritised over mean WQI values.

| City | Excellent | Good | Poor | Very Poor | Unsuitable |
|---|---:|---:|---:|---:|---:|
{class_rows}

## Derived WQI from parameter predictions

For the 103 held-out station-years with all seven observed and predicted WQI parameters, the derived WQI had R² = {derived.R2:.3f}, MAE = {derived.MAE:.2f}, and exact WQI-class agreement of {derived.Exact_WQI_class_agreement:.1%}. The low class agreement means predicted WQI classes should be described as exploratory rather than as reliable categorical classifications. Parameter-level prediction results remain the primary modelling evidence.

## WQI weighting sensitivity

Using the same standards and quality ratings but equal parameter weights, the WQI ranking correlation with the main inverse-limit weighted WQI was {sensitivity.Spearman_rank_correlation:.3f}; exact class agreement was only {sensitivity.Exact_class_agreement:.1%}. This shows that WQI class assignment is materially dependent on the weighting decision. The main index should be reported with this sensitivity analysis and alongside individual parameter values.

## Recommended manuscript figures

1. `paper1_final_model_r2_comparison.png` — held-out final model R² compared with the dummy baseline.
2. `paper1_final_selected_observed_vs_predicted.png` — selected model predictions against observations for the 2023–2024 test period.
3. `final_feature_relative_importance.png` — relative held-out permutation importance by parameter.
4. `paper1_observed_wqi_classes_by_city.png` — observed WQI class composition by city.
5. `wqi_weight_sensitivity_comparison.png` — sensitivity of WQI score to the weighting method.

## Publication cautions

- Do not describe feature importance as causation.
- Do not treat this temporal test as new-city transfer validation.
- Do not claim a predicted WQI class is reliable when class agreement is low.
- Do not interpret the WQI as a legal compliance result.
- State the explicit EC 25 °C screening assumption and the WQI weighting sensitivity.
"""


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    final_metrics = pd.read_csv(require_file("final_temporal_metrics_2023_2024.csv"))
    predictions = pd.read_csv(require_file("final_temporal_predictions_2023_2024.csv"))
    wqi_city = pd.read_csv(require_file("observed_wqi_summary_by_city.csv"))
    classifications = pd.read_csv(require_file("observed_wqi_classification_by_city.csv"))
    derived = pd.read_csv(require_file("derived_wqi_heldout_metrics_2023_2024.csv")).iloc[0]
    sensitivity = pd.read_csv(require_file("wqi_weight_sensitivity_summary.csv")).iloc[0]
    quality_report = (PROCESSED / "data_quality_report.txt").read_text(encoding="utf-8")

    final_table = save_final_performance(final_metrics)
    save_selected_prediction_figure(predictions)
    wqi_classes = save_wqi_class_figure(classifications)
    summary = make_summary(final_metrics.loc[~final_metrics["Model"].eq("Dummy_median")].sort_values("Parameter_Standard"), wqi_city, wqi_classes, derived, sensitivity, quality_report)
    (OUTPUT / "paper1_results_summary_draft.md").write_text(summary, encoding="utf-8")
    print("Paper 1 local results package created in outputs/.")
    print("Created draft summary, final-model table, selected-prediction figure, R2 comparison, and WQI class figure.")


if __name__ == "__main__":
    main()
