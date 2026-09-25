# Methodology summary

## Preparation

The preparation script validates the frozen Paper 1 annual-long, WQI-complete, and climate workbooks; reads the DWD monthly climate ZIP files; derives annual city climate features; calculates city-specific December-ending SPI-12; and joins climate features to parameter records.

Rare missing DWD climate values are imputed only for modelling features using the city and calendar-month median from the 1991–2024 DWD period. The affected values are listed in `data_processed/climate_imputed_months.csv`. Source chemistry measurements are not imputed.

## Parameter modelling

Seven separate regressors predict annual medians. Candidate algorithms are selected using 2015–2021 training and 2022 validation data only. The selected model for each parameter is refitted on 2015–2022, then evaluated once on the held-out 2023–2024 period. A station-aware random forest baseline and a dummy median baseline are retained for comparison.

This is primarily a temporal prediction task for known cities and mostly known wells. It is not an evaluation of prediction in a previously unseen city.

## WQI calculation

WQI is calculated only from complete observed records, using the seven individual parameters and the formula:

```text
WQI = sum(Qi × wi) / sum(wi)
```

For concentration parameters, `Qi = 100 × value / limit`. For pH, `Qi` is the deviation from the ideal pH of 7 relative to the relevant lower or upper boundary. The main weight is `wi = 1 / limit` (with pH weight basis 9.5). The exact values, units, assumptions, and URL are stored in `wqi_standards_trinkwv_2023.json`.

The WQI is a screening index aligned to drinking-water parameters. It does not establish legal compliance and must be interpreted with individual parameter concentrations and their uncertainty.

## Derived WQI evaluation and sensitivity

For held-out 2023–2024 station-years with all seven parameter predictions, the same WQI formula is applied to predicted parameters and compared against the observed WQI. This assesses an index derived from parameter predictions; it does not train a model directly on WQI.

An equal-weight WQI is calculated only as a sensitivity analysis. It uses the same standards and quality ratings, so differences show the dependence of the final score and class on the weighting rule.

