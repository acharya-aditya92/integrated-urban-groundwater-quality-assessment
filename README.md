# Paper 1: urban groundwater quality, climate, WQI, and machine learning

This repository contains the reproducible analysis framework for Paper 1 of an urban groundwater-quality study covering Cologne, Leipzig, and Stuttgart, Germany.

The workflow combines annual groundwater chemistry with city-level climate features, evaluates temporal machine-learning models for seven individual parameters, and calculates a documented rule-based Water Quality Index (WQI). It is designed for transparent research reporting; it is not a drinking-water compliance tool.

## Research scope

- Cities: Cologne, Leipzig, Stuttgart
- Analysis years: 2015–2024
- Climate reference data: DWD monthly climate observations, 1991–2024
- Prediction targets: Chloride, EC, Iron, Manganese, Nitrate, Sulfate, and pH
- ML evaluation: train on 2015–2022 and hold out 2023–2024
- WQI: calculated from complete observed seven-parameter station-year records using a rule-based formula; never used as a direct ML target

The next study stage is a separate transferability analysis using a fourth city. It should not be treated as part of the Paper 1 test set.

## Repository structure

```text
Paper1_ML_Framework/
├── src/                 Reproducible Python analysis scripts
├── data_raw/            Local source workbooks and DWD ZIP files — not published
├── data_processed/      Regenerated analysis-ready files — not published
├── outputs/             Regenerated tables, figures, and reports — not published by default
├── docs/                Method, variable, and release documentation
├── requirements.txt     Python packages
└── wqi_standards_trinkwv_2023.json   WQI standards and assumptions
```

## Setup

Python 3.13 was used in the initial run. Other recent Python 3 versions may work, but should be recorded if used for a reproduction.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Required local input files

The following files must be placed locally in `data_raw/`. They are intentionally excluded from GitHub until their provenance, licences, and any sharing restrictions have been confirmed.

- `Paper1_ML_Annual_Long_Master_Workbook.xlsx`
- `Paper1_Climate_City_Year_Master.xlsx`
- `Paper1_WQI_Master_Workbook.xlsx`
- `DWD_Climate_Raw/monatswerte_KL_02667_19570901_20251231_hist.zip` (Cologne)
- `DWD_Climate_Raw/monatswerte_KL_02928_18510101_20251231_hist.zip` (Leipzig)
- `DWD_Climate_Raw/monatswerte_KL_04931_19400501_20251231_hist.zip` (Stuttgart)

See [data_raw/README.md](data_raw/README.md) for file roles and [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) for required fields.

## Run the workflow

Run scripts from the repository root, in numerical order:

```powershell
python src\01_prepare_ml_data.py
python src\02_run_baseline_models.py
python src\03_create_baseline_diagnostics.py
python src\04_select_final_models.py
python src\05_interpret_final_models.py
python src\06_calculate_observed_wqi.py
python src\07_evaluate_derived_wqi.py
python src\08_wqi_weight_sensitivity.py
```

The scripts write generated data, reports, tables, and figures to `data_processed/` and `outputs/`.

## Core methodological safeguards

- Each parameter is predicted separately; WQI is derived only after parameter prediction.
- The temporal test years 2023–2024 are not used while selecting candidate models.
- The final WQI comparison evaluates only held-out station-years with all seven observed and predicted parameters.
- Missing DWD values are imputed only for modelling features using a city-calendar-month median from 1991–2024; every imputation is logged.
- Concentration targets with blank source measurements remain in the source data and are excluded only from supervised-model fitting for that target.
- The WQI is a standards-aligned screening index, not a legal determination of drinking-water compliance.

## WQI interpretation

The main WQI uses the weighted arithmetic, inverse-limit weighting rule documented in `wqi_standards_trinkwv_2023.json`. It uses German Drinking Water Ordinance (TrinkwV) values as screening reference values. Because inverse-limit weighting gives iron and manganese high influence, the repository includes an equal-weight sensitivity analysis. The two weighting methods must be reported together; the WQI class should not be presented as an unquestionable measurement of overall groundwater quality.

The derived WQI from ML parameter predictions is exploratory. Parameter-level held-out results remain the primary modelling evidence.

## Sharing on GitHub

This framework is ready to publish as code and documentation. Do **not** publish `data_raw/`, `data_processed/`, or `outputs/` automatically. First complete [docs/REPOSITORY_RELEASE_CHECKLIST.md](docs/REPOSITORY_RELEASE_CHECKLIST.md), including data-rights checks and removal of any sensitive identifiers.

## Citation and acknowledgement

Before public release, add a project citation, the authors, institutional affiliation, a licence, and the exact data-source citations. A licence is intentionally not selected here because it is an author/institution decision.

