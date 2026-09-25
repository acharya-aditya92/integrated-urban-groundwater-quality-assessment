"""Build the Paper 1 ML-ready annual-long dataset.

This script reads the frozen Excel master tables and the original DWD monthly
archives. It calculates city-level SPI-12 at each December, validates the
30 city-year climate records, and attaches climate features to every annual
groundwater parameter observation. No model is trained here.
"""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data_raw"
PROCESSED = PROJECT / "data_processed"

ML_MASTER_FILE = RAW / "Paper1_ML_Annual_Long_Master_Workbook.xlsx"
WQI_MASTER_FILE = RAW / "Paper1_WQI_Master_Workbook.xlsx"
CLIMATE_FILE = RAW / "Paper1_Climate_City_Year_Master.xlsx"

DWD_FILES = {
    "Cologne": RAW / "DWD_Climate_Raw" / "monatswerte_KL_02667_19570901_20251231_hist.zip",
    "Leipzig": RAW / "DWD_Climate_Raw" / "monatswerte_KL_02928_18510101_20251231_hist.zip",
    "Stuttgart": RAW / "DWD_Climate_Raw" / "monatswerte_KL_04931_19400501_20251231_hist.zip",
}

GROUNDWATER_COLUMNS = [
    "City",
    "Station_ID",
    "Stable_Station_ID",
    "Year",
    "Parameter_Standard",
    "Annual_Median",
    "Days_Measured",
]

WQI_PARAMETERS = ["EC", "pH", "Manganese", "Iron", "Nitrate", "Chloride", "Sulfate"]


def fail(message: str) -> None:
    raise ValueError(message)


def read_dwd_monthly(city: str, zip_path: Path) -> pd.DataFrame:
    """Read one DWD monthly KL archive directly, without altering the ZIP."""
    if not zip_path.exists():
        fail(f"Missing DWD archive for {city}: {zip_path.name}")

    with ZipFile(zip_path) as archive:
        product_files = [
            name for name in archive.namelist() if name.startswith("produkt_klima_monat_")
        ]
        if len(product_files) != 1:
            fail(f"Could not identify one monthly product file in {zip_path.name}")
        data = pd.read_csv(archive.open(product_files[0]), sep=";", na_values=-999)

    required = {"MESS_DATUM_BEGINN", "MO_TT", "MO_RR"}
    missing = required.difference(data.columns)
    if missing:
        fail(f"{city} DWD file is missing columns: {sorted(missing)}")

    data["Month"] = pd.to_datetime(
        data["MESS_DATUM_BEGINN"].astype("Int64").astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    data["City"] = city
    data["Year"] = data["Month"].dt.year
    data["MO_TT"] = pd.to_numeric(data["MO_TT"], errors="coerce")
    data["MO_RR"] = pd.to_numeric(data["MO_RR"], errors="coerce")

    data = data.loc[data["Year"].between(1991, 2024)].copy()
    data = data[["City", "Month", "Year", "MO_TT", "MO_RR"]]
    return data


def impute_missing_months(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill rare missing DWD monthly values with the city-month median.

    The original DWD values remain unchanged in the raw output.  This copy is
    used only for complete annual climate features and SPI-12; every fill is
    returned so it can be documented in the quality report.
    """
    filled = monthly.copy()
    rows: list[dict[str, object]] = []
    filled["Calendar_Month"] = filled["Month"].dt.month
    for variable in ["MO_TT", "MO_RR"]:
        medians = filled.groupby(["City", "Calendar_Month"])[variable].transform("median")
        missing = filled[variable].isna()
        for _, row in filled.loc[missing, ["City", "Year", "Calendar_Month"]].iterrows():
            rows.append(
                {
                    "City": row["City"],
                    "Year": int(row["Year"]),
                    "Month": int(row["Calendar_Month"]),
                    "Variable": variable,
                    "Method": "city calendar-month median, 1991-2024",
                }
            )
        filled[variable] = filled[variable].fillna(medians)

    if filled[["MO_TT", "MO_RR"]].isna().any().any():
        fail("A missing DWD monthly value could not be imputed")
    return filled.drop(columns="Calendar_Month"), pd.DataFrame(rows)


def calculate_spi12(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return calendar-year SPI-12 based on the December 12-month total.

    The 1991-2024 monthly series is the station-specific reference period.
    A gamma distribution is fitted to each city's annual (December-ending)
    precipitation totals, then transformed to a standard-normal SPI scale.
    """
    rows: list[pd.DataFrame] = []
    for city, city_data in monthly.groupby("City", sort=True):
        city_data = city_data.sort_values("Month").copy()
        if city_data["MO_RR"].isna().any():
            fail(f"{city} has missing monthly precipitation values in 1991-2024")

        city_data["rainfall_12m"] = city_data["MO_RR"].rolling(12, min_periods=12).sum()
        december = city_data.loc[city_data["Month"].dt.month.eq(12), ["City", "Year", "rainfall_12m"]].copy()

        if len(december) != 34 or december["rainfall_12m"].isna().any():
            fail(f"{city} does not have a complete 1991-2024 December SPI-12 series")

        values = december["rainfall_12m"].to_numpy(dtype=float)
        zero_probability = float(np.mean(values == 0))
        positive = values[values > 0]
        if len(positive) < 30:
            fail(f"{city} has too few positive annual rainfall totals for SPI-12")

        shape, _, scale = gamma.fit(positive, floc=0)
        probabilities = zero_probability + (1 - zero_probability) * gamma.cdf(
            values, shape, loc=0, scale=scale
        )
        december["SPI_12"] = norm.ppf(np.clip(probabilities, 1e-6, 1 - 1e-6))
        rows.append(december[["City", "Year", "SPI_12"]])

    return pd.concat(rows, ignore_index=True)


def annual_climate(monthly: pd.DataFrame, require_complete_values: bool = False) -> pd.DataFrame:
    """Create annual mean temperature and annual total precipitation."""
    coverage = monthly.groupby(["City", "Year"]).agg(
        months=("Month", "count"),
        temperature_months=("MO_TT", "count"),
        rainfall_months=("MO_RR", "count"),
    )
    if not coverage["months"].eq(12).all():
        fail("DWD monthly files do not contain 12 dated rows for every city-year")
    if require_complete_values and not (coverage[["temperature_months", "rainfall_months"]] == 12).all().all():
        fail("Climate values are incomplete after imputation")

    annual = (
        monthly.groupby(["City", "Year"], as_index=False)
        .agg(AirTemp_C=("MO_TT", "mean"), Rainfall_mm=("MO_RR", "sum"))
        .sort_values(["City", "Year"])
    )
    return annual


def read_climate_base() -> pd.DataFrame:
    base = pd.read_excel(CLIMATE_FILE, sheet_name="Climate_City_Year_Base")
    expected = {"City", "Year", "AirTemp_C", "Rainfall_mm"}
    if not expected.issubset(base.columns):
        fail("Climate_City_Year_Base does not have the expected four columns")
    base = base[list(expected)].copy()
    base["Year"] = pd.to_numeric(base["Year"], errors="raise").astype(int)
    return base.sort_values(["City", "Year"])


def validate_annual_climate(rebuilt: pd.DataFrame, workbook_base: pd.DataFrame) -> pd.DataFrame:
    target = rebuilt.loc[rebuilt["Year"].between(2015, 2024)].copy()
    if len(target) != 30:
        fail(f"Expected 30 city-year climate rows, found {len(target)}")

    comparison = target.merge(
        workbook_base,
        on=["City", "Year"],
        how="outer",
        suffixes=("_rebuilt", "_workbook"),
        indicator=True,
    )
    if not comparison["_merge"].eq("both").all():
        fail("The climate workbook and DWD-derived city-year table use different keys")

    for column in ["AirTemp_C", "Rainfall_mm"]:
        difference = (
            comparison[f"{column}_rebuilt"] - comparison[f"{column}_workbook"]
        ).abs()
    differences = comparison.loc[
        (comparison["AirTemp_C_rebuilt"] - comparison["AirTemp_C_workbook"]).abs().gt(0.001)
        | (comparison["Rainfall_mm_rebuilt"] - comparison["Rainfall_mm_workbook"]).abs().gt(0.001)
    ].copy()
    # Differences are retained as an audit record rather than stopping the
    # pipeline. They arise where the workbook retained DWD's -999 missing code;
    # this script correctly converts it to missing before calculating climate.
    return differences


def read_ml_master() -> pd.DataFrame:
    data = pd.read_excel(ML_MASTER_FILE, sheet_name="Paper1_ML_Annual_Long_Master")
    missing = set(GROUNDWATER_COLUMNS).difference(data.columns)
    if missing:
        fail(f"ML master is missing columns: {sorted(missing)}")

    data = data[GROUNDWATER_COLUMNS].copy()
    data["Year"] = pd.to_numeric(data["Year"], errors="raise").astype(int)
    data["Annual_Median"] = pd.to_numeric(data["Annual_Median"], errors="raise")
    data["Days_Measured"] = pd.to_numeric(data["Days_Measured"], errors="raise").astype(int)

    # Stable IDs are authoritative: this restores valid leading zeroes and alphanumeric IDs.
    if data["Stable_Station_ID"].isna().any():
        fail("ML master has blank Stable_Station_ID values")
    data["Station_ID"] = data["Stable_Station_ID"].astype(str).str.split("_", n=1).str[1]

    if data["Station_ID"].isna().any():
        fail("Could not reconstruct Station_ID from Stable_Station_ID")
    key = ["Stable_Station_ID", "Year", "Parameter_Standard"]
    duplicates = int(data.duplicated(key).sum())
    if duplicates:
        fail(f"ML master has {duplicates} duplicate station-year-parameter records")
    if len(data) != 5052:
        fail(f"Expected 5,052 ML records, found {len(data):,}")
    return data


def validate_wqi_master() -> int:
    wqi = pd.read_excel(WQI_MASTER_FILE, sheet_name="Paper1_WQI_Master")
    required = {"City", "Stable_Station_ID", "Year", *WQI_PARAMETERS}
    missing = required.difference(wqi.columns)
    if missing:
        fail(f"WQI master is missing columns: {sorted(missing)}")
    if wqi[WQI_PARAMETERS].isna().any().any():
        fail("WQI master contains missing required WQI parameter values")
    key = ["Stable_Station_ID", "Year"]
    duplicates = int(wqi.duplicated(key).sum())
    if duplicates:
        fail(f"WQI master has {duplicates} duplicate station-year records")
    return len(wqi)


def main() -> None:
    PROCESSED.mkdir(exist_ok=True)

    monthly = pd.concat(
        [read_dwd_monthly(city, path) for city, path in DWD_FILES.items()], ignore_index=True
    )
    climate_base = read_climate_base()
    # The Excel base table was created from DWD observed values; blank DWD
    # cells were skipped by Power Query. Validate this original calculation
    # before making the clearly documented modelling-only imputations below.
    observed_annual = annual_climate(monthly)
    climate_corrections = validate_annual_climate(observed_annual, climate_base)

    monthly_for_model, imputed = impute_missing_months(monthly)
    annual = annual_climate(monthly_for_model, require_complete_values=True)
    spi12 = calculate_spi12(monthly_for_model)

    climate_ready = (
        annual.loc[annual["Year"].between(2015, 2024)]
        .merge(spi12, on=["City", "Year"], how="inner", validate="one_to_one")
        .sort_values(["City", "Year"])
        .reset_index(drop=True)
    )
    if len(climate_ready) != 30 or climate_ready.isna().any().any():
        fail("Climate-ready table must contain 30 complete city-year rows")

    ml = read_ml_master()
    blank_ml_targets = int(ml["Annual_Median"].isna().sum())
    ml_ready = ml.merge(climate_ready, on=["City", "Year"], how="left", validate="many_to_one")
    if len(ml_ready) != len(ml) or ml_ready[["AirTemp_C", "Rainfall_mm", "SPI_12"]].isna().any().any():
        fail("Climate merge failed or left missing climate features")

    wqi_rows = validate_wqi_master()

    monthly.to_csv(PROCESSED / "dwd_monthly_1991_2024.csv", index=False)
    imputed.to_csv(PROCESSED / "climate_imputed_months.csv", index=False)
    climate_corrections.to_csv(PROCESSED / "climate_workbook_corrections.csv", index=False)
    climate_ready.to_csv(PROCESSED / "climate_city_year_2015_2024_with_spi12.csv", index=False)
    ml_ready.to_csv(PROCESSED / "paper1_ml_annual_long_ready.csv", index=False)

    report = "\n".join(
        [
            "Paper 1 data-preparation report",
            "===============================",
            f"DWD monthly records (1991-2024): {len(monthly):,}",
            f"Climate city-year records (2015-2024): {len(climate_ready):,}",
            f"ML annual-long records after climate join: {len(ml_ready):,}",
            f"ML records with blank Annual_Median retained in source: {blank_ml_targets:,}",
            f"Unique ML stations: {ml_ready['Stable_Station_ID'].nunique():,}",
            f"WQI-complete records validated: {wqi_rows:,}",
            f"Monthly DWD values imputed for climate features/SPI-12: {len(imputed):,}",
            "Imputation method: city calendar-month median from 1991-2024.",
            f"Climate workbook rows corrected from raw DWD values: {len(climate_corrections):,}",
            "SPI-12: December-ending rolling 12-month precipitation index,",
            "        fitted separately by city using the 1991-2024 reference period.",
        ]
    )
    (PROCESSED / "data_quality_report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print("\nCreated files in data_processed/.")


if __name__ == "__main__":
    main()
