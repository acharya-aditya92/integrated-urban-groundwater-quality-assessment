# Data dictionary

## Annual-long chemistry data

| Field | Meaning | Expected type |
|---|---|---|
| `City` | Cologne, Leipzig, or Stuttgart | text |
| `Station_ID` | Source station identifier | text |
| `Stable_Station_ID` | City-prefixed station identifier used by models | text |
| `Year` | Calendar year | integer |
| `Parameter_Standard` | Harmonised target parameter | text |
| `Annual_Median` | Annual median result for the parameter | numeric |
| `Days_Measured` | Number of measurements contributing to the annual value | integer |

## Climate features

| Field | Meaning | Expected type |
|---|---|---|
| `AirTemp_C` | Annual mean air temperature | numeric |
| `Rainfall_mm` | Annual precipitation total | numeric |
| `SPI_12` | December-ending 12-month Standardized Precipitation Index | numeric |

`SPI_12` is fitted separately by city using the 1991–2024 monthly precipitation reference period.

## WQI complete data

Each WQI record is one station-year with complete values for all seven parameters:

| Parameter | Unit used in the framework |
|---|---|
| EC | µS/cm |
| pH | pH units |
| Manganese | mg/L |
| Iron | mg/L |
| Nitrate | mg/L |
| Chloride | mg/L |
| Sulfate | mg/L |

The WQI calculation also requires `City`, `Station_ID`, `Stable_Station_ID`, and `Year`.

## Important constraints

- `Annual_Median` must not be silently converted between units.
- Blank source target values are retained in the source data; they are not treated as zero.
- EC is evaluated against the 25 °C reference value in the WQI configuration. The original source workbooks do not state the EC reference temperature, so this is an explicit screening assumption.

