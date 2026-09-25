# Repository release checklist

Complete this checklist before creating a public GitHub repository.

## Data and ethics

- [ ] Confirm who owns each source workbook and whether it may be redistributed.
- [ ] Confirm whether station identifiers and geographic fields may be shared publicly.
- [ ] Check DWD data terms and add the required citation/acknowledgement.
- [ ] Remove personal paths, names, credentials, or other sensitive metadata.
- [ ] Keep raw and processed data excluded unless an explicit data-release decision is made.

## Research reporting

- [ ] Add author names, institutional affiliation, supervisor-approved title, and a contact method to the README.
- [ ] Add exact source citations for groundwater data, DWD climate data, and WQI standards.
- [ ] Decide the repository licence with the institution/supervisor.
- [ ] Review every generated figure and table before selectively releasing it.
- [ ] State that feature importance is predictive, not causal.
- [ ] State that WQI weighting sensitivity is a limitation.

## Technical release

- [ ] Create a new empty GitHub repository; do not upload the whole parent project folder.
- [ ] Upload only the contents of `Paper1_ML_Framework/` after confirming `.gitignore` is active.
- [ ] Run the scripts from a clean local copy and confirm the README instructions work.
- [ ] Record the Python version and package versions used for the release.
- [ ] Tag a version only after the paper analysis is frozen.

