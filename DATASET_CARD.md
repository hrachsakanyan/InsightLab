# Dataset Card — UCI Heart Disease

## Overview

| | |
|---|---|
| **Name** | Heart Disease |
| **Source** | [UCI Machine Learning Repository, dataset 45](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Collected** | 1988–1991 |
| **Donated** | 1 July 1988 by David W. Aha (UC Irvine) |
| **Records** | 920 raw → **918** after cleaning |
| **Columns used** | 14 of 76 available (the subset used by all published work) |
| **Task** | Binary classification — presence of >50% coronary vessel narrowing |
| **Retrieved** | 11 August 2026, via `python -m src.data` |

## Provenance

The collection combines four independently gathered databases that share an
identical 14-column format:

| Site | Institution | n | Principal investigator |
|---|---|---|---|
| Cleveland | Cleveland Clinic Foundation, Ohio, USA | 303 | Robert Detrano, MD, PhD |
| Hungary | Hungarian Institute of Cardiology, Budapest | 293 | Andras Janosi, MD |
| Switzerland | University Hospital, Zurich & Basel | 123 | William Steinbrunn, MD; Matthias Pfisterer, MD |
| Long Beach VA | V.A. Medical Center, Long Beach, California, USA | 199 | Robert Detrano, MD, PhD |

Counts are post-deduplication; the raw files hold 303 / 294 / 123 / 200.

The UCI documentation carries an explicit publication request from the data
authors: **any publication using this data should name the principal
investigators above.** This card and the notebook honour that request.

## How it was collected

Every patient in these files was **referred for coronary angiography** — the
invasive catheter procedure that directly visualises vessel narrowing. Each
record pairs a panel of non-invasive measurements (patient history, resting ECG,
and in most cases a treadmill or bicycle exercise stress test) with the
angiographic result as ground truth.

The original research question was whether the angiogram outcome could be
predicted without performing the angiogram (Detrano et al., 1989).

**This is not a population sample.** The 55% disease rate in this data is the hit
rate of four cardiology referral pipelines, not the prevalence of coronary
disease in any population. Nothing derived from it estimates population risk.

## Fields

| Column | Type | Description | Coding |
|---|---|---|---|
| `site` | categorical | Collecting hospital | added during load; not in the original files |
| `age` | continuous | Age in years | 28–77 |
| `sex` | categorical | Biological sex | 1 = Male, 0 = Female |
| `cp` | categorical | Chest pain type on admission | 1 typical angina · 2 atypical angina · 3 non-anginal pain · 4 asymptomatic |
| `trestbps` | continuous | Resting blood pressure on admission (mm Hg) | |
| `chol` | continuous | Serum cholesterol (mg/dl) | |
| `fbs` | categorical | Fasting blood sugar > 120 mg/dl | 1 = true, 0 = false |
| `restecg` | categorical | Resting ECG result | 0 normal · 1 ST-T abnormality · 2 LV hypertrophy |
| `thalach` | continuous | Maximum heart rate achieved (bpm) | |
| `exang` | categorical | Exercise-induced angina | 1 = yes, 0 = no |
| `oldpeak` | continuous | Exercise-induced ST depression vs rest (mm) | negative = ST elevation |
| `slope` | categorical | Slope of peak exercise ST segment | 1 upsloping · 2 flat · 3 downsloping |
| `ca` | ordinal | Major vessels (0–3) coloured by fluoroscopy | |
| `thal` | categorical | Thallium stress test | 3 normal · 6 fixed defect · 7 reversible defect |
| `num` | ordinal | **Target.** Angiographic disease status | 0 = <50% narrowing; 1–4 grade severity above it |
| `disease` | binary | **Derived target.** `num > 0` | |

## Known data quality issues

These are the reasons this dataset repays an EDA pass before any modelling.
Section references point at [`notebooks/eda.ipynb`](notebooks/eda.ipynb).

1. **Disguised missing values (§4.2).** 172 patients have `chol = 0` and one has
   `trestbps = 0`. Both are physiologically impossible and are the "not
   recorded" code for two sites: **100% of Switzerland** and 24.5% of Long Beach
   VA. Because Switzerland also has the highest disease rate, keeping these
   zeros *reverses the sign* of the cholesterol–disease association (§9.1).

2. **Missingness is site-determined, not random (§4.1).** `ca` is 99% missing at
   Hungary and Long Beach VA; `thal` 90% and 83%; Switzerland skipped `fbs` for
   61% of patients; Long Beach VA is missing ~28% of the entire exercise-test
   block. Missingness identifies the hospital almost perfectly, so imputation
   that ignores `site` fabricates Cleveland data for other hospitals.

3. **Complete-case analysis collapses to one site (§5.1).** Of the 299 rows
   complete on all 13 clinical fields, **297 are Cleveland**. `dropna()` silently
   converts this into a single-site study.

4. **Strong site heterogeneity (§5).** Disease rate ranges from 36% (Hungary) to
   94% (Switzerland); male share from 68% (Cleveland) to 97% (Long Beach VA);
   mean age from 47.8 to 59.4. Cramér's V between `site` and `restecg` is 0.43 —
   the hospitals graded resting ECGs differently.

5. **Two exact duplicate records** across all 13 clinical fields. Dropped; with
   no patient identifier, a genuine double-entry cannot be distinguished from
   two coincidentally identical patients.

6. **Severity grading is not comparable across sites.** `num` 1–4 is collapsed to
   binary, as in all published work on this data.

7. **12 negative `oldpeak` values** (min −2.6), 11 of them from Switzerland.
   These represent ST *elevation*, a genuine if unusual finding. Retained.

8. **Corrupted source files.** UCI ships a `WARNING` file noting that
   `cleveland.data` was damaged. We use only the four `processed.*` files, which
   are unaffected.

9. **Underpowered subgroups.** 193 women in total (21%), only 15 in the 66–80
   band; Switzerland contributes just 8 non-diseased patients.

## Appropriate and inappropriate use

**Reasonable uses.** Teaching EDA and data-quality auditing; benchmarking
classification methods; methodological work on missing data and multi-site
heterogeneity.

**Not appropriate for.** Clinical decision-making or risk calculators of any
kind. The data is 35+ years old; diagnostic thresholds, imaging technology and
treatment have all moved substantially. It describes referred cardiology
patients at four specific hospitals, is 79% male, and its base rate reflects
referral policy rather than disease prevalence. Do not use it to estimate any
individual's or population's cardiac risk.

**Privacy.** Names and social security numbers were removed by the donor and
replaced with dummy values before release. No re-identification was attempted in
this project, and none should be.

## Citation

> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).
> *Heart Disease* [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C52P4X

Original analysis:

> Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S.,
> Guppy, K., Lee, S., & Froelicher, V. (1989). International application of a new
> probability algorithm for the diagnosis of coronary artery disease.
> *American Journal of Cardiology*, 64(5), 304–310.
