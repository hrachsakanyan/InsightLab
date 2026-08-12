"""Download, load and clean the UCI Heart Disease databases.

The UCI archive ships four separate files, one per collecting hospital. They
share an identical 14-column format, so we stack them into a single frame and
keep a ``site`` column — the site turns out to explain a lot of the data
quality, so throwing it away would hide the most interesting story in the data.

Run as a script to (re)build ``data/heart_disease.csv``::

    python -m src.data
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_CSV = PROJECT_ROOT / "data" / "heart_disease.csv"

SOURCE_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

#: Processed file -> hospital that collected it.
SITE_FILES = {
    "processed.cleveland.data": "Cleveland",
    "processed.hungarian.data": "Hungary",
    "processed.switzerland.data": "Switzerland",
    "processed.va.data": "Long Beach VA",
}

#: The 14 columns used by every published experiment on this dataset.
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

#: Sentinel the UCI documentation declares for missing values, alongside "?".
MISSING_SENTINELS = ["?", "-9", "-9.0"]

# --------------------------------------------------------------------------
# Code -> label maps, transcribed from heart-disease.names
# --------------------------------------------------------------------------

SEX_LABELS = {0: "Female", 1: "Male"}
CP_LABELS = {
    1: "Typical angina",
    2: "Atypical angina",
    3: "Non-anginal pain",
    4: "Asymptomatic",
}
RESTECG_LABELS = {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}
SLOPE_LABELS = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
THAL_LABELS = {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}
YES_NO_LABELS = {0: "No", 1: "Yes"}

#: Human-readable descriptions, used for the data dictionary in the notebook.
COLUMN_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Biological sex",
    "cp": "Chest pain type reported on admission",
    "trestbps": "Resting blood pressure on admission (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting electrocardiographic result",
    "thalach": "Maximum heart rate achieved during exercise test (bpm)",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression induced by exercise relative to rest (mm)",
    "slope": "Slope of the peak exercise ST segment",
    "ca": "Number of major vessels (0-3) coloured by fluoroscopy",
    "thal": "Thallium stress test result",
    "num": "Angiographic disease status, 0-4 (0 = <50% narrowing)",
    "site": "Hospital that collected the record",
    "disease": "Binary target: any vessel with >50% narrowing",
}

#: Variables that are genuinely continuous, as opposed to coded categories.
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

#: Coded categorical variables, in their labelled form.
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

#: Physiologically impossible zeros — a zero here means "not recorded",
#: not "the patient measured zero". Documented per-column so the cleaning
#: step stays auditable rather than a magic number sprinkled through code.
ZERO_IS_MISSING = {
    "chol": "Serum cholesterol of 0 mg/dl is incompatible with life.",
    "trestbps": "A resting blood pressure of 0 mm Hg is incompatible with life.",
}


# --------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------

def download_raw(dest: Path = RAW_DIR, force: bool = False) -> Path:
    """Fetch the UCI archive and extract the four processed site files.

    Returns the directory holding them. Skips the network entirely if the
    files are already present, so notebooks stay runnable offline.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if not force and all((dest / name).exists() for name in SITE_FILES):
        return dest

    import urllib.request

    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in SITE_FILES:
            (dest / name).write_bytes(archive.read(name))

    return dest


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Stack the four site files into one frame, untouched apart from sentinels.

    Missing-value sentinels are converted to ``NaN`` here because they are an
    encoding detail of the file format, not an analytical decision. Everything
    else is left exactly as distributed so the notebook can show the mess.
    """
    download_raw(raw_dir)

    frames = []
    for filename, site in SITE_FILES.items():
        frame = pd.read_csv(
            raw_dir / filename,
            names=RAW_COLUMNS,
            na_values=MISSING_SENTINELS,
        )
        frame.insert(0, "site", site)
        frames.append(frame)

    stacked = pd.concat(frames, ignore_index=True)
    # Fix the site order up front so every groupby and figure — including those
    # built on the raw frame — lists the hospitals in the same sequence.
    stacked["site"] = pd.Categorical(
        stacked["site"], categories=list(SITE_FILES.values()), ordered=False
    )
    return stacked


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the documented cleaning steps.

    Returns ``(clean_frame, log)`` where ``log`` records how many values each
    step changed — the notebook prints it so no step is invisible.
    """
    df = raw.copy()
    log: list[dict] = []

    def record(step: str, column: str, n: int, note: str) -> None:
        log.append({"step": step, "column": column, "values_affected": n, "note": note})

    # 1. Disguised missing values: impossible zeros.
    for column, reason in ZERO_IS_MISSING.items():
        mask = df[column].eq(0)
        if mask.any():
            df.loc[mask, column] = np.nan
        record("zero -> NaN", column, int(mask.sum()), reason)

    # 2. Binary target. Codes 1-4 all mean ">50% narrowing in some vessel";
    #    the severity grading is not comparable across sites, so we collapse it.
    df["disease"] = (df["num"] > 0).astype(int)
    record("derive", "disease", int(df["disease"].sum()), "num > 0 -> disease = 1")

    # 3. Duplicate patient records (same values on every clinical field).
    clinical = [c for c in RAW_COLUMNS if c != "num"]
    duplicates = df.duplicated(subset=clinical, keep="first")
    if duplicates.any():
        df = df.loc[~duplicates].reset_index(drop=True)
    record("drop rows", "-", int(duplicates.sum()), "Exact duplicates across all clinical fields")

    # 4. Human-readable labels for the coded categoricals. Kept as ordered
    #    categoricals where the codes have a natural order, so plots and
    #    groupbys come out in a sensible sequence instead of alphabetically.
    label_maps = {
        "sex": SEX_LABELS,
        "cp": CP_LABELS,
        "restecg": RESTECG_LABELS,
        "slope": SLOPE_LABELS,
        "thal": THAL_LABELS,
        "fbs": YES_NO_LABELS,
        "exang": YES_NO_LABELS,
    }
    for column, mapping in label_maps.items():
        codes = df[column]
        unexpected = int(codes.dropna()[~codes.dropna().isin(mapping)].shape[0])
        df[column] = pd.Categorical(
            codes.map(mapping),
            categories=list(mapping.values()),
            ordered=True,
        )
        if unexpected:
            record("unmapped code -> NaN", column, unexpected, f"Codes outside {sorted(mapping)}")

    # `ca` stays numeric (it is a count of vessels, 0-3) but is genuinely ordinal.
    df["ca"] = pd.to_numeric(df["ca"], errors="coerce")

    df["site"] = pd.Categorical(df["site"], categories=list(SITE_FILES.values()), ordered=False)

    ordered = ["site"] + RAW_COLUMNS + ["disease"]
    return df[ordered], pd.DataFrame(log)


def load_clean(csv_path: Path = CLEAN_CSV) -> pd.DataFrame:
    """Load the cleaned dataset, rebuilding it from raw if it is missing."""
    if not csv_path.exists():
        build()

    df = pd.read_csv(csv_path)
    for column, mapping in [
        ("sex", SEX_LABELS), ("cp", CP_LABELS), ("restecg", RESTECG_LABELS),
        ("slope", SLOPE_LABELS), ("thal", THAL_LABELS),
        ("fbs", YES_NO_LABELS), ("exang", YES_NO_LABELS),
    ]:
        df[column] = pd.Categorical(
            df[column], categories=list(mapping.values()), ordered=True
        )
    df["site"] = pd.Categorical(df["site"], categories=list(SITE_FILES.values()))
    return df


def build(csv_path: Path = CLEAN_CSV) -> pd.DataFrame:
    """Run the full raw -> clean pipeline and write the result to disk."""
    raw = load_raw()
    df, log = clean(raw)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Raw:   {raw.shape[0]:>4} rows x {raw.shape[1]} cols")
    print(f"Clean: {df.shape[0]:>4} rows x {df.shape[1]} cols  ->  {csv_path}")
    print()
    print(log.to_string(index=False))
    return df


if __name__ == "__main__":
    build()
