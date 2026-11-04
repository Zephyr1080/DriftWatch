"""
DriftWatch - data preparation (src/data.py)

Partitions (all splits are by whole dates, never by rows):

  Historical pool = first 60% of dates (Jan 2011 - mid Mar 2012)
      reference     60% of the pool   -> model training + drift reference
      calibration   20% of the pool   -> detector threshold calibration
      stable_test   20% of the pool   -> no-drift control, base for simulated
                                         drift, held-out accuracy
  natural_eval    = last 40% of dates -> real, naturally drifted period
                                         (demand grows strongly in 2012);
                                         used only as a separate real-drift check

Within the pool, dates are grouped into 7-day blocks and each calendar month
is split separately (stratified), so reference, calibration and stable_test
all cover the same seasons. Blocks (not single days) are used so neighbouring
days with near-identical weather do not leak between partitions.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hour.csv"

RANDOM_STATE = 42

HISTORICAL_RATIO = 0.60     # share of dates in the historical pool
CALIBRATION_RATIO = 0.20    # share of the pool used for calibration
STABLE_TEST_RATIO = 0.20    # share of the pool used as stable test
BLOCK_DAYS = 7              # block length used when splitting the pool

BATCH_SIZE = 500            # rows per monitoring batch

# yr is deliberately excluded because it becomes constant
# in the later portion of this dataset.
# cnt, casual and registered are excluded to avoid target leakage.
FEATURE_COLUMNS = [
    "season",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed"
]

PARTITION_NAMES = [
    "reference",
    "calibration",
    "stable_test",
    "natural_eval"
]


# ==================================================
# 1. LOAD DATASET
# ==================================================

def load_dataset():
    """Load and sort the Bike Sharing dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df["dteday"] = pd.to_datetime(df["dteday"])

    df = df.sort_values(
        by=["dteday", "hr"]
    ).reset_index(drop=True)

    return df


# ==================================================
# 2. SPLIT DATASET
# ==================================================

def split_dataset(df):
    """
    Split the dataset into four partitions by whole dates.

    Returns a dict with keys:
        reference, calibration, stable_test, natural_eval
    """

    dates = pd.Series(sorted(df["dteday"].unique()))

    pool_end = int(len(dates) * HISTORICAL_RATIO)

    pool_dates = dates.iloc[:pool_end]
    natural_dates = dates.iloc[pool_end:]

    # ---- group pool dates into 7-day blocks ----
    blocks = pd.DataFrame({"date": pool_dates})

    blocks["block"] = (
        (blocks["date"] - blocks["date"].iloc[0]).dt.days
        // BLOCK_DAYS
    )

    block_info = (
        blocks.groupby("block")["date"]
        .min()
        .dt.to_period("M")
        .rename("month_key")
        .reset_index()
    )

    # ---- split each month's blocks (stratified by month) ----
    rng = np.random.default_rng(RANDOM_STATE)

    block_partition = {}

    for _, month_blocks in block_info.groupby("month_key"):

        block_ids = rng.permutation(
            sorted(month_blocks["block"].tolist())
        )

        n = len(block_ids)
        n_cal = int(round(n * CALIBRATION_RATIO))
        n_test = int(round(n * STABLE_TEST_RATIO))

        for block_id in block_ids[:n_cal]:
            block_partition[block_id] = "calibration"

        for block_id in block_ids[n_cal:n_cal + n_test]:
            block_partition[block_id] = "stable_test"

        for block_id in block_ids[n_cal + n_test:]:
            block_partition[block_id] = "reference"

    blocks["partition"] = blocks["block"].map(block_partition)

    date_to_partition = dict(
        zip(blocks["date"], blocks["partition"])
    )

    for date in natural_dates:
        date_to_partition[date] = "natural_eval"

    partition_per_row = df["dteday"].map(date_to_partition)

    return {
        name: df[partition_per_row == name].reset_index(drop=True)
        for name in PARTITION_NAMES
    }


# ==================================================
# 3. PREPARE FEATURES AND TARGET
# ==================================================

def prepare_features(df, target_median):
    """
    Select model input features and create binary labels.

    0 = Low demand
    1 = High demand
    """

    X = df[FEATURE_COLUMNS].copy()

    y = (
        df["cnt"] > target_median
    ).astype(int)

    return X, y


# ==================================================
# 4. PREPARE COMPLETE DATASET
# ==================================================

def prepare_data():
    """
    Load, split and prepare the dataset.

    The target threshold is calculated using reference
    data only and reused across all partitions.

    Returns a dict with X_/y_ pairs for: train, calibration,
    stable_test, evaluation (natural), plus target_median.
    """

    df = load_dataset()

    partitions = split_dataset(df)

    # Target threshold from reference data only
    target_median = partitions["reference"]["cnt"].median()

    names = {
        "reference": "train",
        "calibration": "calibration",
        "stable_test": "stable_test",
        "natural_eval": "evaluation"
    }

    data = {}

    for partition_name, short_name in names.items():

        X, y = prepare_features(
            partitions[partition_name],
            target_median
        )

        data[f"X_{short_name}"] = X
        data[f"y_{short_name}"] = y

    data["target_median"] = target_median

    return data


# ==================================================
# 5. BATCHING
# ==================================================

def sample_batches(
    X,
    y,
    n_batches,
    batch_size=BATCH_SIZE,
    seed=RANDOM_STATE
):
    """
    Draw n_batches random batches (without replacement inside a
    batch, independently between batches) from stable data.

    Use this for calibration and for the no-drift control, where
    many batches are needed from a limited amount of stable data.
    Batches overlap, so treat them as a resampling scheme.

    Returns a list of (X_batch, y_batch) tuples.
    """

    if batch_size > len(X):
        raise ValueError(
            f"batch_size ({batch_size}) is larger than the data ({len(X)})."
        )

    rng = np.random.default_rng(seed)

    batches = []

    for _ in range(n_batches):

        idx = rng.choice(len(X), size=batch_size, replace=False)

        batches.append((
            X.iloc[idx].reset_index(drop=True),
            y.iloc[idx].reset_index(drop=True)
        ))

    return batches


def chronological_batches(X, y, batch_size=BATCH_SIZE):
    """
    Cut time-ordered data into consecutive batches (the incomplete
    last batch is dropped). Use this for the natural evaluation set.

    Returns a list of (X_batch, y_batch) tuples.
    """

    n_batches = len(X) // batch_size

    return [
        (
            X.iloc[i * batch_size:(i + 1) * batch_size].reset_index(drop=True),
            y.iloc[i * batch_size:(i + 1) * batch_size].reset_index(drop=True)
        )
        for i in range(n_batches)
    ]


# ==================================================
# 6. TEMPORAL DIAGNOSTICS
# ==================================================

def display_temporal_diagnostics(partitions, target_median):
    """Display partition dates, sizes and demand distributions."""

    print("\n========== PARTITION DIAGNOSTICS ==========")

    print("\nTarget threshold (reference median):", target_median)

    for name in PARTITION_NAMES:

        part = partitions[name]

        high = (part["cnt"] > target_median).mean() * 100

        print(
            f"\n{name}: {len(part)} records, "
            f"{part['dteday'].nunique()} days, "
            f"{part['dteday'].min().date()} to "
            f"{part['dteday'].max().date()}"
        )
        print(f"  mean cnt: {part['cnt'].mean():.1f}")
        print(f"  high-demand: {high:.2f}%")

    print("\nOverlap check (should all be 0):")

    names = PARTITION_NAMES

    for i in range(len(names)):
        for j in range(i + 1, len(names)):

            shared = set(partitions[names[i]]["dteday"]) & set(
                partitions[names[j]]["dteday"]
            )

            print(f"  {names[i]} & {names[j]}: {len(shared)}")


# ==================================================
# 7. FEATURE DISTRIBUTION COMPARISON
# ==================================================

def compare_feature_distributions(partitions):
    """
    Compare feature means across partitions.

    This is a descriptive diagnostic, not a formal
    statistical drift test.
    """

    print("\n========== FEATURE MEANS BY PARTITION ==========\n")

    table = pd.DataFrame({
        name: partitions[name][FEATURE_COLUMNS].mean()
        for name in PARTITION_NAMES
    })

    print(table.round(3).to_string())


# ==================================================
# 8. DISPLAY PREPARED DATA
# ==================================================

def display_prepared_data(data):
    """Display shapes and target distributions."""

    print("\n========== DATA PREPARATION RESULTS ==========")

    for name, values in data.items():

        if isinstance(values, pd.DataFrame):

            print(f"\n{name}: {values.shape}")

        elif isinstance(values, pd.Series):

            print(f"\n{name}: {values.shape}")
            print(values.value_counts().to_string())

        else:

            print(f"\n{name}: {values}")


# ==================================================
# MAIN EXECUTION
# ==================================================

if __name__ == "__main__":

    df = load_dataset()

    partitions = split_dataset(df)

    data = prepare_data()

    display_temporal_diagnostics(
        partitions,
        data["target_median"]
    )

    compare_feature_distributions(partitions)

    display_prepared_data(data)

    cal_batches = sample_batches(
        data["X_calibration"],
        data["y_calibration"],
        n_batches=100
    )

    eval_batches = chronological_batches(
        data["X_evaluation"],
        data["y_evaluation"]
    )

    print("\n========== BATCHING ==========")
    print("Calibration batches:", len(cal_batches), "x", len(cal_batches[0][0]), "rows")
    print("Natural evaluation batches:", len(eval_batches), "x", len(eval_batches[0][0]), "rows")

    print("\n========== INSPECTION COMPLETE ==========")
