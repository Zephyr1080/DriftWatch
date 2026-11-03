
import pandas as pd
from pathlib import Path


# ==================================================
# CONFIGURATION
# ==================================================

DATA_PATH = Path("data/raw/hour.csv")

RANDOM_STATE = 42

REFERENCE_PERIOD_RATIO = 0.60
CALIBRATION_RATIO = 0.20

# yr is deliberately excluded because it becomes constant
# in the later portion of this dataset.
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
    Create three partitions:

    1. Reference/training: approximately 48%
    2. Calibration: approximately 12%
    3. Natural evaluation: approximately 40%

    Calibration dates are sampled from the initial
    reference period, using a reproducible seed.

    Splitting is performed by whole dates.
    """

    unique_dates = sorted(df["dteday"].unique())

    # First 60% of dates form the historical pool
    reference_period_end = int(
        len(unique_dates) * REFERENCE_PERIOD_RATIO
    )

    reference_period_dates = unique_dates[
        :reference_period_end
    ]

    # Remaining 40% are kept for chronological evaluation
    evaluation_dates = unique_dates[
        reference_period_end:
    ]

    # Sample 20% of historical dates for calibration
    calibration_dates = pd.Series(
        reference_period_dates
    ).sample(
        frac=CALIBRATION_RATIO,
        random_state=RANDOM_STATE
    ).tolist()

    calibration_date_set = set(calibration_dates)

    # Remaining historical dates form reference data
    reference_dates = [
        date for date in reference_period_dates
        if date not in calibration_date_set
    ]

    reference_date_set = set(reference_dates)
    evaluation_date_set = set(evaluation_dates)

    reference_df = df[
        df["dteday"].isin(reference_date_set)
    ].copy()

    calibration_df = df[
        df["dteday"].isin(calibration_date_set)
    ].copy()

    evaluation_df = df[
        df["dteday"].isin(evaluation_date_set)
    ].copy()

    return (
        reference_df.reset_index(drop=True),
        calibration_df.reset_index(drop=True),
        evaluation_df.reset_index(drop=True)
    )


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
    """

    df = load_dataset()

    (
        reference_df,
        calibration_df,
        evaluation_df
    ) = split_dataset(df)

    # Target threshold from reference data only
    target_median = reference_df["cnt"].median()

    # Reference / model training data
    X_train, y_train = prepare_features(
        reference_df,
        target_median
    )

    # Calibration data
    X_calibration, y_calibration = prepare_features(
        calibration_df,
        target_median
    )

    # Natural chronological evaluation data
    X_evaluation, y_evaluation = prepare_features(
        evaluation_df,
        target_median
    )

    return {
        "X_train": X_train,
        "y_train": y_train,

        "X_calibration": X_calibration,
        "y_calibration": y_calibration,

        "X_evaluation": X_evaluation,
        "y_evaluation": y_evaluation,

        "target_median": target_median
    }


# ==================================================
# 5. TEMPORAL DIAGNOSTICS
# ==================================================

def display_temporal_diagnostics(
    reference_df,
    calibration_df,
    evaluation_df,
    target_median
):
    """Display partition dates and demand distributions."""

    print("\n========== TEMPORAL DIAGNOSTICS ==========")

    partitions = [
        ("Reference / Training", reference_df),
        ("Calibration", calibration_df),
        ("Natural Evaluation", evaluation_df)
    ]

    print("\nPartition date ranges:")

    for name, partition in partitions:

        print(
            f"{name}: "
            f"{partition['dteday'].min()} to "
            f"{partition['dteday'].max()}"
        )

    print("\nPartition sizes:")

    for name, partition in partitions:
        print(f"{name}: {len(partition)} records")

    print("\nTarget threshold:", target_median)

    print("\nHigh-demand percentage:")

    for name, partition in partitions:

        percentage = (
            (partition["cnt"] > target_median).mean() * 100
        )

        print(f"{name}: {percentage:.2f}%")


# ==================================================
# 6. FEATURE DISTRIBUTION COMPARISON
# ==================================================

def compare_feature_distributions(
    reference_df,
    calibration_df
):
    """
    Compare reference and calibration feature statistics.

    This is a descriptive diagnostic, not a formal
    statistical drift test.
    """

    print(
        "\n========== FEATURE DISTRIBUTION COMPARISON =========="
    )

    for feature in FEATURE_COLUMNS:

        reference_mean = reference_df[feature].mean()
        calibration_mean = calibration_df[feature].mean()

        reference_std = reference_df[feature].std()
        calibration_std = calibration_df[feature].std()

        print(f"\nFeature: {feature}")

        print(f"Reference mean: {reference_mean:.4f}")
        print(f"Calibration mean: {calibration_mean:.4f}")

        print(f"Reference std: {reference_std:.4f}")
        print(f"Calibration std: {calibration_std:.4f}")


# ==================================================
# 7. DISPLAY PREPARED DATA
# ==================================================

def display_prepared_data(data):
    """Display shapes and target distributions."""

    print("\n========== DATA PREPARATION RESULTS ==========")

    for name, values in data.items():

        if isinstance(values, pd.DataFrame):

            print(f"\n{name}: {values.shape}")

        elif isinstance(values, pd.Series):

            print(f"\n{name}: {values.shape}")
            print(values.value_counts())

        else:

            print(f"\n{name}: {values}")


# ==================================================
# MAIN EXECUTION
# ==================================================

if __name__ == "__main__":

    # Load dataset
    df = load_dataset()

    # Create partitions
    (
        reference_df,
        calibration_df,
        evaluation_df
    ) = split_dataset(df)

    # Prepare features and labels
    data = prepare_data()

    # Display temporal diagnostics
    display_temporal_diagnostics(
        reference_df,
        calibration_df,
        evaluation_df,
        data["target_median"]
    )

    # Compare reference and calibration distributions
    compare_feature_distributions(
        reference_df,
        calibration_df
    )

    # Display final prepared data
    display_prepared_data(data)

    print("\n========== INSPECTION COMPLETE ==========")
