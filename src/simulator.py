"""
DriftWatch - controlled drift simulator (src/simulator.py)

Real drift happens at an unknown time, so to measure detection delay we
create drift ourselves, starting at a KNOWN batch.

Drift types
    none             no-drift control
    covariate_shift  weather moves: temp/atemp warmer, hum drier (inputs shift)
    hour_shift       hr moves later in the day (clipped at 23). The models rely
                     heavily on hr, so this is the drift most likely to hurt accuracy
    noise            random noise added to the weather features (sensor noise)
    missing          values randomly removed from the weather features

Severities: low, medium, high. Parameters are in feature units. temp, atemp,
hum and windspeed are normalised to 0-1 in this dataset (std about 0.2).

Labels are never changed. Only the inputs are damaged, which is covariate
drift and corruption, not concept drift.

Reproducibility: every function takes a seed. Same seed, same output.

Missing values
    "missing" drift returns NaNs. Detectors fill NaNs with the reference
    median internally. Before calling a model, use impute_with_reference().
"""

import numpy as np
import pandas as pd

try:
    from src.data import sample_batches, BATCH_SIZE
except ImportError:
    from data import sample_batches, BATCH_SIZE


# ==================================================
# CONFIGURATION
# ==================================================

DRIFT_TYPES = ["none", "covariate_shift", "hour_shift", "noise", "missing"]

SEVERITIES = ["low", "medium", "high"]

WEATHER_FEATURES = ["temp", "atemp", "hum", "windspeed"]

# severity -> parameter for each drift type
PARAMS = {
    # temp/atemp shifted up by this, hum shifted down by this
    "covariate_shift": {"low": 0.05, "medium": 0.10, "high": 0.20},
    # hr shifted later by this many hours (clipped at 23)
    "hour_shift": {"low": 1, "medium": 3, "high": 6},
    # standard deviation of Gaussian noise added to weather features
    "noise": {"low": 0.05, "medium": 0.15, "high": 0.30},
    # probability that each weather value is removed
    "missing": {"low": 0.05, "medium": 0.15, "high": 0.30},
}


# ==================================================
# 1. DRIFT ONE BATCH
# ==================================================

def make_drifted_batch(df, drift_type, severity="medium", seed=0):
    """
    Return a drifted copy of df (a feature DataFrame). df is not modified.

    drift_type: one of DRIFT_TYPES
    severity:   one of SEVERITIES (ignored for 'none')
    """

    if drift_type not in DRIFT_TYPES:
        raise ValueError(f"drift_type must be one of {DRIFT_TYPES}")

    batch = df.copy()

    if drift_type == "none":
        return batch

    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}")

    amount = PARAMS[drift_type][severity]
    rng = np.random.default_rng(seed)

    columns = [c for c in WEATHER_FEATURES if c in batch.columns]

    if drift_type == "covariate_shift":

        for column in ("temp", "atemp"):
            if column in batch.columns:
                batch[column] = (batch[column] + amount).clip(0, 1)

        if "hum" in batch.columns:
            batch["hum"] = (batch["hum"] - amount).clip(0, 1)

    elif drift_type == "hour_shift":

        batch["hr"] = (batch["hr"] + amount).clip(upper=23)

    elif drift_type == "noise":

        for column in columns:
            noise = rng.normal(0, amount, len(batch))
            batch[column] = (batch[column] + noise).clip(0, 1)

    elif drift_type == "missing":

        for column in columns:
            removed = rng.random(len(batch)) < amount
            batch[column] = batch[column].mask(removed)

    return batch


# ==================================================
# 2. DRIFT STREAM WITH A KNOWN ONSET
# ==================================================

def make_stream(
    X,
    y,
    drift_type,
    severity="medium",
    n_batches=40,
    onset_batch=20,
    batch_size=BATCH_SIZE,
    seed=0
):
    """
    A sequence of batches where drift starts at a known batch.

    X, y: STABLE data (use the stable_test partition) the batches are drawn from.
    Batches 0 .. onset_batch-1 are clean; batches from onset_batch on are drifted.
    With drift_type='none' no batch is drifted (the no-drift control).

    Returns a list of dicts:
        {"batch_id", "X", "y", "is_drifted"}
    """

    if not 0 <= onset_batch <= n_batches:
        raise ValueError("onset_batch must be between 0 and n_batches.")

    batches = sample_batches(X, y, n_batches, batch_size, seed)

    stream = []

    for batch_id, (X_batch, y_batch) in enumerate(batches):

        is_drifted = (drift_type != "none") and (batch_id >= onset_batch)

        if is_drifted:
            X_batch = make_drifted_batch(
                X_batch,
                drift_type,
                severity,
                seed=seed * 100_000 + batch_id
            )

        stream.append({
            "batch_id": batch_id,
            "X": X_batch,
            "y": y_batch,
            "is_drifted": is_drifted
        })

    return stream


# ==================================================
# 3. HELPER FOR MODELS
# ==================================================

def impute_with_reference(batch, reference_df):
    """Fill NaNs with reference medians so a model can predict on the batch."""

    return batch.fillna(reference_df.median())


# ==================================================
# MAIN EXECUTION (sanity check)
# ==================================================

if __name__ == "__main__":

    try:
        from src.data import prepare_data
        from src.detectors import detect_psi, detect_ks
        from src.model import build_model
    except ImportError:
        from data import prepare_data
        from detectors import detect_psi, detect_ks
        from model import build_model

    data = prepare_data()

    reference = data["X_train"]

    model = build_model("random_forest").fit(data["X_train"], data["y_train"])

    N_RUNS = 20

    rows = []

    for drift_type in DRIFT_TYPES:

        severities = ["-"] if drift_type == "none" else SEVERITIES

        for severity in severities:

            psi_scores, ks_scores, accuracies = [], [], []

            for run in range(N_RUNS):

                (X_b, y_b), = sample_batches(
                    data["X_stable_test"], data["y_stable_test"],
                    n_batches=1, seed=run
                )

                drifted = make_drifted_batch(X_b, drift_type, severity, seed=run)

                psi_scores.append(detect_psi(reference, drifted)["score"])
                ks_scores.append(detect_ks(reference, drifted)["score"])

                accuracies.append(
                    model.accuracy(impute_with_reference(drifted, reference), y_b)
                )

            rows.append({
                "drift_type": drift_type,
                "severity": severity,
                "PSI": np.mean(psi_scores),
                "KS": np.mean(ks_scores),
                "RF_accuracy": np.mean(accuracies)
            })

    table = pd.DataFrame(rows)

    baseline = table.loc[table["drift_type"] == "none", "RF_accuracy"].iloc[0]
    table["accuracy_drop"] = baseline - table["RF_accuracy"]

    print(f"\n========== SIMULATOR CHECK (mean of {N_RUNS} batches each) ==========\n")
    print(table.round(4).to_string(index=False))

    stream = make_stream(
        data["X_stable_test"], data["y_stable_test"],
        "covariate_shift", "high", n_batches=10, onset_batch=6, seed=1
    )

    print("\nStream example (covariate_shift, high, onset at batch 6):")
    print("is_drifted per batch:", [int(s["is_drifted"]) for s in stream])

    print("\n========== SIMULATOR CHECK COMPLETE ==========")