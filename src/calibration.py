import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.data import sample_batches, BATCH_SIZE
    from src.detectors import DETECTORS
except ImportError:
    from data import sample_batches, BATCH_SIZE
    from detectors import DETECTORS


# ==================================================
# CONFIGURATION
# ==================================================

TARGET_FAR = 0.05          # target false-alarm rate
N_CALIBRATION_BATCHES = 300
N_CONTROL_BATCHES = 300
RANDOM_STATE = 42

THRESHOLDS_PATH = Path(__file__).resolve().parents[1] / "results" / "thresholds.json"


# ==================================================
# 0. THE STABLE POOL
# ==================================================

def stable_pool(data):
    """
    Calibration + stable_test partitions combined: the stable data used both
    for calibration and as the base for the control and drifted experiment batches.
    Returns (X, y).
    """

    X = pd.concat(
        [data["X_calibration"], data["X_stable_test"]], ignore_index=True
    )

    y = pd.concat(
        [data["y_calibration"], data["y_stable_test"]], ignore_index=True
    )

    return X, y


# ==================================================
# 1. SCORES ON STABLE BATCHES
# ==================================================

def stable_scores(
    detector_name,
    reference,
    X,
    y,
    n_batches,
    batch_size=BATCH_SIZE,
    seed=RANDOM_STATE
):
    """Dataset-level drift scores of many stable batches against the reference."""

    detector = DETECTORS[detector_name]

    batches = sample_batches(X, y, n_batches, batch_size, seed)

    return np.array([
        detector(reference, X_batch)["score"] for X_batch, _ in batches
    ])


# ==================================================
# 2. CALIBRATE
# ==================================================

def calibrate_detector(
    detector_name,
    reference,
    X_calibration,
    y_calibration,
    target_far=TARGET_FAR,
    n_batches=N_CALIBRATION_BATCHES,
    batch_size=BATCH_SIZE,
    seed=RANDOM_STATE
):
    """Threshold at the (1 - target_far) percentile of stable-batch scores."""

    scores = stable_scores(
        detector_name, reference,
        X_calibration, y_calibration,
        n_batches, batch_size, seed
    )

    return {
        "threshold": float(np.quantile(scores, 1 - target_far)),
        "target_far": target_far,
        "n_batches": n_batches,
        "batch_size": batch_size,
        "stable_score_mean": float(scores.mean()),
        "stable_score_std": float(scores.std())
    }


def calibrate_all(
    reference,
    X_calibration,
    y_calibration,
    target_far=TARGET_FAR,
    n_batches=N_CALIBRATION_BATCHES,
    batch_size=BATCH_SIZE,
    seed=RANDOM_STATE
):
    """Calibrate every detector. Returns {detector_name: calibration dict}."""

    return {
        name: calibrate_detector(
            name, reference, X_calibration, y_calibration,
            target_far, n_batches, batch_size, seed
        )
        for name in DETECTORS
    }


# ==================================================
# 3. ACHIEVED FALSE-ALARM RATE ON UNSEEN STABLE DATA
# ==================================================

def false_alarm_rate(
    detector_name,
    threshold,
    reference,
    X_stable,
    y_stable,
    n_batches=N_CONTROL_BATCHES,
    batch_size=BATCH_SIZE,
    seed=RANDOM_STATE + 1
):
    """Share of stable batches whose score exceeds the threshold."""

    scores = stable_scores(
        detector_name, reference,
        X_stable, y_stable,
        n_batches, batch_size, seed
    )

    return float((scores > threshold).mean())


def evaluate_calibration(
    thresholds,
    reference,
    X_stable,
    y_stable,
    n_batches=N_CONTROL_BATCHES,
    batch_size=BATCH_SIZE,
    seed=RANDOM_STATE + 1
):
    """Achieved false-alarm rate per detector on the stable_test partition."""

    return {
        name: false_alarm_rate(
            name, info["threshold"], reference,
            X_stable, y_stable, n_batches, batch_size, seed
        )
        for name, info in thresholds.items()
    }


# ==================================================
# 4. SAVE / LOAD (so the pipeline and dashboard just read the file)
# ==================================================

def save_thresholds(thresholds, achieved_far=None, path=THRESHOLDS_PATH):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"thresholds": thresholds}

    if achieved_far is not None:
        payload["achieved_false_alarm_rate"] = achieved_far

    path.write_text(json.dumps(payload, indent=2))


def load_thresholds(path=THRESHOLDS_PATH):
    """Return {detector_name: threshold_float}."""

    payload = json.loads(Path(path).read_text())

    return {
        name: info["threshold"]
        for name, info in payload["thresholds"].items()
    }


# ==================================================
# MAIN EXECUTION
# ==================================================

if __name__ == "__main__":

    try:
        from src.data import prepare_data
    except ImportError:
        from data import prepare_data

    data = prepare_data()

    reference = data["X_train"]

    X_pool, y_pool = stable_pool(data)

    thresholds = calibrate_all(reference, X_pool, y_pool)

    # fresh batches (different seed) from the same stable pool
    achieved = evaluate_calibration(thresholds, reference, X_pool, y_pool)

    print("\n========== CALIBRATION (target false-alarm rate = "
          f"{TARGET_FAR:.0%}) ==========\n")

    print(f"{'detector':9s} {'threshold':>10s} {'stable mean':>12s} {'achieved FAR (fresh pool batches)':>36s}")

    for name, info in thresholds.items():
        print(
            f"{name.upper():9s} {info['threshold']:10.4f} "
            f"{info['stable_score_mean']:12.4f} {achieved[name]:36.1%}"
        )

    # diagnostic: calibrate on one set of days, test on a DIFFERENT set of days
    cross = calibrate_all(
        reference, data["X_calibration"], data["y_calibration"]
    )

    cross_far = evaluate_calibration(
        cross, reference, data["X_stable_test"], data["y_stable_test"]
    )

    print("\nDiagnostic (calibrate on calibration days, test on stable_test days):")

    for name in cross_far:
        print(f"  {name.upper():4s} false-alarm rate = {cross_far[name]:.1%}")

    print(
        "  -> large gaps here mean thresholds depend on WHICH stable days are "
        "used; report this as a limitation."
    )

    save_thresholds(thresholds, achieved)

    print(f"\nSaved to {THRESHOLDS_PATH}")

    print("\n========== CALIBRATION COMPLETE ==========")