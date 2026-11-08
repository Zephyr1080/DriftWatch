"""
DriftWatch - experiment runner (scripts/run_experiments.py)

For every drift type and severity (plus the no-drift control) it builds many
streams of batches where drift starts at a KNOWN batch, then records for each
batch:
    - PSI and KS drift scores and whether each detector alerts
      (alert = score > calibrated threshold from results/thresholds.json)
    - accuracy of each model (random forest, SVM, logistic regression)

Outputs (in results/):
    batch_results.csv   one row per batch (raw data for the analysis)
    experiments.csv     summary per detector / drift type / severity

Metrics (per detector)
    false_alarm_rate  share of CLEAN batches that alert (target about 5%)
    detection_rate    share of streams where at least one alert fires at or
                      after the onset batch
    mean_delay        batches between onset and first alert, over detected
                      streams (0 = alert on the very first drifted batch)
    post_alert_rate   share of drifted batches that alert
    acc_drop_<model>  mean accuracy before onset minus after onset

Run from the project root:
    python scripts/run_experiments.py
Optional:
    python scripts/run_experiments.py --seeds 20 --batches 30 --onset 15 --jobs -1
    (--jobs -1 uses every CPU core; --jobs 1 runs on one core)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import prepare_data, BATCH_SIZE
from model import train_all_models, MODEL_NAMES
from detectors import detect_psi, detect_ks
from calibration import (
    stable_pool, calibrate_all, load_thresholds, save_thresholds,
    evaluate_calibration, THRESHOLDS_PATH
)
from simulator import (
    make_stream, impute_with_reference, DRIFT_TYPES, SEVERITIES
)

RESULTS_DIR = PROJECT_ROOT / "results"


# ==================================================
# 1. RUN ONE STREAM
# ==================================================

def run_stream(stream, reference, models, thresholds, drift_type, severity, seed):
    """Score every batch of one stream. Returns a list of row dicts."""

    rows = []

    for item in stream:

        X_batch, y_batch = item["X"], item["y"]

        psi = detect_psi(reference, X_batch)["score"]
        ks = detect_ks(reference, X_batch)["score"]

        X_model = impute_with_reference(X_batch, reference)

        row = {
            "drift_type": drift_type,
            "severity": severity,
            "seed": seed,
            "batch_id": item["batch_id"],
            "is_drifted": int(item["is_drifted"]),
            "psi_score": psi,
            "ks_score": ks,
            "psi_alert": int(psi > thresholds["psi"]),
            "ks_alert": int(ks > thresholds["ks"]),
        }

        for name, model in models.items():
            row[f"acc_{name}"] = model.accuracy(X_model, y_batch)

        rows.append(row)

    return rows


def run_task(
    X_pool, y_pool, reference, models, thresholds,
    drift_type, severity, seed, n_batches, onset, batch_size
):
    """One stream (one condition, one seed). Runs in a worker process."""

    stream = make_stream(
        X_pool, y_pool, drift_type,
        severity if drift_type != "none" else "medium",
        n_batches=n_batches,
        onset_batch=onset,
        batch_size=batch_size,
        seed=seed
    )

    return run_stream(
        stream, reference, models, thresholds, drift_type, severity, seed
    )


# ==================================================
# 2. SUMMARISE
# ==================================================

def summarise(batch_df, onset):
    """One summary row per detector, drift type and severity."""

    summary = []

    for (drift_type, severity), group in batch_df.groupby(
        ["drift_type", "severity"], sort=False
    ):

        is_control = drift_type == "none"

        for detector in ("psi", "ks"):

            alert_col = f"{detector}_alert"

            per_stream = []

            for _, stream in group.groupby("seed"):

                stream = stream.sort_values("batch_id")

                clean = stream if is_control else stream[stream["batch_id"] < onset]
                drifted = stream[stream["batch_id"] >= onset]

                first_alert = drifted.loc[drifted[alert_col] == 1, "batch_id"]

                record = {
                    "clean_alerts": clean[alert_col].mean(),
                    "post_alert_rate": drifted[alert_col].mean(),
                    "detected": int(len(first_alert) > 0),
                    "delay": (first_alert.iloc[0] - onset) if len(first_alert) else np.nan,
                }

                for name in MODEL_NAMES:
                    record[f"acc_drop_{name}"] = (
                        clean[f"acc_{name}"].mean() - drifted[f"acc_{name}"].mean()
                    )

                per_stream.append(record)

            per_stream = pd.DataFrame(per_stream)

            row = {
                "detector": detector.upper(),
                "drift_type": drift_type,
                "severity": severity,
                "n_streams": len(per_stream),
                "false_alarm_rate": per_stream["clean_alerts"].mean(),
            }

            if is_control:
                row["detection_rate"] = np.nan
                row["mean_delay"] = np.nan
                row["post_alert_rate"] = np.nan
            else:
                row["detection_rate"] = per_stream["detected"].mean()
                row["mean_delay"] = per_stream["delay"].mean()
                row["post_alert_rate"] = per_stream["post_alert_rate"].mean()

            for name in MODEL_NAMES:
                row[f"acc_drop_{name}"] = per_stream[f"acc_drop_{name}"].mean()

            summary.append(row)

    return pd.DataFrame(summary)


# ==================================================
# 3. MAIN
# ==================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20, help="streams per condition")
    parser.add_argument("--batches", type=int, default=30, help="batches per stream")
    parser.add_argument("--onset", type=int, default=15, help="first drifted batch")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--jobs", type=int, default=-1, help="CPU cores to use (-1 = all)")
    args = parser.parse_args()

    start = time.time()

    data = prepare_data()

    reference = data["X_train"]

    X_pool, y_pool = stable_pool(data)

    # thresholds: use the saved file, or calibrate now
    if THRESHOLDS_PATH.exists():
        thresholds = load_thresholds()
    else:
        calibrated = calibrate_all(reference, X_pool, y_pool)
        save_thresholds(
            calibrated,
            evaluate_calibration(calibrated, reference, X_pool, y_pool)
        )
        thresholds = load_thresholds()

    print("Thresholds:", {k: round(v, 4) for k, v in thresholds.items()})

    models = train_all_models(data["X_train"], data["y_train"])

    conditions = [("none", "-")] + [
        (d, s) for d in DRIFT_TYPES if d != "none" for s in SEVERITIES
    ]

    # avoid oversubscribing cores: the forest runs single-threaded inside workers
    models["random_forest"].estimator.set_params(n_jobs=1)

    tasks = [
        (drift_type, severity, seed)
        for drift_type, severity in conditions
        for seed in range(args.seeds)
    ]

    print(f"Running {len(tasks)} streams x {args.batches} batches "
          f"on {args.jobs} job(s)...")

    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_task)(
            X_pool, y_pool, reference, models, thresholds,
            drift_type, severity, seed,
            args.batches, args.onset, args.batch_size
        )
        for drift_type, severity, seed in tasks
    )

    rows = [row for stream_rows in results for row in stream_rows]

    batch_df = pd.DataFrame(rows)

    summary = summarise(batch_df, args.onset)

    RESULTS_DIR.mkdir(exist_ok=True)

    batch_df.to_csv(RESULTS_DIR / "batch_results.csv", index=False)
    summary.to_csv(RESULTS_DIR / "experiments.csv", index=False)

    # ---- readable comparison ----
    show = summary.copy()

    pivot = show.pivot_table(
        index=["drift_type", "severity"],
        columns="detector",
        values=["detection_rate", "mean_delay", "false_alarm_rate"],
        sort=False
    )

    rf_drop = (
        show[show["detector"] == "PSI"]
        .set_index(["drift_type", "severity"])["acc_drop_random_forest"]
    )

    pivot[("acc_drop", "RF")] = rf_drop

    pd.set_option("display.width", 200)

    print("\n========== PSI vs KS (same false-alarm target) ==========\n")
    print(pivot.round(3).to_string())

    print(f"\nSaved: {RESULTS_DIR / 'batch_results.csv'}")
    print(f"Saved: {RESULTS_DIR / 'experiments.csv'}")
    print(f"Total time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()