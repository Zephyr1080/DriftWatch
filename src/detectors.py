"""
DriftWatch - drift detectors (src/detectors.py)

Two detectors with the SAME interface:

    detect_psi(reference_df, current_df) -> {"score": float, "features": {name: float}}
    detect_ks(reference_df, current_df)  -> {"score": float, "features": {name: float}}

Both take only feature DataFrames. They never see a model, so they work
unchanged whichever classifier is being monitored (model-agnostic).

Dataset-level score
    Each feature gets its own score. The dataset-level "score" is the MAXIMUM
    over features (aggregation="max", the default) or the mean (aggregation="mean").
    Max is more sensitive to drift in a single feature, but it also raises the
    false-alarm rate as more features are monitored. This is why thresholds are
    calibrated empirically on stable data (calibration.py), per detector and
    per aggregation, instead of using textbook cut-offs.

Missing values
    Corruption drift can introduce NaNs. Before scoring, NaNs in the current
    batch are filled with the REFERENCE median of that feature. This is
    documented behaviour: heavy missingness therefore shows up as a pile-up
    at the median.

Feature types
    - Continuous features (non-integer values, e.g. temp, hum): PSI bins come
      from reference quantiles.
    - Discrete features (integer-valued with few unique values, e.g. hr,
      weekday, season): PSI uses one bin per category seen in the reference,
      plus an "unseen" bin for new categories.
    - KS is applied to every feature as-is. On discrete features KS has many
      ties; the statistic is still a valid distance between distributions
      (so we use it as the score), but its p-values are conservative.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# ==================================================
# CONFIGURATION
# ==================================================

N_BINS = 10                # quantile bins for continuous features (PSI)
EPSILON = 1e-4             # floor for empty bins (PSI)
DISCRETE_MAX_UNIQUE = 25   # integer-valued features with <= this many values are discrete


# ==================================================
# HELPERS
# ==================================================

def _is_discrete(series):
    """Integer-valued feature with few unique values."""

    values = series.dropna()

    return (
        values.nunique() <= DISCRETE_MAX_UNIQUE
        and np.allclose(values, np.round(values))
    )


def _prepare(reference_df, current_df):
    """Align columns and fill NaNs in the current batch with reference medians."""

    columns = [c for c in reference_df.columns if c in current_df.columns]

    if not columns:
        raise ValueError("reference_df and current_df share no columns.")

    reference = reference_df[columns]
    current = current_df[columns].fillna(reference.median())

    return reference, current


def _aggregate(feature_scores, aggregation):

    values = list(feature_scores.values())

    if aggregation == "max":
        return float(np.max(values))

    if aggregation == "mean":
        return float(np.mean(values))

    raise ValueError("aggregation must be 'max' or 'mean'.")


# ==================================================
# 1. PSI
# ==================================================

def _psi_from_proportions(ref_p, cur_p):
    """PSI = sum((cur - ref) * ln(cur / ref)), empty bins floored at EPSILON."""

    ref_p = np.clip(ref_p, EPSILON, None)
    cur_p = np.clip(cur_p, EPSILON, None)

    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def _psi_continuous(ref_values, cur_values, n_bins=N_BINS):
    """Quantile bins from the reference; outer bins are open-ended."""

    edges = np.unique(
        np.quantile(ref_values, np.linspace(0, 1, n_bins + 1))
    )

    if len(edges) < 2:
        return 0.0                      # constant reference feature

    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref_values, bins=edges)
    cur_counts, _ = np.histogram(cur_values, bins=edges)

    return _psi_from_proportions(
        ref_counts / len(ref_values),
        cur_counts / len(cur_values)
    )


def _psi_discrete(ref_values, cur_values):
    """One bin per reference category, plus one 'unseen' bin."""

    categories = np.unique(ref_values)

    ref_p = np.array([(ref_values == c).mean() for c in categories])
    cur_p = np.array([(cur_values == c).mean() for c in categories])

    unseen = 1.0 - cur_p.sum()

    ref_p = np.append(ref_p, 0.0)
    cur_p = np.append(cur_p, max(unseen, 0.0))

    return _psi_from_proportions(ref_p, cur_p)


def detect_psi(reference_df, current_df, aggregation="max"):
    """Population Stability Index per feature, aggregated to one score."""

    reference, current = _prepare(reference_df, current_df)

    features = {}

    for column in reference.columns:

        ref_values = reference[column].to_numpy(dtype=float)
        cur_values = current[column].to_numpy(dtype=float)

        if _is_discrete(reference[column]):
            features[column] = _psi_discrete(ref_values, cur_values)
        else:
            features[column] = _psi_continuous(ref_values, cur_values)

    return {
        "score": _aggregate(features, aggregation),
        "features": features
    }


# ==================================================
# 2. KS TWO-SAMPLE TEST
# ==================================================

def detect_ks(reference_df, current_df, aggregation="max"):
    """
    Kolmogorov-Smirnov statistic per feature, aggregated to one score.

    The score is the KS statistic (largest gap between the two empirical
    CDFs, between 0 and 1). P-values are returned too, under "p_values",
    for reference, but alerting uses the statistic with a calibrated
    threshold so PSI and KS can be compared at the same false-alarm rate.
    """

    reference, current = _prepare(reference_df, current_df)

    features = {}
    p_values = {}

    for column in reference.columns:

        result = ks_2samp(
            reference[column].to_numpy(dtype=float),
            current[column].to_numpy(dtype=float)
        )

        features[column] = float(result.statistic)
        p_values[column] = float(result.pvalue)

    return {
        "score": _aggregate(features, aggregation),
        "features": features,
        "p_values": p_values
    }


# ==================================================
# 3. REGISTRY (used by calibration, pipeline and runner)
# ==================================================

DETECTORS = {
    "psi": detect_psi,
    "ks": detect_ks
}


# ==================================================
# MAIN EXECUTION (quick sanity check)
# ==================================================

if __name__ == "__main__":

    try:
        from src.data import prepare_data, sample_batches, chronological_batches
    except ImportError:
        from data import prepare_data, sample_batches, chronological_batches

    data = prepare_data()

    reference = data["X_train"]

    print("\n========== SCORES ON STABLE BATCHES (no drift) ==========")

    stable_batches = sample_batches(
        data["X_calibration"], data["y_calibration"], n_batches=100
    )

    for name, detector in DETECTORS.items():

        scores = np.array([
            detector(reference, X)["score"] for X, _ in stable_batches
        ])

        print(
            f"{name.upper():4s} mean={scores.mean():.4f}  "
            f"95th percentile={np.percentile(scores, 95):.4f}  "
            f"max={scores.max():.4f}"
        )

    print("\n========== SCORES ON DRIFTED BATCHES ==========")

    X_batch, _ = stable_batches[0]

    scenarios = {
        "no drift (stable batch)": X_batch,
        "temp +0.15 (covariate shift)": X_batch.assign(temp=X_batch["temp"] + 0.15),
        "hum noise (corruption)": X_batch.assign(
            hum=X_batch["hum"] + np.random.default_rng(0).normal(0, 0.15, len(X_batch))
        ),
        "30% missing temp (filled w/ median)": X_batch.assign(
            temp=X_batch["temp"].mask(np.random.default_rng(1).random(len(X_batch)) < 0.3)
        ),
    }

    natural_batch, _ = chronological_batches(
        data["X_evaluation"], data["y_evaluation"]
    )[-1]

    scenarios["natural drift (last 2012 batch)"] = natural_batch

    for label, batch in scenarios.items():

        psi = detect_psi(reference, batch)
        ks = detect_ks(reference, batch)

        top_psi = max(psi["features"], key=psi["features"].get)
        top_ks = max(ks["features"], key=ks["features"].get)

        print(
            f"{label:38s} PSI={psi['score']:.4f} ({top_psi})   "
            f"KS={ks['score']:.4f} ({top_ks})"
        )

    print("\n========== DETECTOR CHECK COMPLETE ==========")