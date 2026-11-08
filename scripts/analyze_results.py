"""
DriftWatch - analysis of the experiment results (scripts/analyze_results.py)

Reads results/batch_results.csv and results/thresholds.json (written by
run_experiments.py and calibration.py) and writes, to results/analysis/:

    tables:  detection_summary.csv    one row per detector / drift type / severity
             psi_vs_ks_tests.csv      paired comparison of the two detectors
             alert_vs_accuracy.csv    harmless vs damaging, caught vs missed
             natural_eval_check.csv   the real (natural) drift period
    plots:   fig_detection_rate.png, fig_detection_delay.png,
             fig_alert_vs_accuracy.png, fig_stream_example.png
    text:    findings.md              numbers and wording generated from the tables

Stricter detection rule
    Counting ANY alert after the onset flatters weak drift, because even clean
    data alerts about 5% of the time per batch. So a drift is "confirmed" only
    when CONFIRM_K consecutive batches alert (default 2) at or after the onset.
    The detection delay is the number of batches after the onset at which the
    drift is confirmed. The same rule is applied to the no-drift control
    streams, which gives the chance level to compare against.

Comparing PSI and KS
    Both detectors saw exactly the same batches, so detection is compared per
    stream with an exact McNemar-style test (binomial test on the streams where
    only one detector confirmed). With 12 drifted conditions, expect about one
    p < 0.05 by chance alone: read p-values as a guide, not as proof.

Run from the project root:
    python scripts/analyze_results.py
Options:
    --example-type hour_shift --example-severity low --example-seed 0
    --skip-natural        skip the natural-drift check (needs data and models)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = RESULTS_DIR / "analysis"


# ==================================================
# CONFIGURATION
# ==================================================

CONFIRM_K = 2            # consecutive alerting batches needed to confirm drift
DAMAGE_DROP = 0.03       # accuracy drop (3 points) counted as damaging
CAUGHT_RATE = 0.80       # confirmed-detection rate counted as "caught"
DAMAGE_MODEL = "random_forest"
ALPHA = 0.05

DRIFT_ORDER = ["covariate_shift", "hour_shift", "noise", "missing"]
SEVERITY_ORDER = ["low", "medium", "high"]

DRIFT_LABELS = {
    "covariate_shift": "Covariate shift (weather)",
    "hour_shift": "Hour shift",
    "noise": "Noise",
    "missing": "Missing values",
    "none": "No drift (control)",
}

COLORS = {"PSI": "#1f77b4", "KS": "#d62728"}


# ==================================================
# 1. LOAD
# ==================================================

def load_results():

    batch_path = RESULTS_DIR / "batch_results.csv"
    threshold_path = RESULTS_DIR / "thresholds.json"

    if not batch_path.exists():
        raise FileNotFoundError(
            f"{batch_path} not found. Run scripts/run_experiments.py first."
        )

    batch_df = pd.read_csv(batch_path)

    thresholds = json.loads(threshold_path.read_text())

    onset = int(batch_df.loc[batch_df["is_drifted"] == 1, "batch_id"].min())

    return batch_df, thresholds, onset


# ==================================================
# 2. PER-STREAM RESULTS WITH THE STRICT RULE
# ==================================================

def first_confirmed(alerts, k):
    """Index where k consecutive alerts first complete, as the START of the run (or None)."""

    alerts = np.asarray(alerts, dtype=int)

    if len(alerts) < k:
        return None

    runs = np.convolve(alerts, np.ones(k, dtype=int), mode="valid") == k

    return int(np.argmax(runs)) if runs.any() else None


def stream_table(batch_df, onset, k=CONFIRM_K):
    """One row per stream and detector."""

    model_cols = [c for c in batch_df.columns if c.startswith("acc_")]

    rows = []

    for (drift_type, severity, seed), stream in batch_df.groupby(
        ["drift_type", "severity", "seed"], sort=False
    ):

        stream = stream.sort_values("batch_id")

        is_control = drift_type == "none"

        clean = stream if is_control else stream[stream["batch_id"] < onset]
        post = stream[stream["batch_id"] >= onset]

        for detector in ("psi", "ks"):

            alerts = post[f"{detector}_alert"].to_numpy()

            start = first_confirmed(alerts, k)

            row = {
                "drift_type": drift_type,
                "severity": severity,
                "seed": seed,
                "detector": detector.upper(),
                "detected": int(start is not None),
                # batches after the onset at which the drift is confirmed
                "delay": (start + k - 1) if start is not None else np.nan,
                "post_alert_rate": alerts.mean(),
                "clean_alert_rate": clean[f"{detector}_alert"].mean(),
            }

            for col in model_cols:
                row[f"drop_{col[4:]}"] = clean[col].mean() - post[col].mean()

            rows.append(row)

    return pd.DataFrame(rows)


def condition_summary(streams):
    """Average over streams for each detector / drift type / severity."""

    drop_cols = [c for c in streams.columns if c.startswith("drop_")]

    summary = (
        streams.groupby(["drift_type", "severity", "detector"], sort=False)
        .agg(
            n_streams=("seed", "nunique"),
            detection_rate=("detected", "mean"),
            mean_delay=("delay", "mean"),
            post_alert_rate=("post_alert_rate", "mean"),
            false_alarm_rate=("clean_alert_rate", "mean"),
            **{c: (c, "mean") for c in drop_cols},
        )
        .reset_index()
    )

    return summary


def chance_level(summary):
    """Detection rate of the control streams = chance of confirming drift on clean data."""

    control = summary[summary["drift_type"] == "none"]

    return dict(zip(control["detector"], control["detection_rate"]))


# ==================================================
# 3. PSI vs KS TESTS
# ==================================================

def paired_tests(streams):

    rows = []

    drifted = streams[streams["drift_type"] != "none"]

    for (drift_type, severity), group in drifted.groupby(
        ["drift_type", "severity"], sort=False
    ):

        wide = group.pivot(index="seed", columns="detector", values="detected")
        delays = group.pivot(index="seed", columns="detector", values="delay")

        psi_only = int(((wide["PSI"] == 1) & (wide["KS"] == 0)).sum())
        ks_only = int(((wide["PSI"] == 0) & (wide["KS"] == 1)).sum())

        discordant = psi_only + ks_only

        p_value = (
            binomtest(min(psi_only, ks_only), discordant, 0.5).pvalue
            if discordant else 1.0
        )

        if p_value < ALPHA:
            verdict = "PSI better" if psi_only > ks_only else "KS better"
        else:
            verdict = "no clear difference"

        rows.append({
            "drift_type": drift_type,
            "severity": severity,
            "psi_detection_rate": wide["PSI"].mean(),
            "ks_detection_rate": wide["KS"].mean(),
            "psi_only_streams": psi_only,
            "ks_only_streams": ks_only,
            "p_value": p_value,
            "psi_mean_delay": delays["PSI"].mean(),
            "ks_mean_delay": delays["KS"].mean(),
            "verdict": verdict,
        })

    return pd.DataFrame(rows)


# ==================================================
# 4. ALERT vs ACCURACY
# ==================================================

def alert_vs_accuracy(summary):

    drifted = summary[summary["drift_type"] != "none"].copy()

    drifted["accuracy_drop"] = drifted[f"drop_{DAMAGE_MODEL}"]

    drifted["damaging"] = drifted["accuracy_drop"] >= DAMAGE_DROP
    drifted["caught"] = drifted["detection_rate"] >= CAUGHT_RATE

    def label(row):
        if row["damaging"] and row["caught"]:
            return "damaging, caught"
        if row["damaging"]:
            return "damaging, MISSED"
        if row["caught"]:
            return "harmless drift, alert fired"
        return "harmless drift, quiet"

    drifted["case"] = drifted.apply(label, axis=1)

    return drifted[[
        "drift_type", "severity", "detector", "detection_rate",
        "accuracy_drop", "damaging", "caught", "case"
    ]]


# ==================================================
# 5. PLOTS
# ==================================================

def _condition_axes(n=4):

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6), sharey=True)

    return fig, axes


def plot_lines(summary, value, ylabel, title, filename, chance=None):

    fig, axes = _condition_axes(len(DRIFT_ORDER))

    for ax, drift_type in zip(axes, DRIFT_ORDER):

        for detector in ("PSI", "KS"):

            part = summary[
                (summary["drift_type"] == drift_type)
                & (summary["detector"] == detector)
            ].set_index("severity").reindex(SEVERITY_ORDER)

            ax.plot(
                SEVERITY_ORDER, part[value],
                marker="o" if detector == "PSI" else "D",
                markersize=9 if detector == "PSI" else 5,
                color=COLORS[detector], label=detector,
                linewidth=4 if detector == "PSI" else 1.5,
                linestyle="-" if detector == "PSI" else "--",
                alpha=0.9
            )

        if chance:
            ax.axhline(
                np.mean(list(chance.values())), color="gray",
                linestyle="--", linewidth=1,
                label="chance (no-drift control)"
            )

        ax.set_title(DRIFT_LABELS[drift_type], fontsize=10)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(ylabel)

    axes[-1].legend(fontsize=8, loc="best")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150)
    plt.close(fig)


def plot_alert_vs_accuracy(table):

    fig, ax = plt.subplots(figsize=(7, 5))

    markers = {"low": "o", "medium": "s", "high": "^"}

    for detector in ("PSI", "KS"):

        for severity in SEVERITY_ORDER:

            part = table[
                (table["detector"] == detector) & (table["severity"] == severity)
            ]

            ax.scatter(
                part["accuracy_drop"] * 100, part["detection_rate"] * 100,
                color=COLORS[detector], marker=markers[severity],
                s=70, alpha=0.8,
                label=f"{detector} / {severity}"
            )

    ax.axvline(DAMAGE_DROP * 100, color="gray", linestyle="--", linewidth=1)
    ax.axhline(CAUGHT_RATE * 100, color="gray", linestyle="--", linewidth=1)

    ax.text(DAMAGE_DROP * 100 + 0.5, 3, "damaging", fontsize=8, color="gray")
    ax.text(0.3, CAUGHT_RATE * 100 + 2, "alert fires", fontsize=8, color="gray")

    ax.set_xlabel(f"Accuracy drop after drift ({DAMAGE_MODEL.replace('_', ' ')}, points)")
    ax.set_ylabel("Drift confirmed (% of streams)")
    ax.set_title("Do alerts match real damage?")
    ax.set_ylim(-3, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_alert_vs_accuracy.png", dpi=150)
    plt.close(fig)


def plot_stream(batch_df, thresholds, onset, drift_type, severity, seed):

    stream = batch_df[
        (batch_df["drift_type"] == drift_type)
        & (batch_df["severity"] == severity)
        & (batch_df["seed"] == seed)
    ].sort_values("batch_id")

    if stream.empty:
        print("Example stream not found, skipping fig_stream_example.png")
        return

    psi_thr = thresholds["thresholds"]["psi"]["threshold"]
    ks_thr = thresholds["thresholds"]["ks"]["threshold"]

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]}
    )

    top.plot(stream["batch_id"], stream["psi_score"] / psi_thr,
             color=COLORS["PSI"], marker="o", markersize=3, label="PSI / threshold")
    top.plot(stream["batch_id"], stream["ks_score"] / ks_thr,
             color=COLORS["KS"], marker="o", markersize=3, label="KS / threshold")
    top.axhline(1, color="black", linestyle="--", linewidth=1)
    top.text(0.2, 1.03, "alert threshold", fontsize=8)
    top.axvline(onset - 0.5, color="gray", linestyle=":")
    top.set_ylabel("Drift score / threshold")
    top.legend(fontsize=8, loc="upper left")
    top.grid(alpha=0.3)

    bottom.plot(stream["batch_id"], stream["acc_random_forest"],
                color="green", marker="o", markersize=3)
    bottom.axvline(onset - 0.5, color="gray", linestyle=":")
    bottom.set_ylabel("Random forest accuracy")
    bottom.set_xlabel("Batch")
    bottom.grid(alpha=0.3)

    top.set_title(
        f"One stream: {DRIFT_LABELS[drift_type]}, {severity} severity "
        f"(seed {seed}); drift starts at batch {onset}"
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_stream_example.png", dpi=150)
    plt.close(fig)


# ==================================================
# 6. NATURAL DRIFT CHECK
# ==================================================

def natural_check(thresholds):
    """Real drift: score the natural 2012 period against the reference."""

    from data import prepare_data, chronological_batches
    from model import train_all_models
    from detectors import detect_psi, detect_ks

    data = prepare_data()

    reference = data["X_train"]

    models = train_all_models(data["X_train"], data["y_train"])

    psi_thr = thresholds["thresholds"]["psi"]["threshold"]
    ks_thr = thresholds["thresholds"]["ks"]["threshold"]

    rows = []

    for index, (X, y) in enumerate(
        chronological_batches(data["X_evaluation"], data["y_evaluation"])
    ):

        psi = detect_psi(reference, X)["score"]
        ks = detect_ks(reference, X)["score"]

        row = {
            "batch": index,
            "psi_score": psi, "ks_score": ks,
            "psi_alert": int(psi > psi_thr), "ks_alert": int(ks > ks_thr),
        }

        for name, model in models.items():
            row[f"acc_{name}"] = model.accuracy(X, y)

        rows.append(row)

    table = pd.DataFrame(rows)

    baseline = {
        name: model.accuracy(data["X_stable_test"], data["y_stable_test"])
        for name, model in models.items()
    }

    return table, baseline


# ==================================================
# 7. FINDINGS TEXT
# ==================================================

def pct(x):
    return f"{100 * x:.0f}%"


def build_findings(
    summary, tests, cases, chance, thresholds, natural, baseline, onset, n_streams
):

    lines = []

    add = lines.append

    add("# DriftWatch: findings\n")

    add("_Generated by `scripts/analyze_results.py` from `results/batch_results.csv`. "
        "All numbers come from simulated drift on the Bike Sharing data._\n")

    # ---- setup ----
    psi_t = thresholds["thresholds"]["psi"]["threshold"]
    ks_t = thresholds["thresholds"]["ks"]["threshold"]

    add("## Setup\n")
    add(f"- {n_streams} streams per condition, drift starting at batch {onset}; "
        f"batches of 500 rows.")
    target = thresholds["thresholds"]["psi"].get("target_far", 0.05)

    add(f"- Thresholds calibrated for a {pct(target)} false-alarm rate: "
        f"PSI {psi_t:.4f}, KS {ks_t:.4f}.")

    achieved = thresholds.get("achieved_false_alarm_rate")

    if achieved:
        add(f"- Achieved false-alarm rate on fresh stable batches: "
            f"PSI {pct(achieved['psi'])}, KS {pct(achieved['ks'])}.")

    add(f"- Drift counts as confirmed after {CONFIRM_K} alerting batches in a row. "
        f"On clean control streams that happens by chance in "
        f"{pct(chance['PSI'])} (PSI) and {pct(chance['KS'])} (KS) of streams.\n")

    # ---- PSI vs KS ----
    add("## PSI vs KS\n")

    clear = tests[tests["verdict"] != "no clear difference"]

    if clear.empty:
        add("- No condition shows a statistically clear difference between PSI and KS.")
    else:
        for _, r in clear.iterrows():
            better, worse = ("PSI", "KS") if r["verdict"] == "PSI better" else ("KS", "PSI")
            rate_b = r[f"{better.lower()}_detection_rate"]
            rate_w = r[f"{worse.lower()}_detection_rate"]
            add(f"- **{DRIFT_LABELS[r['drift_type']]}, {r['severity']}**: {better} confirmed "
                f"drift in {pct(rate_b)} of streams vs {pct(rate_w)} for {worse} "
                f"(paired exact test p = {r['p_value']:.3f}).")

    saturated = tests[
        (tests["psi_detection_rate"] >= 0.95) & (tests["ks_detection_rate"] >= 0.95)
    ]

    if not saturated.empty:
        names = ", ".join(
            f"{DRIFT_LABELS[r['drift_type']].lower()} ({r['severity']})"
            for _, r in saturated.iterrows()
        )
        add(f"- Both detectors confirm drift in at least 95% of streams for: {names}. "
            "These conditions do not separate the two methods.")

    faster = tests[
        ((tests["psi_mean_delay"] - tests["ks_mean_delay"]).abs() >= 1)
        & (tests["psi_detection_rate"] >= 0.5) & (tests["ks_detection_rate"] >= 0.5)
    ]

    for _, r in faster.iterrows():
        who = "PSI" if r["psi_mean_delay"] < r["ks_mean_delay"] else "KS"
        add(f"- {DRIFT_LABELS[r['drift_type']]}, {r['severity']}: {who} confirmed drift "
            f"earlier (mean delay PSI {r['psi_mean_delay']:.1f} vs KS {r['ks_mean_delay']:.1f} batches).")

    add("")

    # ---- alert vs accuracy ----
    add("## Alerts vs real damage\n")
    add(f"Damaging = {DAMAGE_MODEL.replace('_', ' ')} loses at least "
        f"{DAMAGE_DROP * 100:.0f} accuracy points; caught = drift confirmed in at least "
        f"{pct(CAUGHT_RATE)} of streams.\n")

    harmless = cases[cases["case"] == "harmless drift, alert fired"]
    missed = cases[cases["case"] == "damaging, MISSED"]
    useful = cases[cases["case"] == "damaging, caught"]

    def names(frame):
        return "; ".join(
            f"{DRIFT_LABELS[r.drift_type].lower()} ({r.severity}, {r.detector})"
            for r in frame.itertuples()
        )

    add(f"- **Damaging and caught:** {names(useful) or 'none'}.")
    add(f"- **Damaging but missed:** {names(missed) or 'none'}.")
    add(f"- **Harmless drift that still alerts:** {names(harmless) or 'none'}.\n")

    # ---- natural ----
    if natural is not None:

        add("## Real (natural) drift in 2012\n")

        add(f"- PSI alerted on {pct(natural['psi_alert'].mean())} and KS on "
            f"{pct(natural['ks_alert'].mean())} of the {len(natural)} natural-period batches.")

        for model in baseline:
            now = natural[f"acc_{model}"].mean()
            add(f"- {model.replace('_', ' ')}: accuracy {now:.3f} in the drifted period vs "
                f"{baseline[model]:.3f} on stable data.")

        held_up = all(
            natural[f"acc_{m}"].mean() >= baseline[m] - 0.01 for m in baseline
        )

        if held_up and natural[["psi_alert", "ks_alert"]].mean().max() >= 0.5:
            add("- So the inputs drifted strongly while accuracy did not fall: an alert is a "
                "reason to investigate, not proof the model is broken.\n")
        elif not held_up:
            add("- Accuracy fell in the drifted period for at least one model, so here "
                "the drift was not harmless.\n")
        else:
            add("")

    # ---- limitations ----
    add("## Limitations\n")
    add("- Only about 175 stable days exist, so thresholds depend on which days are used "
        "(a threshold calibrated on one set of days can give a very different "
        "false-alarm rate on another). Calibration therefore used the same stable pool as the "
        "experiments, and batches reuse rows; treat results as a controlled demonstration.")
    add("- Drift is simulated, one kind at a time, with fixed severity levels; real drift "
        "is messier. Labels never change (no concept drift).")
    add("- Both detectors use the maximum score over 11 features; other aggregations could "
        "change the ranking.")
    add(f"- {n_streams} streams per condition gives coarse rates. With 12 drifted conditions "
        "about one p < 0.05 is expected by chance.")
    add("- The accuracy comparison uses a 0.03 damage cutoff chosen by us.\n")

    return "\n".join(lines)


# ==================================================
# MAIN
# ==================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--example-type", default="hour_shift")
    parser.add_argument("--example-severity", default="low")
    parser.add_argument("--example-seed", type=int, default=0)
    parser.add_argument("--skip-natural", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    batch_df, thresholds, onset = load_results()

    streams = stream_table(batch_df, onset)
    summary = condition_summary(streams)
    chance = chance_level(summary)
    tests = paired_tests(streams)
    cases = alert_vs_accuracy(summary)

    n_streams = int(batch_df["seed"].nunique())

    summary.to_csv(OUT_DIR / "detection_summary.csv", index=False)
    tests.to_csv(OUT_DIR / "psi_vs_ks_tests.csv", index=False)
    cases.to_csv(OUT_DIR / "alert_vs_accuracy.csv", index=False)

    plot_lines(
        summary, "detection_rate", "Drift confirmed (share of streams)",
        f"Detection rate (drift confirmed after {CONFIRM_K} alerting batches in a row)",
        "fig_detection_rate.png", chance
    )

    plot_lines(
        summary[summary["drift_type"] != "none"], "mean_delay",
        "Batches until confirmed",
        "Detection delay (streams where drift was confirmed)",
        "fig_detection_delay.png"
    )

    plot_alert_vs_accuracy(cases)

    plot_stream(
        batch_df, thresholds, onset,
        args.example_type, args.example_severity, args.example_seed
    )

    natural, baseline = (None, None)

    if not args.skip_natural:
        natural, baseline = natural_check(thresholds)
        natural.to_csv(OUT_DIR / "natural_eval_check.csv", index=False)

    findings = build_findings(
        summary, tests, cases, chance, thresholds,
        natural, baseline, onset, n_streams
    )

    (OUT_DIR / "findings.md").write_text(findings, encoding="utf-8")

    pd.set_option("display.width", 200)

    print(f"\n========== PSI vs KS ({CONFIRM_K} alerting batches in a row) ==========\n")

    show = tests[[
        "drift_type", "severity", "psi_detection_rate", "ks_detection_rate",
        "psi_mean_delay", "ks_mean_delay", "p_value", "verdict"
    ]]

    print(show.round(3).to_string(index=False))

    print("\nChance level on clean control streams:",
          {k: round(v, 3) for k, v in chance.items()})

    print("\n========== ALERT vs ACCURACY ==========\n")

    print(cases.round(3).to_string(index=False))

    print(f"\nSaved tables, figures and findings.md to {OUT_DIR}")


if __name__ == "__main__":
    main()