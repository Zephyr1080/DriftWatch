# DriftWatch

A model-agnostic framework for monitoring data drift and comparing two statistical drift-detection methods: the **Population Stability Index (PSI)** and the **Kolmogorov–Smirnov (KS) two-sample test**.

A deployed model keeps predicting while its input data slowly changes, and accuracy can fall without anyone noticing. DriftWatch watches the incoming data, raises an alert when it stops looking like the training data, and measures which detector does that job better, at the same false-alarm rate.

## Key results

Setup: UCI Bike Sharing data, three classifiers, 13 simulated-drift conditions, 20 streams per condition, batches of 500 rows, drift starting at batch 15. Drift counts as confirmed after 2 alerting batches in a row. Both detectors are calibrated to about a 5% false-alarm rate. On clean control streams, the confirmation rule fires by chance in 0% (PSI) and 10% (KS) of streams.

| Drift | Severity | PSI detects | KS detects | Random forest accuracy drop |
|---|---|---|---|---|
| Covariate shift (weather) | low / medium / high | 100% / 100% / 100% | 100% / 100% / 100% | 1.1 / 3.0 / 6.3 pts |
| **Hour shift** | **low** | **100%** | **10%** | **4.2 pts** |
| Hour shift | medium / high | 100% / 100% | 100% / 100% | 16.8 / 31.5 pts |
| Noise | low | 75% | 85% | 0.4 pts |
| Noise | medium / high | 100% / 100% | 100% / 100% | 2.5 / 5.9 pts |
| **Missing values** | **low** | **75%** | **15%** | **0.3 pts** |
| Missing values | medium / high | 100% / 100% | 100% / 100% | 0.9 / 1.9 pts |

What this shows:

- **Strong drift:** PSI and KS agree. Both confirm it on the first possible batch.
- **Subtle drift in discrete features:** PSI is clearly better. It confirms a 1-hour shift of `hr` in every stream, while KS stays at chance level (p < 0.001), even though this drift costs the model about 4 accuracy points. PSI also beats KS on low missing values (p = 0.004), but that drift is harmless.
- **A likely explanation (not tested):** KS measures the largest gap between cumulative distributions, and a 1-hour shift of a 24-level feature moves too little mass across any boundary for KS to cross its threshold. PSI sees one category emptying and another doubling. Part of the effect comes from how the simulator clips `hr` at 23, so the gap may be smaller for other kinds of hour shift.
- **Alerts are not damage:** noise, missing values and weak weather shifts often alert while accuracy barely moves. An alert is a reason to investigate, not an automatic reason to retrain.
- **Real drift:** in the natural 2012 period, features shifted heavily and both detectors alerted on all 13 batches, yet accuracy did not fall for any of the three models (for example random forest 0.913 versus 0.899 on stable data), because demand growth made the "high demand" label easier to predict.

Full tables, figures and the generated write-up are in `results/analysis/`.

## How it works

1. **Data** (`src/data.py`): hourly bike-rental data with 11 input features. The task is binary: is demand above the reference median? Dates are split into four partitions: `reference` (training and drift reference), `calibration`, `stable_test` (clean data with no drift) and `natural_eval` (the last 40% of dates, where real drift occurs). The stable partitions are split by 3-day blocks inside each calendar month, so all three cover the same seasons.
2. **Models** (`src/model.py`): one interface for random forest, SVM and logistic regression. Scaling is fitted on training data only.
3. **Detectors** (`src/detectors.py`): `detect_psi` and `detect_ks` compare feature tables only and never see a model, so they work with any classifier. Each feature gets a score, and the dataset-level score is the maximum over features.
4. **Simulator** (`src/simulator.py`): creates drift at a known batch so detection delay can be measured. Types: covariate shift, hour shift, noise, missing values, each at low, medium and high severity, plus a no-drift control.
5. **Calibration** (`src/calibration.py`): PSI and KS live on different scales and never score zero on clean data, so each gets its own threshold at the 95th percentile of stable-batch scores (PSI 0.0854, KS 0.1085). This makes the comparison fair.
6. **Experiments** (`scripts/run_experiments.py`) and **analysis** (`scripts/analyze_results.py`): run every condition and compare detection rate, delay, false alarms and accuracy damage.

## Project structure

```
DriftWatch/
├── data/raw/hour.csv            UCI Bike Sharing hourly data
├── src/
│   ├── data.py                  loading, partitions, batching
│   ├── model.py                 RF / SVM / logistic regression wrapper
│   ├── detectors.py             detect_psi, detect_ks
│   ├── simulator.py             drift simulation
│   ├── calibration.py           threshold calibration
│   ├── pipeline.py, alerts.py, runner.py   monitoring pipeline (in progress)
├── app/                         Streamlit dashboard (in progress)
├── tests/                       tests (in progress)
├── scripts/
│   ├── inspect_data.py          data diagnostics
│   ├── run_experiments.py       run all drift conditions
│   └── analyze_results.py       tables, figures, findings
└── results/                     thresholds.json, experiment CSVs, analysis/
```

## Setup and usage

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt   # pandas, numpy, scikit-learn, scipy, matplotlib
```

Run in this order from the project root (each `src` file also prints a quick sanity check):

```bash
python src/data.py                      # partitions and diagnostics
python src/model.py                     # baseline accuracy
python src/detectors.py                 # detector sanity check
python src/simulator.py                 # simulated drift check
python src/calibration.py               # writes results/thresholds.json
python scripts/run_experiments.py       # full experiment (about 10 min on 8 cores)
python scripts/analyze_results.py       # writes results/analysis/
```

For a quick test run, use `python scripts/run_experiments.py --seeds 5`.

### Interfaces for the pipeline and dashboard

```python
detect_psi(reference_df, current_df)  -> {"score": float, "features": {name: float}}
detect_ks(reference_df, current_df)   -> {"score": float, "features": {name: float}, "p_values": {...}}
make_drifted_batch(df, drift_type, severity, seed) -> DataFrame
make_stream(X, y, drift_type, severity, n_batches, onset_batch, batch_size, seed)
build_model(name).fit(X, y); model.predict(X); model.accuracy(X, y)
load_thresholds() -> {"psi": float, "ks": float}      # alert when score > threshold
```

## Limitations

- Only about 175 stable days exist. Thresholds calibrated on one set of days can give a very different false-alarm rate on another (the script prints this check). Calibration therefore uses the same stable pool as the experiments, and batches reuse rows, so the results are a controlled demonstration, not a guarantee for brand-new data.
- Drift is simulated one kind at a time at fixed severities, and labels never change (no concept drift).
- The dataset-level score is the maximum over 11 features, and other aggregations could change the ranking.
- 20 streams per condition gives coarse rates. With 12 drifted conditions, about one p < 0.05 is expected by chance, so the paired tests are a guide, not proof.
- The "damaging" cutoff (3 accuracy points) is our own choice.

## Current status

- [x] Dataset downloaded and inspected
- [x] Feature selection and target definition
- [x] Reference, calibration, stable-test and natural-evaluation partitions
- [x] Baseline models (random forest, SVM, logistic regression)
- [x] Drift simulation (covariate shift, hour shift, noise, missing values)
- [x] PSI and KS detectors
- [x] Threshold calibration
- [x] Comparative experiments and analysis
- [ ] Monitoring pipeline, alerts and Streamlit dashboard
- [ ] Tests and final evaluation

## Data

[UCI Machine Learning Repository: Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), hourly data for 2011–2012 (`hour.csv`, 17,379 rows).