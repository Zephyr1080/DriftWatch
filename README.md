# DriftWatch

## Overview

DriftWatch is a model-agnostic framework for monitoring data drift and comparing two statistical drift-detection methods: Population Stability Index (PSI) and the Kolmogorov–Smirnov (KS) two-sample test.

The project uses the UCI Bike Sharing hourly dataset as its initial use case. It is designed to compare how effectively PSI and KS identify changes in incoming feature distributions while monitoring different machine-learning models.

Feature drift and model performance degradation are evaluated separately: a change in input data does not necessarily mean that prediction performance has worsened.

## Objectives

- Implement PSI- and KS-based data drift detection.
- Design the detection layer independently of the machine-learning model.
- Support multiple classification models through a common interface.
- Calibrate detector thresholds using a separate calibration dataset.
- Compare detection rate, false-alarm rate, and detection delay.
- Examine the relationship between detected feature drift and model performance.
- Present monitoring results through a dashboard.

## Tech Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Statistical testing | SciPy |
| Dashboard | Streamlit |
| Development environment | VS Code |
| Version control | Git and GitHub |

## Current Status

**In progress: comparative experiments and analysis**

- [x] Dataset downloaded and inspected.
- [x] Feature selection and target definition established.
- [x] Reference, calibration, stable-test, and natural-evaluation splits implemented.
- [x] Initial data diagnostics implemented.
- [x] Baseline model training (Random Forest, SVM, Logistic Regression).
- [x] Drift simulation (covariate shift, hour shift, noise, missing values).
- [x] PSI and KS detector implementation.
- [x] Threshold calibration.
- [ ] Comparative experiments (script written; full run and analysis in progress).
- [ ] Monitoring pipeline, alerts, and Streamlit dashboard.
- [ ] Testing and final evaluation.

The current implementation focuses on establishing a reliable data-preparation workflow. The remaining components will be developed and integrated in subsequent stages.
