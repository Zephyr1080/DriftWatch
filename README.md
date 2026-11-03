# DriftWatch Lite

### Comparing PSI and KS Drift Detectors at Matched False-Alarm Rates

## Overview

DriftWatch Lite is a machine learning monitoring system
that compares PSI (Population Stability Index) and
the Kolmogorov-Smirnov test for detecting data drift.

The project evaluates both detectors at matched
false-alarm rates and investigates whether detected
distribution changes correspond to declining model accuracy.

## Objectives

- Train a baseline Random Forest model.
- Simulate controlled data drift.
- Implement PSI and KS detectors.
- Calibrate thresholds at a target 5% false-alarm rate.
- Compare detection delay and detection rates.
- Visualize results using Streamlit.

## Tech Stack

- Python
- NumPy and Pandas
- Scikit-learn
- SciPy
- Streamlit
- Plotly
- Pytest

## Status

Under development.