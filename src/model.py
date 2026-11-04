"""
DriftWatch - model interface (src/model.py)

One small wrapper gives every classifier the same interface:

    model = build_model("random_forest")
    model.fit(X_train, y_train)
    predictions = model.predict(X_batch)
    accuracy = model.accuracy(X_batch, y_batch)

The drift detectors (PSI, KS) never import this file. They only compare
feature DataFrames, so they work the same whatever model is being monitored.
Only the performance side (predictions, accuracy per batch) uses the model.

Preprocessing (scaling for SVM and Logistic Regression) lives inside a
scikit-learn Pipeline. The scaler is fitted on the training data only, then
reused unchanged on calibration, stable-test and monitoring data, so there
is no leakage.
"""

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from src.data import prepare_data      # python -m src.model (project root)
except ImportError:
    from data import prepare_data          # python src/model.py


# ==================================================
# CONFIGURATION
# ==================================================

RANDOM_STATE = 42

MODEL_NAMES = [
    "random_forest",
    "svm",
    "logistic_regression"
]


# ==================================================
# 1. MODEL WRAPPER
# ==================================================

class DriftWatchModel:
    """Common interface around any scikit-learn style classifier."""

    def __init__(self, name, estimator):
        self.name = name
        self.estimator = estimator
        self.is_fitted = False

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        return self.estimator.predict(X)

    def accuracy(self, X, y):
        """Accuracy on one labelled batch (or a whole partition)."""
        return accuracy_score(y, self.predict(X))

    def batch_accuracies(self, batches):
        """
        Accuracy for each batch.

        batches: list of (X_batch, y_batch) tuples, as produced by
        data.sample_batches() and data.chronological_batches().
        """
        return [self.accuracy(X, y) for X, y in batches]

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                f"Model '{self.name}' has not been fitted yet. Call fit() first."
            )

    def __repr__(self):
        return f"DriftWatchModel(name='{self.name}', fitted={self.is_fitted})"


# ==================================================
# 2. MODEL FACTORY
# ==================================================

def build_model(name, random_state=RANDOM_STATE):
    """Create an unfitted model by name."""

    if name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        )

    elif name == "svm":
        estimator = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale")
        )

    elif name == "logistic_regression":
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=random_state)
        )

    else:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {MODEL_NAMES}"
        )

    return DriftWatchModel(name, estimator)


def train_all_models(X_train, y_train, names=MODEL_NAMES):
    """Build and fit every model on the training partition only."""

    models = {}

    for name in names:
        models[name] = build_model(name).fit(X_train, y_train)

    return models


# ==================================================
# 3. EVALUATION
# ==================================================

def evaluate_models(models, partitions):
    """
    Accuracy of each model on each partition.

    partitions: dict of {partition_name: (X, y)}
    Returns a DataFrame with one row per model.
    """

    rows = {}

    for name, model in models.items():
        rows[name] = {
            partition_name: model.accuracy(X, y)
            for partition_name, (X, y) in partitions.items()
        }

    return pd.DataFrame(rows).T


# ==================================================
# MAIN EXECUTION
# ==================================================

if __name__ == "__main__":

    data = prepare_data()

    models = train_all_models(
        data["X_train"],
        data["y_train"]
    )

    partitions = {
        "train": (data["X_train"], data["y_train"]),
        "stable_test": (data["X_stable_test"], data["y_stable_test"]),
        "natural_eval": (data["X_evaluation"], data["y_evaluation"])
    }

    results = evaluate_models(models, partitions)

    print("\n========== BASELINE ACCURACY ==========\n")
    print(results.round(4).to_string())

    print(
        "\nstable_test shows accuracy without drift. natural_eval is the "
        "naturally drifted period: compare it with stable_test to see whether "
        "drift changed accuracy (it can go up, down or stay flat)."
    )

    print("\n========== MODEL TRAINING COMPLETE ==========")
