"""TODO PEP 257"""
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from learning_machines_drift import FileBackend, Monitor, Registry, datasets
from learning_machines_drift.backends import Backend


def generate_features_labels_latents(
    numrows: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:

    """This generates data and returns features, labels and latents"""

    features, labels, latents = datasets.logistic_model(
        x_mu=np.array([0.0, 0.0, 0.0]), size=numrows, return_latents=True
    )

    features_df: pd.DataFrame = pd.DataFrame(
        {"age": features[:, 0], "height": features[:, 1], "ground-truth-label": labels}
    )

    predictions_series: pd.Series = pd.Series(labels)
    latents_df: pd.DataFrame = pd.DataFrame({"latents": latents})
    return (features_df, predictions_series, latents_df)


def register_reference(
    tag: str = "simple_example", backend: Backend = FileBackend("my-data")
) -> Registry:
    """Generate data, register data to detector and return detector"""
    features_df, predictions_series, latents_df = generate_features_labels_latents(10)
    detector = Registry(tag=tag, backend=backend)
    detector.register_ref_dataset(
        features=features_df, labels=predictions_series, latents=latents_df
    )
    return detector


def register_new(detector: Registry) -> None:
    """Generate data and log using Registry."""

    def log_new_data(
        detector: Registry,
        features_df: pd.DataFrame,
        predictions_series: pd.Series,
        latents_df: pd.DataFrame,
    ) -> None:
        """Log features, labels and latents using Registry."""
        with detector:
            detector.log_features(features_df)
            detector.log_labels(predictions_series)
            detector.log_latents(latents_df)

    num_iterations = 1
    for _ in range(num_iterations):
        (
            new_features_df,
            new_predictions_series,
            new_latents_df,
        ) = generate_features_labels_latents(5)
        log_new_data(detector, new_features_df, new_predictions_series, new_latents_df)


def load_data(
    tag: str = "simple_example", backend: Backend = FileBackend("my-data")
) -> Monitor:
    """Load data and return Monitor"""
    measure = Monitor(tag=tag, backend=backend)
    measure.load_data(drift_filter=None)
    return measure


def main() -> None:
    """Generating data, diff data, visualise results"""
    # 1. Generate and store reference data
    # detector = register_reference()

    # 2. Generate and store log data
    # register_new(detector)

    # # 3. Load log data
    measure = load_data()

    # # 4. Run Test
    print(f"Boundary Adherence: {measure.metrics.get_boundary_adherence()}")
    print(f"Range Coverage: {measure.metrics.get_range_coverage()}")

    # measure.hypothesis_tests.scipy_kolmogorov_smirnov()


main()
