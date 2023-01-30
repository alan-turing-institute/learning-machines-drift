"""Simple synthetic example with predictors Age and Height for binary labels."""
from typing import Any, Dict, List, Optional

import pandas as pd

from learning_machines_drift import FileBackend, Monitor, Registry
from learning_machines_drift.datasets import example_dataset
from learning_machines_drift.display import Display
from learning_machines_drift.drift_filter import Filter
from learning_machines_drift.types import StructuredResult


def get_detector_reference(
    features_df: pd.DataFrame, predictions_series: pd.Series, latents_df: pd.DataFrame
) -> Registry:
    """Register reference data, returns a Registry"""
    detector = Registry(
        tag="simple_example",
        backend=FileBackend("my-data"),
        clear_logged=True,
        clear_reference=True,
    )
    detector.register_ref_dataset(
        features=features_df, labels=predictions_series, latents=latents_df
    )
    return detector


def log_new_data(
    detector: Registry,
    features_df: pd.DataFrame,
    predictions_series: pd.Series,
    latents_df: pd.DataFrame,
) -> None:
    """Log features, labels and latents using Registry"""
    with detector:
        detector.log_features(features_df)
        detector.log_labels(predictions_series)
        detector.log_latents(latents_df)


def load_data(drift_filter: Optional[Filter] = None) -> Monitor:
    """Load data and return Monitor"""
    measure = Monitor(tag="simple_example", backend=FileBackend("my-data"))
    measure.load_data(drift_filter)
    return measure


def register_reference() -> Registry:
    """Generate data, register data to detector and return detector"""
    features_df, predictions_series, latents_df = example_dataset(10)
    detector: Registry = get_detector_reference(
        features_df, predictions_series, latents_df
    )
    return detector


def store_logs(detector: Registry) -> None:
    """Generate data and log using Registry"""
    num_iterations = 1
    for _ in range(num_iterations):
        (
            new_features_df,
            new_predictions_series,
            new_latents_df,
        ) = example_dataset(5)
        log_new_data(detector, new_features_df, new_predictions_series, new_latents_df)


def display_diff_results(results: List[StructuredResult]) -> None:
    """Display list of results"""
    for res in results:
        # print(res)
        Display().table(res)
        # Display().plot(res, score_type="statistic")
        # plt.show()
        # Display().plot(res, score_type="pvalue")
        # plt.show()


def main() -> None:
    """Generating data, diff data, visualise results"""

    # 1. Generate and store reference data
    registry: Registry = register_reference()

    # 2. Generate and store log data
    store_logs(registry)

    measure: Monitor = load_data(None)

    test_dispatcher: Dict[str, Any] = {
        "scipy_kolmogorov_smirnov": measure.hypothesis_tests.scipy_kolmogorov_smirnov,
        "scipy_mannwhitneyu": measure.hypothesis_tests.scipy_mannwhitneyu,
        "boundary_adherence": measure.hypothesis_tests.get_boundary_adherence,
        "range_coverage": measure.hypothesis_tests.get_range_coverage,
        "logistic_detection": measure.hypothesis_tests.logistic_detection,
        "get_boundary_adherence": measure.hypothesis_tests.get_boundary_adherence,
        "get_range_coverage": measure.hypothesis_tests.get_range_coverage,
        # "binary_classifier_efficacy":measure.hypothesis_tests.binary_classifier_efficacy
    }

    results: List[StructuredResult] = []

    for h_test_name, h_test_fn in test_dispatcher.items():
        results.append(h_test_fn())

    display_diff_results(results=results)


if __name__ == "__main__":
    main()
    # mock_test()
