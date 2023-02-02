"""Simple synthetic example with predictors Age and Height for binary labels."""
from typing import Any, Dict, List, Optional

import pandas as pd

from learning_machines_drift import (
    Display,
    FileBackend,
    Filter,
    Monitor,
    Registry,
    StructuredResult,
)
from learning_machines_drift.datasets import example_dataset


def get_detector_reference(
    features_df: pd.DataFrame, predictions_series: pd.Series, latents_df: pd.DataFrame
) -> Registry:
    """Register reference data, returns a Registry"""
    detector = Registry(
        tag="simple_example",
        backend=FileBackend("example-backend"),
        clear_logged=True,
        clear_reference=True,
    )
    detector.register_ref_dataset(
        features=features_df, labels=predictions_series, latents=latents_df
    )
    return detector


def log_new_data(
    registry: Registry,
    features_df: pd.DataFrame,
    predictions_series: pd.Series,
    latents_df: pd.DataFrame,
) -> None:
    """Log features, labels and latents using Registry"""
    with registry:
        registry.log_features(features_df)
        registry.log_labels(predictions_series)
        registry.log_latents(latents_df)


def load_data(drift_filter: Optional[Filter] = None) -> Monitor:
    """Load data and return Monitor"""
    monitor = Monitor(tag="simple_example", backend=FileBackend("example-backend"))
    monitor.load_data(drift_filter)
    return monitor


def register_reference() -> Registry:
    """Generate data, register data to detector and return detector"""
    features_df, predictions_series, latents_df = example_dataset(10)
    registry: Registry = get_detector_reference(
        features_df, predictions_series, latents_df
    )
    return registry


def store_logs(registry: Registry) -> None:
    """Generate data and log using Registry"""
    num_iterations = 1
    for _ in range(num_iterations):
        (
            new_features_df,
            new_predictions_series,
            new_latents_df,
        ) = example_dataset(5)
        log_new_data(registry, new_features_df, new_predictions_series, new_latents_df)


def display_diff_results(results: List[StructuredResult]) -> None:
    """Display list of results"""
    for res in results:
        print(res)
        Display().table(res)


def main() -> None:
    """Generating data, diff data, visualise results"""

    # 1. Generate and store reference data
    registry: Registry = register_reference()

    # 2. Generate and store log data
    store_logs(registry)

    # 3. Construct monitor
    monitor: Monitor = load_data(None)

    # 4. Get results from metrics
    test_dispatcher: Dict[str, Any] = {
        "scipy_kolmogorov_smirnov": monitor.metrics.scipy_kolmogorov_smirnov,
        "scipy_mannwhitneyu": monitor.metrics.scipy_mannwhitneyu,
        "boundary_adherence": monitor.metrics.get_boundary_adherence,
        "range_coverage": monitor.metrics.get_range_coverage,
        "logistic_detection": monitor.metrics.logistic_detection,
        "get_boundary_adherence": monitor.metrics.get_boundary_adherence,
        "get_range_coverage": monitor.metrics.get_range_coverage,
    }
    results: List[StructuredResult] = []
    for h_test_name, h_test_fn in test_dispatcher.items():
        print(f"Test '{h_test_name}' completed.")
        results.append(h_test_fn())

    # 5. Display table of results
    display_diff_results(results=results)


if __name__ == "__main__":
    main()
