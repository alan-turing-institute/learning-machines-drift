"""Simple synthetic example with predictors Age and Height for binary labels."""
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from learning_machines_drift import FileBackend, Monitor, Registry, datasets
from learning_machines_drift.display import Display
from learning_machines_drift.drift_filter import Condition, Filter
from learning_machines_drift.types import StructuredResult
from learning_machines_drift.datasets import example_dataset

def generate_features_labels_latents(
    numrows: int,
) -> Tuple[pd.DataFrame, pd.Series,pd.DataFrame]:

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


def get_detector_reference(
    features_df: pd.DataFrame, predictions_series: pd.Series, latents_df: pd.DataFrame
) -> Registry:
    """Register reference data, returns a Registry"""
    detector = Registry(tag="simple_example", backend=FileBackend("my-data"))
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
    features_df, predictions_series, latents_df = generate_features_labels_latents(10)
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
        ) = generate_features_labels_latents(5)
        log_new_data(detector, new_features_df, new_predictions_series, new_latents_df)


# def perform_diff_tests(drift_filter: Optional[Filter] = None) -> List[Any]:
#     """Load data, perform hypothesis tests"""
#     measure = load_data(drift_filter)
#     # ks_results = measure.hypothesis_tests.scipy_kolmogorov_smirnov()
#     boundary_results = measure.hypothesis_tests.sdv_boundary_adherence()
#     return [boundary_results]


def display_diff_results(results: List[Any]) -> None:
    """Display list of results"""
    for res in results:
        print(res)
        Display().table(res)
        # Display().plot(res, score_type="statistic")
        # plt.show()
        # Display().plot(res, score_type="pvalue")
        # plt.show()

def mock_test() -> None:
    features_df, labels_df, latents_df = example_dataset(10)
    det = Registry(tag="test", backend=FileBackend('test-data'))
    det.register_ref_dataset(features=features_df, labels=labels_df, latents=latents_df)

    # And we have features and predicted labels
    num_iterations = 1
    for _ in range(num_iterations):
        (
            new_features_df,
            new_predictions_series,
            new_latents_df,
        ) = generate_features_labels_latents(5)
        log_new_data(det, new_features_df, new_predictions_series, new_latents_df)
    

    meas = Monitor(tag="test", backend=FileBackend("test-data"))
    meas.load_data()

    h_test_dispatcher: Dict[str, Any] = {
        "scipy_kolmogorov_smirnov": meas.hypothesis_tests.scipy_kolmogorov_smirnov,
        # "scipy_mannwhitneyu": meas.hypothesis_tests.scipy_mannwhitneyu,
        # "boundary_adherence": meas.hypothesis_tests.get_boundary_adherence
    }

    for h_test_name, h_test_fn in h_test_dispatcher.items():
        res = h_test_fn()

        # Check res is a StructureResult
        assert isinstance(res, StructuredResult)
        assert (res.method_name==h_test_name)
        if list(res.results.keys())[0] == 'single_value':
            assert isinstance(res.results['single_value'], dict)
        else:
            print(list(res.results.keys()))
            print(list(det.registered_dataset.unify().columns))

            #assert list(results.results.keys())==list(registry.registered_dataset.unify().columns)
            assert list(res.results.keys())==list(det.registered_dataset.unify().columns)

def main() -> None:
    """Generating data, diff data, visualise results"""
    # 1. Generate and store reference data
    registry = register_reference()

    # 2. Generate and store log data
    store_logs(registry)
    measure = load_data(None)

    results = measure.hypothesis_tests.get_boundary_adherence()
    print(results)

    # print(list(results.results.keys()))
    
    # print(list(registry.registered_dataset.unify().columns))

    # assert list(results.results.keys())==list(registry.registered_dataset.unify().columns)
    # results = measure.hypothesis_tests.get_range_coverage()
    # print(results)

    # results = measure.hypothesis_tests.binary_classifier_efficacy(target_variable="ground-truth-label")
    # print(results)

    # results = measure.hypothesis_tests.logistic_detection_custom()
    # print(results)

    # {'methodname':
    #     {'statistic': 
    #         {'age': {'statistic': 0.8666666666666667},
    #         'height': {'statistic': 0.8666666666666667}
    #         }
    #     }
    # }

    # {'methodname':
    #     {'age': {'statistic': 0.8666666666666667},
    #     'height': {'statistic': 0.8666666666666667}
    #     }
    # }

    # {'methodname':
    #     {'single_value':
    #         {'statistic': 0.999, 'pvalue':0.005}
    #     }
    # }
    # results = measure.hypothesis_tests.logistic_detection()
    # print(results)

    # results = measure.hypothesis_tests.scipy_permutation()
    # print(results)

    # results = measure.hypothesis_tests.scipy_kolmogorov_smirnov()
    # print(results)

    # results = measure.hypothesis_tests.scipy_mannwhitneyu()
    # print(results)

    # 3. Load all data with filter and perform tests
    # drift_filter = Filter(
    #     {
    #         "age": [Condition("less", 0.0)],
    #         "height": [Condition("greater", -1.0), Condition("less", 1.0)],
    #     }
    # )
    # results = perform_diff_tests(None)

    # 4. Display results
    # display_diff_results(results)


# measure = Monitor(tag="simple_example", backend=FileBackend("my-data"))
# measure.load_data()
# # print(measure.hypothesis_tests.scipy_kolmogorov_smirnov())
# # print(measure.hypothesis_tests.scipy_permutation())
# # print(measure.hypothesis_tests.scipy_mannwhitneyu())
# # print(measure.hypothesis_tests.scipy_chisquare())
# # print(measure.hypothesis_tests.gaussian_mixture_log_likelihood())
# # print(measure.hypothesis_tests.gaussian_mixture_log_likelihood(normalize=True))
# # print(measure.hypothesis_tests.logistic_detection())
# # print(measure.hypothesis_tests.logistic_detection_custom())
# # print(measure.hypothesis_tests.logistic_detection_custom(score_type="f1"))
# # print(measure.hypothesis_tests.logistic_detection_custom(score_type="roc_auc"))


# ks_results:Dict[str,any] = measure.hypothesis_tests.scipy_kolmogorov_smirnov()
# perm_results = measure.hypothesis_tests.scipy_permutation()
# log_results = measure.hypothesis_tests.logistic_detection_custom(score_type="roc_auc")

# for res in [ks_results, perm_results, log_results]:
#     df = Display().table(res)
#     fig, axs = Display().plot(res, score_type="statistic")


# print(measure.hypothesis_tests.sd_evaluate())


# logged_datasets = detector.backend.load_logged_dataset("simple_example")
# print(logged_datasets.labels)


# print(detector.registered_features)
# print(detector.registered_labels)
# print(detector.ref_dataset)
# print(measure.hypothesis_tests.kolmogorov_smirnov())
# print(measure.hypothesis_tests.sdv_evaluate())


# logged_datasets = detector.backend.load_logged_dataset("simple_example")

# print(logged_datasets.features)
# print(logged_datasets.labels)


if __name__ == "__main__":
    mock_test()
