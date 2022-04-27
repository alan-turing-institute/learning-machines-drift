import pathlib
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import pytest
from pytest_mock import mocker, MockerFixture

from this import d

from learning_machines_drift import DriftDetector, ReferenceDatasetMissing, datasets

N_FEATURES = 3
N_LABELS = 2


@pytest.fixture()
def detector(mocker: MockerFixture) -> DriftDetector:
    """Return a DriftDetector which writes data to a temporary directory"""

    detector = DriftDetector(tag="test")
    mocker.patch.object(detector, "backend")
    return detector


def example_dataset(n_rows: int) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Given we have a reference dataset
    X_reference, Y_reference = datasets.logistic_model(size=n_rows)

    features_df = pd.DataFrame(
        {
            "age": X_reference[:, 0],
            "height": X_reference[:, 1],
            "bp": X_reference[:, 2],
        }
    )

    labels_df = pd.Series(Y_reference, name="y")

    return (features_df, labels_df)


@pytest.fixture()
def detector_with_ref_data(
    tmp_path: pathlib.Path, detector: DriftDetector
) -> Callable[[int], DriftDetector]:
    """Return a DriftDetector with a reference dataset registered which writes data to a temporary directory"""

    def _detector_with_ref_data(n_rows: int) -> DriftDetector:

        features_df, labels_df = example_dataset(n_rows)

        # When we register the dataset
        detector.register_ref_dataset(features=features_df, labels=labels_df)

        return detector

    return _detector_with_ref_data


@pytest.mark.parametrize("n_rows", [5, 10, 100, 1000, 10000])
def test_register_dataset(
    detector_with_ref_data: Callable[[int], DriftDetector], n_rows: int
) -> None:

    # Given we have a reference dataset
    detector = detector_with_ref_data(n_rows)

    # When we get a summary of the reference set
    summary = detector.ref_summary()

    # Then we can access summary information
    assert summary.shapes.features.n_rows == n_rows
    assert summary.shapes.features.n_features == N_FEATURES
    assert summary.shapes.labels.n_rows == n_rows
    assert summary.shapes.labels.n_labels == N_LABELS

    # And we saved the data to the backend
    detector.backend.save_reference_dataset.assert_called_once()


def test_ref_summary_no_dataset(detector: DriftDetector) -> None:

    # Given a detector with no reference dataset registered

    # When we get the reference dataset summary
    # Then raise an exception
    with pytest.raises(ReferenceDatasetMissing):
        _ = detector.ref_summary()

    # And we should not have saved any reference data
    detector.backend.save_reference_dataset.assert_not_called()


def test_all_registered(
    detector_with_ref_data: Callable[[int], DriftDetector], tmp_path: pathlib.Path
) -> None:

    # Given we have registered a reference dataset
    detector = detector_with_ref_data(100)

    # And we have features and predicted labels
    X, Y_pred = datasets.logistic_model()
    latent_x = X.mean(axis=0)

    # When we log features and labels of new data
    with detector:

        # ToDo: I set these false here as the backend doesn't support them yet
        # Should discuss how we want to handle this
        detector.expect_labels = False
        detector.expect_latent = False

        detector.log_features(
            pd.DataFrame(
                {
                    "age": X[:, 0],
                    "height": X[:, 1],
                    "bp": X[:, 2],
                }
            )
        )
        # detector.log_labels(pd.Series(Y_pred, name="y"))
        # detector.log_latent(
        #     pd.DataFrame(
        #         {
        #             "mean_age": latent_x[0],
        #             "mean_height": latent_x[1],
        #             "mean_bp": latent_x[2],
        #         },
        #         index=[0],
        #     )
        # )

    # Then we can ensure that everything is registered
    assert detector.all_registered()

    # And we saved a reference dataset
    detector.backend.save_reference_dataset.assert_called_once()

    # And we saved the logged features
    detector.backend.save_logged_features.assert_called_once()


def test_statistics_summary(detector: DriftDetector) -> None:

    # Given we have registered a reference dataset
    features_df, labels_df = example_dataset(100)

    # And we have logged features, labels and latent
    with detector:

        detector.register_ref_dataset(features=features_df, labels=labels_df)

        # And we have features and predicted labels
        X, Y_pred = datasets.logistic_model()
        latent_x = X.mean(axis=0)

        detector.log_features(
            pd.DataFrame(
                {
                    "age": X[:, 0],
                    "height": X[:, 1],
                    "bp": X[:, 2],
                }
            )
        )
        detector.log_labels(pd.Series(Y_pred, name="y"))
        detector.log_latent(
            pd.DataFrame(
                {
                    "mean_age": latent_x[0],
                    "mean_height": latent_x[1],
                    "mean_bp": latent_x[2],
                },
                index=[0],
            )
        )

        res = detector.hypothesis_tests.kolmogorov_smirnov()

        # Check we get a dictionary with an entry for every feature column
        assert isinstance(res, dict)
        assert res.keys() == set(features_df.columns)


# def test_monitor_drift(detector_with_ref_data: DriftDetector) -> None:

#     # Given we have registered a reference dataset
#     detector = detector_with_ref_data

#     # When we log features and labels of new data
#     with DriftDetector(
#         tag="test", expect_features=True, expect_labels=True, expect_latent=False
#     ) as detector:

#         X, Y = datasets.logistic_model()

#         detector.log_features(X)
#         detector.log_labels(Y)
#         detector.log_latent(latent_vars)

#     # Then we can get a summary of drift
#     detector.drift_summary()
