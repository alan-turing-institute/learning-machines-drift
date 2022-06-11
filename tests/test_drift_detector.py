"""TODO PEP 257"""
# pylint: disable=W0621

import pathlib
from typing import Callable, Tuple

# import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from learning_machines_drift import (  # DriftDetector,; DriftMeasure,
    Monitor,
    ReferenceDatasetMissing,
    Registry,
    datasets,
)
from learning_machines_drift.backends import FileBackend

# , mocker
# from this import d


N_FEATURES = 3
N_LABELS = 2


@pytest.fixture()
def detector(mocker: MockerFixture) -> Registry:
    """Return a DriftDetector which writes data to a temporary directory"""

    det = Registry(tag="test")
    mocker.patch.object(det, "backend")
    return det


@pytest.fixture()
def measure(mocker: MockerFixture) -> Monitor:
    """Return a DriftDetector which writes data to a temporary directory"""

    meas = Monitor(tag="test")
    mocker.patch.object(meas, "backend")
    return meas


def example_dataset(n_rows: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """TODO PEP 257"""
    # Given we have a reference dataset
    x_reference, y_reference = datasets.logistic_model(size=n_rows)
    features_df = pd.DataFrame(
        {
            "age": x_reference[:, 0],
            "height": x_reference[:, 1],
            "bp": x_reference[:, 2],
        }
    )

    labels_df = pd.Series(y_reference, name="y")

    return (features_df, labels_df)


@pytest.fixture()
def detector_with_ref_data(  # type: ignore
    tmp_path: pathlib.Path, detector
) -> Callable[[int], Registry]:
    """Return a DriftDetector with a reference dataset registered
    which writes data to a temporary directory"""
    print(tmp_path)

    def _detector_with_ref_data(n_rows: int) -> Registry:
        """TODO PEP 257"""

        features_df, labels_df = example_dataset(n_rows)

        # When we register the dataset
        detector.register_ref_dataset(features=features_df, labels=labels_df)

        return detector  # type: ignore

    return _detector_with_ref_data


@pytest.mark.parametrize("n_rows", [5, 10, 100, 1000, 10000])
def test_register_dataset(
    detector_with_ref_data: Callable[[int], Registry], n_rows: int
) -> None:
    """TODO PEP 257"""

    # Given we have a reference dataset
    det: Registry = detector_with_ref_data(n_rows)

    # When we get a summary of the reference set
    summary = det.ref_summary()

    # Then we can access summary information
    assert summary.shapes.features.n_rows == n_rows
    assert summary.shapes.features.n_features == N_FEATURES
    assert summary.shapes.labels.n_rows == n_rows
    assert summary.shapes.labels.n_labels == N_LABELS

    # And we saved the data to the backend
    det.backend.save_reference_dataset.assert_called_once()  # type: ignore


def test_ref_summary_no_dataset(detector) -> None:  # type: ignore
    """TODO PEP 257"""

    # Given a detector with no reference dataset registered

    # When we get the reference dataset summary
    # Then raise an exception
    with pytest.raises(ReferenceDatasetMissing):
        _ = detector.ref_summary()

    # And we should not have saved any reference data
    detector.backend.save_reference_dataset.assert_not_called()


def test_all_registered(
    detector_with_ref_data: Callable[[int], Registry], tmp_path: pathlib.Path
) -> None:
    """TODO PEP 257"""
    print(tmp_path)
    # Given we have registered a reference dataset
    det = detector_with_ref_data(100)

    # And we have features and predicted labels
    x_pred, y_pred = datasets.logistic_model()
    # latent_x = x_pred.mean(axis=0)

    # When we log features and labels of new data
    with det:
        # I set these false here as the backend doesn't support them yet
        # Should discuss how we want to handle this
        det.expect_labels = False

        det.log_features(
            pd.DataFrame(
                {
                    "age": x_pred[:, 0],
                    "height": x_pred[:, 1],
                    "bp": x_pred[:, 2],
                }
            )
        )
        det.log_labels(pd.Series(y_pred, name="y"))
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
    assert det.all_registered()

    # And we saved a reference dataset
    det.backend.save_reference_dataset.assert_called_once()  # type: ignore

    # And we saved the logged features
    det.backend.save_logged_features.assert_called_once()  # type: ignore

    # And we saved the logged labels
    det.backend.save_logged_labels.assert_called_once()  # type: ignore


def test_statistics_summary(tmp_path) -> None:  # type: ignore
    """TODO PEP 257"""

    # Given we have registered a reference dataset
    features_df, labels_df = example_dataset(100)
    det = Registry(tag="test", backend=FileBackend(tmp_path))
    det.register_ref_dataset(features=features_df, labels=labels_df)

    # And we have features and predicted labels
    x_monitor, y_monitor = datasets.logistic_model()
    # latent_x = x_monitor.mean(axis=0)
    features_monitor_df = pd.DataFrame(
        {
            "age": x_monitor[:, 0],
            "height": x_monitor[:, 1],
            "bp": x_monitor[:, 2],
        }
    )
    labels_monitor_df = pd.DataFrame({"y": y_monitor})

    with det:
        # And we have logged features, labels and latent
        det.log_features(features_monitor_df)
        det.log_labels(labels_monitor_df)
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

    meas = Monitor(tag="test", backend=FileBackend(tmp_path))
    meas.load_data()
    res = meas.hypothesis_tests.scipy_kolmogorov_smirnov(verbose=True)

    # Check we get a dictionary with an entry for every feature column
    assert isinstance(res, dict)
    # print(set(features_df.columns))
    # print(res)
    assert res.keys() == set(features_df.columns)


def test_load_all_logged_data(  # type: ignore
    detector_with_ref_data: Callable[[int], Registry], tmp_path
) -> None:
    """TODO PEP 257"""
    # pylint: disable=unused-argument
    # Given we have registered a reference dataset
    det = Registry(tag="test", backend=FileBackend(tmp_path))
    features_df, labels_df = example_dataset(20)
    det.register_ref_dataset(features=features_df, labels=labels_df)

    # And we have features and predicted labels
    x_pred, y_pred = datasets.logistic_model(size=2)

    # When we log features and labels of new data
    with det:
        det.log_features(
            pd.DataFrame(
                {
                    "age": x_pred[:, 0],
                    "height": x_pred[:, 1],
                    "bp": x_pred[:, 2],
                }
            )
        )
        det.log_labels(pd.Series(y_pred, name="y"))

    # And we have features and predicted labels
    x_pred, y_pred = datasets.logistic_model(size=2)

    with det:
        det.log_features(
            pd.DataFrame(
                {
                    "age": x_pred[:, 0],
                    "height": x_pred[:, 1],
                    "bp": x_pred[:, 2],
                }
            )
        )
        det.log_labels(pd.Series(y_pred, name="y"))

    # Load data
    meas = Monitor(tag="test", backend=FileBackend(tmp_path))
    recovered_dataset = meas.load_data()

    dimensions = recovered_dataset.unify().shape
    assert dimensions[0] == 4
    assert dimensions[1] == 4

    print(recovered_dataset.unify())


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
