"""TODO PEP 257"""
# pylint: disable=W0621

import pathlib
from functools import partial
from typing import Any, Callable, Dict, Tuple

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
N_LATENTS = 1


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


def example_dataset(n_rows: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """TODO PEP 257"""
    # Given we have a reference dataset
    x_reference, y_reference, latents_reference = datasets.logistic_model(
        size=n_rows, return_latents=True
    )
    # x_reference, _ = datasets.logistic_model(size=n_rows)
    features_df = pd.DataFrame(
        {
            "age": x_reference[:, 0],
            "height": x_reference[:, 1],
            "bp": x_reference[:, 2],
        }
    )

    labels_df = pd.Series(y_reference, name="y")
    latents_df = pd.DataFrame({"latents": latents_reference})

    return (features_df, labels_df, latents_df)


@pytest.fixture()
def detector_with_ref_data(
    tmp_path: pathlib.Path, detector: Registry
) -> Callable[[int], Registry]:
    """Return a DriftDetector with a reference dataset registered
    which writes data to a temporary directory"""
    print(tmp_path)

    def _detector_with_ref_data(n_rows: int) -> Registry:
        """TODO PEP 257"""

        features_df, labels_df, latents_df = example_dataset(n_rows)

        # When we register the dataset
        detector.register_ref_dataset(
            features=features_df, labels=labels_df, latents=latents_df
        )

        return detector

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
    if summary.shapes.latents is not None:
        assert summary.shapes.latents.n_rows == n_rows
        assert summary.shapes.latents.n_latents == N_LATENTS

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
    # Given we have registered a reference dataset
    det = detector_with_ref_data(100)

    # And we have features and predicted labels
    x_pred, y_pred, latents_pred = datasets.logistic_model(return_latents=True)
    # latent_x = x_pred.mean(axis=0)

    # When we log features and labels of new data
    with det:
        # I set these false here as the backend doesn't support them yet
        # Should discuss how we want to handle this
        det.expect_labels = True
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
        det.log_latents(
            pd.DataFrame(
                {
                    "latents": latents_pred,
                }
            )
        )

    # Then we can ensure that everything is registered
    assert det.all_registered()

    # And we saved a reference dataset
    det.backend.save_reference_dataset.assert_called_once()  # type: ignore

    # And we saved the logged features
    det.backend.save_logged_features.assert_called_once()  # type: ignore

    # And we saved the logged labels
    det.backend.save_logged_labels.assert_called_once()  # type: ignore


def test_summary_statistic_list(tmp_path: pathlib.Path) -> None:
    """TODO PEP 257"""
    features_df, labels_df, latents_df = example_dataset(10)
    det = Registry(tag="test", backend=FileBackend(tmp_path))
    det.register_ref_dataset(features=features_df, labels=labels_df, latents=latents_df)

    # And we have features and predicted labels
    x_monitor, y_monitor, latents_monitor = datasets.logistic_model(return_latents=True)
    features_monitor_df = pd.DataFrame(
        {
            "age": x_monitor[:, 0],
            "height": x_monitor[:, 1],
            "bp": x_monitor[:, 2],
        }
    )
    labels_monitor_df = pd.DataFrame({"y": y_monitor})
    latents_monitor_df = pd.DataFrame({"latents": latents_monitor})

    with det:
        # And we have logged features, labels and latent
        det.log_features(features_monitor_df)
        det.log_labels(labels_monitor_df)
        det.log_latents(latents_monitor_df)

    meas = Monitor(tag="test", backend=FileBackend(tmp_path))
    meas.load_data()
    h_test_dispatcher: Dict[str, Any] = {
        "scipy_kolmogorov_smirnov": meas.hypothesis_tests.scipy_kolmogorov_smirnov,
        "scipy_chisquare": meas.hypothesis_tests.scipy_chisquare,
        "scipy_mannwhitneyu": meas.hypothesis_tests.scipy_mannwhitneyu,
        "scipy_permutation": meas.hypothesis_tests.scipy_permutation,
        "gaussian_mixture_log_likelihood": meas.hypothesis_tests.gaussian_mixture_log_likelihood,
        "logistic_detection": meas.hypothesis_tests.logistic_detection,
        "logistic_detection_f1": partial(
            meas.hypothesis_tests.logistic_detection_custom, score_type="f1"
        ),
        "logistic_detection_roc_auc": partial(
            meas.hypothesis_tests.logistic_detection_custom, score_type="roc_auc"
        ),
    }

    for h_test_name, h_test_fn in h_test_dispatcher.items():
        res = h_test_fn(verbose=False)

        # Check res is a dict
        assert isinstance(res, dict)

        # If `chisquare`, not compatible with any of the test dataset dtypes
        # so skip
        if h_test_name == "scipy_chisquare":
            continue

        # If `scipy` test, it will be column-wise
        if h_test_name.startswith("scipy"):
            assert res.keys() == set(features_df.columns)
        # Otherwise, there should be single item in returned dictionary that
        # matches the specified names in the test
        else:
            assert list(res.keys()) == [h_test_name]


def test_statistics_summary(tmp_path) -> None:  # type: ignore
    """TODO PEP 257"""

    # Given we have registered a reference dataset
    features_df, labels_df, latents_df = example_dataset(100)
    det = Registry(tag="test", backend=FileBackend(tmp_path))
    det.register_ref_dataset(features=features_df, labels=labels_df, latents=latents_df)

    # And we have features and predicted labels
    x_monitor, y_monitor, latents_monitor = datasets.logistic_model(return_latents=True)
    # latent_x = x_monitor.mean(axis=0)
    features_monitor_df = pd.DataFrame(
        {
            "age": x_monitor[:, 0],
            "height": x_monitor[:, 1],
            "bp": x_monitor[:, 2],
        }
    )
    labels_monitor_df = pd.DataFrame({"y": y_monitor})
    latents_monitor_df = pd.DataFrame({"latents": latents_monitor})

    with det:
        # And we have logged features, labels and latent
        det.log_features(features_monitor_df)
        det.log_labels(labels_monitor_df)
        det.log_latents(latents_monitor_df)

    meas = Monitor(tag="test", backend=FileBackend(tmp_path))
    meas.load_data()
    res = meas.hypothesis_tests.scipy_kolmogorov_smirnov(verbose=True)

    # Check we get a dictionary with an entry for every feature column
    assert isinstance(res, dict)
    assert res.keys() == set(features_df.columns)


def test_load_all_logged_data(  # type: ignore
    detector_with_ref_data: Callable[[int], Registry], tmp_path
) -> None:
    """TODO PEP 257"""
    # pylint: disable=unused-argument
    # Given we have registered a reference dataset
    det = Registry(tag="test", backend=FileBackend(tmp_path))
    features_df, labels_df, latents_df = example_dataset(20)
    det.register_ref_dataset(features=features_df, labels=labels_df, latents=latents_df)

    # And we have features and predicted labels
    x_pred, y_pred, latents_pred = datasets.logistic_model(size=2, return_latents=True)

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
        det.log_latents(pd.DataFrame({"latents": latents_pred}))

    # And we have features and predicted labels
    x_pred, y_pred, latents_pred = datasets.logistic_model(size=2, return_latents=True)

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
        det.log_latents(pd.DataFrame({"latents": latents_pred}))

    # Load data
    meas = Monitor(tag="test", backend=FileBackend(tmp_path))
    recovered_dataset = meas.load_data()

    dimensions = recovered_dataset.unify().shape
    assert dimensions[0] == 4
    assert dimensions[1] == 5


def test_category_columns(tmp_path: pathlib.Path) -> None:
    """TODO PEP 257"""
    features_df, labels_df, latents_df = example_dataset(10)
    det = Registry(tag="test", backend=FileBackend(tmp_path))
    det.register_ref_dataset(features=features_df, labels=labels_df, latents=latents_df)

    # And we have features and predicted labels
    x_monitor, y_monitor, latents_monitor = datasets.logistic_model(return_latents=True)
    features_monitor_df = pd.DataFrame(
        {
            "age": x_monitor[:, 0],
            "height": x_monitor[:, 1],
            "bp": x_monitor[:, 2],
        }
    )
    labels_monitor_df = pd.DataFrame({"y": y_monitor})
    latents_monitor_df = pd.DataFrame({"latents": latents_monitor})

    with det:
        # And we have logged features, labels and latent
        det.log_features(features_monitor_df)
        det.log_labels(labels_monitor_df)
        det.log_latents(latents_monitor_df)

    meas = Monitor(tag="test", backend=FileBackend(tmp_path))
    meas.load_data()
    h_test_dispatcher: Dict[str, Any] = {
        "scipy_chisquare": meas.hypothesis_tests.scipy_chisquare
    }

    for h_test_name, h_test_fn in h_test_dispatcher.items():
        res = h_test_fn(verbose=False)

        # Check res is a dict
        assert isinstance(res, dict)

        # If `chisquare`, not compatible with any of the test dataset dtypes
        # so skip
        if h_test_name == "scipy_chisquare":
            assert len(list(res.keys())) == 1


def test_dataset_types(tmp_path: pathlib.Path) -> None:
    """TODO PEP 257"""
    features_df, labels_df, latents_df = example_dataset(10)
    det = Registry(tag="test", backend=FileBackend(tmp_path))
    det.register_ref_dataset(features=features_df, labels=labels_df, latents=latents_df)

    # And we have features and predicted labels
    x_monitor, y_monitor, latents_monitor = datasets.logistic_model(return_latents=True)
    features_monitor_df = pd.DataFrame(
        {
            "age": x_monitor[:, 0],
            "height": x_monitor[:, 1],
            "bp": x_monitor[:, 2],
        }
    )
    labels_monitor_df = pd.DataFrame({"y": y_monitor})
    latents_monitor_df = pd.DataFrame({"latents": latents_monitor})

    with det:
        # And we have logged features, labels and latent
        det.log_features(features_monitor_df)
        det.log_labels(labels_monitor_df)
        det.log_latents(latents_monitor_df)

    meas = Monitor(tag="test", backend=FileBackend(tmp_path))
    meas.load_data()
    if meas.ref_dataset is not None:
        assert isinstance(meas.ref_dataset.features, pd.DataFrame)
        assert isinstance(meas.ref_dataset.labels, pd.Series)
        assert isinstance(meas.ref_dataset.latents, pd.DataFrame)
