"""Tests for drift detector."""
# pylint: disable=W0621,too-many-locals

import pathlib
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from learning_machines_drift import Monitor, ReferenceDatasetMissing, Registry, datasets
from learning_machines_drift.backends import FileBackend
from learning_machines_drift.datasets import generate_features_labels_latents, logistic_model
from learning_machines_drift.display import Display
from learning_machines_drift.drift_filter import Comparison, Condition, Filter
from learning_machines_drift.types import StructuredResult
N_FEATURES = 3
N_LABELS = 2
N_LATENTS = 1


@pytest.fixture()
def detector(mocker: MockerFixture) -> Registry:
    """Returns a DriftDetector that writes data to a temporary directory."""
    det = Registry(tag="test", expect_latent=True)
    mocker.patch.object(det, "backend")
    return det

@pytest.fixture()
def measure(mocker: MockerFixture) -> Monitor:
    """Returns a DriftDetector that writes data to a temporary directory."""
    meas = Monitor(tag="test")
    mocker.patch.object(meas, "backend")
    return meas

@pytest.fixture()
def detector_with_ref_data(
    detector: Registry
) -> Callable[[int], Registry]:
    """Returns a DriftDetector with a reference dataset registered which
    writes data to a temporary directory.
    """
    def _detector_with_ref_data(n_rows: int) -> Registry:
        """Returns registry with saved reference data."""
        features_df, labels_df, latents_df = generate_features_labels_latents(n_rows)
        # When we register the dataset
        detector.register_ref_dataset(
            features=features_df, labels=labels_df, latents=latents_df
        )
        return detector
    return _detector_with_ref_data


def detector_with_log_data(
    detector: Registry,
    num_rows:int
) -> Registry:
    """Returns a DriftDetector with a logged datasets registered which
    writes data to a temporary directory.
    """
    num_iterations = 1
    for _ in range(num_iterations):
        (new_features_df,
        new_predictions_series,
        new_latents_df,
        ) = generate_features_labels_latents(num_rows)
        with detector:
            # And we have logged features, labels and latent
            detector.log_features(new_features_df)
            detector.log_labels(new_predictions_series)
            detector.log_latents(new_latents_df)
    return detector
    


@pytest.mark.parametrize("n_rows", [5, 10, 100, 1000])
def test_register_dataset(
    detector_with_ref_data: Callable[[int], Registry], 
    n_rows: int
) -> None:
    """Tests whether registry has expected reference data."""

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
    """Tests that the correct exception is raised upon detector with no
    reference dataset."""

    # Given a detector with no reference dataset registered

    # When we get the reference dataset summary
    # Then raise an exception
    with pytest.raises(ReferenceDatasetMissing):
        _ = detector.ref_summary()

    # And we should not have saved any reference data
    detector.backend.save_reference_dataset.assert_not_called()

@pytest.mark.parametrize("n_rows", [5, 10, 100, 1000])
def test_all_registered(
    detector_with_ref_data: Callable[[int], Registry], 
    n_rows: int) -> None:
    """Tests whether all expected data is registered."""
    # Given we have registered a reference dataset
    det = detector_with_ref_data(n_rows)
    det = detector_with_log_data(det,n_rows)

    # Then we can ensure that everything is registered
    assert det.all_registered()

    # And we saved a reference dataset
    det.backend.save_reference_dataset.assert_called_once()  # type: ignore

    # And we saved the logged features
    det.backend.save_logged_features.assert_called_once()  # type: ignore

    # And we saved the logged labels
    det.backend.save_logged_labels.assert_called_once()  # type: ignore

def test_summary_statistic_list(
    tmp_path: pathlib.Path,
) -> None:
    """Tests application of hypothesis tests from a monitor."""
    # TODO fix to use 'detector_with_ref_data' # pylint: disable=fixme
    n_rows=10
    features_df, labels_df, latents_df = generate_features_labels_latents(n_rows)
    det = Registry(tag="test", backend=FileBackend(tmp_path))
    det.register_ref_dataset(features=features_df, labels=labels_df, latents=latents_df)    
    det = detector_with_log_data(det,n_rows)
    measure = Monitor(tag="test", backend=FileBackend(tmp_path))
    measure.load_data()

    h_test_dispatcher: Dict[str, Any] = {
        "scipy_kolmogorov_smirnov": measure.hypothesis_tests.scipy_kolmogorov_smirnov,
        "scipy_mannwhitneyu": measure.hypothesis_tests.scipy_mannwhitneyu,
        "boundary_adherence": measure.hypothesis_tests.get_boundary_adherence,
        "logistic_detection": measure.hypothesis_tests.logistic_detection,
    }
    for h_test_name, h_test_fn in h_test_dispatcher.items():
        res = h_test_fn()
        # Check res is a StructureResult
        assert isinstance(res, StructuredResult)
        assert (res.method_name==h_test_name)
        if list(res.results.keys())[0] == 'single_value':
            assert isinstance(res.results['single_value'], dict)
        else:
            # print(list(res.results.keys()))
            # print(list(det.registered_dataset.unify().columns))
            assert list(res.results.keys())==list(det.registered_dataset.unify().columns)

def test_statistics_summary(tmp_path) -> None:  # type: ignore
    """Tests whether hypothesis tests are applied to all columns of unifed
    dataset as expected."""

    # Given we have registered a reference dataset
    n_rows=10
    features_df, labels_df, latents_df = generate_features_labels_latents(n_rows)
    det = Registry(tag="test", backend=FileBackend(tmp_path))
    det.register_ref_dataset(features=features_df, labels=labels_df, latents=latents_df)    
    det = detector_with_log_data(det, n_rows)
    measure = Monitor(tag="test", backend=FileBackend(tmp_path))
    measure.load_data()
    res = measure.hypothesis_tests.scipy_kolmogorov_smirnov()

    # Check we get a dictionary with an entry for every feature column
    assert isinstance(res, StructuredResult)
    assert res.method_name == "scipy_kolmogorov_smirnov"
    assert list(res.results.keys())==list(det.registered_dataset.unify().columns)

def test_with_noncommon_columns(tmp_path) -> None:  # type: ignore
    """Tests the application of hypothesis tests to the intersection of columns
    when there are differing columns in the reference and registered dataset.
    """

    # Given we have registered a reference dataset
    features_df, labels_df, latents_df = generate_features_labels_latents(10)
    det = Registry(tag="test", backend=FileBackend(tmp_path))
    det.register_ref_dataset(features=features_df, labels=labels_df, latents=latents_df)    
    
    num_iterations = 1
    for _ in range(num_iterations):
        (new_features_df,
        new_predictions_series,
        new_latents_df,
        ) = generate_features_labels_latents(10)
        new_features_df = new_features_df.drop(['age'], axis=1)
        with det:
            # And we have logged features, labels and latent
            det.log_features(new_features_df)
            det.log_labels(new_predictions_series)
            det.log_latents(new_latents_df)
    
    measure = Monitor(tag="test", backend=FileBackend(tmp_path))
    measure.load_data()

    h_test_dispatcher: Dict[str, Any] = {
        "scipy_kolmogorov_smirnov": measure.hypothesis_tests.scipy_kolmogorov_smirnov,
        "scipy_mannwhitneyu": measure.hypothesis_tests.scipy_mannwhitneyu,
        "scipy_permutation": measure.hypothesis_tests.scipy_permutation,
        "logistic_detection": measure.hypothesis_tests.logistic_detection,
        # "logistic_detection_f1": partial(
        #     measure.hypothesis_tests.logistic_detection_custom, score_type="f1"
        # ),
        # "logistic_detection_roc_auc": partial(
        #     measure.hypothesis_tests.logistic_detection_custom, score_type="roc_auc"
        # ),
    }

    for h_test_name, h_test_fn in h_test_dispatcher.items():
        res = h_test_fn()
        # Check res is a dict
        assert isinstance(res, StructuredResult)
        assert (res.method_name==h_test_name)
        if list(res.results.keys())[0] == 'single_value':
            assert isinstance(res.results['single_value'], dict)
        else:
            assert list(res.results.keys())==list(det.registered_dataset.unify().columns)

        # # If `scipy` test, it will be column-wise
        # if h_test_name.startswith("scipy"):
        #     # Get unified subsets with columns common to both registered and reference.
        #     # Only these columns should hav test results.
        #     (
        #         unified_ref_subset,
        #         unified_reg_subset,
        #     ) = meas.hypothesis_tests._get_unified_subsets()
        #     method = res[h_test_name]
        #     statistic = method['statistic']
        #     assert statistic.keys() == set(unified_reg_subset.columns)
        #     assert statistic.keys() == set(unified_ref_subset.columns)
        # # Otherwise, there should be single item in returned dictionary that
        # # matches the specified names in the test
        # else:
        #     assert list(res.keys()) == [h_test_name]


def test_display(tmp_path: pathlib.Path) -> None:
    """Tests whether the display class returns the expected types."""

    n_rows=10
    features_df, labels_df, latents_df = generate_features_labels_latents(n_rows)
    det = Registry(tag="test", backend=FileBackend(tmp_path))
    det.register_ref_dataset(features=features_df, labels=labels_df, latents=latents_df)    
    det = detector_with_log_data(det,n_rows)
    measure = Monitor(tag="test", backend=FileBackend(tmp_path))
    measure.load_data()

    # Subset of h_tests sufficient for testing plotting
    h_test_dispatcher: Dict[str, Any] = {
        "scipy_kolmogorov_smirnov": measure.hypothesis_tests.scipy_kolmogorov_smirnov,
        "scipy_mannwhitneyu": measure.hypothesis_tests.scipy_mannwhitneyu,
    }

    # Loop over h_tests
    for _, h_test_fn in h_test_dispatcher.items():
        # Get result from scoring
        res = h_test_fn(verbose=False)

        # Check Display produces a dataframe
        df = Display().table(res, verbose=False)
        assert isinstance(df, pd.DataFrame)
        # Check the correct shape
        assert df.shape == (len(res), 2)

        # Check Display returns a plot and array of axes
        fig, axs = Display().plot(res)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axs, np.ndarray)
        for ax in axs.flatten():
            assert isinstance(ax, plt.Axes)


def test_load_all_logged_data(tmp_path: pathlib.Path) -> None:
    """Tests whether logging of data and that upon reload the data has the
    correct shape."""
    num_rows = 10
    features_df, labels_df, latents_df = generate_features_labels_latents(num_rows)
    det = Registry(tag="test", backend=FileBackend(tmp_path))
    det.register_ref_dataset(features=features_df, labels=labels_df, latents=latents_df)    
    det = detector_with_log_data(det,num_rows)
    measure = Monitor(tag="test", backend=FileBackend(tmp_path))
    recovered_dataset = measure.load_data()
    dimensions:Tuple[int, int] = recovered_dataset.unify().shape
    assert dimensions[0] == num_rows
    assert dimensions[1] == (3+1+1)


def test_condition() -> None:
    """Test for creating condition instances."""
    condition = Condition("less", 5)
    assert condition.comparison == Comparison.LESS
    assert condition.value == 5

    with pytest.raises(Exception):
        condition = Condition("Equal", 5)

    with pytest.raises(Exception):
        condition = Condition("Greater", 5)


@pytest.mark.parametrize("n_rows", [10, 100, 1000])
def test_load_data_filtered(tmp_path: pathlib.Path, n_rows: int) -> None:
    """Tests whether a filter applied to load data from registry correctly
    filters data."""
    features_df, labels_df, latents_df = generate_features_labels_latents(n_rows)
    det = Registry(tag="test", backend=FileBackend(tmp_path))
    det.register_ref_dataset(features=features_df, labels=labels_df, latents=latents_df)    
    det = detector_with_log_data(det,n_rows)
    measure = Monitor(tag="test", backend=FileBackend(tmp_path))
    measure.load_data()

    # Make a drift filter
    filter_dict: Dict[str, List[Condition]] = dict(
        [
            ("age", [Condition("less", 0.2)]),
            ("height", [Condition("greater", -0.1), Condition("less", 0.5)]),
            ("ground-truth-label", [Condition("equal", 0)]),
            ("latents", [Condition("greater", 0.6)]),
        ]
    )
    drift_filter = Filter(filter_dict)

    # Load data (unfiltered datasets)
    monitor_unfiltered = Monitor(tag="test", backend=FileBackend(tmp_path))
    _ = monitor_unfiltered.load_data()

    # Load data (filtered datasets)
    monitor = Monitor(tag="test", backend=FileBackend(tmp_path))
    _ = monitor.load_data(drift_filter)

    # Assert conditions on unfiltered and filtered datasets
    for (ref_dataset, reg_dataset, assertion_bool) in [
        (monitor_unfiltered.ref_dataset, monitor_unfiltered.registered_dataset, False),
        (monitor.ref_dataset, monitor.registered_dataset, True),
    ]:
        assert ref_dataset is not None and reg_dataset is not None
        assert ref_dataset.latents is not None and reg_dataset.latents is not None
        assert ref_dataset.features.shape[0] == ref_dataset.labels.shape[0]
        assert ref_dataset.labels.shape[0] == ref_dataset.latents.shape[0]
        assert reg_dataset.features.shape[0] == reg_dataset.labels.shape[0]
        assert reg_dataset.labels.shape[0] == reg_dataset.latents.shape[0]
        assert ref_dataset.features["age"].lt(0.2).all() == assertion_bool
        assert reg_dataset.features["age"].lt(0.2).all() == assertion_bool
        assert ref_dataset.features["height"].gt(-0.1).all() == assertion_bool
        assert reg_dataset.features["height"].gt(-0.1).all() == assertion_bool
        assert ref_dataset.features["height"].lt(0.5).all() == assertion_bool
        assert reg_dataset.features["height"].lt(0.5).all() == assertion_bool
        assert ref_dataset.labels.eq(0).all() == assertion_bool
        assert reg_dataset.labels.eq(0).all() == assertion_bool
        assert ref_dataset.latents["latents"].gt(0.6).all() == assertion_bool
        assert reg_dataset.latents["latents"].gt(0.6).all() == assertion_bool


def test_category_columns(tmp_path: pathlib.Path) -> None:
    """Tests the returned result structure for categorical hypothesis test."""
    n_rows=10
    features_df, labels_df, latents_df = generate_features_labels_latents(n_rows)
    det = Registry(tag="test", backend=FileBackend(tmp_path))
    det.register_ref_dataset(features=features_df, labels=labels_df, latents=latents_df)    
    det = detector_with_log_data(det,n_rows)
    measure = Monitor(tag="test", backend=FileBackend(tmp_path))
    measure.load_data()

    h_test_dispatcher: Dict[str, Any] = {
        # "scipy_chisquare": meas.hypothesis_tests.scipy_chisquare
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
    """Tests whether the reference dataset components have the expected types."""
    n_rows=10
    features_df, labels_df, latents_df = generate_features_labels_latents(n_rows)
    det = Registry(tag="test", backend=FileBackend(tmp_path))
    det.register_ref_dataset(features=features_df, labels=labels_df, latents=latents_df)    
    det = detector_with_log_data(det,n_rows)
    measure = Monitor(tag="test", backend=FileBackend(tmp_path))
    measure.load_data()

    if measure.ref_dataset is not None:
        assert isinstance(measure.ref_dataset.features, pd.DataFrame)
        assert isinstance(measure.ref_dataset.labels, pd.Series)
        assert isinstance(measure.ref_dataset.latents, pd.DataFrame)


def test_sdmetrics(tmp_path: pathlib.Path) -> None:
    """TODO PEP 257"""
    n_rows=10
    features_df, labels_df, latents_df = generate_features_labels_latents(n_rows)
    det = Registry(tag="test", backend=FileBackend(tmp_path))
    det.register_ref_dataset(features=features_df, labels=labels_df, latents=latents_df)    
    det = detector_with_log_data(det,n_rows)
    measure = Monitor(tag="test", backend=FileBackend(tmp_path))
    measure.load_data()

    result:StructuredResult = measure.hypothesis_tests.get_boundary_adherence()
    assert result.method_name=="boundary_adherence"
    assert type(result.results) is dict
    assert len(result.results.keys()) == 5
    assert next(iter(result.results)) == "age"
    result_age = result.results["age"]
    assert type(result_age) is dict
    assert next(iter(result_age.keys())) == "statistic"
    assert result_age["statistic"] >= 0.0 and result_age["statistic"] <= 1.0
