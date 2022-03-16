import pytest

from learning_machines_drift import DriftDetector, datasets

N_ROWS = 10
N_FEATURES = 3
N_LABELS = 2


@pytest.fixture()
def detector_with_ref_data() -> DriftDetector:

    # Given we have a reference dataset
    X_reference, Y_reference = datasets.logistic_model(size=N_ROWS)

    # When we register the dataset
    detector = DriftDetector(tag="test")
    detector.register_ref_dataset(features=X_reference, labels=Y_reference)

    return detector


def test_register_dataset(detector_with_ref_data: DriftDetector) -> None:

    # Given we have a reference dataset
    detector = detector_with_ref_data

    # Then we can see a summary of the dataset
    summary = detector.ref_summary()

    assert summary.shapes.features.n_rows == N_ROWS
    assert summary.shapes.features.n_features == N_FEATURES

    assert summary.shapes.labels.n_rows == N_ROWS
    assert summary.shapes.labels.n_labels == N_LABELS


# def test_register_features_and_labels(detector_with_ref_data: DriftDetector) -> None:

#     # Given we have registered a reference dataset
#     detector = detector_with_ref_data

#     # When we log features and labels of new data
#     with DriftDetector(
#         tag="test", expect_features=True, expect_labels=True, expect_latent=False
#     ) as detector:

#         X, Y = datasets.logistic_model()

#         detector.log_features(X)
#         detector.log_labels(Y)

#     # Then we can get a summary of registrations
#     detector.log_summary()


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

#     # Then we can get a summary of drift
#     detector.drift_summary()
