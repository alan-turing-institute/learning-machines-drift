import pathlib
from typing import Tuple

import pandas as pd
from pandas.testing import assert_frame_equal

from learning_machines_drift import Dataset, FileBackend, datasets


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


def test_file_backend_reference(tmp_path: pathlib.Path) -> None:

    # Given
    tag = "test_tag"
    features_df, labels_df = example_dataset(100)
    reference_dataset = Dataset(features_df, labels_df)

    # When
    backend = FileBackend(tmp_path)
    backend.save_reference_dataset(tag, reference_dataset)

    # Then
    recovered_dataset = backend.load_reference_dataset(tag)
    assert recovered_dataset.features.equals(recovered_dataset.features)
    assert recovered_dataset.labels.equals(recovered_dataset.labels)


def test_file_backend_features(tmp_path: pathlib.Path) -> None:

    # Given
    tag = "test_tag"
    features_df_1, _ = example_dataset(5)
    features_df_2, _ = example_dataset(5)
    features_df_3, _ = example_dataset(5)
    features_df_4, _ = example_dataset(5)

    expected_df = (
        pd.concat(
            [
                features_df_1,
                features_df_2,
                features_df_3,
                features_df_4,
            ]
        )
        .sort_values(by="age")
        .reset_index(drop=True)
    )

    # When
    backend = FileBackend(tmp_path)
    backend.save_logged_features(tag, features_df_1)
    backend.save_logged_features(tag, features_df_2)
    backend.save_logged_features(tag, features_df_3)
    backend.save_logged_features(tag, features_df_4)

    # Then
    recovered_df = backend.load_logged_features(tag)
    assert_frame_equal(
        expected_df,
        recovered_df.sort_values(by="age").reset_index(drop=True),
        check_exact=False,
    )
