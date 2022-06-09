"""TODO PEP 257"""
import pathlib
import re
from typing import Tuple
from uuid import uuid4

import pandas as pd
from pandas.testing import assert_frame_equal

from learning_machines_drift import Dataset, FileBackend, datasets
from learning_machines_drift.backends import get_identifier


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


def test_file_backend_reference(tmp_path: pathlib.Path) -> None:
    """TODO PEP 257"""

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
    """TODO PEP 257"""

    # Given
    tag = "test_tag"
    features_df_1, labels_df_1 = example_dataset(5)
    features_df_2, labels_df_2 = example_dataset(5)
    features_df_3, labels_df_3 = example_dataset(5)

    expected_df = (
        pd.concat(
            [
                pd.concat([features_df_1, labels_df_1], axis=1),
                pd.concat([features_df_2, labels_df_2], axis=1),
                pd.concat([features_df_3, labels_df_3], axis=1),
            ]
        )
        .sort_values(by="age")
        .reset_index(drop=True)
    )

    # When we log feature and labels
    backend = FileBackend(tmp_path)

    identifier_1 = uuid4()
    backend.save_logged_features(tag, identifier_1, features_df_1)
    backend.save_logged_labels(tag, identifier_1, labels_df_1)

    identifier_2 = uuid4()
    backend.save_logged_features(tag, identifier_2, features_df_2)
    backend.save_logged_labels(tag, identifier_2, labels_df_2)

    identifier_3 = uuid4()
    backend.save_logged_features(tag, identifier_3, features_df_3)
    backend.save_logged_labels(tag, identifier_3, labels_df_3)

    # Then we should be able to load them all back
    recovered_dataset: Dataset = backend.load_logged_dataset(tag)
    recovered_df = recovered_dataset.unify()

    assert_frame_equal(
        expected_df,
        recovered_df.sort_values(by="age").reset_index(drop=True),
        check_exact=False,
    )


def test_get_identifier():
    """TODO PEP 257"""

    expected_uuid = uuid4()

    string_with_id = f"{expected_uuid}_any_other_text.csv"
    assert get_identifier(string_with_id) == expected_uuid


def test_labels():
    """TODO PEP 257"""
    re_label = re.compile("labels", re.I)
    print(re_label.search("asdfsa_labels"))


# def test_file_backend_labels(tmp_path: pathlib.Path) -> None:

#     # Given
#     tag = "test_tag"
#     _, labels_df_1 = example_dataset(5)
#     _, labels_df_2 = example_dataset(5)
#     _, labels_df_3 = example_dataset(5)
#     _, labels_df_4 = example_dataset(5)

#     expected_df = (
#         pd.concat(
#             [
#                 labels_df_1,
#                 labels_df_2,
#                 labels_df_3,
#                 labels_df_4,
#             ]
#         )
#         .sort_values(by="age")
#         .reset_index(drop=True)
#     )

#     # When
#     backend = FileBackend(tmp_path)
#     backend.save_logged_labels(tag, labels_df_1)
#     backend.save_logged_labels(tag, labels_df_2)
#     backend.save_logged_labels(tag, labels_df_3)
#     backend.save_logged_labels(tag, labels_df_4)

#     # Then
#     recovered_df = backend.load_logged_labels(tag)
#     assert_frame_equal(
#         expected_df,
#         recovered_df.sort_values(by="age").reset_index(drop=True),
#         check_exact=False,
#     )
