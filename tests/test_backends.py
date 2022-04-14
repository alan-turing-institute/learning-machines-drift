import pathlib
import pytest

from learning_machines_drift import FileBackend, Dataset, datasets
import pandas as pd
from typing import Tuple


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


def test_file_backend(tmp_path: pathlib.Path) -> None:

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
