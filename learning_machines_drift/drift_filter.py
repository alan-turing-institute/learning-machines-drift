# pylint: disable=C0103
# pylint: disable=W0621
# pylint: disable=R0913

"""Module to filter a dataset."""
from enum import Enum, auto
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from learning_machines_drift.types import Dataset


class Operator(Enum):
    """Operator class for 'LESS', 'GREATER' and 'EQUAL' cases."""

    LESS = auto()
    GREATER = auto()
    EQUAL = auto()


class Condition:  # pylint: disable=too-few-public-methods
    """Condition class comprising of an 'operator' and a 'value'."""

    operator: Operator
    value: Any

    def __init__(self, operator: str, value: Any):
        """Init condition from passed string and value.

        Args:
            operator (str): One of 'equal', 'less', 'greater'.
            value (Any): The value for comparison.

        """
        if operator == "less":
            self.operator = Operator.LESS
        elif operator == "greater":
            self.operator = Operator.GREATER
        elif operator == "equal":
            self.operator = Operator.EQUAL
        else:
            raise ValueError(
                f"'{operator}' is not implemented. "
                "Please choose one of 'equal', 'less', 'greater'."
            )
        self.value = value


class Filter:  # pylint: disable=too-few-public-methods
    """Filter class.

    Filters a given dataset through an AND operation applied across all passed
    conditions.

    Args:
        conditions (dict[str, List[Condition]]): Dict with key (variable) and value
            as a list of (condition, value) to be used for filtering.

    Attributes:
        conditions (dict[str, List[Condition]]): Dict with key (variable) and value
            as a list of (condition, value) to be used for filtering.
    """

    conditions: Optional[dict[str, List[Condition]]]

    def __init__(self, conditions: Optional[dict[str, List[Condition]]]):
        """Initialize a dict with variable keys and (condition, value) values."""
        self.conditions = conditions

    def _filter_df(
        self, df: pd.DataFrame, variable: str, condition: Condition
    ) -> pd.DataFrame:
        """Subset a dataframe given condition and value."""
        if condition.operator == Operator.LESS:
            return df.loc[df[variable] < condition.value]
        if condition.operator == Operator.GREATER:
            return df.loc[df[variable] > condition.value]
        if condition.operator == Operator.EQUAL:
            return df.loc[df[variable] == condition.value]
        raise ValueError(
            f"'{condition.operator}' is not implemented. "
            "Please choose one of 'equal', 'less', 'greater'."
        )

    def _filter_series(
        self,
        series: pd.Series,
        condition: Condition,
    ) -> pd.Series:
        """Subset a series given condition and value."""
        if condition.operator == Operator.LESS:
            return series[series < condition.value]
        if condition.operator == Operator.GREATER:
            return series[series > condition.value]
        if condition.operator == Operator.EQUAL:
            return series[series == condition.value]
        raise ValueError(
            f"'{condition.operator}' is not implemented. "
            "Please choose one of 'equal', 'less', 'greater'."
        )

    def transform(self, dataset: Dataset) -> Dataset:
        """Transform the passed dataset given filter.

        Args:
            dataset (Dataset): the dataset to be filtered.

        Returns:
            Dataset: transformed dataset given filters
        """
        if self.conditions is None:
            return dataset

        # Filter the dataframe or series of the variables
        for (variable, conditions) in self.conditions.items():
            for condition in conditions:
                if variable in dataset.features.columns:
                    dataset.features = self._filter_df(
                        dataset.features, variable, condition
                    )
                elif variable == dataset.labels.name:
                    dataset.labels = self._filter_series(dataset.labels, condition)
                elif dataset.latents is not None:
                    if variable in dataset.latents.columns:
                        dataset.latents = self._filter_df(
                            dataset.latents, variable, condition
                        )

        # Select only the common idx
        common_idx = np.intersect1d(
            dataset.features.index.to_numpy(),
            dataset.labels.index.to_numpy(),
        )
        if dataset.latents is not None:
            common_idx = np.intersect1d(
                common_idx,
                dataset.latents.index.to_numpy(),
            )

        # Update dataset with common idx
        dataset.features = dataset.features.loc[common_idx]
        dataset.labels = dataset.labels.loc[common_idx]
        if dataset.latents is not None:
            dataset.latents = dataset.latents.loc[common_idx]

        # Return filtered dataset
        return dataset
