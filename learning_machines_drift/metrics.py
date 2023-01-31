"""Class for scoring drift between reference and registered datasets."""

import textwrap
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from sdmetrics.single_column import BoundaryAdherence, RangeCoverage
from sdmetrics.single_table import LogisticDetection
from sdmetrics.utils import HyperTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from learning_machines_drift.types import Dataset, StructuredResult


class Wrapper(Enum):
    """Enum for specifying the calculation type."""

    TYPE_TUPLE = 1
    TYPE_OTHER = 2
    TYPE_SDMETRIC = 3


class Metrics:
    """A class with metrics for scoring data drift between registered and
    reference datasets.

    Attributes:
        reference_dataset (Dataset): Reference datastet for drift measures.
        registered_dataset (Dataset): Registered/logged datastet for drift
            measures.
        random_state (Optional[int]): Optional seeding for reproducibility.
    """

    def __init__(
        self,
        reference_dataset: Dataset,
        registered_dataset: Dataset,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize with registered and reference and optional seed.

        Args:
            reference_dataset (Dataset): Reference datastet for drift measures.
            registered_dataset (Dataset): Registered/logged datastet for drift
                measures.
            random_state (Optional[int]): Optional seeding for reproducibility.

        """
        self.reference_dataset = reference_dataset
        self.registered_dataset = registered_dataset
        self.random_state = random_state

    @staticmethod
    def _format_about_str(method: str, description: str, line_len: int = 79) -> str:
        """Takes methods and description and returns a string to print.

        Args:
            method (str): Name of the method.
            description (str): Description of the method.

        Returns:
            str: Formatted string.
        """
        return "\n".join(
            [
                "",
                textwrap.fill(f"Method: {method}".replace("\n", " "), line_len),
                textwrap.fill(
                    f"Description: {description}".replace("\n", " "), line_len
                ),
                "",
            ]
        )

    @staticmethod
    def _to_dict(obj: object) -> Dict[str, float]:
        """Convert an object to a dict."""
        obj_as_dict = {}
        attributes = [
            attribute for attribute in dir(obj) if attribute in ["statistic", "pvalue"]
        ]
        for attribute in attributes:
            obj_as_dict[attribute] = getattr(obj, attribute)
        return obj_as_dict

    def _calc(
        self,
        func: Callable[..., Any],
        subset: Optional[List[str]] = None,
        wrapper: Wrapper = Wrapper.TYPE_OTHER,
    ) -> Any:
        """Method for calculating statistic and pvalue from a passed scoring
        function.

        Args:
            func (Callable[..., Any]): Function that takes two series and
                returns a dictionary or named tuple containing scores.
            subset (List[str]): List of feature names to subset scores over.

        Returns:
            results (dict): Dictionary with features as keys and scores as
                values.
        """

        def call_func(
            ref_col: pd.Series,
            reg_col: pd.Series,
            wrapper: Wrapper = Wrapper.TYPE_OTHER,
        ) -> Dict[str, float]:
            """Returns a dict of statistic and p-value (if available)."""
            if wrapper is Wrapper.TYPE_TUPLE:
                result = func((ref_col, reg_col))
            elif wrapper is Wrapper.TYPE_SDMETRIC:
                result = func(real_data=ref_col, synthetic_data=reg_col)
                result = {"statistic": result}
            else:
                result = func(ref_col, reg_col)

            if not isinstance(result, dict):
                result = self._to_dict(result)

            result_dict: Dict[str, float] = result
            return result_dict

        results: Dict[str, Dict[str, float]] = {}

        if subset is not None:
            # If subset, only loop over the subset
            columns_to_calc: List[str] = subset
        else:
            # Get all columns from features, labels, latents
            columns_to_calc = self.reference_dataset.unify().columns.to_list()

        for col_name in columns_to_calc:
            # Check if in features
            if self.reference_dataset is not None:
                if (col_name in self.reference_dataset.features.columns) and (
                    col_name in self.registered_dataset.features.columns
                ):
                    ref_col = self.reference_dataset.features[col_name]
                    reg_col = self.registered_dataset.features[col_name]
                # Check if labels
                elif (col_name == self.reference_dataset.labels.name) and (
                    col_name == self.registered_dataset.labels.name
                ):
                    ref_col = self.reference_dataset.labels
                    reg_col = self.registered_dataset.labels
                # Check if in latents
                elif (
                    (self.reference_dataset.latents is not None)
                    and (col_name in self.reference_dataset.latents.columns)
                    and (self.registered_dataset.latents is not None)
                    and (col_name in self.registered_dataset.latents.columns)
                ):
                    ref_col = self.reference_dataset.latents[col_name]
                    reg_col = self.registered_dataset.latents[col_name]
                else:
                    print(
                        "Error:",
                        ValueError(
                            f"'{col_name}' is not in features, labels or latents."
                        ),
                    )
                    continue
            else:
                raise ValueError("Reference dataset is None.")
            # Run calc and update dictionary
            results[col_name] = call_func(ref_col, reg_col, wrapper)

        return results

    def _get_unified_subsets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Unify both datasets and return dataframes with common columns."""

        def get_intersect(data1: pd.DataFrame, data2: pd.DataFrame) -> List[str]:
            """Get list of common columns to both dataframes."""
            return list(set(data1.columns) & set(data2.columns))

        unified_ref = self.reference_dataset.unify()
        unified_reg = self.registered_dataset.unify()
        subset: List[str] = get_intersect(unified_ref, unified_reg)
        return (unified_ref[subset], unified_reg[subset])

    def scipy_kolmogorov_smirnov(self, verbose: bool = True) -> StructuredResult:
        """Calculates feature-wise two-sample Kolmogorov-Smirnov test for
        goodness of fit. Assumes continuous underlying distributions but
        scores are still interpretable if data is approximately continuous.

        Args:
            verbose (bool): Boolean for verbose output to stdout.

        Returns:
            results (dict): Dictionary of statistics and  p-values by feature.
        """
        method = "SciPy Kolmogorov Smirnov"
        description = ""
        about_str = self._format_about_str(method=method, description=description)

        results = self._calc(stats.ks_2samp)
        if verbose:
            print(about_str)

        result_dict: Dict[str, Dict[str, float]] = results
        structured_result = StructuredResult("scipy_kolmogorov_smirnov", result_dict)
        return structured_result

    def scipy_mannwhitneyu(self, verbose: bool = True) -> StructuredResult:
        """Calculates feature-wise Mann-Whitney U test, a nonparametric test of
        the null hypothesis that the distribution underlying sample x is the
        same as the distribution underlying sample y. Provides a test for the
        difference in location of two distributions. Assumes continuous
        underlying distributions but scores are still interpretable if data
        is approximately continuous.

        Args:
            verbose (bool): Boolean for verbose output to stdout.

        Returns:
            results (dict): Dictionary of statistics and  p-values by feature.

        """
        method = "SciPy Mann-Whitney U"
        description = (
            "Non-parameric test between independent samples comparing their location."
        )
        about_str = self._format_about_str(method=method, description=description)

        results = self._calc(stats.mannwhitneyu)
        if verbose:
            print(about_str)

        result_dict: Dict[str, Dict[str, float]] = results
        structured_result = StructuredResult("scipy_mannwhitneyu", result_dict)
        return structured_result

    def scipy_permutation(
        self,
        agg_func: Callable[..., float] = np.mean,
        verbose: bool = True,
    ) -> StructuredResult:
        """Performs feature-wise permutation test with default statistic to
        measure differences under permutations of labels as the mean.

        Args:
            func (Callable[..., float]): Function for comparing two samples.
            verbose (bool): Print outputs
        Returns:
            results (dict): Dictionary with keys as features and values as
            scipy.stats.permutation_test object with test results.
        """
        method = f"SciPy Permutation Test (test function: {agg_func.__name__})"
        description = (
            "Performs permutation test on all features with passed stat_fn "
            "measuring the difference between samples."
        )
        about_str = self._format_about_str(method=method, description=description)

        def statistic(
            lhs: npt.ArrayLike,
            rhs: npt.ArrayLike,
            axis: int = 0,
        ) -> float:
            """Statistic for evaluating the difference between permuted
            samples.
            """
            return agg_func(lhs, axis=axis) - agg_func(rhs, axis=axis)

        func = partial(
            stats.permutation_test,
            statistic=statistic,
            permutation_type="independent",
            alternative="two-sided",
            n_resamples=9999,
            random_state=self.random_state,
        )

        results = self._calc(func, wrapper=Wrapper.TYPE_TUPLE)

        if verbose:
            print(about_str)

        result_dict: Dict[str, Dict[str, float]] = results
        structured_result = StructuredResult("scipy_permutation", result_dict)
        return structured_result

    # pylint: disable=invalid-name
    def logistic_detection(  # pylint: disable=too-many-locals, too-many-branches
        self,
        normalize: bool = False,
        score_type: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> StructuredResult:
        """Calculates a measure of similarity using fitted logistic regression
        to predict reference or registered label. SD metrics package
        `source <https://github.com/sdv-dev/SDMetrics/blob/v0.6.0/sdmetrics/single_table/detection/base.py#L46-L91>`_ # pylint: disable=line-too-long
        is adapted to permit optional `score_type` and `seed` to be given allowing
        alternative and reproducible metrics.

        `score_type` can be:
            - None: defaults to scoring of `logistic_detection` method.
            - "f1": Cross-validated F1 score with  0.5 threshold.
            - "roc_auc": Cross-validated receiver operating characteristic (area under the curve).

        Args:
            score_type (Optional[str]): None for default or string; "f1" and
                "roc_auc" currently implemented.
            seed (Optional[int]): Optional integer for reproducibility of
                scoring as cross-validation performed.
            verbose (bool): Boolean for verbose output to stdout.

        Returns:
            results (float): Score providing an overall similarity measure of
                reference and registered datasets.
        """
        if score_type is None:
            method = f"Logistic Detection (scoring: standard, normalize: {normalize})"
        else:
            method = (
                f"Logistic Detection (scoring: {score_type}, normalize: {normalize})"
            )
        description = (
            "Detection metric based on a LogisticRegression classifier from "
            "scikit-learn with custom scoring."
        )
        about_str = self._format_about_str(method=method, description=description)

        if verbose:
            print(about_str)

        # Get unified subsets
        unified_ref_subset, unified_reg_subset = self._get_unified_subsets()

        # Transform data for fitting using SD metrics HyperTransformer
        ht = HyperTransformer()
        transformed_reference_data = ht.fit_transform(unified_ref_subset).to_numpy()
        transformed_registered_data = ht.transform(unified_reg_subset).to_numpy()

        X = np.concatenate([transformed_reference_data, transformed_registered_data])
        y = np.hstack(
            [
                np.ones(len(transformed_reference_data)),
                np.zeros(len(transformed_registered_data)),
            ]
        )
        if np.isin(X, [np.inf, -np.inf]).any():
            X[np.isin(X, [np.inf, -np.inf])] = np.nan

        try:
            scores = []
            kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            lr = LogisticRegression(solver="lbfgs")
            for train_index, test_index in kf.split(X, y):
                lr.fit(X[train_index], y[train_index])
                y_pred = lr.predict_proba(X[test_index])[:, 1]

                if score_type is None:
                    roc_auc = roc_auc_score(y[test_index], y_pred)
                    score = max(0.5, roc_auc) * 2 - 1
                elif score_type == "roc_auc":
                    score = roc_auc_score(y[test_index], y_pred)
                elif score_type == "f1":
                    score = f1_score(y[test_index], (y_pred > 0.5) * 1)
                else:
                    raise NotImplementedError(f"{score_type} not implemented.")
                scores.append(score)
        except ValueError as err:
            raise ValueError(
                f"DetectionMetric: Unable to be fit with error {err}"
            ) from err

        if score_type is None:
            # SDMetrics approach to scoring takes 1 - mean:
            # https://github.com/sdv-dev/SDMetrics/blob/master/sdmetrics/single_table/detection/base.py#L89
            results_key: str = "logistic_detection"
            results: float = 1 - np.mean(scores)
        else:
            # Custom metrics assume the mean of the scores
            results_key = f"logistic_detection_{score_type}"
            results = np.mean(scores)

        if normalize and score_type is None:
            results = LogisticDetection.normalize(results)

        result_dict: Dict[str, Dict[str, float]] = {
            "single_value": {"statistic": results}
        }
        structured_result = StructuredResult(results_key, result_dict)
        return structured_result

    # pylint: enable=invalid-name

    def get_boundary_adherence(
        self,
    ) -> StructuredResult:
        """For each feature the proportion of registered data that lies within
        the minimum and maximum of the reference dataset.

        See `SDMetrics <https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/boundaryadherence>`_
        for further details.

        Returns:
            StructuredResult: The boundary adherence of the registered dataset
                compared to the reference dataset.

        """
        results: Dict[str, Dict[str, float]] = self._calc(
            BoundaryAdherence.compute, wrapper=Wrapper.TYPE_SDMETRIC
        )
        structured_result = StructuredResult("boundary_adherence", results)
        return structured_result

    def get_range_coverage(self) -> StructuredResult:
        """For each feature the proportion of the range of the registered data
        that is covered by the reference dataset.

        See `SDMetrics <https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/rangecoverage>`_
        for further details.

        Returns:
            StructuredResult: The range of the registered dataset compared
                to the reference dataset.

        """
        results: Dict[str, Dict[str, float]] = self._calc(
            RangeCoverage.compute, wrapper=Wrapper.TYPE_SDMETRIC
        )
        structured_result = StructuredResult("range_coverage", results)
        return structured_result
