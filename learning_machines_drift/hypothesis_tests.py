"""Class for scoring drift between reference and registered datasets."""

import textwrap
from collections import Counter
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from sdmetrics.single_table import LogisticDetection
from sdmetrics.utils import HyperTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sdmetrics.single_column import BoundaryAdherence, RangeCoverage
from sdmetrics.single_table import BinaryAdaBoostClassifier, BinaryDecisionTreeClassifier, BinaryLogisticRegression, BinaryMLPClassifier

from learning_machines_drift.types import Dataset

from enum import Enum
 
class Wrapper(Enum):
    TYPE_TUPLE = 1
    TYPE_OTHER = 2
    TYPE_SDMETRIC = 3
    
class HypothesisTests:
    """
    A class for performing hypothesis tests and scoring between registered and
    reference datasets.
    """

    def __init__(
        self,
        reference_dataset: Dataset,
        registered_dataset: Dataset,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize with registered and reference and optional seed."""
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
            feature: str,
            ref_col: pd.Series,
            reg_col: pd.Series,
            results: Dict[str, Any],
            wrapper: Wrapper = Wrapper.TYPE_OTHER,
        ) -> Dict[str, Any]:
            
            if wrapper is Wrapper.TYPE_TUPLE:
                result = func((ref_col, reg_col))
            elif wrapper is Wrapper.TYPE_SDMETRIC:
                result = func(real_data=ref_col, synthetic_data=reg_col)
                result = {"statistic":result}
            else:
                result = func(ref_col, reg_col)
              
            if not isinstance(result, dict):
                results[feature] = self._to_dict(result)
            else:
                results[feature] = result
            
            return results

        results: Dict[str, Any] = {}

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
            results = call_func(col_name, ref_col, reg_col, results, wrapper)

        return results

    # # Skip the ones that are not calculable
    # @staticmethod
    # def _get_category_columns(dataset: Dataset) -> List[str]:
    #     """Get a list of feature names that have category-like features.
    #     Category-like features are defined as:
    #         - Unit or Binary (less than two values)
    #         OR
    #         - Categorical (category dtype)

    #     Args:
    #         data (pd.DataFrame): data to be have features checked.

    #     Returns:
    #         List[str]: List of feature names that are category-like.
    #     """
    #     # Unify all features, labels and latents
    #     data: pd.DataFrame = dataset.unify()
    #     # Get the number of unique values by feature
    #     nunique: pd.Series = data.nunique()
    #     # Unit or binary features
    #     unit_or_bin_features: List[str] = nunique[nunique <= 2].index.to_list()
    #     # Integer or category features
    #     cat_features: List[str] = data.dtypes[
    #         data.dtypes.eq("category")
    #     ].index.to_list()
    #     # Get list of unique features for output
    #     out_features: List[str] = list(np.unique(unit_or_bin_features + cat_features))

    #     return out_features

    # def get_unified_subsets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #     """Unify both datasets and return dataframes with common columns."""

    #     def get_intersect(data1: pd.DataFrame, data2: pd.DataFrame) -> List[str]:
    #         """Get list of common columns to both dataframes."""
    #         return list(set(data1.columns) & set(data2.columns))

    #     unified_ref = self.reference_dataset.unify()
    #     unified_reg = self.registered_dataset.unify()
    #     subset: List[str] = get_intersect(unified_ref, unified_reg)
    #     return (unified_ref[subset], unified_reg[subset])
    
    
    def scipy_kolmogorov_smirnov(self, verbose: bool = True) -> Any:
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
        return results

    def scipy_mannwhitneyu(self, verbose: bool = True) -> Any:
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
        return results

    # @staticmethod
    # def _chi_square(data1: pd.Series, data2: pd.Series) -> dict[str, float]:
    #     """Perform a chi-square test on two category-like series.

    #     Args:
    #         data1 (pd.Series): First series.
    #         data2 (pd.Series): Second series.
    #     Returns:
    #         dict[str, float]: dict of chi-square statistic and p-value.
    #     """
    #     # Get unique elements across all data
    #     base: npt.NDArray[Any] = np.unique(np.append(data1, data2))
    #     # Get counts of values in data1
    #     d1_counter: Counter[Any] = Counter(data1)
    #     # Get counts of values in data2
    #     d2_counter: Counter[Any] = Counter(data2)
    #     # Get counts in order of base for both counters
    #     d1_counts: List[int] = [d1_counter[el] for el in base]
    #     d2_counts: List[int] = [d2_counter[el] for el in base]
    #     # Calculate chi-square
    #     statistic, pvalue, _, _ = stats.chi2_contingency(
    #         np.stack([d1_counts, d2_counts])
    #     )
    #     return {"statistic": statistic, "pvalue": pvalue}

    # def scipy_chisquare(self, verbose: bool = True) -> Any:
    #     """Calculates feature-wise chi-square statistic and p-value for
    #     the hypothesis test of independence of the observed frequencies.
    #     Provides a test for the independence of two count distributions.
    #     Assumes categorical underlying distributions.

    #     Args:
    #         verbose (bool): Boolean for verbose output to stdout.

    #     Returns:
    #         results (dict): Dictionary of statistics and  p-values by feature.
    #     """
    #     method = (
    #         "SciPy chi-square test of independence of variables in a "
    #         "contingency table."
    #     )
    #     description = (
    #         "Chi-square test for categorical-like data comparing counts in "
    #         "registered and reference data."
    #     )
    #     about_str = self._format_about_str(method=method, description=description)

    #     results = self._calc(
    #         self._chi_square,
    #         subset=self._get_category_columns(self.registered_dataset),
    #     )
    #     if verbose:
    #         print(about_str)
    #     return results

    def scipy_permutation(
        self,
        agg_func: Callable[..., float] = np.mean,
        verbose: bool = True,
    ) -> Any:
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

        return results

    

    def logistic_detection(
        self, normalize: bool = False, verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """Calculates a measure of similarity using fitted logistic regression
        to predict reference or registered label using Synthetic Data
        Vault package.

        A value of one indicates indistinguishable data, while a value of zero
        indicates the converse.

        Args:
            verbose (bool): Boolean for verbose output to stdout.
            normalize (bool): Normalize raw_score to interval [0, 1].

        Returns:
            results (Dict[str, Dict[str, float]]): Score providing an overall similarity measure of
                reference and registered datasets.
        """
        method = f"Logistic Detection (normalize: {normalize})"
        description = (
            "Detection metric based on a LogisticRegression classifier from "
            "scikit-learn."
        )
        about_str = self._format_about_str(method=method, description=description)

        if verbose:
            print(about_str)

        results: float = LogisticDetection.compute(*self.get_unified_subsets())

        if normalize:
            results = LogisticDetection.normalize(results)

        return {"logistic_detection": {"statistic": results, "pvalue": np.nan}}

    # pylint: disable=invalid-name
    def logistic_detection_custom(  # pylint: disable=too-many-locals, too-many-branches
        self,
        normalize: bool = False,
        score_type: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """Calculates a measure of similarity using fitted logistic regression
        to predict reference or registered label using Synthetic Data
        Vault package. Optional `score_type` and `seed` can be passed to provide
        interpretable metrics for the user.

        `score_type` can be:
            None: defaults to scoring of `logistic_detection` method.
            "f1": Cross-validated F1 score with  0.5 threshold.
            "roc_auc": Cross-validated receiver operating characteristic (area
                under the curve)

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
        unified_ref_subset, unified_reg_subset = self.get_unified_subsets()

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

        return {results_key: {"statistic": results, "pvalue": np.nan}}

    # TODO: add test for this method if developed further pylint: disable=fixme
    def binary_classifier_efficacy(
        self,
        target_variable: str,
        clf: Union[
            BinaryAdaBoostClassifier,
            BinaryDecisionTreeClassifier,
            BinaryLogisticRegression,
            BinaryMLPClassifier,
        ] = BinaryLogisticRegression,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """Calculates accuracy of classifier trained on reference data and
        tested on registered data.

        Args:
            target_variable (str): Target (ground truth label) variable name.
            clf (Union[BinaryAdaBoostClassifier, BinaryDecisionTreeClassifier,
                BinaryLogisticRegression, BinaryMLPClassifier]): SDV binary classifier class.
            verbose (bool): Boolean for verbose output to stdout.

        Returns:
            results (Dict[str, Dict[str, float]]): Score providing an overall similarity measure of
                reference and registered dataset.
        """
        method = f"Binary classification (ML efficacy): ({clf.__str__})"
        description = (
            "Efficacy metric using accuracy of classifier trained on "
            "reference dataset and tested on registered dataset."
        )
        about_str = self._format_about_str(method=method, description=description)

        if verbose:
            print(about_str)

        result: float = clf.compute(
            test_data=self.registered_dataset.features,
            train_data=self.reference_dataset.features,
            target=target_variable,
            metadata=None,
        )

        return {"binary_classifier_efficacy": {"statistic": result, "pvalue": np.nan}}

    # pylint: enable=invalid-name

    def get_boundary_adherence(self)->Dict:
        results:Dict = self._calc(BoundaryAdherence.compute, wrapper=Wrapper.TYPE_SDMETRIC)
        return results

    def get_range_coverage(self)->Dict:
        results:Dict = self._calc(RangeCoverage.compute, wrapper=Wrapper.TYPE_SDMETRIC)
        return results
