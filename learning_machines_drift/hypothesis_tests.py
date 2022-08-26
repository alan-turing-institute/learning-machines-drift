"""Class for scoring drift between reference and registered datasets."""

import textwrap
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from sdmetrics.errors import IncomputableMetricError
from sdmetrics.single_table import CSTest, GMLogLikelihood
from sdmetrics.single_table import KSComplement as KSTest
from sdmetrics.single_table import LogisticDetection
from sdmetrics.utils import HyperTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from learning_machines_drift.types import Dataset


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
        results = {}
        for feature in self.reference_dataset.feature_names:
            if subset is not None:
                if feature not in subset:
                    continue
            ref_col = self.reference_dataset.features[feature]
            reg_col = self.registered_dataset.features[feature]

            result = func(ref_col, reg_col)
            results[feature] = self._to_dict(result)
        return results

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

    @staticmethod
    def _chi_square(data1: pd.Series, data2: pd.Series) -> dict[str, float]:
        """Perform a chi-square test on two category-like series.

        Args:
            data1 (pd.Series): First series.
            data2 (pd.Series): Second series.
        Returns:
            dict[str, float]: dict of chi-square statistic and p-value.
        """
        # Get unique elements across all data
        base: npt.NDArray[Any] = np.unique(np.append(data1, data2))
        # Get counts of values in data1
        d1_counter: Counter[Any] = Counter(data1)
        # Get counts of values in data2
        d2_counter: Counter[Any] = Counter(data2)
        # Get counts in order of base for both counters
        d1_counts: List[int] = [d1_counter[el] for el in base]
        d2_counts: List[int] = [d2_counter[el] for el in base]
        # Calculate chi-square
        statistic, pvalue, _, _ = stats.chi2_contingency(
            np.stack([d1_counts, d2_counts])
        )
        return {"statistic": statistic, "pvalue": pvalue}

    def scipy_chisquare(self, verbose: bool = True) -> Any:
        """Calculates feature-wise chi-square statistic and p-value for
        the hypothesis test of independence of the observed frequencies.
        Provides a test for the independence of two count distributions.
        Assumes categorical underlying distributions.

        Args:
            verbose (bool): Boolean for verbose output to stdout.

        Returns:
            results (dict): Dictionary of statistics and  p-values by feature.
        """
        method = (
            "SciPy chi-square test of independence of variables in a "
            "contingency table."
        )
        description = (
            "Chi-square test for categorical-like data comparing counts in "
            "registered and reference data."
        )
        about_str = self._format_about_str(method=method, description=description)

        results = self._calc(
            self._chi_square,
            subset=self._get_categorylike_features(self.registered_dataset.features),
        )
        if verbose:
            print(about_str)
        return results

    def scipy_permutation(
        self,
        func: Callable[..., float] = np.mean,
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
        method = f"SciPy Permutation Test (test function: {func.__name__})"
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
            return func(lhs, axis=axis) - func(rhs, axis=axis)

        results = {}
        for feature in self.reference_dataset.feature_names:
            ref_col = self.reference_dataset.features[feature]
            reg_col = self.registered_dataset.features[feature]
            result = stats.permutation_test(
                (ref_col, reg_col),
                statistic,
                permutation_type="independent",
                alternative="two-sided",
                n_resamples=9999,
                random_state=self.random_state,
            )
            results[feature] = self._to_dict(result)
        if verbose:
            print(about_str)

        return results

    def sdv_kolmogorov_smirnov(
        self, normalize: bool = False, verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """Calculates Synthetic Data Vault package version of the
        Kolmogorov-Smirnov (KS) two-sample test.

        Args:
            verbose (bool): Boolean for verbose output to stdout.
            normalize (bool): Normalize raw_score to interval [0, 1].

        Returns:
            results (Dict[str, Dict[str, float]]): 1 - the mean KS
            statistic across features.
        """
        method = f"SDV Kolmogorov Smirnov (normalize: {normalize})"
        description = (
            "This metric uses the two-sample Kolmogorov–Smirnov test to "
            "compare the distributions of continuous columns using the "
            "empirical CDF. The output for each column is 1 minus the KS "
            "Test D statistic, which indicates the maximum distance "
            "between the expected CDF and the observed CDF values."
        )
        about_str = self._format_about_str(method=method, description=description)

        if verbose:
            print(about_str)

        # Only run computation on features; exclude labels
        results: float = KSTest.compute(
            self.reference_dataset.features, self.registered_dataset.features
        )

        if normalize:
            results = KSTest.normalize(results)

        return {"sdv_kolmogorov_smirnov": {"statistic": results, "pvalue": np.nan}}

    # Skip the ones that are not calculable
    @staticmethod
    def _get_categorylike_features(data: pd.DataFrame) -> List[str]:
        """Get a list of feature names that have category-like features.
        Category-like features are defined as:
            - Unit or Binary (less than two values)
            OR
            - Categorical (category dtype)

        Args:
            data (pd.DataFrame): data to be have features checked.

        Returns:
            List[str]: List of feature names that are category-like.
        """
        # Get the number of unique values by feature
        nunique: pd.Series = data.nunique()
        # Unit or binary features
        unit_or_bin_features: List[str] = nunique[nunique <= 2].index.to_list()
        # Integer or category features
        cat_features: List[str] = data.dtypes[
            data.dtypes.eq("category")
        ].index.to_list()
        # Get list of unique features for output
        out_features: List[str] = list(np.unique(unit_or_bin_features + cat_features))

        return out_features

    def subset_to_categories(self, data: pd.DataFrame) -> pd.DataFrame:
        """Subset a dataframe to only category-like features.

        Args:
            data (pd.DataDrame): Input dataframe
        Returns:
            pd.DataDrame: Output dataframe subsetted to category-like
            features and category type.

        """
        out_features = self._get_categorylike_features(data)
        return data[out_features].astype("category")

    def sdv_chisquare(
        self, normalize: bool = False, verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """Calculates average chi-square statistic and p-value for
        the hypothesis test of independence of the observed frequencies for
        categorical features using Synthetic Data Vault.

        Args:
            verbose (bool): Boolean for verbose output to stdout.
            normalize (bool): Normalize raw_score to interval [0, 1].

        Returns:
            results (Optional[Dict[str, Dict[str, float]]]): Dictionary of
            statistics and  p-values by feature.
        """
        method = f"SDV CS Test (normalize: {normalize})"
        description = (
            "This metric uses the Chi-Squared test to compare the "
            "distributions of two discrete columns, with the mean score taken "
            "across categorical and boolean columns."
        )
        about_str = self._format_about_str(method=method, description=description)

        # Convert to dataframes with catgeory type for CSTest compatibility
        ref_cat = self.subset_to_categories(self.reference_dataset.features)
        reg_cat = self.subset_to_categories(self.registered_dataset.features)

        # Print about_str
        if verbose:
            print(about_str)

        # Score only computable if non-zero number of columns
        try:
            results: float = CSTest.compute(ref_cat, reg_cat)
            if normalize:
                results = CSTest.normalize(results)
            return {"sdv_chisquare": {"statistic": results, "pvalue": np.nan}}

        except IncomputableMetricError:
            return {"sdv_chisquare": {}}

    def gaussian_mixture_log_likelihood(
        self, verbose: bool = True, normalize: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """Calculates the log-likelihood of reference data given Gaussian
        Mixture Model (GMM) fits on the reference data using Synthetic Data
        Vault package.

        Args:
            verbose (bool): Boolean for verbose output to stdout.
            normalize (bool): Normalize raw_score to interval [0, 1].

        Returns:
            results (Dict[str, Dict[str, float]]): Log-likelihood of reference
            data with fitted model that has lowest Bayesian Information
            Criterion (BIC).
        """
        method: str = f"Gaussian Mixture Log Likelihood (normalize: {normalize})"
        description: str = (
            "This metric fits multiple GaussianMixture models to the real "
            "data and then evaluates the average log likelihood of the "
            "synthetic data on them."
        )
        about_str = self._format_about_str(method=method, description=description)

        if verbose:
            print(about_str)

        results: float = GMLogLikelihood.compute(
            self.reference_dataset.unify(), self.registered_dataset.unify()
        )
        if normalize:
            results = GMLogLikelihood.normalize(results)

        return {
            "gaussian_mixture_log_likelihood": {"statistic": results, "pvalue": np.nan}
        }

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

        results: float = LogisticDetection.compute(
            self.reference_dataset.unify(), self.registered_dataset.unify()
        )

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

        # Transform data for fitting using SD metrics HyperTransformer
        ht = HyperTransformer()
        transformed_reference_data = ht.fit_transform(
            self.reference_dataset.unify()
        ).to_numpy()
        transformed_registered_data = ht.transform(
            self.registered_dataset.unify()
        ).to_numpy()

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

    # pylint: enable=invalid-name

    # def sd_evaluate(self, verbose=True) -> Any:
    #     method = "SD Evaluate"
    #     description = "Detection metric based on a LogisticRegression
    #     classifier from scikit-learn"
    #     about_str = "\nMethod: {method}\nDescription:{description}"
    #     about_str = about_str.format(method=method, description=description)

    #     if verbose:
    #         print(about_str)
    #     results = evaluate(
    #         self.reference_dataset.unify(),
    #         self.registered_dataset.unify(),
    #         aggregate=False,
    #     )
    #     return results
