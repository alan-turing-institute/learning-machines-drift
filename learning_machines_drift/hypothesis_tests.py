"""TODO PEP 257"""

from collections import Counter
from typing import Any, Callable, List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from sdmetrics.errors import IncomputableMetricError
from sdmetrics.single_table import CSTest, GMLogLikelihood, KSTest, LogisticDetection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# from sdmetrics.utils import HyperTransformer
from learning_machines_drift.hypertransformer import HyperTransformer
from learning_machines_drift.types import Dataset


# TODO: write function for standard formatting of description strings #pylint: disable=fixme
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
        """
        Initialize with registered and reference and optional seed.
        """
        self.reference_dataset = reference_dataset
        self.registered_dataset = registered_dataset
        self.random_state = random_state

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
                    results[feature] = {"statistic": np.nan, "pvalue": np.nan}
                    continue
            ref_col = self.reference_dataset.features[feature]
            reg_col = self.registered_dataset.features[feature]
            results[feature] = func(ref_col, reg_col)
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
        about_str = "\nMethods: {method}\nDescription:{description}"
        about_str = about_str.format(method=method, description=description)

        results = self._calc(stats.ks_2samp)
        if verbose:
            print(about_str)
        return results

    def scipy_mannwhitneyu(self, verbose=True) -> Any:  # type: ignore
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
        about_str = "\nMethods: {method}\nDescription:{description}"
        about_str = about_str.format(method=method, description=description)

        results = self._calc(stats.mannwhitneyu)
        if verbose:
            print(about_str)
        return results

    def scipy_chisquare(self, verbose=True) -> Any:  # type: ignore
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
            "SciPy chi-square test of independence of variables in a"
            "contingency table."
        )
        description = """
        Chi-square test for categorical-like data comparing counts in
        registered and reference data."""
        about_str = "\nMethods: {method}\nDescription:{description}"
        about_str = about_str.format(method=method, description=description)

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
        method = "SciPy Permutation Test"
        description = """
            Performs permutation test on all features with passed stat_fn measuring
            the difference between samples.
        """
        about_str = "\nMethods: {method}\nDescription:{description}"
        about_str = about_str.format(method=method, description=description)

        # Statistic for evaluating the difference between permuted samples
        def statistic(
            lhs: npt.ArrayLike,
            rhs: npt.ArrayLike,
            axis: int = 0,
        ) -> float:
            return func(lhs, axis=axis) - func(rhs, axis=axis)

        results = {}
        for feature in self.reference_dataset.feature_names:
            ref_col = self.reference_dataset.features[feature]
            reg_col = self.registered_dataset.features[feature]
            results[feature] = stats.permutation_test(
                (ref_col, reg_col),
                statistic,
                permutation_type="independent",
                alternative="two-sided",
                n_resamples=9999,
                random_state=self.random_state,
            )
        if verbose:
            print(about_str)

        return results

    def sdv_kolmogorov_smirnov(self, verbose=True) -> Any:  # type: ignore
        """Calculates Synthetic Data Vault package version of the
        Kolmogorov-Smirnov (KS) two-sample test.

        Args:
            verbose (bool): Boolean for verbose output to stdout.

        Returns:
            results (float): 1 - the mean KS statistic across features.
        """
        method = "SDV Kolmogorov Smirnov"
        description = """This metric uses the two-sample Kolmogorov–Smirnov
                        test to compare the distributions
                        of continuous columns using the empirical CDF.
                        \n The output for each column is 1 minus the KS
                        Test D statistic, which indicates the
                        maximum distance between the expected CDF and the
                        observed CDF values."""
        about_str = "\nMethod: {method}\nDescription:{description}"
        about_str = about_str.format(method=method, description=description)

        # Only run computation on features; exclude labels
        results = KSTest.compute(
            self.reference_dataset.features, self.registered_dataset.features
        )

        if verbose:
            print(about_str)
        return results

    @staticmethod
    def _get_categorylike_features(data: pd.DataFrame) -> List[str]:
        """TODO PEP 257"""
        # Get the number of unique values by feature
        nunique: pd.Series = data.nunique()
        # Unit or binary features
        bin_features: List[str] = nunique[nunique <= 2].index.to_list()
        # Integer or category features
        int_or_cat_features: List[str] = data.dtypes[
            data.dtypes.eq(int) | data.dtypes.eq("category")
        ].index.to_list()
        out_features: List[str] = list(np.unique(bin_features + int_or_cat_features))
        return out_features

    def subset_to_categories(self, data: pd.DataFrame) -> pd.DataFrame:
        """Subset a dataframe to only categorical-like features. Convert to
        categorical columns if feature is:
            - Binary (two values)
            OR
            - Interger (int dtype)
            OR
            - Categorical (category dtype)

        Args:
            data (pd.DataDrame): Input dataframe
        Returns:
            pd.DataDrame: Output dataframe subsetted to categorical-like
            features.

        """
        out_features = self._get_categorylike_features(data)
        return data[out_features].astype("category")

    def sdv_cs_test(self, verbose=True) -> Any:  # type: ignore
        """TODO PEP 257"""
        method = "SDV CS Test"
        description = """\nThis metric uses the Chi-Squared test to compare the distributions of
        two discrete columns, with the mean score taken across categorical and
        boolean columns.
        """
        about_str = "\nMethod: {method}\nDescription:{description}"
        about_str = about_str.format(method=method, description=description)

        # Convert to dataframes with catgeory type for CSTest compatibility
        ref_cat = self.subset_to_categories(self.reference_dataset.features)
        reg_cat = self.subset_to_categories(self.registered_dataset.features)

        # Score only computable if non-zero number of columns
        try:
            results = CSTest.compute(ref_cat, reg_cat)

            if verbose:
                print(about_str)

            return results
        except IncomputableMetricError:
            return None

    @staticmethod
    # TODO: fix variable types #pylint: disable=fixme
    def _chi_square(data1: Any, data2: Any) -> Any:
        """TODO PEP 257"""
        base = np.unique(np.append(data1, data2))
        d1_counter = Counter(data1)
        d2_counter = Counter(data2)
        d1_counts = [d1_counter[el] for el in base]
        d2_counts = [d2_counter[el] for el in base]
        statistic, pvalue, _, _ = stats.chi2_contingency(
            np.stack([d1_counts, d2_counts])
        )
        return {"statistic": statistic, "pvalue": pvalue}

    def gaussian_mixture_log_likelihood(self, verbose=True) -> Any:  # type: ignore
        """TODO PEP 257"""
        method: str = "Gaussian Mixture Log Likelihood"
        description = """This metric fits multiple GaussianMixture models
        to the real data and then evaluates
        the average log likelihood of the synthetic data on them."""
        about_str = "\nMethod: {method}\nDescription:{description}"
        about_str = about_str.format(method=method, description=description)

        if verbose:
            print(about_str)

        results = GMLogLikelihood.compute(
            self.reference_dataset.unify(), self.registered_dataset.unify()
        )
        return results

    def logistic_detection(self, verbose=True) -> Any:  # type: ignore
        """TODO PEP 257"""
        method = "Logistic Detection"
        description = "Detection metric based on a LogisticRegression classifier from scikit-learn."
        about_str = "\nMethod: {method}\nDescription:{description}"
        about_str = about_str.format(method=method, description=description)

        if verbose:
            print(about_str)
        results = LogisticDetection.compute(
            self.reference_dataset.unify(), self.registered_dataset.unify()
        )
        return results

    # pylint: disable=invalid-name
    def logistic_detection_custom(self, score_type=None, seed=None, verbose=True) -> Any:  # type: ignore # pylint: disable=too-many-locals
        """TODO PEP 257"""
        if score_type is None:
            method = "Logistic Detection (standard scoring)"
        else:
            method = f"Logistic Detection ({score_type} scoring)"
        description = "Detection metric based on a LogisticRegression classifier from scikit-learn with custom scoring."  # pylint: disable=line-too-long
        about_str = "\nMethod: {method}\nDescription:{description}"
        about_str = about_str.format(method=method, description=description)

        if verbose:
            print(about_str)

        # From: https://github.com/sdv-dev/SDMetrics/blob/master/sdmetrics/single_table/detection/base.py#L69-L91 # pylint: disable=line-too-long
        ht = HyperTransformer()
        transformed_reference_data = ht.fit_transform(  # type: ignore[no-untyped-call]
            self.reference_dataset.unify()
        ).to_numpy()
        transformed_registered_data = ht.transform(  # type: ignore[no-untyped-call]
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
            return 1 - np.mean(scores)

        # Custom metrics assume the mean of the scores
        return np.mean(scores)

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
