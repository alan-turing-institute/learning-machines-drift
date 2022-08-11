"""TODO PEP 257"""

from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from sdmetrics.errors import IncomputableMetricError
from sdmetrics.single_table import CSTest, GMLogLikelihood, KSTest, LogisticDetection

from learning_machines_drift.types import Dataset


# TODO: write function for standard formatting of description strings #pylint: disable=fixme
class HypothesisTests:
    """TODO PEP 257"""

    def __init__(
        self,
        reference_dataset: Dataset,
        registered_dataset: Dataset,
        random_state: Optional[int] = None,
    ) -> None:
        """TODO PEP 257"""
        self.reference_dataset = reference_dataset
        self.registered_dataset = registered_dataset
        self.random_state = random_state

    def _calc(self, func: Callable[[npt.ArrayLike, npt.ArrayLike], Any]) -> Any:
        """TODO PEP 257"""
        results = {}
        for feature in self.reference_dataset.feature_names:
            ref_col = self.reference_dataset.features[feature]
            reg_col = self.registered_dataset.features[feature]
            results[feature] = func(ref_col, reg_col)
        return results

    def scipy_kolmogorov_smirnov(self, verbose=True) -> Any:  # type: ignore
        """TODO PEP 257"""
        method = "SciPy Kolmogorov Smirnov"
        description = ""
        about_str = "\nMethods: {method}\nDescription:{description}"
        about_str = about_str.format(method=method, description=description)

        results = self._calc(stats.ks_2samp)
        if verbose:
            print(about_str)
        return results

    def scipy_mannwhitneyu(self, verbose=True) -> Any:  # type: ignore
        """TODO PEP 257"""
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

    def scipy_permutation(
        self,
        func: Callable[..., float] = np.mean,
        verbose: bool = True,
    ) -> Any:
        """
        Performs permutation test on all features.

        Args:
            func: Function for comparing two samples.
            verbose: Print outputs
        Returns:
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
        """TODO PEP 257"""
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

        # Only run computation on predictors; exclude labels
        results = KSTest.compute(
            self.reference_dataset.features, self.registered_dataset.features
        )

        if verbose:
            print(about_str)
        return results

    def sdv_cs_test(self, verbose=True) -> Any:  # type: ignore
        """TODO PEP 257"""
        method = "SDV CS Test"
        description = """\nThis metric uses the Chi-Squared test to compare the distributions of
        two discrete columns, with the mean score taken across categorical and
        boolean columns.
        """
        about_str = "\nMethod: {method}\nDescription:{description}"
        about_str = about_str.format(method=method, description=description)

        def subset_to_categories(data: pd.DataFrame) -> pd.DataFrame:
            """
            Get only categorical-like features. Convert to categorical columns if:
               - Binary (two values)
               - Interger (int dtype)
               - Categorical (category dtype)
            """
            nunique = data.nunique()
            # Unit or binary features
            bin_features = nunique[nunique <= 2].index.to_list()
            # Integer or category features
            int_or_cat_features = data.dtypes[
                data.dtypes.eq(int) | data.dtypes.eq("category")
            ].index.to_list()
            out_features = np.unique(bin_features + int_or_cat_features)
            return data[out_features].astype("category")

        # Convert to dataframes with catgeory type for CSTest compatibility
        ref_cat = subset_to_categories(self.reference_dataset.features)
        reg_cat = subset_to_categories(self.registered_dataset.features)

        # Score only computable if non-zero number of columns
        try:
            results = CSTest.compute(ref_cat, reg_cat)

            if verbose:
                print(about_str)

            return results
        except IncomputableMetricError:
            return None

    @staticmethod
    def _chi_square(data1: npt.ArrayLike) -> Any:
        """TODO PEP 257"""
        d1_counts = np.unique(data1, return_counts=True)
        d2_counts = np.unique(data1, return_counts=True)

        return stats.chisquare(d1_counts, d2_counts)

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
