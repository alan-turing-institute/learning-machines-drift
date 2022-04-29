from learning_machines_drift.types import Dataset
from typing import Any, Callable
from scipy import stats
import numpy.typing as npt
import numpy as np
# from sdmetrics.single_table import LogisticDetection
from sdmetrics.single_table import GMLogLikelihood

class HypothesisTests:
    def __init__(self, reference_dataset: Dataset, registered_dataset: Dataset):
        self.reference_dataset = reference_dataset
        self.registered_dataset = registered_dataset

    def _calc(self, f: Callable[[npt.ArrayLike, npt.ArrayLike], Any]) -> Any:

        results = {}
        for feature in self.reference_dataset.feature_names:
            ref_col = self.reference_dataset.features[feature]
            reg_col = self.registered_dataset.features[feature]
            results[feature] = f(ref_col, reg_col)

        return results

    def kolmogorov_smirnov(self) -> Any:

        return self._calc(stats.ks_2samp)

    @staticmethod
    def _chi_square(data1: npt.ArrayLike, data2: npt.ArrayLike) -> Any:

        d1_unique, d1_counts = np.unique(data1, return_counts=True)
        d2_unique, d2_counts = np.unique(data1, return_counts=True)

        return stats.chisquare(d1_counts, d2_counts)
    
    def sdv_evaluate(self) -> Any:
        # return LogisticDetection.compute(self.reference_dataset.unify(), self.registered_dataset.unify())
        return GMLogLikelihood.compute(self.reference_dataset.unify(), self.registered_dataset.unify())
