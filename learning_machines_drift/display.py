"""Class for scoring drift between reference and registered datasets."""

# import textwrap
# from collections import Counter
# from functools import partial
# from typing import Any, Callable, Dict, List, Optional, Tuple

# import numpy as np
# import numpy.typing as npt
# import pandas as pd
# from scipy import stats
# from sdmetrics.single_table import GMLogLikelihood, LogisticDetection
# from sdmetrics.utils import HyperTransformer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import f1_score, roc_auc_score
# from sklearn.model_selection import StratifiedKFold

# from learning_machines_drift.types import Dataset

from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# import numpy.typing as npt


class Display:
    """A class for converting a dictionary of hypothesis test scores to displayed output."""

    def __init__(self) -> None:
        """Initialize with registered and reference and optional seed."""

    @classmethod
    def plot(
        cls,
        score_dict: Dict[str, Dict[str, float]],
        score_type: str = "pvalue",
        score_name: str = "KS_test",
    ) -> Tuple[plt.Figure, Any]:
        """Plot method for displaying a set of scores on a subplot grid.

        Args:
            score_dict (Dict[str, Dict[str, float]]): Dictionary of scores from
                a hypothesis test output.
            score_type (str): Either "statistic" or "pvalue".
            together (bool): Whether to plot on same subplot.

        Returns:
            Tuple[plt.Figure, Any]: tuple of fig and subplot array.
        """
        fig: plt.Figure[...] = plt.figure(figsize=(5, 4))
        axs: plt.Axes[...] = fig.subplots(1, 1, squeeze=False)
        x_vals, y_vals, xticklabels, colors = [], [], [], []
        for i, (key, scores) in enumerate(score_dict.items()):
            xticklabels.append(key)
            x_vals.append(i)
            y_vals.append(scores[score_type])
            colors.append(f"C{i}")

        ax = axs[0, 0]
        ax.scatter(x_vals, y_vals, marker="o", label=score_name, color=colors)
        ax.set(
            xlabel="Variable",
            ylabel=score_type,
            title=score_name,
            xticks=x_vals,
        )
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")
        return fig, axs

    @classmethod
    def table(
        cls, score_dict: Dict[str, Dict[str, float]], verbose: bool = True
    ) -> pd.DataFrame:
        """Gets a pandas dataframe and optionally prints a table of hypothesis
        test results.

        Args:
            score_dict (Dict[str, Dict[str, float]]): Dictionary of scores
                from a hypothesis test output.

        Returns:
            pd.DataFrame
        """
        df: pd.DataFrame = pd.DataFrame.from_dict(score_dict, orient="index")
        if verbose:
            print(df.to_markdown())
        return df
