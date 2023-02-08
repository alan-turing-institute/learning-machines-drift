"""Class for scoring drift between reference and registered datasets."""

from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from learning_machines_drift.types import StructuredResult


class Display:
    """A class for converting a dictionary of drift scores to displayed output."""

    @classmethod
    def plot(
        cls,
        result: StructuredResult,
        score_name: Optional[str] = None,
        score_type: str = "pvalue",
        alpha: float = 0.05,
    ) -> Tuple[plt.Figure, Any]:
        """Plot method for displaying a set of scores on a subplot grid.

        Args:
            result (StructuredResult): Structured result from a drift
                score measurement.
            score_type (str): Either "statistic" or "pvalue".
            score_name (str): Name of score to be plotted and used as plot title.
            alpha (float): Value of alpha to be used in p-value plots.

        Returns:
            Tuple[plt.Figure, Any]: tuple of fig and subplot array.
        """
        # Set-up plots
        fig: plt.Figure[...] = plt.figure(figsize=(5, 4))
        axs: plt.Axes[...] = fig.subplots(1, 1, squeeze=False)

        # Get lists of values to use in plots
        x_vals, y_vals, xticklabels, colors = [], [], [], []
        for i, (key, scores) in enumerate(result.results.items()):
            try:
                xticklabels.append(key)
                x_vals.append(i)
                y_vals.append(scores[score_type])
                colors.append(f"C{0}")
            except KeyError:
                print(f"'{score_type}' not in result.")
                y_vals.append(np.nan)

        # Plot
        ax = axs[0, 0]
        ax.scatter(x_vals, y_vals, marker="o", label="_no_label_", color=colors)

        if score_type == "pvalue":
            ax.hlines(
                min(x_vals),
                max(x_vals),
                alpha,
                ls=":",
                label=r"$\alpha$" f"={alpha:.3f}",
            )

        # Labels
        ax.set(
            xlabel="Variable",
            ylabel=score_type,
            title=score_name,
            xticks=x_vals,
        )
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")

        if score_type == "pvalue":
            ax.legend(prop={"size": "small"})

        # Return figure and axes
        return fig, axs

    @classmethod
    def table(cls, result: StructuredResult, verbose: bool = True) -> pd.DataFrame:
        """Gets a pandas dataframe and optionally prints a table of results from
        drift scoring.

        Args:
            structured_result (StructuredResult): Structured result from a drift
                score measurement.

        Returns:
            pd.DataFrame: Dataframe of scores.
        """
        # Convert dict to pandas dataframe
        df: pd.DataFrame = pd.DataFrame.from_dict(result.results, orient="index")

        # Print to stdout if verbose
        if verbose:
            print(df.to_markdown())

        # Return the dataframe version
        return df
