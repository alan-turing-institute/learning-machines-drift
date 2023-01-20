"""Class for scoring drift between reference and registered datasets."""

from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd


class Display:
    """A class for converting a dictionary of hypothesis test scores to
    displayed output.

    """

    @classmethod
    def plot(
        cls,
        score_dict: Dict[str, Dict[str, float]],
        score_type: str = "pvalue",
        score_name: str = "KS_test",
        alpha: float = 0.05,
    ) -> Tuple[plt.Figure, Any]:
        """Plot method for displaying a set of scores on a subplot grid.

        Args:
            score_dict (Dict[str, Dict[str, float]]): Dictionary of scores from
                a hypothesis test output.
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
        for i, (key, scores) in enumerate(score_dict.items()):
            xticklabels.append(key)
            x_vals.append(i)
            y_vals.append(scores[score_type])
            # Currently set all colors identical
            colors.append(f"C{0}")

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
    def table(
        cls, score_dict: Dict[str, Dict[str, float]], verbose: bool = True
    ) -> pd.DataFrame:
        """Gets a pandas dataframe and optionally prints a table of hypothesis
        test results.

        Args:
            score_dict (Dict[str, Dict[str, float]]): Dictionary of scores
                from a hypothesis test output.

        Returns:
            pd.DataFrame: Dataframe of scores.
        """
        # Convert dict to pandas dataframe
        df: pd.DataFrame = pd.DataFrame.from_dict(score_dict, orient="index")

        # Print to stdout if verbose
        if verbose:
            print(df.to_markdown())

        # Return the dataframe version
        return df
