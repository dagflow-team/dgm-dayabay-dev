from itertools import product

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from numpy.typing import NDArray
from yaml import add_representer

from dagflow.bundles.load_hist import load_hist
from dagflow.core import NodeStorage
from dagflow.parameters import Parameter


class FFormatter(ticker.ScalarFormatter):

    def __init__(self, fformat="%1.1f", offset=True, mathText=True, *args, **kwargs):
        self.fformat = fformat
        ticker.ScalarFormatter.__init__(
            self, useOffset=offset, useMathText=mathText, *args, **kwargs
        )

    def _set_format(self):
        self.format = self.fformat


add_representer(
    np.ndarray,
    lambda representer, obj: representer.represent_str(np.array_repr(obj)),
)


def update_dict_parameters(
    dict_parameters: dict[str, Parameter],
    groups: list[str],
    model_parameters: NodeStorage,
) -> None:
    """Update dictionary of minimization parameters

    Parameters
    ----------
        dict_parameters : dict[str, Parameter])
            Dictionary of parameters
        groups : list[str]
            List of groups of parameters to be added to dict_parameters
        model_parameters : NodeStorage
            storage of model parameters

    Returns
    -------
    None
    """
    for group in groups:
        dict_parameters.update(
            {
                f"{group}.{path}": parameter
                for path, parameter in model_parameters[group].walkjoineditems()
            }
        )


def load_model_from_file(
    filename: str, node_name: str, name_pattern: str, groups: list[str]
) -> NodeStorage:
    """Update dictionary of minimization parameters

    Parameters
    ----------
        filename : str
            Path to file that contains model observations
        node_name : str
            Name of node where outputs model observations will be stored
        name_pattern : str
            Pattern uses two placeholders: for detector and for item from `groups`
        groups : list[str]
            list of groups to be added to NodeStorage

    Returns
    -------
    NodeStorage
        Storage that contains model observations
    """
    comparison_storage = load_hist(
        name=node_name,
        x="erec",
        y="fine",
        merge_x=True,
        filenames=filename,
        replicate_outputs=list(
            product(["AD11", "AD12", "AD21", "AD22", "AD31", "AD32", "AD33", "AD34"], groups)
        ),
        skip=({"AD22", "6AD"}, {"AD34", "6AD"}, {"AD11", "7AD"}),
        name_function=lambda _, idx: name_pattern.format(*idx),
    )
    return comparison_storage["outputs"]


def filter_fit(src: dict, keys_to_fiter: list[str]) -> None:
    """Remove keys from fit dictionary

    Parameters
    ----------
        src : dict
            Dictionary of fit
        keys_to_filter : list[str]
            List of keys to be deleted from fit dictionary

    Returns
    -------
    None
    """
    keys = list(src.keys())
    for key in keys:
        if key in keys_to_fiter:
            del src[key]
            continue
        if isinstance(src[key], dict):
            filter_fit(src[key], keys_to_fiter)


def convert_numpy_to_lists(src: dict) -> None:
    """Convert recursively numpy array in dictionary

    Parameters
    ----------
        src : dict
            Dictionary that may contains numpy arrays as value

    Returns
    -------
    None
    """
    for key, value in src.items():
        if isinstance(value, np.ndarray):
            src[key] = value.tolist()
        elif isinstance(value, dict):
            convert_numpy_to_lists(value)


def calculate_ratio_error(data_a: NDArray | float, data_b: NDArray | float) -> NDArray | float:
    r"""Calculate error of ratio of two observables
    .. math::
        \sigma\left(\frac{a}{b}\right) = \sqrt{\left(\frac{\sigma_a}{b}\right)^2 + \left(\frac{\sigma_b}{b^2}\right)^2} =
        = \frac{1}{b}\sqrt{\frac{a}{b}\left(a + b\right)}

    Parameters
    ----------
        data_a : NDArray
            Numerator
        data_b : NDArray
            Denominator

    Returns
    -------
    NDArray
        Error of ratio
    """
    ratio = data_a / data_b
    return 1 / data_b * (ratio * (data_a + data_b)) ** 0.5


def calculate_difference_error(data_a: NDArray | float, data_b: NDArray | float) -> NDArray | float:
    r"""Calculate error of difference of two observables
    .. math::
        \sigma\left(a - b\right) = \sqrt{\left(\sigma_a\right)^2 + \left(\sigma_b\right)^2} =
        = \sqrt{a + b\right)}

    Parameters
    ----------
        data_a : NDArray
            First operand
        data_b : NDArray
            Second operand

    Returns
    -------
    NDArray
        Error of difference
    """
    return (data_a + data_b) ** 0.5


def plot_spectra_ratio(
    data_a: NDArray,
    data_b: NDArray,
    edges: NDArray,
    title: str,
    plot_diff: bool = False,
    label_a: str = "A: fit",
    label_b: str = "B: data",
    legend_title: str = "",
    ylim_ratio: tuple[float] | tuple = (),
) -> None:
    """

    Parameters
    ----------
        data_a : NDArray
            Observation of model
        data_b : NDArray
            (Pseudo-)data
        edges : NDArray
            Edges of bins where data_a and data_b are determined
        title : str
            Title for plot
        plot_diff : bool
            Plot difference of data_a and data_b
        label_a : str
            Label for data_a
        label_b : str
            Label for data_b
        ylim_ratio : tuple[float] | tuple[None]
            Limits for y-axis of ratio plot

    Returns
    -------
    None
    """
    centers = (edges[1:] + edges[:-1]) / 2
    xerrs = (edges[1:] - edges[:-1]) / 2
    if plot_diff:
        fig, axs = plt.subplots(3, 1, height_ratios=[2, 1, 1], sharex=True)
    else:
        fig, axs = plt.subplots(2, 1, height_ratios=[2, 1], sharex=True)
    axs[0].step([edges[0], *edges], [0, *data_a, 0], where="post", label=label_a)
    axs[0].errorbar(
        centers, data_b, yerr=data_b**0.5, marker="o", markersize=4, linestyle="none", label=label_b
    )
    axs[1].errorbar(
        centers,
        data_a / data_b - 1,
        yerr=calculate_ratio_error(data_a, data_b),
        xerr=xerrs,
        marker="o",
        markersize=4,
        linestyle="none",
    )
    if plot_diff:
        axs[2].errorbar(
            centers,
            data_a - data_b,
            yerr=calculate_difference_error(data_a, data_b),
            xerr=xerrs,
            marker="o",
            markersize=4,
            linestyle="none",
        )
    axs[0].set_title(title)
    formatter = FFormatter()
    formatter.set_powerlimits((0, 2))
    axs[0].yaxis.set_major_formatter(formatter)
    axs[0].legend(title=legend_title)
    if plot_diff:
        axs[2].set_xlabel("E, MeV")
        axs[2].set_ylabel("A - B")
    else:
        axs[1].set_xlabel("Reconstructed energy [MeV]")
    axs[0].set_ylabel("Entries")
    # axs[1].yaxis.tick_right()
    axs[1].tick_params(left=True, right=True, labelleft=False, labelright=True)
    axs[1].yaxis.set_label_position("left")
    axs[1].set_ylabel("A / B - 1")
    # axs[1].yaxis.tick_left()
    if ylim_ratio:
        axs[1].set_ylim(ylim_ratio)
    plt.setp(axs[0].get_xticklabels(), visible=False)


def calc_box_around(
    xy_point: tuple[float, float], xy_errors: tuple[float, float], factor: float = 0.1
) -> tuple[list[float], list[float]]:
    x, y = xy_point
    xerr, yerr = xy_errors
    return (
        [
            x - xerr * factor,
            x - xerr * factor,
            x + xerr * factor,
            x + xerr * factor,
            x - xerr * factor,
        ],
        [
            y - yerr * factor,
            y + yerr * factor,
            y + yerr * factor,
            y - yerr * factor,
            y - yerr * factor,
        ],
    )


def plot_spectral_weights(edges, fit) -> None:
    """

    Parameters
    ----------
        data_a : NDArray
            Observation of model
        data_b : NDArray
            (Pseudo-)data
        edges : NDArray
            Edges of bins where data_a and data_b are determined
        title : str
            Title for plot
        ylim_ratio : tuple[float] | tuple[None]
            Limits for y-axis of ratio plot

    Returns
    -------
    None
    """
    data = []
    yerrs = []
    for key in filter(lambda key: "spec" in key, fit["names"]):
        data.append(fit["xdict"][key])
        yerrs.append(fit["errorsdict"][key])
    plt.figure()
    plt.hlines(0, 0, 13, color="black", alpha=0.75)
    plt.errorbar(edges, data, xerr=0.1, yerr=yerrs, linestyle="none")
    plt.title(r"Correction to $\overline{\nu}_{e}$ spectrum")
    plt.xlabel(r"$E_{\nu}$, MeV")
    plt.ylabel("Correction")
    plt.xlim(1.5, 12.5)
