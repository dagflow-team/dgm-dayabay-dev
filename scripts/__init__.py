import numpy as np
from numpy.typing import NDArray
from yaml import add_representer
from matplotlib import pyplot as plt
from itertools import product
from dagflow.bundles.load_hist import load_hist
from dagflow.parameters import Parameter
from dagflow.core import NodeStorage


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
                for path, parameter in model_parameters(group).walkjoineditems()
            }
        )


def load_model_from_file(filename: str, node_name: str, name_pattern: str, groups: list[str]) -> NodeStorage:
    """Update dictionary of minimization parameters

    Parameters
    ----------
        filename : str
            Path to file that contains model observations
        node_name : str
            Name of node where outputs model observations will be stored
        name_pattern : str
            Pattern that determines model observations in file,
            pattern uses two placeholders: for detector and for item from `groups`
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


def plot_spectra_ratio_difference(
    data_a: NDArray, data_b: NDArray, edges: NDArray, title: str,
    ylim_ratio: tuple[float] | tuple = (),
) -> None:
    r"""

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
    NDArray
        Error of difference
    """
    centers = (edges[1:] + edges[:-1]) / 2
    xerrs = (edges[1:] - edges[:-1]) / 2
    fig, axs = plt.subplots(3, 1, figsize=(7, 6), height_ratios=[2, 1, 1], sharex=True)
    axs[0].step([edges[0], *edges], [0, *data_a, 0], where="post", label="A: fit")
    axs[0].errorbar(
        centers, data_b, xerr=xerrs, yerr=data_b**0.5, linestyle="none", label="B: data"
    )
    axs[1].errorbar(
        centers,
        data_a / data_b - 1,
        xerr=xerrs,
        yerr=calculate_ratio_error(data_a, data_b),
        linestyle="none",
    )
    axs[2].errorbar(
        centers,
        data_a - data_b,
        xerr=xerrs,
        yerr=calculate_difference_error(data_a, data_b),
        linestyle="none",
    )
    axs[0].set_title(title)
    axs[0].legend()
    axs[2].set_xlabel("E, MeV")
    axs[0].set_ylabel("Entries")
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_label_position("right")
    axs[1].set_ylabel("A / B - 1")
    axs[2].set_ylabel("A - B")
    if ylim_ratio:
        axs[1].set_ylim(ylim_ratio)
    axs[0].minorticks_on()
    plt.setp(axs[0].get_xticklabels(), visible=False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)


def plot_spectral_weights(edges, fit) -> None:
    data = []
    yerrs = []
    for key in filter(lambda key: "spec" in key, fit["names"]):
        data.append(fit["xdict"][key])
        yerrs.append(fit["errorsdict"][key])
    plt.figure()
    plt.errorbar(edges, data, xerr=0.1, yerr=yerrs, linestyle="none")
    plt.title(r"Spectral weights of $\overline{\nu}_{e}$ spectrum")
    plt.xlabel("E, MeV")
    plt.ylabel("value")
    plt.tight_layout()
