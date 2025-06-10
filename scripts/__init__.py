from itertools import product, zip_longest

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from numpy.typing import NDArray
from yaml import add_representer
from yaml import safe_load as yaml_load

from dagflow.bundles.load_hist import load_hist
from dagflow.core import NodeStorage
from dagflow.parameters import Parameter
from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer


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


def do_fit(minimizer: IMinuitMinimizer, model, is_iterative: bool = False) -> dict:
    fit = minimizer.fit()
    if is_iterative:
        for _ in range(4):
            model.next_sample(mc_parameters=False, mc_statistics=False)
            fit = minimizer.fit()
            if not fit["success"]:
                break
    return fit


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
    axs[0].step([edges[0], *edges], [0, *data_a, 0], where="post", label=label_a, color="C1")
    axs[0].errorbar(
        centers,
        data_b,
        yerr=data_b**0.5,
        marker="o",
        markersize=4,
        linestyle="none",
        label=label_b,
        color="C0",
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
    axs[0].legend(title=legend_title, loc="upper right")
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
        # yerrs.append(fit.get("errorsdict_profiled", fit["errorsdict"])[key])
        yerrs.append(fit["errorsdict"][key])
    plt.figure()
    plt.hlines(0, 0, 13, color="black", alpha=0.75)
    plt.errorbar(edges, data, xerr=0.1, yerr=yerrs, linestyle="none")
    plt.title(r"Correction to $\overline{\nu}_{e}$ spectrum")
    plt.xlabel(r"$E_{\nu}$, MeV")
    plt.ylabel("Correction")
    plt.xlim(1.5, 12.5)


def plot_fit_2d(
    fit_path: str,
    compare_fit_paths: list[str],
    xlim: tuple[float] | None = None,
    ylim: tuple[float] | None = None,
    label_a: str | None = None,
    labels_b: list[str] = [],
    title_legend: str | None = None,
    add_box: bool = False,
    dashed_comparison: bool = False,
    add_global_normalization: bool = False,
    add_nsigma_legend: bool = True,
):
    if add_global_normalization:
        fig, (ax, axgn) = plt.subplots(
            1,
            2,
            width_ratios=(4, 1),
            gridspec_kw={
                "wspace": 0,
            },
            subplot_kw={},
        )
    else:
        fig, ax, = plt.subplots(1, 1)
        axgn = None

    with open(fit_path, "r") as f:
        fit = yaml_load(f)

    xdict = fit["xdict"]
    errorsdict = fit.get("errorsdict_profiled", fit["errorsdict"])

    dm_value, dm_error_left, dm_error_right, _ = get_parameter_fit(xdict, errorsdict, "oscprob.DeltaMSq32")
    sin_value, sin_error_left, sin_error_right, _ = get_parameter_fit(xdict, errorsdict, "oscprob.SinSq2Theta13")

    ax.errorbar(
        sin_value,
        dm_value,
        xerr=[[sin_error_left], [sin_error_right]],
        yerr=[[dm_error_left], [dm_error_right]],
        label=label_a,
    )
    if add_box:
        label = r"$0.1\sigma$ " + label_a if label_a else r"$0.1\sigma$"
        ax.axvspan(
            sin_value - 0.1 * sin_error_left,
            sin_value + 0.1 * sin_error_right,
            -10,
            10,
            color="0.9",
            label=label,
        )
        ax.axhspan(dm_value - 0.1 * dm_error_left, dm_value + 0.1 * dm_error_right, -10, 10, color="0.9")

    if axgn:
        axgn.yaxis.set_label_position("right")
        axgn.set_ylabel("Normalization offset")
        axgn.tick_params(labelleft=False, labelright=True, labelbottom=False)
        axgn.grid(axis="x")
        axgn.set_ylim(-0.15, 0.075)

        gn_value, gn_error_left, gn_error_right, gn_type = get_parameter_fit(
            xdict, errorsdict, "detector.global_normalization"
        )
        axgn.errorbar(
            0,
            gn_value - 1,
            yerr=[[gn_error_left], [gn_error_right]],
            xerr=1,
            fmt="o",
            markerfacecolor="none",
            label=gn_type,
        )

    nsigma_legend = None

    for i, (compare_fit_path, label_b) in enumerate(
        zip_longest(compare_fit_paths, labels_b, fillvalue=None)
    ):
        with open(compare_fit_path, "r") as f:
            compare_fit = yaml_load(f)

        compare_xdict = compare_fit["xdict"]
        compare_errorsdict = compare_fit.get("errorsdict_profiled", compare_fit["errorsdict"])

        sin_value_c, sin_error_left_c, sin_error_right_c, _ = get_parameter_fit(
            compare_xdict, compare_errorsdict, "oscprob.SinSq2Theta13"
        )
        dm_value_c, dm_error_left_c, dm_error_right_c, _ = get_parameter_fit(
            compare_xdict, compare_errorsdict, "oscprob.DeltaMSq32"
        )

        eb = ax.errorbar(
            sin_value_c,
            dm_value_c,
            xerr=[[sin_error_left_c], [sin_error_right_c]],
            yerr=[[dm_error_left_c], [dm_error_right_c]],
            label=label_b,
            capsize=4,
        )

        # if dashed_comparison:
        #     eb[2][0].set_linestyle("--")
        #     eb[2][1].set_linestyle("--")

        if axgn:
            gn_value_c, gn_error_c_left, gn_error_c_right, gn_type_c = get_parameter_fit(
                compare_xdict, compare_fit["errorsdict"], "detector.global_normalization"
            )
            xoffset = (i + 1) / 10.0
            axgn.errorbar(
                xoffset,
                gn_value_c - 1,
                yerr=[[gn_error_c_left], [gn_error_c_right]],
                xerr=1,
                fmt="o",
                markerfacecolor="none",
                label=gn_type_c,
            )

        if add_nsigma_legend and not nsigma_legend:
            dm_error = (dm_error_right + dm_error_left) / 2
            sin_error = (sin_error_right + sin_error_left) / 2
            gn_error = (gn_error_right + gn_error_left) / 2
            labels = [
                r"$\sin^2 2\theta_{13} = " + f"{(sin_value - sin_value_c) / sin_error * 100:1.3f}$",
                r"$\Delta m^2_{32} = " + f"{(dm_value - dm_value_c) / dm_error * 100:1.3f}$",
            ]
            if gn_error_c_left:
                labels.append(
                    r"$N^{\text{global}} = " + f"{(gn_value - gn_value_c) / gn_error * 100:1.3f}$"
                )
            handles = [plt.Line2D([], [], color="none", label=label) for label in labels]
            nsigma_legend = plt.legend(
                handles=handles,
                title=r"$n\sigma$ difference, %",
                loc="lower right",
                handlelength=0,
                handletextpad=0,
                bbox_to_anchor=(0, 0),
            )

    ax.legend(title=title_legend, loc="upper right")
    ax.set_xlabel(r"$\sin^22\theta_{13}$")
    ax.set_ylabel(r"$\Delta m^2_{32}$ [eV$^2$]")
    ax.set_title("")
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 2))
    ax.yaxis.set_major_formatter(formatter)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if nsigma_legend:
        plt.gca().add_artist(nsigma_legend)

    plt.subplots_adjust(left=0.17, right=0.86, bottom=0.1, top=0.95)


def get_parameter_fit(xdict: dict, errorsdict: dict, key: str) -> tuple[float, float, float, str]:
    try:
        if isinstance(errorsdict[key], float):
            return xdict[key], errorsdict[key], errorsdict[key], "fit"
        return xdict[key], -1 * errorsdict[key][0], errorsdict[key][1], "fit"
    except KeyError:
        pass

    names = [name for name in xdict if name.startswith("neutrino_per_fission_factor.spec_scale")]
    scale = np.array([xdict[name] for name in names])
    unc = np.array([errorsdict[name] for name in names])
    w = unc**-2
    wsum = w.sum()
    res = (scale * w).sum() / wsum
    # res_unc = wsum**-0.5 # incorrect since scales are correlated

    return 1 + res, 0.0, 0.0, "calc"
