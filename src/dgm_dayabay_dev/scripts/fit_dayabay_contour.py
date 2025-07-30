#!/usr/bin/env python
r"""Script for contour plot of best fit value.

Examples
--------
Example of call

.. code-block:: shell

    ./scripts/fit_dayabay_contour.py --version v0e \
      --scan-par oscprob.SinSq2Theta13 0.07 0.1 31 \
      --scan-par oscprob.DeltaMSq32 2.2e-3 2.8e-3 61 \
      --chi2 full.chi2n_covmat \
      --output-contour output/contour.pdf \
      --output-map output/contour.npz
"""
import itertools
from argparse import Namespace
from typing import Any

import numpy as np
from dag_modelling.parameters.gaussian_parameter import Parameter
from dag_modelling.tools.logger import DEBUG as INFO4
from dag_modelling.tools.logger import INFO1, INFO2, INFO3, set_level
from dgm_fit.iminuit_minimizer import IMinuitMinimizer
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.stats import chi2, norm

from ..models import available_models, load_model
from . import update_dict_parameters

set_level(INFO1)

DATA_INDICES = {"model": 0, "loaded": 1}


def convert_sigmas_to_chi2(df: int, sigmas: list[float] | NDArray) -> NDArray:
    """Convert deviation of normal unit distribution N(0, 1) to critical value
    of chi-squared.

    Parameters
    ----------
    df : int
        Degree of freedom of chi-squared distribution.
    sigmas : list[float] | NDArray
        List or array deviations from 0 in terms of standard deviation of normal unit distribution N(0, 1).

    Returns
    -------
    NDArray
        Array of critical values of chi-squared.
    """
    percentiles = 2 * norm(0, 1).cdf(sigmas) - 1
    return chi2(df).ppf(percentiles)


def get_profile_of_chi2(
    parameter_grid: NDArray,
    profile_grid: NDArray,
    chi2_map: NDArray,
    best_fit_value: float,
    best_fit_fun: float,
) -> tuple[NDArray, NDArray]:
    """Make a profile of the chi-squared map using thee minimum value. Works
    with 2-dimensional maps.

    Parameters
    ----------
    parameter_grid : NDArray
        Array of grid to look for best fit value of parameter.
    profile_grid : NDArray
        Array of grid to create profile grid.
    chi2_map : NDArray
        Map of chi-squared values.
    best_fit_value : float
        Value of parameter in best fit point.
    best_fit_fun : float
        Value of the chi-squared in best fit point.

    Returns
    -------
    tuple[NDArray, NDArray]
        Array of profile grid values and array of chi-squared values.
    """
    abs_difference = np.abs(parameter_grid - best_fit_value)
    closest_value = abs_difference.min()
    mask = abs_difference == closest_value
    chi2_profile = chi2_map[mask] - best_fit_fun
    return profile_grid[mask], chi2_profile


def prepare_axes(
    ax: plt.Axes,
    limits: list[tuple[float, float]],
    profile: tuple[NDArray, NDArray],
    xlabel: str = "",
    ylabel: str = "",
    ticks: list[float] = [5, 10, 15, 20],
    levels: list[float] = [1, 4, 9, 16],
):
    """Update axis labels, limits, ticks, and plot levels.

    Parameters
    ----------
    ax : plt.Axes
        Element of (sub-)plot.
    limits : list[tuple[float, float], tuple[float, float]]
        Tuples of xlimits and ylimits.
    profile : tuple[NDArray, NDArray]
        Array of x values and y values (profile grid and chi-squared values or reversed).
    xlabel : str, optional
        Label of x axis.
    ylabel : str, optional
        Label of y axis.
    ticks : list[float], optional
        Ticks of chi-squared axis.
    levels : list[float], optional
        Levels of constant chi-squared.
    """
    xlim, ylim = limits
    if xlabel:
        ax.set_xticks(ticks, ticks)
        ax.set_yticks([], [])
        ax.set_xlabel(xlabel)
        ax.vlines(levels, *ylim, linestyle="--", alpha=0.25, colors="black")
    elif ylabel:
        ax.set_xticks([], [])
        ax.set_yticks(ticks, ticks)
        ax.set_ylabel(ylabel)
        ax.hlines(levels, *xlim, linestyle="--", alpha=0.25, colors="black")
    ax.plot(profile[0], profile[1], color="black")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.minorticks_on()


def cartesian_product(
    grid_opts: list[tuple[str, float, float, int]],
) -> tuple[list[str], list[NDArray], NDArray]:
    """Create cartesian products of several axes.

    Parameters
    ----------
    grid_opts : list[tuple[str, float, float, int]]
        Tuple of parameter name, left and right bounds,
        and the number of points with equal distance between the bounds.

    Returns
    -------
    tuple[list[str], NDArray]
        List of parameter names, list of arrays for cartersian product, and cartesian products of arrays.
    """
    parameters = []
    grids = []
    for parameter, l_bound, r_bound, num in grid_opts:
        parameters.append(parameter)
        grids.append(np.linspace(float(l_bound), float(r_bound), int(num)))
    grid = np.array(list(itertools.product(*grids)))
    return parameters, grids, grid


def main(args: Namespace) -> None:

    model = load_model(
        args.version,
        source_type=args.source_type,
        model_options=args.model_options,
    )

    model.storage["nodes.data.proxy"].switch_input(DATA_INDICES[args.data])

    parameters_free = model.storage("parameters.free")
    parameters_constrained = model.storage("parameters.constrained")
    statistic = model.storage("outputs.statistic")

    stat_chi2 = statistic[f"{args.chi2}"]
    minimization_parameters: dict[str, Parameter] = {}
    update_dict_parameters(minimization_parameters, args.free_parameters, parameters_free)
    if "covmat" not in args.chi2:
        update_dict_parameters(
            minimization_parameters,
            args.constrained_parameters,
            parameters_constrained,
        )

    model.next_sample(mc_parameters=False, mc_statistics=False)
    minimizer = IMinuitMinimizer(stat_chi2, parameters=minimization_parameters, nbins=model.nbins)
    global_fit = minimizer.fit()
    print(global_fit)
    fun = global_fit["fun"]
    bf_xdict = global_fit["xdict"]
    best_fit_x = bf_xdict["oscprob.SinSq2Theta13"]
    best_fit_y = bf_xdict["oscprob.DeltaMSq32"]
    model.set_parameters(bf_xdict)

    parameters, grids, xy_grid = cartesian_product(args.scan_par)
    grid_parameters = []
    minimization_parameters_2d = minimization_parameters.copy()
    for parameter in parameters:
        grid_parameter = minimization_parameters_2d.pop(parameter)
        grid_parameters.append(parameter)

    model.next_sample(mc_parameters=False, mc_statistics=False)
    minimizer_scan_2d = IMinuitMinimizer(stat_chi2, parameters=minimization_parameters_2d)
    chi2_map = np.zeros(xy_grid.shape[0])
    for idx, grid_values in enumerate(xy_grid):
        model.set_parameters(dict(zip(grid_parameters, grid_values)))
        fit = minimizer_scan_2d.fit()
        minimizer_scan_2d.push_initial_values()
        chi2_map[idx] = fit["fun"]

    chi2_map_1d: dict[str, NDArray | Any] = dict.fromkeys(grid_parameters)
    for parameter, grid_1d in zip(grid_parameters, grids):
        model.set_parameters(bf_xdict)
        chi2_map_1d[parameter] = np.zeros(grid_1d.shape[0])
        minimization_parameters_1d = minimization_parameters.copy()
        minimization_parameters_1d.pop(parameter)
        minimizer_scan_1d = IMinuitMinimizer(stat_chi2, parameters=minimization_parameters_1d)
        for idx, grid_value in enumerate(grid_1d):
            model.set_parameters({parameter: grid_value})
            fit = minimizer_scan_1d.fit()
            minimizer_scan_1d.push_initial_values()
            chi2_map_1d[parameter][idx] = fit["fun"]

    fig, axes = plt.subplots(2, 2, gridspec_kw={"width_ratios": [3, 1], "height_ratios": [1, 3]})
    sinSqD13_profile, chi2_profile = get_profile_of_chi2(
        xy_grid[:, 1], xy_grid[:, 0], chi2_map, best_fit_y, fun
    )

    # TODO: profile 1d

    label = r"$\Delta\chi^2$"
    prepare_axes(
        axes[0, 0],
        limits=[(xy_grid[:, 0].min(), xy_grid[:, 0].max()), (0, 20)],
        ylabel=label,
        profile=(sinSqD13_profile, chi2_profile),
    )

    dm32_profile, chi2_profile = get_profile_of_chi2(
        xy_grid[:, 0],
        xy_grid[:, 1],
        chi2_map,
        best_fit_x,
        fun,
    )
    prepare_axes(
        axes[1, 1],
        limits=[(0, 20), (xy_grid[:, 1].min(), xy_grid[:, 1].max())],
        xlabel=label,
        profile=(chi2_profile, dm32_profile),
    )

    ndof = len(parameters)
    levels = convert_sigmas_to_chi2(ndof, [0, 1, 2, 3])
    axes[1, 0].grid(linestyle="--")
    axes[1, 0].tricontourf(xy_grid[:, 0], xy_grid[:, 1], chi2_map - fun, levels=levels, cmap="GnBu")
    bf_x_error, bf_y_error, *_ = best_fit_errors
    axes[1, 0].errorbar(
        best_fit_x,
        best_fit_y,
        xerr=bf_x_error,
        yerr=bf_y_error,
        color="black",
        marker="o",
        markersize=3,
        capsize=3,
    )

    axes[1, 0].set_ylabel(r"$\Delta m^2_{32}$, [eV$^2$]")
    axes[1, 0].set_xlabel(r"$\sin^22\theta_{13}$")
    axes[1, 0].minorticks_on()
    fig.delaxes(axes[0, 1])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if args.output_contour:
        plt.savefig(args.output_contour)
    plt.show()

    if args.output_map:
        np.save(args.output_map, np.stack((*grid.T, chi2_map), axis=1))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", default=0, action="count", help="verbosity level")

    model = parser.add_argument_group("model", "model related options")
    model.add_argument(
        "--version",
        default="v0",
        choices=available_models(),
        help="model version",
    )
    model.add_argument(
        "-s",
        "--source-type",
        "--source",
        choices=("tsv", "hdf5", "root", "npz"),
        default="npz",
        help="Data source type",
    )
    model.add_argument("--model-options", "--mo", default={}, help="Model options as yaml dict")

    fit_options = parser.add_argument_group("fit", "Set fit procedure")
    fit_options.add_argument(
        "--data",
        default="model",
        choices=DATA_INDICES.keys(),
        help="Choose data for fit: 0th and 1st output",
    )
    fit_options.add_argument(
        "--scan-par",
        nargs=4,
        action="append",
        default=[],
        help="linspace of parameter",
    )
    fit_options.add_argument(
        "--chi2",
        default="stat.chi2p",
        choices=[
            "stat.chi2p_iterative",
            "stat.chi2n",
            "stat.chi2p",
            "stat.chi2cnp",
            "stat.chi2p_unbiased",
            "stat.chi2poisson",
            "full.covmat.chi2p_iterative",
            "full.covmat.chi2n",
            "full.covmat.chi2p",
            "full.covmat.chi2p_unbiased",
            "full.covmat.chi2cnp",
            "full.covmat.chi2cnp_alt",
            "full.pull.chi2p_iterative",
            "full.pull.chi2p",
            "full.pull.chi2cnp",
            "full.pull.chi2p_unbiased",
            "full.pull.chi2poisson",
        ],
        help="Choose chi-squared function for minimizer",
    )
    fit_options.add_argument(
        "--free-parameters",
        default=[],
        nargs="*",
        help="Add free parameters to minimization process",
    )
    fit_options.add_argument(
        "--constrained-parameters",
        default=[],
        nargs="*",
        help="Add constrained parameters to minimization process",
    )

    outputs = parser.add_argument_group("outputs", "set outputs")
    outputs.add_argument(
        "--output-contour",
        help="path to save plot of contour plots",
    )
    outputs.add_argument(
        "--output-map",
        help="path to save data of contour plots",
    )

    args = parser.parse_args()

    main(args)
