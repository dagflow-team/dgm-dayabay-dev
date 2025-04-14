#!/usr/bin/env python
"""
Script for contour plot of best fit value

Example of call:
```
./scripts/fit_dayabay_contour.py --version v0e \
    --scan-par oscprob.SinSq2Theta13 0.07 0.1 31 \
    --scan-par oscprob.DeltaMSq32 2.2e-3 2.8e-3 61 \
    --chi2 full.chi2n_covmat \
    --output-contour output/contour.pdf \
    --output-map output/contour.npz
```
"""
import itertools
from argparse import Namespace

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.stats import chi2, norm

from dagflow.parameters.gaussian_parameter import GaussianParameter
from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer
from models import available_models, load_model
from scripts import update_dict_parameters

set_level(INFO1)

DATA_INDICES = {"model": 0, "loaded": 1}


def convert_sigmas_to_chi2(df: int, sigmas: list[float] | NDArray) -> NDArray:
    """Convert deviation of normal unit distribution N(0, 1) to critical value of chi-squared

    Parameters
    ----------
    df : int
        degree of freedom of chi-squared distribution
    sigmas : list[float] | NDArray
        list or array deviations from 0 in terms of standard deviation of normal unit distribution N(0, 1)

    Returns
    -------
    NDArray
        array of critical values of chi-squared
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
    """Make a profile of the chi-squared map using thee minimum value.
    Works with 2-dimensional maps.

    Parameters
    ----------
    parameter_grid : NDArray
        array of grid to look for best fit value of parameter
    profile_grid : NDArray
        array of grid to create profile grid
    chi2_map : NDArray
        map of chi-squared values
    best_fit_value : float
        value of parameter in best fit point
    best_fit_fun : float
        value of the chi-squared in best fit point

    Returns
    -------
    tuple[NDArray, NDArray]
        array of profile grid values and array of chi-squared values
    """
    abs_difference = np.abs(parameter_grid - best_fit_value)
    closest_value = abs_difference.min()
    mask = abs_difference == closest_value
    chi2_profile = chi2_map[mask] - best_fit_fun
    return profile_grid[mask], chi2_profile


def prepare_axes(
    ax: plt.Axes,
    limits: list[tuple[float, float], tuple[float, float]],
    profile: tuple[NDArray, NDArray],
    xlabel: str = "",
    ylabel: str = "",
    ticks: list[float] = [5, 10, 15, 20],
    levels: list[float] = [1, 4, 9, 16],
):
    """Update axis labels, limits, ticks, and plot levels

    Parameters
    ----------
    ax : plt.Axes
        element of (sub-)plot
    limits : list[tuple[float, float], tuple[float, float]]
        tuples of xlimits and ylimits
    profile : tuple[NDArray, NDArray]
        array of x values and y values (profile grid and chi-squared values or reversed)
    xlabel : str, optional
        label of x axis, by default ""
    ylabel : str, optional
        label of y axis, by default ""
    ticks : list[float], optional
        ticks of chi-squared axis, by default [5, 10, 15, 20]
    levels : list[float], optional
        levels of constant chi-squared, by default [1, 4, 9, 16]
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


def cartesian_product(grid_opts: list[tuple[str, float, float, int]]) -> tuple[list[str], NDArray]:
    """Create cartesian products of several axes

    Parameters
    ----------
    grid_opts : list[tuple[str, float, float, int]]
        tuple of parameter name, left and right bounds, and the number of points with equal distance between the bounds

    Returns
    -------
    tuple[list[str], NDArray]
        list of parameter names, and array of cartesian products of grids
    """
    parameters = []
    grids = []
    for parameter, l_bound, r_bound, num in grid_opts:
        parameters.append(parameter)
        grids.append(np.linspace(float(l_bound), float(r_bound), int(num)))
    grid = np.array(list(itertools.product(*grids)))
    return parameters, grid


def push_parameters(parameters: list[GaussianParameter], values: NDArray) -> None:
    """Push values of parameters

    Parameters
    ----------
    parameters : list[GaussianParameter]
        list of parameters to be changed
    values : NDArray
        array of values to be pushed in parameters
    """
    for parameter, value in zip(parameters, values):
        parameter.push(value)


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

    parameters_groups = {
        "free": ["oscprob"],
        "constrained": ["oscprob", "reactor", "detector", "bkg"],
    }
    if args.use_free_spec:
        parameters_groups["free"].append("neutrino_per_fission_factor")
    else:
        parameters_groups["free"].append("detector")
    if args.use_hm_unc_pull_terms:
        parameters_groups["constrained"].append("reactor_anue")

    stat_chi2 = statistic[f"{args.chi2}"]
    minimization_parameters: dict[str, GaussianParameter] = {}
    update_dict_parameters(minimization_parameters, parameters_groups["free"], parameters_free)
    if "covmat" not in args.chi2:
        update_dict_parameters(
            minimization_parameters,
            parameters_groups["constrained"],
            parameters_constrained,
        )

    model.next_sample()
    minimizer = IMinuitMinimizer(stat_chi2, parameters=minimization_parameters)
    global_fit = minimizer.fit()
    best_fit_fun = global_fit["fun"]
    best_fit_values = global_fit["x"]
    best_fit_x, best_fit_y, *_ = best_fit_values
    best_fit_errors = global_fit["errors"]
    push_parameters(minimization_parameters.values(), best_fit_values)

    parameters, grid = cartesian_product(args.scan_par)
    grid_parameters = []
    for parameter in parameters:
        grid_parameter = minimization_parameters.pop(parameter)
        grid_parameters.append(grid_parameter)

    model.next_sample()
    scan_minimizer = IMinuitMinimizer(stat_chi2, parameters=minimization_parameters)
    chi2_map = np.zeros(grid.shape[0])
    for idx, grid_values in enumerate(grid):
        push_parameters(grid_parameters, grid_values)
        fit = scan_minimizer.fit()
        scan_minimizer.push_initial_values()
        chi2_map[idx] = fit["fun"]

    fig, axes = plt.subplots(2, 2, gridspec_kw={"width_ratios": [3, 1], "height_ratios": [1, 3]})
    sinSqD13_profile, chi2_profile = get_profile_of_chi2(
        grid[:, 1], grid[:, 0], chi2_map, best_fit_y, best_fit_fun
    )

    label = r"$\Delta\chi^2$"
    prepare_axes(
        axes[0, 0],
        limits=[(grid[:, 0].min(), grid[:, 0].max()), (0, 20)],
        ylabel=label,
        profile=(sinSqD13_profile, chi2_profile),
    )

    dm32_profile, chi2_profile = get_profile_of_chi2(
        grid[:, 0],
        grid[:, 1],
        chi2_map,
        best_fit_x,
        best_fit_fun,
    )
    prepare_axes(
        axes[1, 1],
        limits=[(0, 20), (grid[:, 1].min(), grid[:, 1].max())],
        xlabel=label,
        profile=(chi2_profile, dm32_profile),
    )

    ndof = len(parameters)
    levels = convert_sigmas_to_chi2(ndof, [0, 1, 2, 3])
    axes[1, 0].grid(linestyle="--")
    axes[1, 0].tricontourf(
        grid[:, 0], grid[:, 1], chi2_map - best_fit_fun, levels=levels, cmap="GnBu"
    )
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

    pars = parser.add_argument_group("fit", "Set fit procedure")
    pars.add_argument(
        "--data",
        default="model",
        choices=DATA_INDICES.keys(),
        help="Choose data for fit: 0th and 1st output",
    )
    pars.add_argument(
        "--scan-par",
        nargs=4,
        action="append",
        default=[],
        help="linspace of parameter",
    )
    pars.add_argument(
        "--chi2",
        default="stat.chi2p",
        choices=[
            "stat.chi2p_iterative",
            "stat.chi2n",
            "stat.chi2p",
            "stat.chi2cnp",
            "stat.chi2p_unbiased",
            "full.chi2p_covmat_fixed",
            "full.chi2n_covmat",
            "full.chi2p_covmat_variable",
            "full.chi2p_iterative",
            "full.chi2cnp",
            "full.chi2p_unbiased",
            "full.chi2cnp_covmat",
        ],
        help="Choose chi-squared function for minimizer",
    )
    pars.add_argument(
        "--use-free-spec",
        action="store_true",
        help="Add antineutrino spectrum parameters to minimizer",
    )
    pars.add_argument(
        "--use-hm-unc-pull-terms",
        action="store_true",
        help="Add uncertainties of antineutrino spectra (HM model) to minimizer",
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
