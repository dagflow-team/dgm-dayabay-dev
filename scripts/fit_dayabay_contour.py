#!/usr/bin/env python
from argparse import Namespace

import numpy as np
import itertools
from numpy.typing import NDArray
from scipy.stats import norm, chi2
from matplotlib import pyplot as plt

from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from dagflow.parameters.gaussian_parameter import GaussianParameter
from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer
from models import available_models, load_model

set_level(INFO1)

DATA_INDICES = {"asimov": 0, "data-a": 1}


def convert_sigmas_to_chi2(ndof, nsigmas) -> NDArray:
    percentiles = 2 * norm(0, 1).cdf(range(nsigmas + 1)) - 1
    return chi2(ndof).ppf(percentiles)


def get_profile_of_chi2(
    parameter_grid: NDArray,
    profile_grid: NDArray,
    chi2_map: NDArray,
    best_fit_value: float,
    best_fit_fun: float,
) -> tuple[NDArray, NDArray]:
    abs_difference = np.abs(parameter_grid - best_fit_value)
    closest_value = abs_difference.min()
    mask = abs_difference == closest_value
    chi2_profile = chi2_map[mask] - best_fit_fun
    return profile_grid[mask], chi2_profile


def prepare_axes(
    ax: plt.Axes,
    limits: list[tuple[float, float], tuple[float, float]],
    profile: tuple[NDArray],
    xlabel: str = "",
    ylabel: str = "",
    ticks: list[float] = [5, 10, 15, 20],
    levels: list[float] = [1, 4, 9, 16],
):
    xlim, ylim = limits
    if xlabel:
        ax.set_xticks(ticks, ticks)
        ax.set_yticks([], [])
        ax.set_xlabel(xlabel)
        ax.vlines(levels, *ylim, linestyle="--", alpha=0.25, colors="black")
        ax.plot(profile[1], profile[0], color="black")
    elif ylabel:
        ax.set_xticks([], [])
        ax.set_yticks(ticks, ticks)
        ax.set_ylabel(ylabel)
        ax.hlines(levels, *xlim, linestyle="--", alpha=0.25, colors="black")
        ax.plot(profile[0], profile[1], color="black")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.minorticks_on()


def cartesian_product(grid_opts: list[tuple[GaussianParameter, float, float, int]]):
    parameters = []
    grids = []
    for parameter, l_bound, r_bound, num in grid_opts:
        parameters.append(parameter)
        grids.append(np.linspace(float(l_bound), float(r_bound), int(num)))
    grid = np.array(list(itertools.product(*grids)))
    return parameters, grid


def push_parameters(parameters: list[GaussianParameter], values: NDArray) -> None:
    for parameter, value in zip(parameters, values):
        parameter.push(value)


def main(args: Namespace) -> None:

    model = load_model(
        args.version,
        source_type=args.source_type,
        spectrum_correction_mode=args.spec,
        monte_carlo_mode=args.data_mc_mode,
        seed=args.seed,
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
    minimization_parameters = {
        par_name: par
        for key in parameters_groups["free"]
        for par_name, par in parameters_free(key).walkjoineditems()
    }
    if "covmat" not in args.chi2:
        minimization_parameters.update(
            {
                par_name: par
                for key in parameters_groups["constrained"]
                for par_name, par in parameters_constrained(key).walkjoineditems()
            }
        )

    model.next_sample()
    minimizer = IMinuitMinimizer(stat_chi2, parameters=minimization_parameters)
    global_fit = minimizer.fit()
    best_fit_fun = global_fit["fun"]
    best_fit_values = global_fit["x"]
    best_fit_errors = global_fit["errors"]
    push_parameters(minimization_parameters.values(), best_fit_values)

    parameters, grid = cartesian_product(args.scan_par)
    grid_parameters = []
    for parameter in parameters:
        parameter = minimization_parameters.pop(parameter)
        grid_parameters.append(parameter)

    model.next_sample()
    scan_minimizer = IMinuitMinimizer(stat_chi2, parameters=minimization_parameters)
    chi2_map = []
    for grid_values in grid:
        push_parameters(grid_parameters, grid_values)
        fit = scan_minimizer.fit()
        scan_minimizer.push_initial_values()
        chi2_map.append(fit["fun"])

    chi2_map = np.array(chi2_map)

    fig, axes = plt.subplots(2, 2, gridspec_kw={"width_ratios": [3, 1], "height_ratios": [1, 3]})
    ndof = len(parameters)
    profiles = []
    for i in range(ndof):
        x_profile, y_profile = get_profile_of_chi2(
            grid[:, i], grid[:, (i + 1) % 2], chi2_map, best_fit_values[i], best_fit_fun,
        )
        profiles.append((x_profile, y_profile))


    label = r"$\Delta\chi^2$"
    prepare_axes(
        axes[0, 0],
        limits=[(grid[:, 0].min(), grid[:, 0].max()), (0, 20)],
        ylabel=label,
        profile=profiles[1],
    )

    prepare_axes(
        axes[1, 1],
        limits=[(0, 20), (grid[:, 1].min(), grid[:, 1].max())],
        xlabel=r"$\Delta\chi^2$",
        profile=profiles[0],
    )

    levels = convert_sigmas_to_chi2(ndof, 3)
    axes[1, 0].grid(linestyle="--")
    axes[1, 0].tricontourf(grid[:, 0], grid[:, 1], chi2_map - best_fit_fun, levels=levels, cmap="GnBu")
    bf_x, bf_y, *_ = best_fit_values
    bf_x_error, bf_y_error, *_ = best_fit_errors
    axes[1, 0].errorbar(bf_x, bf_y, xerr=bf_x_error, yerr=bf_y_error, color="black", marker="o", markersize=3, capsize=3)

    axes[1, 0].set_ylabel(r"$\Delta m^2_{32}$, [eV$^2$]")
    axes[1, 0].set_xlabel(r"$\sin^22\theta_{13}$")
    axes[1, 0].minorticks_on()
    fig.delaxes(axes[0, 1])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f"/tmp/contour-profile.pdf")
    plt.cla(), plt.clf()



    import IPython; IPython.embed()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", default=0, action="count", help="verbosity level"
    )

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
    model.add_argument(
        "--spec",
        choices=("linear", "exponential"),
        default="exponential",
        help="antineutrino spectrum correction mode",
    )
    model.add_argument("--seed", default=0, type=int, help="seed of randomization")
    model.add_argument(
        "--data-mc-mode",
        default="asimov",
        choices=["asimov", "normal-stats", "poisson"],
        help="type of data to be analyzed",
    )
    model.add_argument(
        "--model-options", "--mo", default={}, help="Model options as yaml dict"
    )

    pars = parser.add_argument_group("fit", "Set fit procedure")
    pars.add_argument(
        "--data",
        default="asimov",
        choices=["asimov", "data-a"],
        help="Choose data for fit",
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
            "stat.chi2p_iterative", "stat.chi2n", "stat.chi2p", "stat.chi2cnp",
            "stat.chi2p_unbiased", "full.chi2p_covmat_fixed", "full.chi2n_covmat",
            "full.chi2p_covmat_variable", "full.chi2p_iterative", "full.chi2cnp",
            "full.chi2p_unbiased", "full.chi2cnp_covmat",
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
        "--output-fit",
        help="path to save full fit, yaml format",
    )
    outputs.add_argument(
        "--output-plot-pars",
        help="path to save plot of normalized values",
    )
    outputs.add_argument(
        "--output-plot-covmat",
        help="path to save plot of normalized values",
    )

    args = parser.parse_args()

    main(args)
