#!/usr/bin/env python
from argparse import Namespace

import numpy as np
from matplotlib import pyplot as plt
from yaml import dump as yaml_dump
from yaml import safe_load as yaml_load

from dagflow.parameters import GaussianParameter, Parameter
from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer
from models import LATEX_SYMBOLS, available_models, load_model
from scripts import convert_numpy_to_lists, filter_fit, update_dict_parameters

set_level(INFO1)

DATA_INDICES = {"asimov": 0, "data": 1}


plt.rcParams.update(
    {
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.grid": True,
    }
)


def main(args: Namespace) -> None:

    if args.verbose:
        args.verbose = min(args.verbose, 3)
        set_level(globals()[f"INFO{args.verbose}"])

    model = load_model(
        args.version,
        source_type=args.source_type,
        monte_carlo_mode=args.data_mc_mode,
        seed=args.seed,
        model_options=args.model_options,
    )

    storage = model.storage
    storage["nodes.data.proxy"].switch_input(DATA_INDICES[args.data])
    parameters_free = storage("parameters.free")
    parameters_constrained = storage("parameters.constrained")
    statistic = storage("outputs.statistic")

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

    chi2 = statistic[f"{args.chi2}"]
    minimization_parameters: dict[str, GaussianParameter] = {}
    update_dict_parameters(minimization_parameters, parameters_groups["free"], parameters_free)
    if "covmat" not in args.chi2:
        update_dict_parameters(
            minimization_parameters,
            parameters_groups["constrained"],
            parameters_constrained,
        )

    model.next_sample(mc_parameters=False, mc_statistics=False)
    minimizer = IMinuitMinimizer(chi2, parameters=minimization_parameters)

    model.next_sample()
    minimizer = IMinuitMinimizer(chi2, parameters=minimization_parameters)
    fit = minimizer.fit()
    filter_fit(fit, ["summary"])
    print(fit)
    convert_numpy_to_lists(fit)
    if args.output_fit:
        with open(f"{args.output_fit}", "w") as f:
            yaml_dump(fit, f)

    if args.output_plot_spectra:
        edges = model.storage["outputs.edges.energy_final"].data
        centers = (edges[1:] + edges[:-1]) / 2
        xerrs = (edges[1:] - edges[:-1]) / 2
        for key, data in model.storage["outputs.data.real.final.detector_period"].walkjoineditems():
            data = data.data
            obs = model.storage[f"outputs.eventscount.final.detector_period.{key}"].data
            fig, axs = plt.subplots(3, 1, figsize=(7, 6), height_ratios=[2, 1, 1], sharex=True)
            axs[0].step([edges[0], *edges], [0, *obs, 0], where="post", label="A: fit")
            axs[0].errorbar(
                centers, data, xerr=xerrs, yerr=data**0.5, linestyle="none", label="B: data"
            )
            axs[1].errorbar(
                centers,
                obs / data - 1,
                xerr=xerrs,
                yerr=(obs / data**2 + obs**2 / data**4 * data) ** 0.5,
                linestyle="none",
            )
            axs[2].errorbar(
                centers,
                obs - data,
                xerr=xerrs,
                yerr=(data**0.5 + obs**0.5) ** 0.5,
                linestyle="none",
            )
            axs[0].set_title(key)
            axs[0].legend()
            axs[2].set_xlabel("E, MeV")
            axs[0].set_ylabel("Entries")
            axs[1].yaxis.tick_right()
            axs[1].yaxis.set_label_position("right")
            axs[1].set_ylabel("A / B - 1")
            axs[2].set_ylabel("A - B")
            axs[0].minorticks_on()
            plt.setp(axs[0].get_xticklabels(), visible=False)
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.0)
            plt.savefig(args.output_plot_spectra.format(key.replace(".", "-")))

        if args.use_free_spec:
            edges = model.storage[
                "outputs.reactor_anue.spectrum_free_correction.spec_model_edges"
            ].data
            data = []
            yerrs = []
            for key in filter(lambda key: True if "spec" in key else False, fit["names"]):
                data.append(fit["xdict"][key])
                yerrs.append(fit["errorsdict"][key])
            plt.figure()
            plt.errorbar(edges, data, xerr=0.1, yerr=yerrs, linestyle="none")
            plt.title(r"Spectral weights of $\overline{\nu}_{e}$ spectrum")
            plt.xlabel("E, MeV")
            plt.ylabel("value")
            plt.tight_layout()
            plt.savefig(args.output_plot_spectra.format("sw"))

    plt.figure()
    plt.errorbar(
        fit["xdict"]["oscprob.SinSq2Theta13"],
        fit["xdict"]["oscprob.DeltaMSq32"],
        xerr=fit["errorsdict"]["oscprob.SinSq2Theta13"],
        yerr=fit["errorsdict"]["oscprob.DeltaMSq32"],
        label="dag-flow",
    )
    if args.compare_input:
        with open(args.compare_input, "r") as f:
            compare_fit = yaml_load(f)
        plt.errorbar(
            compare_fit["SinSq2Theta13"]["value"],
            compare_fit["DeltaMSq32"]["value"],
            xerr=compare_fit["SinSq2Theta13"]["error"],
            yerr=compare_fit["DeltaMSq32"]["error"],
            label="dataset",
        )
    plt.xlabel(r"$\sin^22\theta_{13}$")
    plt.ylabel(r"$\Delta m^2_{32}$, [eV$^2$]")
    plt.title(args.chi2 + f" = {fit['fun']:1.3f}")
    plt.xlim(0.082, 0.090)
    plt.ylim(2.37e-3, 2.53e-3)
    plt.legend()
    plt.tight_layout()
    if args.output_plot_fit:
        plt.savefig(args.output_plot_fit)
    if args.compare_input:
        print(args.chi2)
        for name, par_values in compare_fit.items():
            if name not in fit["xdict"].keys():
                continue
            fit_value = fit["xdict"][name]
            fit_error = fit["errorsdict"][name]
            value = par_values["value"]
            error = par_values["error"]
            print(f"{name:>22}:")
            print(f"{'dataset':>22}: value={value:1.9e}, error={error:1.9e}")
            print(f"{'dag-flow':>22}: value={fit_value:1.9e}, error={fit_error:1.9e}")
            print(
                f"{' '*23} value_diff={(fit_value / value - 1)*100:1.7f}%, error_diff={(fit_error / error - 1)*100:1.7f}%"
            )
            print(f"{' '*23} sigma_diff={(fit_value - value) / error:1.7f}")

    plt.show()

    if args.interactive:
        from IPython import embed

        embed()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", default=0, action="count", help="verbosity level")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start IPython session",
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
    model.add_argument("--seed", default=0, type=int, help="seed of randomization")
    model.add_argument(
        "--data-mc-mode",
        default="asimov",
        choices=["asimov", "normal-stats", "poisson"],
        help="type of data to be analyzed",
    )
    model.add_argument("--model-options", "--mo", default={}, help="Model options as yaml dict")

    pars = parser.add_argument_group("fit", "Set fit procedure")
    pars.add_argument(
        "--data",
        default="asimov",
        choices=["asimov", "data"],
        help="Choose data for fit",
    )
    pars.add_argument(
        "--par",
        nargs=2,
        action="append",
        default=[],
        help="set parameter value",
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

    comparison = parser.add_argument_group("comparison", "Comparison options")
    comparison.add_argument(
        "--compare-input",
        help="path to file with wich compare",
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
        "--output-plot-corrmat",
        help="path to save plot of correlation matrix of fitted parameters",
    )
    outputs.add_argument(
        "--output-plot-spectra",
        help="path to save full plot of fits",
    )
    outputs.add_argument(
        "--output-plot-fit",
        help="path to save full plot of fits",
    )

    args = parser.parse_args()

    main(args)
