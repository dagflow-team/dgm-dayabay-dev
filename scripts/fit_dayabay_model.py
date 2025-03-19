#!/usr/bin/env python
from argparse import Namespace

import numpy as np
from matplotlib import pyplot as plt
from yaml import dump as yaml_dump
from yaml import safe_load as yaml_load
from yaml import add_representer

from dagflow.parameters import GaussianParameter
from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer
from models import LATEX_SYMBOLS, available_models, load_model
from scripts import update_dict_parameters

set_level(INFO1)

DATA_INDICES = {"asimov": 0, "data": 1}


add_representer(
    np.ndarray,
    lambda representer, obj: representer.represent_str(np.array_repr(obj)),
)


def filter_fit(src: dict, keys_to_fiter: list[str]) -> dict:
    keys = list(src.keys())
    for key in keys:
        if key in keys_to_fiter:
            del src[key]
            continue
        if isinstance(src[key], dict):
            filter_fit(src[key], keys_to_fiter)


def convert_numpy_to_lists(src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, np.ndarray):
            src[key] = value.tolist()
        elif isinstance(value, dict):
            convert_numpy_to_lists(value)



def main(args: Namespace) -> None:

    if args.verbose:
        args.verbose = min(args.verbose, 3)
        set_level(globals()[f"INFO{args.verbose}"])

    model = load_model(
        args.version,
        source_type=args.source_type,
        spectrum_correction_mode=args.spec,
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

    stat_chi2 = statistic[f"{args.chi2}"]
    minimization_parameters = {}
    update_dict_parameters(
        minimization_parameters, parameters_groups["free"], parameters_free
    )
    # if "covmat" not in args.chi2:
    #     update_dict_parameters(
    #         minimization_parameters,
    #         parameters_groups["constrained"],
    #         parameters_constrained,
    #     )

    model.next_sample(mc_parameters=False, mc_statistics=False)
    minimizer = IMinuitMinimizer(stat_chi2, parameters=minimization_parameters)

    if args.interactive:
        from IPython import embed
        embed()

    print(len(minimization_parameters))
    fit = minimizer.fit()
    print(fit)

    if args.output_plot_pars:
        values = []
        errors = []
        labels = []
        for i, (parname, par) in enumerate(minimization_parameters.items()):
            if isinstance(par, GaussianParameter):
                values.append((fit["xdict"][parname] - par.central) / par.sigma)
                errors.append(abs(fit["errorsdict"][parname] / par.sigma))
                labels.append(LATEX_SYMBOLS[parname])
        npars = len(values)
        if npars:
            fig, axs = plt.subplots(
                2, 1, figsize=(5, 0.225 * npars),
                gridspec_kw={"height_ratios": [1, npars / 5]},
            )
            for i in range(0, 4):
                axs[1].vlines([-i, i], -1, npars + 1, linestyle="--", color=f"black", alpha=0.25)
            axs[0].hist(values, np.linspace(-3.25, 3.25, 2 * int(npars**0.5)))
            axs[1].scatter(values, labels)
            for ax in axs:
                ax.set_xlim(-3.25, 3.25)
            axs[1].set_ylim(-0.25, npars - 0.75)
            plt.tight_layout()
            plt.subplots_adjust(hspace=0)
            if args.output_plot_pars:
                plt.savefig(args.output_plot_pars)

    if args.output_plot_corrmat and fit["covariance"] is not None:
        covariance = fit["covariance"]
        npars = len(covariance)
        diag = np.diagonal(covariance) ** 0.5
        correlation = covariance / np.outer(diag, diag)
        if npars < 20:
            fig, ax = plt.subplots(1, 1)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(0.225 * npars, 0.185 * npars))
        cs = ax.matshow(correlation, vmin=-1, vmax=1, cmap="RdBu")
        clb = plt.colorbar(cs, ax=ax)
        clb.set_ticks([-1, -0.5, -0.25, 0, 0.25, 0.5, 1])
        clb.set_ticklabels([-1, -0.5, -0.25, 0, 0.25, 0.5, 1], fontsize=npars / 4)
        tick_labels = [LATEX_SYMBOLS[parname] for parname in fit["names"]]
        ax.set_xticks(np.arange(npars), tick_labels, fontsize=8, rotation=90)
        ax.set_yticks(np.arange(npars), tick_labels, fontsize=8, rotation=0)
        plt.tight_layout()
        plt.savefig(args.output_plot_corrmat)

    filter_fit(fit, ["summary"])
    convert_numpy_to_lists(fit)
    if args.output_fit:
        with open(args.output_fit, "w") as f:
            yaml_dump(fit, f)

    if args.compare_input:
        with open(args.compare_input, "r") as f:
            compare_fit = yaml_load(f)
        plt.errorbar(
            fit["xdict"]["SinSq2Theta13"], fit["xdict"]["DeltaMSq32"],
            xerr=fit["errorsdict"]["SinSq2Theta13"], yerr=fit["errorsdict"]["DeltaMSq32"],
            label="dag-flow",
        )
        plt.errorbar(
            compare_fit["SinSq2Theta13"]["value"], compare_fit["DeltaMSq32"]["value"],
            xerr=compare_fit["SinSq2Theta13"]["error"], yerr=compare_fit["DeltaMSq32"]["error"],
            label="dataset",
        )
        plt.xlabel(r"$\sin^22\theta_{13}$")
        plt.ylabel(r"$\Delta m^2_{32}$, [eV$^2$]")
        plt.title(args.chi2 + f" = {fit['fun']:1.3f}")
        plt.legend()
        plt.tight_layout()
        if args.output_plot_fit:
            plt.savefig(args.output_plot_fit)
        print(args.chi2)
        for name, par_values in compare_fit.items():
            if name not in fit["xdict"].keys():
                continue
            fit_value = fit["xdict"][name]
            fit_error = fit["errorsdict"][name]
            value = par_values["value"]
            error = par_values["error"]
            print(f"{name:>22}:")
            print(f"{'dataset':>22}: value={value:1.5e}, error={error:1.5e}")
            print(f"{'dag-flow':>22}: value={fit_value:1.5e}, error={fit_error:1.5e}")
            print(f"{' '*23} value_diff={(fit_value / value - 1)*100:1.3f}%, error_diff={(fit_error / error - 1)*100:1.3f}%")
            print(f"{' '*23} sigma_diff={(fit_value - value) / error:1.3f}")

    plt.show()

    if args.interactive:
        from IPython import embed
        embed()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", default=0, action="count", help="verbosity level"
    )
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
            "stat.chi2p_iterative", "stat.chi2n", "stat.chi2p", "stat.chi2cnp",
            "stat.chi2p_unbiased", "full.chi2p_covmat_fixed",
            "full.chi2n_covmat", "full.chi2p_covmat_variable",
            "full.chi2p_iterative", "full.chi2cnp",
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
        "--output-plot-fit",
        help="path to save full plot of fits",
    )

    args = parser.parse_args()

    main(args)
