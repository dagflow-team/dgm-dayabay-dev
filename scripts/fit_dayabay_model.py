#!/usr/bin/env python
from argparse import Namespace

from yaml import dump as yaml_dump
import numpy as np
from matplotlib import pyplot as plt

from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from dagflow.core import NodeStorage
from dagflow.parameters import Parameter, GaussianParameter
from dagflow.tools.yaml_dumper import convert_numpy_to_lists, filter_fit
from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer
from models import available_models, load_model, LATEX_SYMBOLS

set_level(INFO1)

DATA_INDICES = {"asimov": 0, "data-a": 1}


def update_minimization_paramers(
    minimization_parameters: dict[str, Parameter], groups: list[str], model_parameters: NodeStorage,
) -> None:
    for group in groups:
        minimization_parameters.update(dict(model_parameters(group).walkjoineditems()))


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
    )

    storage: NodeStorage = model.storage
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
    update_minimization_paramers(
        minimization_parameters, parameters_groups["free"], parameters_free
    )
    if "covmat" not in args.chi2:
        update_minimization_paramers(
            minimization_parameters,
            parameters_groups["constrained"],
            parameters_constrained,
        )

    model.next_sample()
    minimizer = IMinuitMinimizer(stat_chi2, parameters=minimization_parameters)
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

    args = parser.parse_args()

    main(args)
