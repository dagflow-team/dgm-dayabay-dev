#!/usr/bin/env python
from argparse import Namespace

<<<<<<< HEAD
from dagflow.logger import DEBUG as INFO4
from dagflow.logger import INFO1, INFO2, INFO3
from dagflow.logger import set_level

from models import load_model, available_models


set_level(INFO1)


def main(args: Namespace) -> None:

    model = load_model(
        args.version,
        source_type=args.source_type,
        spectrum_correction_mode=args.spec,
        monte_carlo_mode=args.data_mc_mode,
        seed=args.seed,
        model_options=args.model_options,
   )

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

    from yaml import safe_dump
    from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer

    chi2 = statistic[f"{args.chi2}"]
    minimization_parameters = [
        par
        for key in parameters_groups["free"]
        for par in parameters_free(key).walkvalues()
    ]
    if "covmat" not in args.chi2:
        minimization_parameters += [
            par
            for key in parameters_groups["constrained"]
            for par in parameters_constrained(key).walkvalues()
        ]

    model.next_sample()
    minimizer = IMinuitMinimizer(chi2, parameters=minimization_parameters)
    fit = minimizer.fit()
    print(fit)
    if args.output:
        dagflow_fit = dict(**dagflow_fit)
        with open(f"{args.full_fit_output}", "w") as f:
            safe_dump(dagflow_fit, f)

    if args.compare_input:
        with open(args.compare_input, "r") as f:
            compare_fit = yaml_load(f)
        plt.errorbar(
            compare_fit["SinSq2Theta13"]["value"], compare_fit["DeltaMSq32"]["value"],
            xerr=compare_fit["SinSq2Theta13"]["error"], yerr=compare_fit["DeltaMSq32"]["error"],
            label="SYSU",
        )
        plt.errorbar(
            fit["xdict"]["SinSq2Theta13"], fit["xdict"]["DeltaMSq32"],
            xerr=fit["errorsdict"]["SinSq2Theta13"], yerr=fit["errorsdict"]["DeltaMSq32"],
            label="dag-flow",
        )
        plt.xlabel(r"$\sin^22\theta_{13}$")
        plt.ylabel(r"$\Delta m^2_{32}$, [eV$^2$]")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # plt.figure(figsize=(8, 5))
    # plt.plot(storage)


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

    pars = parser.add_argument_group("fit", "Set fit procedure")
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
        choices=["stat.chi2p", "full.chi2p", "stat.chi2n", "full.chi2n"],
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
        "--output",
        help="path to save full fit, yaml format",
    )

    args = parser.parse_args()

    main(args)
