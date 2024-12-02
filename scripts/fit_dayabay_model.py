#!/usr/bin/env python
from argparse import Namespace

from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from models import available_models, load_model

set_level(INFO1)

DATA_INDICES = {"asimov": 0, "data-a": 1}


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

    from yaml import safe_dump

    from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer

    chi2 = statistic[f"{args.chi2}"]
    minimization_parameters = {
        par_name: par
        for key in parameters_groups["free"]
        for par_name, par in parameters_free(key).walkjoineditems()
    }
    if "covmat" not in args.chi2:
        minimization_parameters.update({
            par_name: par
            for key in parameters_groups["constrained"]
            for par_name, par in parameters_constrained(key).walkjoineditems()
        })

    model.next_sample()
    minimizer = IMinuitMinimizer(chi2, parameters=minimization_parameters)
    fit = minimizer.fit()
    print(fit)
    if args.output:
        dagflow_fit = dict(**dagflow_fit)
        with open(f"{args.full_fit_output}", "w") as f:
            safe_dump(dagflow_fit, f)


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
            "chi2p_iterative", "chi2n", "chi2p", "chi2cnp", "chi2p_unbiased",
            "chi2p_covmat_fixed", "chi2n_covmat", "chi2p_covmat_variable",
            "chi2p_iterative", "chi2cnp", "chi2p_unbiased", "chi2cnp_covmat"
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
        "--output",
        help="path to save full fit, yaml format",
    )

    args = parser.parse_args()

    main(args)
