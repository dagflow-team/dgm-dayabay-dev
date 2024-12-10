#!/usr/bin/env python
from argparse import Namespace

import yaml
from matplotlib import pyplot as plt

from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from dagflow.parameters.gaussian_parameter import GaussianParameter
from dagflow.tools.yaml_dumper import convert_numpy_to_lists, filter_fit
from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer
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

    chi2 = statistic[f"{args.chi2}"]
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
    minimizer = IMinuitMinimizer(chi2, parameters=minimization_parameters)
    fit = minimizer.fit()
    print(fit)

    if args.output_plot_pars:
        values = []
        errors = []
        labels = []
        for parname, par in minimization_parameters.items():
            if isinstance(par, GaussianParameter):
                values.append((fit["xdict"][parname] - par.central) / par.sigma)
                errors.append(abs(fit["errorsdict"][parname] / par.sigma))
                labels.append(parname)
        npars = len(values)
        plt.figure(figsize=(5, 0.225 * npars))
        f_key = labels[0]
        l_key = labels[-1]
        plt.scatter(values, labels)
        # plt.errorbar(values, labels, xerr=errors, linestyle="None")
        plt.vlines(0, f_key, l_key, linestyle="--", color="black")
        for i in range(1, 4):
            plt.vlines(i, f_key, l_key, linestyle="--", color=f"C{i}")
            plt.vlines(-i, f_key, l_key, linestyle="--", color=f"C{i}")
        plt.ylim(-1, npars + 1)
        plt.tight_layout()
        plt.savefig(args.output_plot_pars)

    if args.output_plot_covmat:
        cs = plt.matshow(fit["covariance"])
        plt.colorbar(cs)
        plt.savefig(args.output_plot_covmat)

    filter_fit(fit)
    convert_numpy_to_lists(fit)
    if args.output_fit:
        with open(args.output_fit, "w") as f:
            yaml.dump(fit, f)


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
