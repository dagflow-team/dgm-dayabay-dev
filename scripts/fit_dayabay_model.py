#!/usr/bin/env python
"""
Script for fit model to observed/model data

Example of call:
```
./scripts/fit_dayabay_model.py --version v0e \
    --mo "{dataset: b, monte_carlo_mode: poisson, seed: 1}" \
    --chi2 full.chi2n_covmat \
    --free-parameters oscprob neutrino_per_fission_factor \
    --constrained-parameters oscprob detector reactor bkg reactor_anue \
    --output-fit output/fit.yaml
```
"""
from argparse import Namespace

from IPython import embed
from yaml import dump as yaml_dump
from yaml import safe_load as yaml_load

from dagflow.parameters import Parameter
from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer
from models import available_models, load_model
from scripts import convert_numpy_to_lists, filter_fit, update_dict_parameters

set_level(INFO1)

DATA_INDICES = {"model": 0, "loaded": 1}


def main(args: Namespace) -> None:

    if args.verbose:
        args.verbose = min(args.verbose, 3)
        set_level(globals()[f"INFO{args.verbose}"])

    model = load_model(
        args.version,
        source_type=args.source_type,
        model_options=args.model_options,
    )

    storage = model.storage
    storage["nodes.data.proxy"].switch_input(DATA_INDICES[args.data])
    parameters_free = storage("parameters.free")
    parameters_constrained = storage("parameters.constrained")
    statistic = storage("outputs.statistic")

    chi2 = statistic[f"{args.chi2}"]
    minimization_parameters: dict[str, Parameter] = {}
    update_dict_parameters(minimization_parameters, args.free_parameters, parameters_free)
    if "covmat" not in args.chi2:
        update_dict_parameters(
            minimization_parameters,
            args.constrained_parameters,
            parameters_constrained,
        )

    minimizer = IMinuitMinimizer(chi2, parameters=minimization_parameters, limits={"oscprob.SinSq2Theta13": (0, 1), "oscprob.DeltaMSq32": (2e-3, 3e-3)}, verbose=True)
    fit = minimizer.fit()
    if "iterative" in args.chi2:
        for _ in range(4):
            model.next_sample(mc_parameters=False, mc_statistics=False)
            fit = minimizer.fit()
    filter_fit(fit, ["summary"])
    print(fit)
    convert_numpy_to_lists(fit)
    if args.output_fit:
        with open(f"{args.output_fit}", "w") as f:
            yaml_dump(fit, f)

    if args.interactive:
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
        default="hdf5",
        help="Data source type",
    )
    model.add_argument("--model-options", "--mo", default={}, help="Model options as yaml dict")

    fit_options = parser.add_argument_group("fit", "Set fit procedure")
    fit_options.add_argument(
        "--data",
        default="model",
        choices=DATA_INDICES.keys(),
        help="Choose data for fit",
    )
    fit_options.add_argument(
        "--par",
        nargs=2,
        action="append",
        default=[],
        help="set parameter value",
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
        "--output-fit",
        help="path to save full fit, yaml format",
    )

    args = parser.parse_args()

    main(args)
