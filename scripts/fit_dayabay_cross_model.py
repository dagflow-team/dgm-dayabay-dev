#!/usr/bin/env python
"""Script for fit model to another copy model. Models are loading from .yaml file.

Examples
--------
Example of call:
.. code::
    ./scripts/fit_dayabay_cross_model.py --config-path scripts/cross-fit-config.yaml \\
      --chi2 full.chi2n_covmat \\
      --output-plot-spectra "output/obs-{}.pdf" \\
      --output-fit output/fit.yaml
```
"""
from argparse import Namespace
from typing import Any

from IPython import embed
from LaTeXDatax import datax as datax_dump
from matplotlib import pyplot as plt
from yaml import safe_dump as yaml_dump
from yaml import safe_load as yaml_load

from dagflow.parameters import Parameter
from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer
from models import load_model
from scripts import convert_numpy_to_lists, do_fit, filter_fit, update_dict_parameters

set_level(INFO1)


def parse_config(config_path: str) -> list[dict[str, Any]]:
    """Load yaml config as python dictionary.

    Parameters
    ----------
    config_path : str
        Path to file with model options for two models.

    Returns
    -------
    list[dict[str, Any]]
        Two dictionaries with model options.
    """
    with open(config_path, "r") as f:
        return yaml_load(f)


def main(args: Namespace) -> None:

    if args.verbose:
        args.verbose = min(args.verbose, 3)
        set_level(globals()[f"INFO{args.verbose}"])

    models = []
    for config in parse_config(args.config_path):
        model = load_model(**config)
        models.append(model)

    model_data = models[0]
    model_fit = models[1]

    storage_data = model_data.storage
    storage_fit = model_fit.storage
    graph_data = model_data.graph
    graph_fit = model_fit.graph
    graph_fit.open()
    graph_data.open()
    storage_fit["nodes.data.proxy"].open()
    storage_data["outputs.data.proxy"] >> storage_fit["nodes.data.proxy"]
    storage_fit["nodes.data.proxy"].close(close_children=False)
    storage_fit["nodes.data.proxy"].switch_input(2)
    graph_fit.close()
    graph_data.close()
    parameters_free = storage_fit("parameters.free")
    parameters_constrained = storage_fit("parameters.constrained")
    statistic = storage_fit("outputs.statistic")

    for model in models:
        model.next_sample(mc_parameters=False, mc_statistics=False)

    chi2 = statistic[f"{args.chi2}"]
    minimization_parameters: dict[str, Parameter] = {}
    update_dict_parameters(minimization_parameters, args.free_parameters, parameters_free)
    if "covmat" not in args.chi2:
        update_dict_parameters(
            minimization_parameters,
            args.constrained_parameters,
            parameters_constrained,
        )

    if args.constrain_osc_parameters:
        minimizer = IMinuitMinimizer(
            chi2,
            parameters=minimization_parameters,
            limits={"oscprob.SinSq2Theta13": (0, 1), "oscprob.DeltaMSq32": (2e-3, 3e-3)},
            nbins=model_fit.nbins,
            verbose=args.verbose > 1,
        )

        fit = do_fit(minimizer, model_fit, "iterative" in args.chi2)
        if args.profile_parameters:
            minos_profile = minimizer.profile_errors(args.profile_parameters)
            fit["errorsdict_profiled"] = minos_profile["errorsdict"]
        filter_fit(fit, ["summary"])
        convert_numpy_to_lists(fit)
        if args.output_fit:
            with open(f"{args.output_fit}.constrained_osc", "w") as f:
                yaml_dump(fit, f)
        if not fit["success"]:
            exit()

    minimizer = IMinuitMinimizer(
        chi2, parameters=minimization_parameters, nbins=model_fit.nbins, verbose=args.verbose > 1
    )

    if args.interactive:
        embed()

    fit = do_fit(minimizer, model_fit, "iterative" in args.chi2)
    if args.profile_parameters:
        minos_profile = minimizer.profile_errors(args.profile_parameters)
        fit["errorsdict_profiled"] = minos_profile["errorsdict"]

    filter_fit(fit, ["summary"])
    convert_numpy_to_lists(fit)
    if args.output_fit:
        with open(f"{args.output_fit}", "w") as f:
            yaml_dump(fit, f)

    if args.output_fit_tex:
        for key, val in fit.copy().items():
            if isinstance(val, dict):
                for key0, val0 in val.items():
                    if isinstance(val0, (list, tuple)) and len(val0) == 2:
                        fit[f"{key}.{key0}.left"] = val0[0]
                        fit[f"{key}.{key0}.right"] = val0[1]
                    else:
                        fit[f"{key}.{key0}"] = val0

        datax_dump(
            args.output_fit_tex,
            **{
                key: val
                for key, val in fit.items()
                if not isinstance(val, (list, dict, type(None)))
            },
        )


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
        "--config-path", required=True, help="Config file with model options as yaml list of dicts"
    )

    fit_options = parser.add_argument_group("fit", "Set fit procedure")
    fit_options.add_argument(
        "--constrain-osc-parameters",
        action="store_true",
        help="Constrain oscillation parameters",
    )
    fit_options.add_argument(
        "--profile-parameters",
        action="extend",
        nargs="*",
        default=[],
        help="choose parameters for Minos profiling",
    )
    fit_options.add_argument(
        "--chi2",
        default="stat.chi2p",
        # TODO: Try to get info about statistics from model
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

    comparison = parser.add_argument_group("comparison", "Comparison options")
    comparison.add_argument(
        "--compare-concatenation",
        choices=["detector", "detector_period"],
        default="detector_period",
        help="Choose concatenation mode for plotting observation",
    )
    comparison.add_argument(
        "--compare-input",
        help="path to file with wich compare",
    )

    outputs = parser.add_argument_group("output", "output related options")
    outputs.add_argument(
        "--output-fit",
        help="path to save full fit, yaml format",
    )
    outputs.add_argument(
        "--output-fit-tex",
        help="path to save full fit, TeX format",
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
