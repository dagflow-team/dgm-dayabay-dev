#!/usr/bin/env python
from argparse import Namespace
from typing import Any

from yaml import add_representer
from yaml import safe_load as yaml_load

from dagflow.parameters import Parameter
from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer
from models import load_model
from scripts import update_dict_parameters

set_level(INFO1)


def parse_config(config_path: str) -> list[dict[str, Any]]:
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

    storage_data = models[0].storage
    storage_fit = models[1].storage
    graph_data = models[0].graph
    graph_fit = models[1].graph
    graph_fit.open()
    graph_data.open()
    storage_fit["nodes.data.proxy"].open(open_children=False)
    storage_data["outputs.data.proxy"] >> storage_fit["nodes.data.proxy"]
    storage_fit["nodes.data.proxy"].close(close_children=False)
    storage_fit["nodes.data.proxy"].switch_input(2)
    graph_fit.close()
    graph_data.close()
    parameters_free = storage_fit("parameters.free")
    parameters_constrained = storage_fit("parameters.constrained")
    statistic = storage_fit("outputs.statistic")

    parameters_groups = {
        "free": ["oscprob"],
        "constrained": ["oscprob", "reactor", "detector", "bkg"],
    }
    if args.use_free_spec:
        parameters_groups["free"].append("neutrino_per_fission_factor")
    else:
        parameters_groups["free"].append("detector")

    stat_chi2 = statistic[f"{args.chi2}"]
    minimization_parameters: dict[str, Parameter] = {}
    update_dict_parameters(minimization_parameters, parameters_groups["free"], parameters_free)

    models[0].next_sample(mc_parameters=False, mc_statistics=False)
    minimizer = IMinuitMinimizer(stat_chi2, parameters=minimization_parameters, limits={"SinSq2Theta13": (0, 1)})

    if args.interactive:
        from IPython import embed
        embed()

    print(len(minimization_parameters))
    fit = minimizer.fit()
    print(fit)

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
    model.add_argument("--config-path", required=True, help="Config file with model options as yaml list of dicts")

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

    args = parser.parse_args()

    main(args)
