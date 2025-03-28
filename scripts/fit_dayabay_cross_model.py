#!/usr/bin/env python
import numpy as np
from argparse import Namespace
from typing import Any

from matplotlib import pyplot as plt
from yaml import add_representer
from yaml import safe_load as yaml_load
from yaml import safe_dump as yaml_dump

from dagflow.parameters import Parameter
from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer
from models import load_model
from scripts import update_dict_parameters, filter_fit, convert_numpy_to_lists

set_level(INFO1)


plt.rcParams.update(
    {
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.grid": True,
    }
)


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

    print(len(minimization_parameters))
    fit = minimizer.fit()
    print(fit)

    if args.interactive:
        from IPython import embed
        embed()

    filter_fit(fit, ["summary"])
    convert_numpy_to_lists(fit)
    if args.output_fit:
        with open(args.output_fit, "w") as f:
            yaml_dump(fit, f)

    if args.output_plot_spectra:
        edges = storage_data["outputs.edges.energy_final"].data
        centers = (edges[1:] + edges[:-1]) / 2
        xerrs = (edges[1:] - edges[:-1]) / 2
        for key, data in storage_data["outputs.eventscount.final.detector_period"].walkjoineditems():
            data = data.data
            obs = storage_fit[f"outputs.eventscount.final.detector_period.{key}"].data
            fig, axs = plt.subplots(3, 1, figsize=(7, 6), height_ratios=[2, 1, 1], sharex=True)
            axs[0].step([edges[0], *edges], [0, *obs, 0], where="post", label="A: fit")
            axs[0].errorbar(centers, data, xerr=xerrs, yerr=data**.5, linestyle="none", label="B: data")
            axs[1].errorbar(centers, obs / data - 1, xerr=xerrs, yerr=(obs / data**2 + obs**2 / data**4 * data)**.5, linestyle="none")
            axs[2].errorbar(centers, obs - data, xerr=xerrs, yerr=(data**.5 + obs**.5)**.5, linestyle="none")
            axs[0].set_title(key)
            axs[0].legend()
            axs[2].set_xlabel("E, MeV")
            axs[0].set_ylabel("Entries")
            axs[1].yaxis.tick_right()
            axs[1].yaxis.set_label_position("right")
            axs[1].set_ylabel("A / B - 1")
            axs[2].set_ylabel("A - B")
            axs[1].set_ylim(-1e-3, 1e-3)
            axs[0].minorticks_on()
            plt.setp(axs[0].get_xticklabels(), visible=False)
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.0)
            plt.savefig(args.output_plot_spectra.format(key.replace(".", "-")))

        if args.use_free_spec:
            edges = model.storage["outputs.reactor_anue.spectrum_free_correction.spec_model_edges"].data
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

    output = parser.add_argument_group("output", "output related options")
    output.add_argument("--output-fit", help="path to save fit in yaml format")
    output.add_argument("--output-plot-spectra", help="path with one placeholder to save spectra of each detector")

    args = parser.parse_args()

    main(args)
