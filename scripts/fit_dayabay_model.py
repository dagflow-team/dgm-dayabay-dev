#!/usr/bin/env python
"""
Script for fit model to observed/model data

Example of call:
```
./scripts/fit_dayabay_model.py --version v0e \
    --mo "{dataset: b, monte_carlo_mode: poisson, seed: 1}" \
    --chi2 full.chi2n_covmat \
    --output-plot-spectra "output/obs-{}.pdf" \
    --output-fit output/fit.yaml
```
"""
from argparse import Namespace

from matplotlib import pyplot as plt
from yaml import dump as yaml_dump
from yaml import safe_load as yaml_load

from dagflow.parameters import Parameter
from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer
from models import available_models, load_model
from scripts import (
    convert_numpy_to_lists,
    filter_fit,
    plot_spectra_ratio_difference,
    plot_spectral_weights,
    update_dict_parameters,
)

set_level(INFO1)

DATA_INDICES = {"model": 0, "loaded": 1}


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
    minimization_parameters: dict[str, Parameter] = {}
    update_dict_parameters(minimization_parameters, parameters_groups["free"], parameters_free)
    if "covmat" not in args.chi2:
        update_dict_parameters(
            minimization_parameters,
            parameters_groups["constrained"],
            parameters_constrained,
        )

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
        for obs_name, data in model.storage[
            "outputs.data.real.final.detector_period"
        ].walkjoineditems():
            plot_spectra_ratio_difference(
                model.storage[f"outputs.eventscount.final.detector_period.{obs_name}"].data,
                data.data,
                edges,
                obs_name,
            )
            plt.savefig(args.output_plot_spectra.format(obs_name.replace(".", "-")))

        if args.use_free_spec:
            edges = model.storage[
                "outputs.reactor_anue.spectrum_free_correction.spec_model_edges"
            ].data
            plot_spectral_weights(edges, fit)
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
        default="hdf5",
        help="Data source type",
    )
    model.add_argument("--model-options", "--mo", default={}, help="Model options as yaml dict")

    pars = parser.add_argument_group("fit", "Set fit procedure")
    pars.add_argument(
        "--data",
        default="model",
        choices=DATA_INDICES.keys(),
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
