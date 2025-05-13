#!/usr/bin/env python
"""
Script for fit model to observed/model data

Example of call:
```
./scripts/plot_fit_dayabay.py --version v0e \
    --mo "{dataset: b, monte_carlo_mode: poisson, seed: 1}" \
    --output-plot-spectra "output/obs-{}.pdf" \
    --output-fit output/fit.yaml
```
"""
from argparse import Namespace

from IPython import embed
from matplotlib import pyplot as plt
from yaml import safe_load as yaml_load

from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from models import available_models, load_model
from models.dayabay_labels import LATEX_SYMBOLS
from scripts import calc_box_around, filter_fit, plot_spectra_ratio, plot_spectral_weights
from scripts.covmatrix_mc import calculate_correlation_matrix

set_level(INFO1)

DATA_INDICES = {"model": 0, "loaded": 1}
AD_TO_EH = {
    "AD11": "EH1",
    "AD12": "EH1",
    "AD21": "EH2",
    "AD22": "EH2",
    "AD31": "EH3",
    "AD32": "EH3",
    "AD33": "EH3",
    "AD34": "EH3",
}


plt.rcParams.update(
    {
        "xtick.top": True,
        "xtick.minor.top": True,
        "xtick.minor.visible": True,
        "axes.grid": True,
        "ytick.left": True,
        "ytick.minor.left": True,
        "ytick.right": True,
        "ytick.minor.right": True,
        "ytick.minor.visible": True,
    }
)


def get_obs(storage_generator):
    result = {}
    for key, obs in storage_generator:
        result[key] = obs.data.copy()
    return result


def sum_by_eh(dict_obs) -> dict:
    result = dict(zip(["EH1", "EH2", "EH3"], [0, 0, 0]))
    for detector, obs in dict_obs.items():
        result[AD_TO_EH[detector]] += obs
    return result


def main(args: Namespace) -> None:

    if args.verbose:
        args.verbose = min(args.verbose, 3)
        set_level(globals()[f"INFO{args.verbose}"])

    with open(args.input_fit, "r") as f:
        fit = yaml_load(f)

    if not fit["success"]:
        print("Fit is not succeed")
        exit()

    model = load_model(
        args.version,
        source_type=args.source_type,
        model_options=args.model_options,
    )

    storage = model.storage

    match args.data:
        case "model":
            data_obs = get_obs(
                storage[f"outputs.eventscount.final.{args.compare_concatenation}"].walkjoineditems()
            )
        case "loaded":
            data_obs = get_obs(
                storage[f"outputs.data.real.final.{args.compare_concatenation}"].walkjoineditems()
            )
    model.set_parameters(fit["xdict"])
    fit_obs = get_obs(
        storage[f"outputs.eventscount.final.{args.compare_concatenation}"].walkjoineditems()
    )

    if args.output_plot_correlation_matrix:
        fig, axs = plt.subplots(1, 1, figsize=(6, 5), sharey=True)
        cs = axs.matshow(
            calculate_correlation_matrix(fit["covariance"]), vmin=-1, vmax=1, cmap="RdBu_r"
        )
        plt.title(args.output_correlation_matrix_title)
        plt.colorbar(cs, ax=axs)
        labels = [LATEX_SYMBOLS[parameter] for parameter in fit["names"]]
        axs.set_xticks(range(fit["npars"]), labels)
        axs.tick_params(axis="x", rotation=90)
        axs.set_yticks(range(fit["npars"]), labels)
        axs.grid(False)
        axs.minorticks_off()
        plt.tight_layout()
        plt.savefig(args.output_plot_correlation_matrix)

    if args.output_plot_spectra:
        edges = storage["outputs.edges.energy_final"].data
        for obs_name, data in data_obs.items():
            plot_spectra_ratio(
                fit_obs[obs_name],
                data,
                edges,
                title=args.output_spectra_title.format(obs_name),
                label_a=args.output_spectra_label_a,
                label_b=args.output_spectra_label_b,
            )
            plt.subplots_adjust(hspace=0.0, left=0.1, right=0.875, bottom=0.125, top=0.925)
            plt.savefig(args.output_plot_spectra.format(obs_name.replace(".", "-")))

        if list(filter(lambda x: "neutrino_per_fission_factor" in x, fit["names"])):
            edges = storage["outputs.reactor_anue.spectrum_free_correction.spec_model_edges"].data
            plot_spectral_weights(edges, fit)
            if args.output_sw_ylim:
                plt.ylim(args.output_sw_ylim)
            plt.savefig(args.output_plot_spectra.format("sw"))

    plt.figure()
    eb0 = plt.errorbar(
        fit["xdict"]["oscprob.SinSq2Theta13"],
        fit["xdict"]["oscprob.DeltaMSq32"],
        xerr=fit["errorsdict"]["oscprob.SinSq2Theta13"],
        yerr=fit["errorsdict"]["oscprob.DeltaMSq32"],
        label=args.output_fit_label_a,
    )
    plt.legend(title=args.output_fit_title_legend + f" = {fit['fun']:1.3f}")
    if args.compare_fit:
        with open(args.compare_fit, "r") as f:
            compare_fit = yaml_load(f)
        eb1 = plt.errorbar(
            compare_fit["SinSq2Theta13"]["value"],
            compare_fit["DeltaMSq32"]["value"],
            xerr=compare_fit["SinSq2Theta13"]["error"],
            yerr=compare_fit["DeltaMSq32"]["error"],
            label=args.output_fit_label_b,
        )
        eb1[2][0].set_linestyle("--")
        eb1[2][1].set_linestyle("--")
        (box,) = plt.plot(
            *calc_box_around(
                (compare_fit["SinSq2Theta13"]["value"], compare_fit["DeltaMSq32"]["value"]),
                (compare_fit["SinSq2Theta13"]["error"], compare_fit["DeltaMSq32"]["error"]),
            ),
            color="C1",
            label=r"$0.1\sigma$",
        )
        plt.legend(
            handles=[eb0, eb1, box], title=args.output_fit_title_legend + f" = {fit['fun']:1.3f}"
        )
    plt.xlabel(r"$\sin^22\theta_{13}$")
    plt.ylabel(r"$\Delta m^2_{32}$ [eV$^2$]")
    plt.title("")
    plt.grid()
    if args.output_fit_xlim:
        plt.xlim(args.output_fit_xlim)
        print(args.output_fit_xlim)
    if args.output_fit_ylim:
        plt.ylim(args.output_fit_ylim)
        print(args.output_fit_ylim)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.95)
    if args.output_plot_fit:
        plt.savefig(args.output_plot_fit)
    if args.compare_fit:
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

    if args.output_show:
        plt.show()

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
    model.add_argument(
        "--data",
        default="model",
        choices=DATA_INDICES.keys(),
        help="Choose data for plotting as observed",
    )

    comparison = parser.add_argument_group("comparison", "Comparison options")
    comparison.add_argument(
        "--compare-concatenation",
        choices=["detector", "detector_period"],
        default="detector_period",
        help="Choose concatenation mode for plotting observation",
    )
    comparison.add_argument(
        "--sum-by-eh",
        action="store_true",
        help="Sum detectors by experimental halls",
    )
    comparison.add_argument(
        "--input-fit",
        help="path to file which load as expected",
    )
    comparison.add_argument(
        "--compare-fit",
        help="path to file with which compare",
    )

    outputs = parser.add_argument_group("outputs", "set outputs")
    outputs.add_argument(
        "--output-plot-pars",
        help="path to save plot of normalized values",
    )
    outputs.add_argument(
        "--output-plot-correlation-matrix",
        help="path to save plot of correlation matrix of fitted parameters",
    )
    outputs.add_argument(
        "--output-correlation-matrix-title",
    )
    outputs.add_argument(
        "--output-plot-spectra",
        help="path to save full plot of fits",
    )
    outputs.add_argument(
        "--output-spectra-title",
    )
    outputs.add_argument(
        "--output-plot-fit",
        help="path to save full plot of fits",
    )
    outputs.add_argument(
        "--output-fit-title-legend",
    )
    outputs.add_argument(
        "--output-fit-xlim",
        nargs=2,
        type=float,
    )
    outputs.add_argument(
        "--output-fit-ylim",
        nargs=2,
        type=float,
    )
    outputs.add_argument(
        "--output-sw-ylim",
        nargs=2,
        type=float,
    )
    outputs.add_argument(
        "--output-spectra-label-a",
    )
    outputs.add_argument(
        "--output-spectra-label-b",
    )
    outputs.add_argument(
        "--output-fit-label-a",
    )
    outputs.add_argument(
        "--output-fit-label-b",
    )
    outputs.add_argument(
        "--output-show",
        action="store_true",
    )

    args = parser.parse_args()

    main(args)
