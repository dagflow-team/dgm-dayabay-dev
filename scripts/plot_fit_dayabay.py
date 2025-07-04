#!/usr/bin/env python
"""Script for fit model to observed/model data.

Examples
--------
Example of call:: 

    ./scripts/plot_fit_dayabay.py --version v0e --data model \\
      --mo "{dataset: a}" \\
      --compare-concatenation detector \\
      --input-fit fit.yaml \\
      --reference-fit reference-fit.yaml \\
      --output-plot-fit "output/fit_2d.pdf" \\
      --output-fit-label-a "Reference fit" \\
      --output-fit-label-b "Public model" \\
      --output-plot-spectra "output/fit-{}.pdf" \\
      --output-spectra-title "Fit to modeled data, {}" \\
      --output-plot-correlation-matrix "output/fit-correlation-matrix.pdf" \\
      --output-correlation-matrix-title "Post-fit parameters correlation matrix" \\
      --output-spectra-label-a "Public model" \\
      --output-spectra-label-b "Modeled data" \\
      --output-sw-ylim -0.5 0.5 \\
      --output-fit-xlim 0.080 0.010 \\
      --output-fit-ylim 0.002 0.003 \\
      --output-fit-sigma-cross \\
      --output-fit-title-legend "legend for 2d plot fit" \\
      --gn \\
      --output-show
"""
from argparse import Namespace

from IPython import embed
from matplotlib import pyplot as plt
from yaml import safe_load as yaml_load

from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from models import AD_TO_EH, available_models, load_model
from models.dayabay_labels import LATEX_SYMBOLS
from scripts import (
    filter_covariance_matrix,
    get_obs,
    plot_fit_2d,
    plot_spectra_ratio,
    plot_spectral_weights,
)
from scripts.covmatrix_mc import calculate_correlation_matrix

set_level(INFO1)

DATA_INDICES = {"model": 0, "loaded": 1}


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

    data_to_observation = {
        "model": "eventscount",
        "loaded": "data.real",
    }
    subname = data_to_observation[args.data]
    data_obs = get_obs(
        storage[f"outputs.{subname}.final.{args.compare_concatenation}"].walkjoineditems()
    )
    model.set_parameters(fit["xdict"])
    fit_obs = get_obs(
        storage[f"outputs.eventscount.final.{args.compare_concatenation}"].walkjoineditems()
    )

    if args.output_plot_correlation_matrix:
        selected_parameters = args.correlation_matrix_parameters
        matrix, parameters = filter_covariance_matrix(
            fit["covariance"], fit["names"], selected_parameters
        )
        npars = len(parameters)
        figsize = None if npars < 20 else (0.24 * npars, 0.2 * npars)
        fig, axs = plt.subplots(1, 1, figsize=figsize)
        cs = axs.matshow(
            calculate_correlation_matrix(fit["covariance"]),
            vmin=-1,
            vmax=1,
            cmap="RdBu_r",
        )
        plt.title(args.output_correlation_matrix_title)
        plt.colorbar(cs, ax=axs)
        labels = [LATEX_SYMBOLS[parameter] for parameter in fit["names"]]
        axs.set_xticks(range(npars), labels)
        axs.tick_params(axis="x", rotation=90)
        axs.set_yticks(range(npars), labels)
        axs.grid(False)
        axs.minorticks_off()
        plt.tight_layout()
        plt.savefig(args.output_plot_correlation_matrix, metadata={"creationDate": None})

    if args.output_plot_spectra:
        edges = storage["outputs.edges.energy_final"].data
        for obs_name, data in data_obs.items():
            title = "{}, {} period".format(*obs_name.split(".")) if "." in obs_name else obs_name
            plot_spectra_ratio(
                fit_obs[obs_name],
                data,
                edges,
                title=args.output_spectra_title.format(title),
                legend_title=args.output_spectra_legend_title,
                label_a=args.output_spectra_label_a,
                label_b=args.output_spectra_label_b,
            )
            plt.subplots_adjust(hspace=0.0, left=0.1, right=0.9, bottom=0.125, top=0.925)
            plt.savefig(
                args.output_plot_spectra.format(obs_name.replace(".", "-")),
                metadata={"creationDate": None},
            )

        if list(filter(lambda x: "neutrino_per_fission_factor" in x, fit["names"])):
            edges = storage["outputs.reactor_anue.spectrum_free_correction.spec_model_edges"].data
            plot_spectral_weights(edges, fit)
            plt.subplots_adjust(left=0.12, right=0.95, bottom=0.10, top=0.90)
            if args.output_sw_ylim:
                plt.ylim(args.output_sw_ylim)
            plt.savefig(args.output_plot_spectra.format("sw"), metadata={"creationDate": None})

    plot_fit_2d(
        args.reference_fit,
        [args.input_fit],
        xlim=args.output_fit_xlim,
        ylim=args.output_fit_ylim,
        label_a=args.output_fit_label_a,
        labels_b=[args.output_fit_label_b],
        legend_title=args.output_fit_legend_title.format(**fit),
        add_sigma_cross=args.output_fit_sigma_cross,
        dashed_comparison=True,
        add_global_normalization=args.global_normalization,
        add_nsigma_legend=True,
    )
    if args.output_plot_fit:
        plt.savefig(args.output_plot_fit, metadata={"creationDate": None})
    if args.reference_fit:
        with open(args.reference_fit, "r") as f:
            reference_fit = yaml_load(f)

        reference_xdict = reference_fit["xdict"]
        reference_errorsdict = reference_fit["errorsdict"]
        for name, par_values in reference_xdict.items():
            if name not in fit["xdict"].keys():
                continue
            fit_value = fit["xdict"][name]
            fit_error = fit["errorsdict"][name]
            value = par_values
            error = reference_errorsdict[name]
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
        help="Model version",
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
        "--input-fit",
        help="Path to file which load as expected",
    )
    comparison.add_argument(
        "--reference-fit",
        help="Path to file with which compare",
    )

    outputs = parser.add_argument_group("outputs", "set outputs")
    outputs.add_argument(  # NOTE: Used nowhere
        "--output-plot-pars",
        help="Path to save plot of normalized values",
    )
    outputs.add_argument(
        "--output-plot-correlation-matrix",
        help="Path to save plot of correlation matrix of fitted parameters",
    )
    outputs.add_argument(
        "--correlation_matrix_parameters",
        default=[],
        nargs="*",
        help="Choose parameters for correlation matrix to be plotted, support filtering",
    )
    outputs.add_argument(
        "--output-correlation-matrix-title",
        help="Set title for correlation matrix plot",
    )
    outputs.add_argument(
        "--output-plot-spectra",
        help="Path to save full plot of fits, needs placeholder",
    )
    outputs.add_argument(
        "--output-spectra-title",
        help="Set title for spectra plot",
    )
    outputs.add_argument(
        "--output-spectra-legend-title",
        default="",
        help="Set title for legend of spectra plot",
    )
    outputs.add_argument(
        "--output-plot-fit",
        help="Path to save full plot of fits (2d)",
    )
    outputs.add_argument(
        "--output-fit-legend-title",
        help="Set title for legend of fit plot",
    )
    outputs.add_argument(
        "--output-fit-sigma-cross",
        action="store_true",
        help="Plot 0.1σ width cross with the respect to reference fit in fit plot",
    )
    outputs.add_argument(
        "--output-fit-xlim",
        nargs=2,
        type=float,
        help="X limits of fit plot",
    )
    outputs.add_argument(
        "--output-fit-ylim",
        nargs=2,
        type=float,
        help="Y limits of fit plot",
    )
    outputs.add_argument(
        "--output-sw-ylim",
        nargs=2,
        type=float,
        help="Y limits of spectral weights plot",
    )
    outputs.add_argument(
        "--output-spectra-label-a",
        help="Set label for expected observation of spectra plot",
    )
    outputs.add_argument(
        "--output-spectra-label-b",
        help="Set label for observed observation of spectra plot",
    )
    outputs.add_argument(
        "--output-fit-label-a",
        help="Set label for observed observation of fit plot (2d)",
    )
    outputs.add_argument(
        "--output-fit-label-b",
        help="Set label for expected observation of fit plot (2d)",
    )
    outputs.add_argument(
        "--output-show",
        action="store_true",
        help="Show plots",
    )

    outputs.add_argument(
        "--global-normalization",
        "--gn",
        action="store_true",
        help="Plot also global normalization in fit plot (2d)",
    )

    args = parser.parse_args()

    main(args)
