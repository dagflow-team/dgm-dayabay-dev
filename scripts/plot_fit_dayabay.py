#!/usr/bin/env python
"""Script for fit model to observed/model data.

Example of call:
```
./scripts/plot_fit_dayabay.py --version v0e \
    --mo "{dataset: b, monte_carlo_mode: poisson, seed: 1}" \
    --output-plot-spectra "output/obs-{}.pdf" \
    --output-fit output/fit.yaml
```
"""
from argparse import Namespace
from collections.abc import Mapping

import numpy as np
from IPython import embed
from matplotlib import pyplot as plt
from matplotlib import ticker
from yaml import safe_load as yaml_load

from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from models import available_models, load_model
from models.dayabay_labels import LATEX_SYMBOLS
from scripts import (
    calc_box_around,
    filter_fit,
    plot_spectra_ratio,
    plot_spectral_weights,
)
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
                storage[
                    f"outputs.eventscount.final.{args.compare_concatenation}"
                ].walkjoineditems()
            )
        case "loaded":
            data_obs = get_obs(
                storage[
                    f"outputs.data.real.final.{args.compare_concatenation}"
                ].walkjoineditems()
            )
    model.set_parameters(fit["xdict"])
    fit_obs = get_obs(
        storage[
            f"outputs.eventscount.final.{args.compare_concatenation}"
        ].walkjoineditems()
    )

    if args.output_plot_correlation_matrix:
        figsize = (
            None if fit["npars"] < 20 else (0.24 * fit["npars"], 0.2 * fit["npars"])
        )
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
            title = (
                "{}, {} period".format(*obs_name.split("."))
                if "." in obs_name
                else obs_name
            )
            plot_spectra_ratio(
                fit_obs[obs_name],
                data,
                edges,
                title=args.output_spectra_title.format(title),
                legend_title=args.output_spectra_legend_title,
                label_a=args.output_spectra_label_a,
                label_b=args.output_spectra_label_b,
            )
            plt.subplots_adjust(
                hspace=0.0, left=0.1, right=0.9, bottom=0.125, top=0.925
            )
            plt.savefig(args.output_plot_spectra.format(obs_name.replace(".", "-")))

        if list(filter(lambda x: "neutrino_per_fission_factor" in x, fit["names"])):
            edges = storage[
                "outputs.reactor_anue.spectrum_free_correction.spec_model_edges"
            ].data
            plot_spectral_weights(edges, fit)
            plt.subplots_adjust(left=0.12, right=0.95, bottom=0.10, top=0.90)
            if args.output_sw_ylim:
                plt.ylim(args.output_sw_ylim)
            plt.savefig(args.output_plot_spectra.format("sw"))

    if args.global_norm:
        fig, (ax, axgn) = plt.subplots(
            1,
            2,
            width_ratios=(4, 1),
            gridspec_kw={
                "wspace": 0,
            },
            subplot_kw={},
        )
    else:
        fig, ax = plt.subplots(1, 1)
        axgn = None

    if args.compare_fit:
        with open(args.compare_fit, "r") as f:
            compare_fit = yaml_load(f)

        compare_xdict = compare_fit["xdict"]
        compare_errorsdict = compare_fit["errorsdict"]

        ax.errorbar(
            compare_xdict["oscprob.SinSq2Theta13"],
            compare_xdict["oscprob.DeltaMSq32"],
            xerr=compare_errorsdict["oscprob.SinSq2Theta13"],
            yerr=compare_errorsdict["oscprob.DeltaMSq32"],
            label=args.output_fit_label_a,
        )
        (box,) = ax.plot(
            *calc_box_around(
                (
                    compare_xdict["oscprob.SinSq2Theta13"],
                    compare_xdict["oscprob.DeltaMSq32"],
                ),
                (
                    compare_errorsdict["oscprob.SinSq2Theta13"],
                    compare_errorsdict["oscprob.DeltaMSq32"],
                ),
            ),
            color="C0",
            label=r"$0.1\sigma$",
        )

        if axgn:
            gn_value, gn_error, gn_type = get_global_normalization(
                compare_xdict, compare_errorsdict
            )
            axgn.errorbar(
                0.1,
                gn_value,
                yerr=gn_error,
                xerr=1,
                fmt="o",
                markerfacecolor="none",
                label=gn_type,
            )

    xdict = fit["xdict"]
    errorsdict = fit["errorsdict"]
    eb = ax.errorbar(
        xdict["oscprob.SinSq2Theta13"],
        xdict["oscprob.DeltaMSq32"],
        xerr=errorsdict["oscprob.SinSq2Theta13"],
        yerr=errorsdict["oscprob.DeltaMSq32"],
        label=args.output_fit_label_b,
    )
    eb[2][0].set_linestyle("--")
    eb[2][1].set_linestyle("--")
    ax.legend(title=args.output_fit_title_legend.format(fun=fit['fun']))
    ax.set_xlabel(r"$\sin^22\theta_{13}$")
    ax.set_ylabel(r"$\Delta m^2_{32}$ [eV$^2$]")
    plt.title("")
    plt.grid()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)
    if args.output_fit_xlim:
        plt.xlim(args.output_fit_xlim)
    if args.output_fit_ylim:
        plt.ylim(args.output_fit_ylim)

    if axgn:
        axgn.yaxis.set_label_position("right")
        axgn.set_ylabel("Normalization offset")
        axgn.tick_params(labelleft=False, labelright=True, labelbottom=False)
        axgn.grid(axis="y")
        axgn.set_ylim(-0.15, 0.05)
        axgn.set_xlim(-1.5, 1.5)

        gn_value, gn_error, gn_type = get_global_normalization(xdict, errorsdict)
        axgn.errorbar(
            0,
            gn_value,
            yerr=gn_error,
            xerr=1,
            fmt="o",
            markerfacecolor="none",
            label=gn_type,
        )

    plt.subplots_adjust(left=0.12, right=0.86, bottom=0.11, top=0.90)
    if args.output_plot_fit:
        plt.savefig(args.output_plot_fit)
    if args.compare_fit:
        for name, par_values in compare_fit["xdict"].items():
            if name not in fit["xdict"].keys():
                continue
            fit_value = fit["xdict"][name]
            fit_error = fit["errorsdict"][name]
            value = par_values
            error = compare_fit["errorsdict"][name]
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


def get_global_normalization(
    xdict: Mapping, errorsdict: Mapping
) -> tuple[float, float, str]:
    key = "detector.global_normalization"
    try:
        return xdict[key] - 1.0, errorsdict[key], "fit"
    except KeyError:
        pass

    names = [
        name
        for name in xdict
        if name.startswith("neutrino_per_fission_factor.spec_scale")
    ]
    scale = np.array([xdict[name] for name in names])
    unc = np.array([errorsdict[name] for name in names])
    w = unc**-2
    wsum = w.sum()
    res = (scale * w).sum() / wsum
    # res_unc = wsum**-0.5 # incorrect since scales are correlated

    return res, 0.0, "calc"


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
        default="hdf5",
        help="Data source type",
    )
    model.add_argument(
        "--model-options", "--mo", default={}, help="Model options as yaml dict"
    )
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
        "--output-spectra-legend-title",
        default="",
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

    outputs.add_argument(
        "--global-norm",
        "--gn",
        action="store_true",
        help="Plot also global normalization",
    )

    args = parser.parse_args()

    main(args)
