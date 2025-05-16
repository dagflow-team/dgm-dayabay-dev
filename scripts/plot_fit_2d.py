#!/usr/bin/env python
"""Script for 2d fit plot.

Example of call:
```
./scripts/plot_fit_2d.py \
    --input-fit fit-a.yaml \
    --output-fit-label-a a \
    --compare-fits fit-b.yaml fit-c.yaml \
    --output-box \
    --output-fit-labels-b b c \
    --output-show
```
"""
from argparse import Namespace
from collections.abc import Mapping
from itertools import zip_longest

import numpy as np
from matplotlib import pyplot as plt
from yaml import safe_load as yaml_load

from scripts import calc_box_around

plt.rcParams.update(
    {
        "xtick.top": True,
        "xtick.minor.top": True,
        "xtick.minor.visible": True,
        "ytick.left": True,
        "ytick.minor.left": True,
        "ytick.right": True,
        "ytick.minor.right": True,
        "ytick.minor.visible": True,
    }
)


def main(args: Namespace) -> None:

    with open(args.input_fit, "r") as f:
        fit = yaml_load(f)

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
        fig, (ax,) = plt.subplots(1, 1)
        axgn = None

    xdict = fit["xdict"]
    errorsdict = fit["errorsdict"]

    ax.errorbar(
        xdict["oscprob.SinSq2Theta13"],
        xdict["oscprob.DeltaMSq32"],
        xerr=errorsdict["oscprob.SinSq2Theta13"],
        yerr=errorsdict["oscprob.DeltaMSq32"],
        label=args.output_fit_label_a,
    )
    if args.output_box:
        (box,) = ax.plot(
            *calc_box_around(
                (
                    xdict["oscprob.SinSq2Theta13"],
                    xdict["oscprob.DeltaMSq32"],
                ),
                (
                    errorsdict["oscprob.SinSq2Theta13"],
                    errorsdict["oscprob.DeltaMSq32"],
                ),
            ),
            color="C0",
            label=r"$0.1\sigma$",
        )

    if axgn:
        axgn.yaxis.set_label_position("right")
        axgn.set_ylabel("Normalization offset")
        axgn.tick_params(labelleft=False, labelright=True, labelbottom=False)
        axgn.grid(axis="y")
        axgn.set_ylim(-0.15, 0.05)

        gn_value, gn_error, gn_type = get_global_normalization(xdict, errorsdict)
        axgn.errorbar(
            0,
            gn_value,
            yerr=gn_error,
            xerr=1,
            fmt="o",
            markerfacecolor="none",
            color="C0",
            label=gn_type
        )

    for compare_fit, label_b in zip_longest(
        args.compare_fits, args.output_fit_labels_b, fillvalue=None
    ):
        with open(compare_fit, "r") as f:
            compare_fit = yaml_load(f)

        compare_xdict = compare_fit["xdict"]
        compare_errorsdict = compare_fit["errorsdict"]

        eb1 = ax.errorbar(
            compare_xdict["oscprob.SinSq2Theta13"],
            compare_xdict["oscprob.DeltaMSq32"],
            xerr=compare_errorsdict["oscprob.SinSq2Theta13"],
            yerr=compare_errorsdict["oscprob.DeltaMSq32"],
            label=label_b,
        )
    ax.legend(title=args.output_fit_title_legend)
    ax.set_xlabel(r"$\sin^22\theta_{13}$")
    ax.set_ylabel(r"$\Delta m^2_{32}$ [eV$^2$]")
    ax.set_title("")
    ax.grid()
    if args.output_fit_xlim:
        ax.xlim(args.output_fit_xlim)
    if args.output_fit_ylim:
        ax.ylim(args.output_fit_ylim)

    plt.subplots_adjust(left=0.17, right=0.86, bottom=0.1, top=0.95)
    if args.output_plot_fit:
        plt.savefig(args.output_plot_fit)

    if args.output_show:
        plt.show()


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

    comparison = parser.add_argument_group("comparison", "Comparison options")
    comparison.add_argument(
        "--input-fit",
        help="path to file with reference fit values",
    )
    comparison.add_argument(
        "--compare-fits",
        default=[],
        nargs="*",
        action="extend",
        help="path to files with which compare",
    )

    outputs = parser.add_argument_group("outputs", "set outputs")
    outputs.add_argument(
        "--output-plot-fit",
        help="path to save full plot of fits",
    )
    outputs.add_argument(
        "--output-box",
        action="store_true",
        help="Draw 0.1sigma box around input_fit",
    )
    outputs.add_argument("--output-fit-title-legend", default="")
    outputs.add_argument(
        "--output-fit-label-a",
    )
    outputs.add_argument(
        "--output-fit-xlim",
        type=float,
    )
    outputs.add_argument(
        "--output-fit-ylim",
        type=float,
    )
    outputs.add_argument(
        "--output-fit-labels-b",
        default=[],
        nargs="*",
        action="extend",
    )
    outputs.add_argument(
        "--output-show",
        action="store_true",
    )

    outputs.add_argument("--global-norm", "--gn", action="store_true")

    args = parser.parse_args()

    main(args)
