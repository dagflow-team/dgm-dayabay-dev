#!/usr/bin/env python
"""
Script for 2d fit plot

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
from itertools import zip_longest

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

    plt.figure()
    plt.errorbar(
        fit["xdict"]["oscprob.SinSq2Theta13"],
        fit["xdict"]["oscprob.DeltaMSq32"],
        xerr=fit["errorsdict"]["oscprob.SinSq2Theta13"],
        yerr=fit["errorsdict"]["oscprob.DeltaMSq32"],
        label=args.output_fit_label_a,
    )
    plt.legend(title=args.output_fit_title_legend + f" = {fit['fun']:1.3f}")
    if args.output_box:
        (box,) = plt.plot(
            *calc_box_around(
                (fit["xdict"]["oscprob.SinSq2Theta13"], fit["xdict"]["oscprob.DeltaMSq32"]),
                (fit["errorsdict"]["oscprob.SinSq2Theta13"], fit["errorsdict"]["oscprob.DeltaMSq32"]),
            ),
            color="C0",
            label=r"$0.1\sigma$",
        )
    for compare_fit, label_b in zip_longest(args.compare_fits, args.output_fit_labels_b, fillvalue=None):
        with open(compare_fit, "r") as f:
            compare_fit = yaml_load(f)
        eb1 = plt.errorbar(
            compare_fit["xdict"]["oscprob.SinSq2Theta13"],
            compare_fit["xdict"]["oscprob.DeltaMSq32"],
            xerr=compare_fit["errorsdict"]["oscprob.SinSq2Theta13"],
            yerr=compare_fit["errorsdict"]["oscprob.DeltaMSq32"],
            label=label_b,
        )
        # plt.legend(title=args.output_fit_title_legend + f" = {fit['fun']:1.3f}")
    plt.xlabel(r"$\sin^22\theta_{13}$")
    plt.ylabel(r"$\Delta m^2_{32}$ [eV$^2$]")
    plt.title("")
    plt.grid()
    if args.output_fit_xlim:
        plt.xlim(args.output_fit_xlim)
    if args.output_fit_ylim:
        plt.ylim(args.output_fit_ylim)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.95)
    if args.output_plot_fit:
        plt.savefig(args.output_plot_fit)

    if args.output_show:
        plt.show()


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
    # outputs.add_argument(
    #     "--output-fit-title-legend",
    # )
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

    args = parser.parse_args()

    main(args)
