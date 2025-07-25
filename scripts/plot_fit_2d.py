#!/usr/bin/env python
r"""Script for 2d fit plot.

Examples
--------
Example of call::

    ./scripts/plot_fit_2d.py \
      --input-fit fit-a.yaml \
      --output-fit-label-a a \
      --compare-fits fit-b.yaml fit-c.yaml \
      --output-box \
      --output-fit-labels-b b c \
      --output-show
"""
from argparse import Namespace

from matplotlib import pyplot as plt
from yaml import safe_load as yaml_load

from scripts import plot_fit_2d

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

    with open(args.input_fit, "r") as f:
        fit = yaml_load(f)

    plot_fit_2d(
        args.input_fit,
        args.compare_fits,
        xlim=args.output_fit_xlim,
        ylim=args.output_fit_ylim,
        label_a=args.output_fit_label_a,
        labels_b=args.output_fit_labels_b,
        legend_title=args.output_fit_legend_title.format(**fit),
        add_box=args.output_box,
        add_global_normalization=args.global_normalization,
        add_nsigma_legend=False,
    )
    if args.output_plot_fit:
        plt.savefig(args.output_plot_fit, metadata={"creationDate": None})

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
    outputs.add_argument("--output-fit-legend-title", default="")
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

    outputs.add_argument(
        "--global-normalization",
        "--gn",
        action="store_true",
        help="Plot also global normalization",
    )

    args = parser.parse_args()

    main(args)
