#!/usr/bin/env python

from __future__ import annotations

from argparse import Namespace

from dag_modelling.tools.logger import set_verbosity
from matplotlib import pyplot as plt
from matplotlib import transforms

from dgm_dayabay_dev.models import load_model

plt.rcParams.update(
    {
        "axes.grid": False,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)


def main(opts: Namespace) -> None:
    if opts.verbose:
        set_verbosity(opts.verbose)

    model = load_model(
        opts.version,
        model_options=opts.model_options,
        parameter_values=opts.par,
    )

    storage = model.storage

    if opts.method:
        method = getattr(model, opts.method)
        assert method

        method()

    days_storage = storage["outputs.daily_data.days"]
    neutrino_rate_storage = storage["outputs.daily_data.reactor.antineutrino_rate_per_s"]

    reactors = ["R1", "R2", "R3", "R4", "R5", "R6"]
    reactors = {ad: i for i, ad in enumerate(reactors)}

    gridspec_kw = {
        "hspace": 0,
        "left": 0.08,
        "right": 0.92,
        "bottom": 0.05,
        "top": 0.95,
    }
    figsize = (12, 10)
    fig_nr, axes_nr = plt.subplots(
        6,
        1,
        sharex=True,
        figsize=figsize,
        subplot_kw={"ylabel": r"rate, $s^{-1}$"},
        gridspec_kw=gridspec_kw,
    )

    text_offset = transforms.ScaledTranslation(0.04, 0.04, fig_nr.dpi_scale_trans)
    axes_nr[0].set_title("Neutrino rate")

    labels_added = set()

    plot_kwargs0 = dict(markersize=0.5)
    for (reactor, period), output in neutrino_rate_storage.walkitems():
        data_days = days_storage[period].data

        reactor_id = reactors[reactor]

        ax_nr = axes_nr[reactor_id]
        nr_data = output.data
        mask = nr_data>0

        ax_nr.plot(
            data_days[mask],
            nr_data[mask] * 100,
            ".",
            **plot_kwargs0,
        )

        ticks_right = bool(reactor_id % 2)
        if reactor not in labels_added:
            ax_nr.text(
                1,
                1,
                reactor,
                transform=ax_nr.transAxes - text_offset,
                va="top",
                ha="right",
            )

        ax_nr.tick_params(
            axis="y",
            which="both",
            left=True,
            right=True,
            labelleft=not ticks_right,
            labelright=ticks_right,
        )
        if ticks_right:
            ax_nr.yaxis.set_label_position("right")

        labels_added.add(reactor)

    ax = axes_nr[-1]
    ax.set_xlabel("Day since start of data taking")
    ax.set_xlim(left=0)

    if opts.output:
        fig_nr.savefig(opts.output)
        print(f"Save plot: {opts.output}")

    if opts.show or not opts.output:
        plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", default=1, action="count", help="verbosity level")

    model = parser.add_argument_group("model", "model related options")
    model.add_argument(
        "--version",
        default="latest",
        choices=["v2"],
        help="model version",
    )
    model.add_argument("--model-options", "--mo", default={}, help="Model options as yaml dict")
    model.add_argument("--method", help="Call model's method")

    pars = parser.add_argument_group("pars", "setup pars")
    pars.add_argument("--par", nargs=2, action="append", default=[], help="set parameter value")

    plot = parser.add_argument_group("pars", "plots")
    plot.add_argument(
        "-o",
        "--output",
        help="output file",
    )
    plot.add_argument("-s", "--show", action="store_true", help="show")

    main(parser.parse_args())
