#!/usr/bin/env python

from __future__ import annotations

from argparse import Namespace

from matplotlib import pyplot as plt
from matplotlib import rcParams, transforms

from dag_modelling.tools.logger import DEBUG as INFO4
from dag_modelling.tools.logger import INFO1, INFO2, INFO3, set_level
from ..models import available_models, load_model

set_level(INFO1)

plt.rcParams.update(
    {
        "axes.grid": False,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)


def main(opts: Namespace) -> None:
    if opts.verbose:
        opts.verbose = min(opts.verbose, 3)
        set_level(globals()[f"INFO{opts.verbose}"])

    model = load_model(
        opts.version,
        model_options=opts.model_options,
        source_type=opts.source_type,
        parameter_values=opts.par,
    )

    storage = model.storage

    if opts.method:
        method = getattr(model, opts.method)
        assert method

        method()

    days_storage = storage["outputs.daily_data.days"]
    power_storage = storage["outputs.daily_data.reactor.power"]
    fission_fraction_storage = storage["outputs.daily_data.reactor.fission_fraction"]

    reactors = ["DB1", "DB2", "LA1", "LA2", "LA3", "LA4"]
    reactors = {ad: i for i, ad in enumerate(reactors)}

    gridspec_kw = {
        "hspace": 0,
        "left": 0.08,
        "right": 0.92,
        "bottom": 0.05,
        "top": 0.95,
    }
    figsize = (12, 10)
    fig_power, axes_power = plt.subplots(
        6,
        1,
        sharex=True,
        figsize=figsize,
        subplot_kw={"ylabel": r"$W_{\rm th}$, %"},
        gridspec_kw=gridspec_kw,
    )
    fig_ff, axes_ff = plt.subplots(
        6,
        1,
        sharex=True,
        figsize=figsize,
        subplot_kw={"ylabel": r"f, %"},
        gridspec_kw=gridspec_kw,
    )
    text_offset = transforms.ScaledTranslation(0.04, 0.04, fig_power.dpi_scale_trans)

    axes_power[0].set_title("Thermal power")
    axes_ff[0].set_title("Fission fraction")

    labels_added = set()

    plot_kwargs0 = dict(markersize=0.5)
    plot_kwargs = dict(color="C0", **plot_kwargs0)
    for (reactor, period), output in power_storage.walkitems():
        data_days = days_storage[period].data
        power_data = output.data

        reactor_id = reactors[reactor]

        ax_power = axes_power[reactor_id]
        mask = power_data > 0
        ax_power.plot(data_days[mask], power_data[mask] * 100, ".", **plot_kwargs)

        ax_ff = axes_ff[reactor_id]
        for iisotope, (isotope, ff_isotope_storage) in enumerate(
            fission_fraction_storage[reactor].items()
        ):
            ff_data = ff_isotope_storage[period].data

            ax_ff.plot(
                data_days[mask],
                ff_data[mask] * 100,
                ".",
                color=f"C{iisotope}",
                label=isotope,
                **plot_kwargs0,
            )

        ticks_right = bool(reactor_id % 2)
        for ax in (ax_power, ax_ff):
            if reactor not in labels_added:
                ax.text(
                    1,
                    1,
                    reactor,
                    transform=ax.transAxes - text_offset,
                    va="top",
                    ha="right",
                )

            ax.tick_params(
                axis="y",
                which="both",
                left=not ticks_right,
                right=ticks_right,
                labelleft=not ticks_right,
                labelright=ticks_right,
            )
            if ticks_right:
                ax.yaxis.set_label_position("right")

        labels_added.add(reactor)

    for axes in (axes_power, axes_ff):
        ax = axes[-1]
        ax.set_xlabel("Day since start of data taking")
        ax.set_xlim(left=0)

    if opts.output:
        for plot_type, fig in {
            "power": fig_power,
            "fission_fraction": fig_ff,
        }.items():
            if "{type" not in opts.output:  # }
                raise RuntimeError("Output format should contain {type} for plot type")

            fname = opts.output.format(type=plot_type)
            fig.savefig(fname)
            print(f"Save plot: {fname}")

    if opts.show or not opts.output:
        plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", default=0, action="count", help="verbosity level"
    )
    parser.add_argument(
        "--source-type",
        "--source",
        choices=("tsv", "hdf5", "root", "npz"),
        default="hdf5",
        help="Data source type",
    )

    model = parser.add_argument_group("model", "model related options")
    model.add_argument(
        "--version",
        default="latest",
        choices=available_models(),
        help="model version",
    )
    model.add_argument(
        "--model-options", "--mo", default={}, help="Model options as yaml dict"
    )
    model.add_argument("--method", help="Call model's method")

    pars = parser.add_argument_group("pars", "setup pars")
    pars.add_argument(
        "--par", nargs=2, action="append", default=[], help="set parameter value"
    )

    plot = parser.add_argument_group("pars", "plots")
    plot.add_argument(
        "-o",
        "--output",
        help='output files (supported format keys: "type")',
    )
    plot.add_argument("-s", "--show", action="store_true", help="show")

    main(parser.parse_args())
