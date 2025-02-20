#!/usr/bin/env python

from __future__ import annotations

from argparse import Namespace

from matplotlib import pyplot as plt
from matplotlib import rcParams, transforms

from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from models import available_models, load_model

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
    eff_storage = storage["outputs.daily_data.detector.eff"]
    efflivetime_storage = storage["outputs.daily_data.detector.efflivetime"]
    rate_acc_storage = storage["outputs.daily_data.detector.rate_acc"]

    ads = ["AD11", "AD12", "AD21", "AD22", "AD31", "AD32", "AD33", "AD34"]
    ads = {ad: i for i, ad in enumerate(ads)}

    gridspec_kw = {
        "hspace": 0,
        "left": 0.08,
        "right": 0.92,
        "bottom": 0.05,
        "top": 0.95,
    }
    figsize = (12, 10)
    fig_eff, axes_eff = plt.subplots(
        8,
        1,
        sharex=True,
        figsize=figsize,
        subplot_kw={"ylabel": r"$\varepsilon$, %"},
        gridspec_kw=gridspec_kw,
    )
    fig_efflivetime, axes_efflivetime = plt.subplots(
        8,
        1,
        sharex=True,
        figsize=figsize,
        subplot_kw={"ylabel": r"T, day"},
        gridspec_kw=gridspec_kw,
    )
    fig_rate_acc, axes_rate_acc = plt.subplots(
        8,
        1,
        sharex=True,
        figsize=figsize,
        subplot_kw={"ylabel": r"R, #/day"},
        gridspec_kw=gridspec_kw,
    )
    text_offset = transforms.ScaledTranslation(0.04, 0.04, fig_eff.dpi_scale_trans)

    axes_eff[0].set_title("Efficiency (muon veto, multiplicity)")
    axes_efflivetime[0].set_title("Effective livetime")
    axes_rate_acc[0].set_title("Accidentals rate")

    labels_added = set()

    seconds_in_day = 60 * 60 * 24
    plot_kwargs = dict(markersize=0.5, color="C0")
    for (period, ad), output in eff_storage.walkitems():
        data_days = days_storage[period].data
        eff_data = output.data
        efflivetime_data = efflivetime_storage[period, ad].data
        rate_acc_data = rate_acc_storage[period, ad].data

        ad_id = ads[ad]

        ax_eff = axes_eff[ad_id]
        mask = eff_data > 0
        ax_eff.plot(data_days[mask], eff_data[mask] * 100, ".", **plot_kwargs)

        ax_efflivetime = axes_efflivetime[ad_id]
        mask = efflivetime_data > 0
        ax_efflivetime.plot(
            data_days[mask], efflivetime_data[mask] / seconds_in_day, ".", **plot_kwargs
        )

        ax_rate_acc = axes_rate_acc[ad_id]
        mask = rate_acc_data > 0
        ax_rate_acc.plot(data_days[mask], rate_acc_data[mask], ".", **plot_kwargs)

        ticks_right = bool(ad_id % 2)
        for ax in (ax_eff, ax_efflivetime, ax_rate_acc):
            if ad not in labels_added:
                ax.text(
                    1, 1, ad, transform=ax.transAxes - text_offset, va="top", ha="right"
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

        labels_added.add(ad)

    for axes in (axes_eff, axes_efflivetime, axes_rate_acc):
        ax = axes[-1]
        ax.set_xlabel("Day since start of data taking")
        ax.set_xlim(left=0)

    if opts.output:
        for plot_type, fig in {
            "eff": fig_eff,
            "efflivetime": fig_efflivetime,
            "rate_acc": fig_rate_acc,
        }.items():
            if "{type" not in opts.output:  # }
                raise RuntimeError("Output format should contain {type} for plot type")

            if "data-a" in model._future:
                selection = "A"
            elif "data-b" in model._future:
                selection = "B"
            else:
                selection = "d"

            fname = opts.output.format(type=plot_type, selection=selection, s=selection)
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
        default="recent",
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
        help='output files (supported format keys: "type", "selection")',
    )
    plot.add_argument("-s", "--show", action="store_true", help="show")

    main(parser.parse_args())
