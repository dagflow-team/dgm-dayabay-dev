#!/usr/bin/env python

from argparse import Namespace

from matplotlib import pyplot as plt
from numpy import log, nanmax, nanmin, where

from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, logger, set_level
from models import available_models, available_sources, load_model
from multikeydict.nestedmkdict import NestedMKDict
from multikeydict.tools import mkmap

plt.style.use(
    {
        "figure.figsize": (6.4, 6.4),
        "axes.formatter.limits": (-2, 4),
        "axes.formatter.use_mathtext": True,
        "axes.grid": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.top": True,
        "ytick.right": True,
        "lines.markerfacecolor": "none",
        "savefig.dpi": 300,
    }
)

set_level(INFO1)


def main(opts: Namespace) -> None:
    if opts.verbose:
        opts.verbose = min(opts.verbose, 3)
        set_level(globals()[f"INFO{opts.verbose}"])

    logger.info(f"Create model A: {opts.version_a}")
    modelA = load_model(
        opts.version_a,
        model_options=opts.model_options_a,
        parameter_values=opts.par0,
    )

    logger.info(f"Create model B: {opts.version_b}")
    modelB = load_model(
        opts.version_b,
        model_options=opts.model_options_b,
        parameter_values=opts.par0,
    )

    get_hist = lambda output: (output.dd.axes_edges[0].data, output.data.copy())

    # source = "outputs.eventscount.final.detector"
    source = "outputs.eventscount.fine.ibd_normalized_detector"
    sourceA = modelA.storage.get_dict(source)
    sourceB = modelB.storage.get_dict(source)
    hists0_A = mkmap(get_hist, sourceA)
    hists0_B = mkmap(get_hist, sourceB)
    plot(hists0_A, hists0_B, opts.version_a, opts.version_b, title=source, opts=opts)

    if opts.par:
        title = "\n".join(f"{par}={value}" for (par, value) in opts.par)
        modelA.set_parameters(opts.par)
        modelB.set_parameters(opts.par)
        hists1_A = mkmap(get_hist, sourceA)
        hists1_B = mkmap(get_hist, sourceB)

        make_diff = lambda eh0, eh1: (eh0[0], eh0[1] - eh1[1])
        histsD_A = mkmap(make_diff, hists1_A, hists0_A)
        histsD_B = mkmap(make_diff, hists1_B, hists0_B)

        plot(
            histsD_A,
            histsD_B,
            opts.version_a,
            opts.version_b,
            title=f"{source}\n{title}",
            ylabel="mod−def",
            opts=opts,
        )

    if opts.show:
        plt.show()


def plot(
    storageA: NestedMKDict,
    storageB: NestedMKDict,
    labelA: str,
    labelB: str,
    *,
    opts: Namespace,
    title: str = "",
    xlabel: str = "E, MeV",
    ylabel: str = "entries",
):
    for key, (edgesA, dataA) in storageA.walkitems():
        _, dataB = storageB[key]

        _, (ax, axd, axr) = plt.subplots(
            3,
            1,
            sharex=True,
            height_ratios=(3, 1, 1),
            gridspec_kw={"hspace": 0},
        )
        plt.sca(ax)
        if title:
            ktitle = f"{title}: {'.'.join(key)}"
        else:
            ktitle = f"{'.'.join(key)}"
        ax.set_title(ktitle)
        if ylabel:
            ax.set_ylabel(ylabel)

        ax.stairs(dataA, edgesA, label=labelA)
        ax.stairs(dataB, edgesA, label=labelB)
        ax.legend()

        diff = dataA - dataB
        lratio = log(dataA / dataB)
        lratio[dataB == 0] = 0.0
        istart_ratioA = where(dataA > 0)[0][0]
        istart_ratioB = where(dataB > 0)[0][0]
        istart_ratio = max(istart_ratioA, istart_ratioB)
        lratiom = lratio[istart_ratio:]

        sdiff = diff.sum()
        tratio = sdiff / dataB.sum()

        plt.sca(axd)
        axd.stairs(diff, edgesA, label=f"sum={sdiff:.2f}")
        axd.set_ylabel(f"{labelA}−{labelB}", size="small")
        if opts.ylim:
            axd.set_ylim(*opts.ylim)
        if opts.dlim:
            axd.set_ylim(*opts.dlim)
        axd.legend(fontsize="small")

        plt.sca(axr)
        axr.stairs(lratiom, edgesA[istart_ratio:], label=f"ratio={tratio*100:.2f} %")
        axr.set_ylabel(f"log({labelA}/{labelB})", size="small")
        axr.set_ylim(
            min(nanmin(lratiom) * 0.9, nanmin(lratiom) * 1.1),
            max(nanmax(lratiom) * 0.9, nanmax(lratiom) * 1.1),
        )
        if opts.ylim:
            axr.set_ylim(*opts.ylim)
        if opts.llim:
            axr.set_ylim(*opts.llim)
        axr.legend(fontsize="small")

        if xlabel:
            ax.set_xlabel(xlabel)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", default=0, action="count", help="verbosity level"
    )

    model = parser.add_argument_group("model", "model related options")
    model.add_argument(
        "version_a",
        choices=available_models(),
        help="model A version",
    )
    model.add_argument(
        "version_b",
        choices=available_models(),
        help="model A version",
    )
    model.add_argument(
        "--model-options-a", "--mo-a", default={}, help="Model options as yaml dict"
    )
    model.add_argument(
        "--model-options-b", "--mo-b", default={}, help="Model options as yaml dict"
    )

    pars = parser.add_argument_group("pars", "setup pars")
    pars.add_argument(
        "--par0",
        nargs=2,
        action="append",
        default=[],
        help="set initial parameter value",
    )
    pars.add_argument(
        "--par",
        nargs=2,
        action="append",
        default=[],
        help="set comparison parameter value",
    )

    plotargs = parser.add_argument_group("plot", "plot related options")
    plotargs.add_argument("--llim", type=float, nargs=2, help="log ratio limits")
    plotargs.add_argument("--dlim", type=float, nargs=2, help="diff limits")
    plotargs.add_argument("--ylim", type=float, nargs=2, help="log ratio/diff limits")
    plotargs.add_argument("-s", "--show", action="store_true", help="show plots")
    main(parser.parse_args())
