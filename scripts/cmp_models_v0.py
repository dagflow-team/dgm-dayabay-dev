#!/usr/bin/env python
from argparse import Namespace

from matplotlib import pyplot as plt

from dagflow.logger import DEBUG as INFO4
from dagflow.logger import INFO1, INFO2, INFO3, logger, set_level
from dagflow.plot import plot_auto
from dagflow.storage import NodeStorage
from models import available_models, available_sources, load_model
from multikeydict.nestedmkdict import NestedMKDict

# from dagflow.plot import plot_auto

set_level(INFO1)


def main(opts: Namespace) -> None:
    if opts.verbose:
        opts.verbose = min(opts.verbose, 3)
        set_level(globals()[f"INFO{opts.verbose}"])

    logger.info(f"Create model A: {opts.version_a}")
    modelA = load_model(
        opts.version_a,
        model_options=opts.model_options_a,
        source_type=opts.source_type,
        parameter_values=opts.par,
    )

    logger.info(f"Create model B: {opts.version_b}")
    modelB = load_model(
        opts.version_b,
        model_options=opts.model_options_b,
        source_type=opts.source_type,
        parameter_values=opts.par,
    )

    source = "outputs.eventscount.final.detector"
    plot(
        modelA.storage.get_dict(source),
        modelB.storage.get_dict(source),
        opts.version_a,
        opts.version_b,
    )


def plot(storageA: NestedMKDict, storageB: NestedMKDict, labelA: str, labelB: str):
    for key, outputA in storageA.walkitems():
        outputB = storageB[key]

        _, (ax, axr) = plt.subplots(
            2,
            1,
            sharex=True,
            height_ratios=(3, 1),
            gridspec_kw={"hspace": 0},
        )
        plt.sca(ax)
        plot_auto(outputA, label=labelA)
        plot_auto(outputB, label=labelB)
        ax.legend()
        ax.set_ylabel("entries")

        plt.sca(axr)
        diff = outputA.data - outputB.data
        edges = outputA.dd.axes_edges[0].data
        axr.stairs(diff, edges)
        axr.set_xlabel(ax.get_xlabel())
        axr.set_ylabel(f"{labelA}—{labelB}")

        import IPython; IPython.embed(colors='neutral')

    plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", default=0, action="count", help="verbosity level"
    )
    parser.add_argument(
        "-s",
        "--source-type",
        "--source",
        choices=available_sources(),
        default="npz",
        help="Data source type",
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
        "--par", nargs=2, action="append", default=[], help="set parameter value"
    )

    main(parser.parse_args())
