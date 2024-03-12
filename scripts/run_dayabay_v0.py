#!/usr/bin/env python
from argparse import Namespace

from dagflow.graph import Graph
from dagflow.logger import INFO1
from dagflow.logger import INFO2
from dagflow.logger import INFO3
from dagflow.logger import set_level
from dagflow.storage import NodeStorage

from models.dayabay_v0 import model_dayabay_v0
# from dagflow.plot import plot_auto

set_level(INFO1)

def main(opts: Namespace) -> None:
    if opts.verbose:
        opts.verbose = min(opts.verbose, 3)
        set_level(globals()[f"INFO{opts.verbose}"])

    override_indices = {idxdef[0]: tuple(idxdef[1:]) for idxdef in opts.index}
    model = model_dayabay_v0(
        close=opts.close,
        strict=opts.strict,
        source_type=opts.source_type,
        override_indices=override_indices,
        spectrum_correction_mode=opts.spec
    )

    graph = model.graph
    storage = model.storage

    if not graph.closed:
        print("Nodes")
        print(storage("nodes").to_table(truncate=True))
        print("Outputs")
        print(storage("outputs").to_table(truncate=True))
        print("Not connected inputs")
        print(storage("inputs").to_table(truncate=True))

        plot_graph(graph, storage)
        return

    if opts.print_all:
        print(storage.to_table(truncate=True))
    for sources in opts.print:
        for source in sources:
            print(storage(source).to_table(truncate=True))
    if len(storage("inputs")) > 0:
        print("Not connected inputs")
        print(storage("inputs").to_table(truncate=True))

    if opts.plot_all:
        storage("outputs").plot(folder=opts.plot_all)

    storage.to_datax("output/dayabay_v0_data.tex")
    plot_graph(graph, storage)


def plot_graph(graph: Graph, storage: NodeStorage) -> None:
    from dagflow.graphviz import GraphDot

    GraphDot.from_graph(graph, show="all").savegraph("output/dayabay_v0.dot")
    GraphDot.from_graph(
        graph,
        show=["type", "mark", "label", "path"],
        filter={
            "reactor": [0],
            "detector": [0, 1],
            "isotope": [0],
            "period": [0],
            "background": [0],
        },
    ).savegraph("output/dayabay_v0_reduced.dot")
    GraphDot.from_graph(
        graph,
        show="all",
        filter={
            "reactor": [0],
            "detector": [0, 1],
            "isotope": [0],
            "period": [0],
            "background": [0],
        },
    ).savegraph("output/dayabay_v0_reduced_full.dot")
    GraphDot.from_node(
        storage["nodes.statistic.nuisance.all"],
        show="all",
        mindepth=-1,
        no_forward=True,
    ).savegraph("output/dayabay_v0_nuisance.dot")
    # GraphDot.from_output(
    #     storage["outputs.edges.energy_evis"], show="all", mindepth=-3, no_forward=True
    # ).savegraph("output/dayabay_v0_top.dot")
    GraphDot.from_output(
        storage["outputs.statistic.stat.chi2p"],
        show="all",
        mindepth=-1,
        filter={
            "reactor": [0],
            "detector": [0],
            "isotope": [0],
            "period": [0],
            "background": [0],
        },
        no_forward=True,
    ).savegraph("output/dayabay_v0_stat.dot")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-v', '--verbose', default=0, action='count', help='verbosity level')
    parser.add_argument(
        "-s",
        "--source-type",
        "--source",
        choices=("tsv", "hdf5", "root", "npz"),
        default="tsv",
        help="Data source type",
    )
    parser.add_argument(
        "--plot-all", help="plot all the nodes to the folder", metavar="folder"
    )

    storage = parser.add_argument_group("storage", "storage related options")
    parser.add_argument("-P", "--print-all", action="store_true", help="print all")
    parser.add_argument("-p", "--print", action="append", nargs="+", default=[], help="print all")

    graph = parser.add_argument_group("graph", "graph related options")
    graph.add_argument(
        "--no-close", action="store_false", dest="close", help="Do not close the graph"
    )
    graph.add_argument(
        "--no-strict", action="store_false", dest="strict", help="Disable strict mode"
    )
    graph.add_argument(
        "-i",
        "--index",
        nargs="+",
        action="append",
        default=[],
        help="override index",
        metavar=("index", "value1"),
    )

    model = parser.add_argument_group("model", "model related options")
    model.add_argument("--spec", choices=("linear", "exponential"), help="antineutrino spectrum correction mode")

    main(parser.parse_args())
