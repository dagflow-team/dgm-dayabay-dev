#!/usr/bin/env python

from argparse import Namespace

from dagflow.graph import Graph
from dagflow.logger import SUBINFO, SUBSUBINFO, set_level
from dagflow.plot import plot_auto
from dagflow.storage import NodeStorage
from models.dayabay_v0 import model_dayabay_v0

set_level(SUBINFO)
# set_level(SUBSUBINFO)


def main(opts: Namespace) -> None:
    model = model_dayabay_v0(
        close=opts.close, strict=opts.strict, source_type=opts.source_type
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

    print(storage.to_table(truncate=True))
    if len(storage("inputs")) > 0:
        print("Not connected inputs")
        print(storage("inputs").to_table(truncate=True))

    if opts.plot_all:
        storage("outputs").plot(folder=opts.plot_all)

    # storage("outputs.oscprob").plot(folder='output/dayabay_v0_auto')
    # storage("outputs.countrate").plot(show_all=True)
    # storage("outputs").plot(
    #     folder = 'output/dayabay_v0_auto',
    #     overlay_priority = (index["isotope"], index["reactor"], index['background'], index["detector"])
    # )

    # storage["parameter.normalized.detector.eres.b_stat"].value = 1
    # storage["parameter.normalized.detector.eres.a_nonuniform"].value = 2
    #
    # # p1 = storage["parameter.normalized.detector.eres.b_stat"]
    # # p2 = storage["parameter.constrained.detector.eres.b_stat"]
    #
    # constrained = storage("parameter.constrained")
    # normalized = storage("parameter.normalized")
    #
    # print("Everything")
    # print(storage.to_table(truncate=True))

    # print("Constants")
    # print(storage("parameter.constant").to_table(truncate=True))
    #
    # print("Constrained")
    # print(constrained.to_table(truncate=True))
    #
    # print("Normalized")
    # print(normalized.to_table(truncate=True))
    #
    # print("Stat")
    # print(storage("stat").to_table(truncate=True))

    # print("Parameters (latex)")
    # print(storage["parameter"].to_latex())
    #
    # print("Constants (latex)")
    # tex = storage["parameter.constant"].to_latex(columns=["path", "value", "label"])
    # print(tex)

    storage.to_datax("output/dayabay_v0_data.tex")
    plot_graph(graph, storage)


def plot_graph(graph: Graph, storage: NodeStorage) -> None:
    from dagflow.graphviz import GraphDot

    GraphDot.from_graph(graph, show="all").savegraph("output/dayabay_v0.dot")
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
    ).savegraph("output/dayabay_v0_reduced.dot")
    GraphDot.from_node(
        storage["stat.nuisance.all"], show="all", mindepth=-1, no_forward=True
    ).savegraph("output/dayabay_v0_nuisance.dot")
    GraphDot.from_output(
        storage["outputs.edges.energy_evis"], show="all", mindepth=-3, no_forward=True
    ).savegraph("output/dayabay_v0_top.dot")
    GraphDot.from_output(
        storage["outputs.statistic.stat.chi2p"], show="all", mindepth=-1, no_forward=True
    ).savegraph("output/dayabay_v0_stat.dot")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-s",
        "--source-type",
        "--source",
        choices=("tsv", "hdf", "root", "npz"),
        default="tsv",
        help="Data source type",
    )
    parser.add_argument(
        "--plot-all", help="plot all the nodes toe the folder", metavar="folder"
    )
    parser.add_argument(
        "--no-close", action="store_false", dest="close", help="Do not close the graph"
    )
    parser.add_argument(
        "--no-strict", action="store_false", dest="strict", help="Disable strict mode"
    )

    main(parser.parse_args())
