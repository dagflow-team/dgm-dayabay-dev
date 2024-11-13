#!/usr/bin/env python

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

from h5py import File

from dagflow.core import Graph, NodeStorage
from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from models import available_models, load_model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas import DataFrame

# from dagflow.plot import plot_auto

set_level(INFO1)


def main(opts: Namespace) -> None:
    if opts.verbose:
        opts.verbose = min(opts.verbose, 3)
        set_level(globals()[f"INFO{opts.verbose}"])

    override_indices = {idxdef[0]: tuple(idxdef[1:]) for idxdef in opts.index}
    model = load_model(
        opts.version,
        model_options=opts.model_options,
        close=opts.close,
        strict=opts.strict,
        source_type=opts.source_type,
        override_indices=override_indices,
        parameter_values=opts.par,
    )

    graph = model.graph
    storage = model.storage

    if opts.interactive:
        from IPython import embed

        embed(colors="neutral")

    if not graph.closed:
        print("Nodes")
        print(storage("nodes").to_table(truncate="auto"))
        print("Outputs")
        print(storage("outputs").to_table(truncate="auto"))
        print("Not connected inputs")
        print(storage("inputs").to_table(truncate="auto"))

        if opts.graph_auto:
            plot_graph(graph, storage, opts)
        return

    if opts.print_all:
        print(storage.to_table(truncate="auto"))
    for sources in opts.print:
        for source in sources:
            print(storage(source).to_table(truncate="auto"))
    if len(storage("inputs")) > 0:
        print("Not connected inputs")
        print(storage("inputs").to_table(truncate="auto"))

    if opts.method:
        method = getattr(model, opts.method)
        assert method

        method()

    if opts.plot_all:
        storage("outputs").plot(folder=opts.plot_all)

    if opts.plot:
        folder, sources = opts.plot[0], opts.plot[1:]
        for source in sources:
            storage(source).plot(folder=f"{folder}/{source.replace('.', '/')}")

    if opts.pars_datax:
        storage["parameters.all"].to_datax_file(
            f"output/dayabay_{opts.version}_pars_datax.tex"
        )

    if opts.pars_latex:
        storage["parameters.all"].to_latex_file(
            f"output/dayabay_{opts.version}_pars.tex"
        )

    if opts.pars_text:
        storage["parameters.all"].to_text_file(
            f"output/dayabay_{opts.version}_pars.txt"
        )

    if opts.summary:
        try:
            summary = model.make_summary_table()
        except AttributeError:
            pass
        else:
            save_summary(summary, opts.summary)

    if opts.graph_auto:
        plot_graph(graph, storage, opts)

    if opts.graphs:
        mindepth = opts.mindepth or -2
        maxdepth = opts.maxdepth or +1
        accept_index = {
            "reactor": [0],
            "detector": [0, 1],
            "isotope": [0],
            "period": [2],
        }
        storage["nodes"].savegraphs(
            opts.graphs,
            mindepth=mindepth,
            maxdepth=maxdepth,
            keep_direction=True,
            show="all",
            accept_index=accept_index,
            filter={
                "reactor": [0],
                "detector": [0, 1],
                "isotope": [0],
                "period": [2],
            },
        )

    if opts.graph_from_node:
        from dagflow.plot.graphviz import GraphDot

        nodepath, filepath = opts.graph_from_node
        node = storage("nodes")[nodepath]
        GraphDot.from_node(
            node,
            show="all",
            mindepth=opts.mindepth,
            maxdepth=opts.maxdepth,
            keep_direction=True,
        ).savegraph(filepath)


def save_summary(summary: DataFrame, filenames: Sequence[str]):
    for ofile in filenames:
        if ofile != "-":
            opath = Path(ofile)
            Path(opath.parent).mkdir(parents=True, exist_ok=True)
        match ofile.split("."):
            case (*_, "-"):
                print(summary)
            case (*_, "txt"):
                summary.to_csv(ofile, sep="\t", index=False)
            case (*_, "pd", "hdf5"):
                summary.to_hdf(ofile, key="summary", index=False, mode="w")
            case (*_, "hdf5"):
                rec = summary.to_records(index=False)
                l1 = summary["name"].str.len().max()
                newdtype = [
                    (rec.dtype.names[0], f"S{l1:d}"),
                    *(
                        (rec.dtype.names[i], rec.dtype[i])
                        for i in range(1, len(rec.dtype))
                    ),
                ]
                rec = rec.astype(newdtype)
                with File(ofile, mode="w") as f:
                    f.create_dataset("summary", data=rec)
            case _:
                raise ValueError(ofile)

        if ofile != "-":
            print(f"Write: {ofile}")


def plot_graph(graph: Graph, storage: NodeStorage, opts: Namespace) -> None:
    from dagflow.plot.graphviz import GraphDot

    GraphDot.from_graph(graph, show="all").savegraph(
        f"output/dayabay_{opts.version}.dot"
    )
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
    ).savegraph(f"output/dayabay_{opts.version}_reduced.dot")
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
    ).savegraph(f"output/dayabay_{opts.version}_reduced_full.dot")
    GraphDot.from_node(
        storage["nodes.statistic.nuisance.all"],
        show="all",
        mindepth=-1,
        maxdepth=0,
    ).savegraph(f"output/dayabay_{opts.version}_nuisance.dot")
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
        maxdepth=0,
    ).savegraph(f"output/dayabay_{opts.version}_stat.dot")


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
        choices=("tsv", "hdf5", "root", "npz"),
        default="hdf5",
        help="Data source type",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start IPython session",
    )

    plot = parser.add_argument_group("plot", "plotting related options")
    plot.add_argument(
        "--plot-all", help="plot all the nodes to the folder", metavar="folder"
    )
    plot.add_argument(
        "--plot",
        nargs="+",
        help="plot the nodes in storages",
        metavar=("folder", "storage"),
    )

    storage = parser.add_argument_group("storage", "storage related options")
    storage.add_argument("-P", "--print-all", action="store_true", help="print all")
    storage.add_argument(
        "-p", "--print", action="append", nargs="+", default=[], help="print all"
    )
    storage.add_argument(
        "--pars-datax", action="store_true", help="print parameters to latex (datax)"
    )
    storage.add_argument(
        "--pars-latex", action="store_true", help="print latex tables with parameters"
    )
    storage.add_argument(
        "--pars-text", action="store_true", help="print text tables with parameters"
    )
    storage.add_argument(
        "--summary",
        nargs="+",
        help="print/save summary data",
    )

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

    dot = parser.add_argument_group("graphviz", "plotting graphs")
    dot.add_argument(
        "-g",
        "--graph-from-node",
        nargs=2,
        help="plot the graph starting from the node",
        metavar=("node", "file"),
    )
    dot.add_argument("--mindepth", "--md", type=int, help="minimal depth")
    dot.add_argument("--maxdepth", "--Md", type=int, help="maximaldepth depth")
    dot.add_argument(
        "--graph-auto", "--ga", action="store_true", help="plot graphs auto"
    )
    dot.add_argument("--graphs", help="save partial graphs from every node")

    model = parser.add_argument_group("model", "model related options")
    model.add_argument(
        "--version",
        default="v0",
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

    main(parser.parse_args())
