#!/usr/bin/env python

# isort: off
from __future__ import annotations
from dag_modelling.tools import disable_implicit_numpy_multithreading

# isort: on

from argparse import Namespace
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from dag_modelling.tools.logger import set_verbosity
from dag_modelling.tools.save_records import save_records
from dgm_dayabay_dev.models import available_models_limited, load_model
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

plt.rcParams.update(
    {
        "axes.formatter.use_mathtext": True,
        "axes.grid": False,
        "xtick.minor.visible": True,
        "xtick.top": True,
        "ytick.minor.visible": True,
        "ytick.right": True,
        "axes.formatter.limits": (-3, 4),
        "figure.max_open_warning": 30,
    }
)


def main(opts: Namespace) -> None:
    if opts.verbose:
        set_verbosity(opts.verbose)

    override_indices = {idxdef[0]: tuple(idxdef[1:]) for idxdef in opts.index}
    model = load_model(
        opts.version,
        model_options=opts.model_options,
        close=opts.close,
        strict=opts.strict,
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
        return

    if opts.print_all:
        print(storage.to_table(truncate="auto", df_kwargs={"columns": opts.print_columns}))
    for sources in opts.print:
        for source in sources:
            print(
                storage(source).to_table(
                    truncate="auto",
                    df_kwargs={
                        "columns": opts.print_columns,
                        "parent_key": source,
                    },
                )
            )
    if len(storage("inputs")) > 0:
        print("Not connected inputs")
        print(
            storage("inputs").to_table(truncate="auto", df_kwargs={"columns": opts.print_columns})
        )

    if opts.method:
        method = getattr(model, opts.method)
        assert method

        method()

    plot_overlay_priority = [
        model.index["isotope"],
        model.index["reactor"],
        model.index["background"],
        model.index["detector"],
        model.index["lsnl"],
    ]
    plot_kwargs = {
        "overlay_priority": plot_overlay_priority,
        "latex_substitutions": latex_substitutions,
        "exact_substitutions": exact_substitutions,
        "savefig_kwargs": {
            "metadata": {"CreationDate": None},
        },
    }
    if opts.plots_all:
        storage("outputs").plot(folder=opts.plots_all, minimal_data_size=10, **plot_kwargs)

    if opts.plots:
        folder, *sources = opts.plots
        for source in sources:
            storage["outputs"](source).plot(
                folder=f"{folder}/{source.replace('.', '/')}",
                minimal_data_size=10,
                **plot_kwargs,
            )

    if opts.export_root:
        storage("outputs").to_root(
            opts.export_root,
            latex_substitutions=latex_substitutions,
        )

    if opts.pars_datax:
        storage["parameters.all"].to_datax(f"output/dayabay_{opts.version}_pars_datax.tex")

    if opts.pars_latex:
        storage["parameters.all"].to_latex_file(opts.pars_latex)

    if opts.pars_latex_split:
        storage["parameters.all"].to_latex_files_split(
            opts.pars_latex_split,
            filter_columns=["central", "count"],
            to_latex_kwargs={
                "float_format": "{:.6g}".format,
                "index": False,
            },
            latex_substitutions=latex_substitutions,
        )

    if opts.pars_text:
        storage["parameters.all"].to_text_file(f"output/dayabay_{opts.version}_pars.txt")

    if opts.summary:
        save_summary(model, opts.summary)

    graph_accept_index = {
        "reactor": [0],
        "detector": [0, 1],
        "isotope": [0],
        "period": [1, 2],
    }
    if opts.graphs_all:
        path = Path(opts.graphs_all)
        storage["parameter_group.all"].savegraphs(
            path / "parameters",
            mindepth=opts.mindepth,
            maxdepth=opts.maxdepth,
            keep_direction=True,
            show="all",
            accept_index=graph_accept_index,
            filter=graph_accept_index,
        )
        with suppress(KeyError):
            storage["parameters.sigma"].savegraphs(
                path / "parameters" / "sigma",
                mindepth=opts.mindepth,
                maxdepth=opts.maxdepth,
                keep_direction=True,
                show="all",
                accept_index=graph_accept_index,
                filter=graph_accept_index,
            )
        storage["nodes"].savegraphs(
            path,
            mindepth=opts.mindepth,
            maxdepth=opts.maxdepth,
            keep_direction=True,
            show="all",
            accept_index=graph_accept_index,
            filter=graph_accept_index,
        )

    if opts.graphs:
        folder, *nodepaths = opts.graphs
        folder = Path(folder)
        for nodepath in nodepaths:
            nodes = storage("nodes")[nodepath]
            nodes.savegraphs(
                f"{folder}/{nodepath.replace('.', '/')}",
                mindepth=opts.mindepth,
                maxdepth=opts.maxdepth,
                keep_direction=True,
                show="all",
                accept_index=graph_accept_index,
                filter=graph_accept_index,
            )


def save_summary(model: Any, filenames: Sequence[str]):
    data = {}
    try:
        for period in ["total", "6AD", "8AD", "7AD"]:
            data[period] = model.make_summary_table(period=period)
    except AttributeError:
        return

    save_records(data, filenames, tsv_allow_no_key=True, to_records_kwargs={"index": False})


latex_substitutions = {
    " U235": r" $^{235}$U",
    " U238": r" $^{238}$U",
    " Pu239": r" $^{239}$Pu",
    " Pu241": r" $^{241}$Pu",
    "U235 ": r"$^{235}$U ",
    "U238 ": r"$^{238}$U ",
    "Pu239 ": r"$^{239}$Pu ",
    "Pu241 ": r"$^{241}$Pu ",
    "²³⁵U": r"$^{235}$U",
    "²³⁸U": r"$^{238}$U",
    "²³⁹Pu": r"$^{239}$Pu",
    "²⁴¹Pu": r"$^{241}$Pu",
    "Eν": r"$E_{\nu}$",
    "Edep": r"$E_{\rm dep}$",
    "Evis": r"$E_{\rm vis}$",
    "Escint": r"$E_{\rm scint}$",
    "Erec": r"$E_{\rm rec}$",
    "cosθ": r"$\cos\theta$",
    "Δm²₃₁": r"$\Delta m^2_{31}$",
    "Δm²₃₂": r"$\Delta m^2_{32}$",
    "Δm²₂₁": r"$\Delta m^2_{21}$",
    "sin²2θ₁₃": r"$\sin^22\theta_{13}$",
    "sin²2θ₁₂": r"$\sin^22\theta_{12}$",
    "sin²θ₁₃": r"$\sin^2\theta_{13}$",
    "sin²θ₁₂": r"$\sin^2\theta_{12}$",
    "sin²2θ₁₂": r"$\sin^22\theta_{12}$",
    "¹³C(α,n)¹⁶O": r"$^{13}{\rm C}(\alpha,n)^{16}{\rm O}$",
    "²⁴¹Am¹³C": r"$^{241}{\rm Am}^{13}{\rm C}$",
    "⁹Li/⁸He": r"$^{9}{\rm Li}/^{8}{\rm He}$",
    "ν̅": r"$\overline{\nu}$",
    "ν": r"$\nu$",
    "α": r"$\alpha$",
    "δ": r"$\delta$",
    "γ": r"$\gamma$",
    "μ": r"$\mu$",
    "σ": r"$\sigma$",
    "π": r"$\pi$",
    "χ²": r"$χ^2$",
    "·": r"$\cdot$",
    "×": r"$\times$",
    "[#": r"[\#",
    " # ": r" \# ",
    "⁻¹": r"$^{-1}$",
    "⁻²": r"$^{-2}$",
    "¹": r"$^1$",
    "²": r"$^2$",
    "³": r"$^3$",
    "⁴": r"$^4$",
    "⁵": r"$^5$",
    "⁶": r"$^6$",
    "⁷": r"$^7$",
    "⁸": r"$^8$",
    "⁹": r"$^9$",
    "⁰": r"$^0$",
    "ᵢ": r"$_i$",
}

exact_substitutions = {
    "U235": r"$^{235}$U",
    "U238": r"$^{238}$U",
    "Pu239": r"$^{239}$Pu",
    "Pu241": r"$^{241}$Pu",
    "acc": r"accidentals",
    "lihe": r"$^{9}{\rm Li}/^{8}{\rm He}$",
    "fastn": r"fast neutrons",
    "alphan": r"$^{13}{\rm C}(α,n)^{16}{\rm O}$",
    "amc": r"$^{241}{\rm Am}^{13}{\rm C}$",
}

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", default=1, action="count", help="verbosity level")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start IPython session",
    )
    parser.add_argument(
        "--export-root", help="Export outputs as graphs and histograms to the ROOT file"
    )

    plot = parser.add_argument_group("plot", "plotting related options")
    plot.add_argument("--plots-all", help="plot all the nodes to the folder", metavar="folder")
    plot.add_argument(
        "--plots",
        nargs="+",
        help="plot the nodes in storages",
        metavar=("folder", "storage"),
    )

    storage = parser.add_argument_group("storage", "storage related options")
    storage.add_argument("-P", "--print-all", action="store_true", help="print all")
    storage.add_argument("-p", "--print", action="append", nargs="+", default=[], help="print all")
    storage.add_argument("--print-columns", "--pc", default=None, nargs="+", help="Print columns")
    storage.add_argument(
        "--pars-datax", action="store_true", help="print parameters to latex (datax)"
    )
    storage.add_argument("--pars-latex", help="print latex table with parameters")
    storage.add_argument("--pars-latex-split", help="print latex tables with parameters")
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
    # dot.add_argument(
    #     "-g",
    #     "--graph-from-node",
    #     nargs=2,
    #     help="plot the graph starting from the node",
    #     metavar=("node", "file"),
    # )
    dot.add_argument("--mindepth", "--md", default=-2, type=int, help="minimal depth")
    dot.add_argument("--maxdepth", "--Md", default=+1, type=int, help="maximaldepth depth")
    dot.add_argument("--graphs-all", help="plot graphs", metavar="folder")
    dot.add_argument(
        "--graphs",
        nargs="+",
        help="save partial graphs from every node",
        metavar=("folder", "storage"),
    )

    model = parser.add_argument_group("model", "model related options")
    model.add_argument(
        "--version",
        default="latest",
        choices=available_models_limited(first="v0f"),
        help="model version",
    )
    model.add_argument("--model-options", "--mo", default={}, help="Model options as yaml dict")
    model.add_argument("--method", help="Call model's method")

    pars = parser.add_argument_group("pars", "setup pars")
    pars.add_argument("--par", nargs=2, action="append", default=[], help="set parameter value")

    main(parser.parse_args())
