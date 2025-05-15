#!/usr/bin/env python

from __future__ import annotations

from argparse import Namespace
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

from dagflow.core import Graph, NodeStorage
from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from dagflow.tools.profiling import (
    FitSimulationProfiler,
    FrameworkProfiler,
    MemoryProfiler,
    NodeProfiler,
)
from dagflow.lib.common import Array
from dagflow.tools.save_records import save_records
from models import available_models, load_model
from scripts import update_dict_parameters

if TYPE_CHECKING:
    from collections.abc import Sequence
    from dagflow.parameters import Parameter
    from typing import Any


set_level(INFO1)

PROFILE_NODES = [
    "Product",
    "Cholesky",
    "SumMatOrDiag",
    "Sum",
    "AxisDistortionMatrixLinearLegacy",
    "RenormalizeDiag",
    "NueSurvivalProbability",
    "Integrator",
    "Division",
    "Interpolator",
    "EnergyResolutionMatrixBC",
    "RebinMatrix",
    "VectorMatrixProduct",
    "NormalizeCorrelatedVarsTwoWays",
    "Concatenation",
    "View",
    "CNPStat",
    "Monotonize",
    "EnergyResolutionSigmaRelABC",
    "ElSumSq",
    "Chi2",
    "LogProdDiag",
]


def profile(model, opts: Namespace, fit_params: NodeStorage, stat):
    print("Running profiling!")

    outpath = Path(opts.outpath) / opts.version
    outpath.mkdir(parents=True, exist_ok=True)
    cur_time = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")

    all_nodes = model.graph._nodes

    # nodes that are executed more than 1 time during the fit process
    many_execution_nodes = list(
        filter(lambda x: type(x).__name__ in PROFILE_NODES, all_nodes)
    )

    # NOTE: maby there is a better way to obtain paramters
    minimization_pars: dict[str, Parameter] = {}
    update_dict_parameters(
        minimization_pars, ["oscprob", "detector"], fit_params["free"]
    )
    update_dict_parameters(
        minimization_pars,
        ["oscprob", "detector", "reactor", "bkg"],
        fit_params["constrained"],
    )

    params = [param.output.node for param in minimization_pars.values()]
    endpoints = [stat.node]

    node_profiler = NodeProfiler(
        target_nodes=many_execution_nodes,
        n_runs=opts.node_prof_runs,
    )
    st = time()
    node_profiler.estimate_target_nodes()
    print(f"\nNode profiling took {time() - st:.2f} seconds.")
    report = node_profiler.print_report(sort_by="%_of_total")
    report.to_csv(outpath / f"node_{cur_time}.csv")

    framework_profiler = FrameworkProfiler(
        sources=params,
        sinks=endpoints,
        n_runs=opts.framework_runs,
    )
    st = time()
    framework_profiler.estimate_framework_time()
    print(f"\nFramework profiling took {time() - st:.2f} seconds.")
    framework_profiler.print_report().to_csv(outpath / f"framework_{cur_time}.csv")

    memory_profiler = MemoryProfiler(
        target_nodes=all_nodes,
    )
    st = time()
    memory_profiler.estimate_target_nodes()
    print(f"\nMemory profiling took {time() - st:.2f} seconds.")
    report = memory_profiler.print_report(sort_by="size_sum")
    report.to_csv(outpath / f"memory_{cur_time}.csv")

    fit_param_wise_profiler = FitSimulationProfiler(
        mode="parameter-wise",
        parameters=params,
        endpoints=endpoints,
        n_runs=opts.fit_param_wise_runs,
    )
    st = time()
    fit_param_wise_profiler.estimate_fit()
    print(f"\nParameter-wise fit profiling took {time() - st:.2f} seconds.")
    report = fit_param_wise_profiler.print_report()
    report.to_csv(outpath / f"fit_param_wise_{cur_time}.csv")

    fit_simultaneous_profiler = FitSimulationProfiler(
        mode="simultaneous",
        parameters=params,
        endpoints=endpoints,
        n_runs=opts.fit_simultaneous_runs,
    )
    st = time()
    fit_simultaneous_profiler.estimate_fit()
    print(f"\nSimultaneous fit profiling took {time() - st:.2f} seconds.")
    report = fit_simultaneous_profiler.print_report()
    report.to_csv(outpath / f"fit_simultaneous_{cur_time}.csv")


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
        storage("outputs").plot(folder=opts.plot_all, minimal_data_size=10)

    if opts.plot:
        folder, sources = opts.plot[0], opts.plot[1:]
        for source in sources:
            storage["outputs"](source).plot(
                folder=f"{folder}/{source.replace('.', '/')}", minimal_data_size=10
            )

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
        save_summary(model, opts.summary)

    profile(
        model, opts, storage["parameters"], storage["outputs.statistic.full.chi2cnp"]
    )


def save_summary(model: Any, filenames: Sequence[str]):
    data = {}
    try:
        for period in ["total", "6AD", "8AD", "7AD"]:
            data[period] = model.make_summary_table(period=period)
    except AttributeError:
        return

    save_records(
        data, filenames, tsv_allow_no_key=True, to_records_kwargs={"index": False}
    )


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

    profiling_specs = parser.add_argument_group("profiling", "profiling options")
    profiling_specs.add_argument(
        "-o",
        "--output-dir",
        default="./output/profiling",
        dest="outpath",
        metavar="/PATH/TO/DIR",
        help="output dir for profiling results ",
    )
    profiling_specs.add_argument(
        "--np-runs",
        default=1,
        dest="node_prof_runs",
        type=int,
        metavar="N_RUNS",
        help="number of runs of NodeProfiling for each node",
    )
    profiling_specs.add_argument(
        "--fw-runs",
        default=10,
        dest="framework_runs",
        type=int,
        metavar="N_RUNS",
        help="number of runs of NodeProfiling for each node",
    )
    profiling_specs.add_argument(
        "--fit-param-wise-runs",
        dest="fit_param_wise_runs",
        default=10,
        type=int,
        metavar="N_RUNS",
        help="number of runs for parameter-wise fit simulation profiling",
    )
    profiling_specs.add_argument(
        "--fit-simultaneous-runs",
        dest="fit_simultaneous_runs",
        default=10,
        type=int,
        metavar="N_RUNS",
        help="number of runs for simultaneous fit simulation profiling",
    )

    main(parser.parse_args())
