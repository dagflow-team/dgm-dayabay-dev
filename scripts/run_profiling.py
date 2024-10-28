#!/usr/bin/env python
from argparse import Namespace
from datetime import datetime
from pathlib import Path

# from dagflow.graph import Graph
from dagflow.logger import DEBUG as INFO4
from dagflow.logger import INFO1, INFO2, INFO3, set_level
# from dagflow.storage import NodeStorage
from dagflow.tools.profiling import NodeProfiler, FrameworkProfiler, MemoryProfiler
from models import available_models, load_model


def profile(model, opts: Namespace):
    nodes = model.graph._nodes

    # capture current time and prepare output dir
    cur_time = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    outpath = Path(opts.outpath) / opts.version
    outpath.mkdir(parents=True, exist_ok=True)

    n_profiling = NodeProfiler(nodes, n_runs=opts.np_runs)
    n_profiling.estimate_target_nodes()
    report = n_profiling.print_report()
    report.to_csv(outpath / f'node_prof_{cur_time}.csv')

    fw_profiling = FrameworkProfiler(nodes, n_runs=opts.fw_runs)
    fw_profiling.estimate_framework_time()
    report = fw_profiling.print_report()
    report.to_csv(outpath / f'framewrok_prof_{cur_time}.csv')

    m_proifling = MemoryProfiler(nodes)
    m_proifling.estimate_target_nodes()
    report = m_proifling.print_report()
    report.to_csv(outpath / f'memory_prof{cur_time}.csv')

def main(opts: Namespace) -> None:
    if opts.verbose:
        opts.verbose = min(opts.verbose, 3)
        set_level(globals()[f"INFO{opts.verbose}"])

    override_indices = {idxdef[0]: tuple(idxdef[1:]) for idxdef in opts.index}
    model = load_model(
        opts.version,
        model_options=opts.model_options,
        close=True,
        strict=opts.strict,
        source_type=opts.source_type,
        override_indices=override_indices,
        parameter_values=opts.par,
    )

    graph = model.graph
    storage = model.storage

    if opts.print_all:
        print(storage.to_table(truncate="auto"))
    for sources in opts.print:
        for source in sources:
            print(storage(source).to_table(truncate="auto"))
    if len(storage("inputs")) > 0:
        print("Not connected inputs")
        print(storage("inputs").to_table(truncate="auto"))

    profile(model, opts)

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
        default="tsv",
        help="Data source type",
    )

    storage = parser.add_argument_group("storage", "storage related options")
    storage.add_argument("-P", "--print-all", action="store_true", help="print all")
    storage.add_argument(
        "-p", "--print", action="append", nargs="+", default=[], help="print all"
    )

    graph = parser.add_argument_group("graph", "graph related options")
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

    pars = parser.add_argument_group("pars", "setup pars")
    pars.add_argument(
        "--par", nargs=2, action="append", default=[], help="set parameter value"
    )

    profiling = parser.add_argument_group("profiling", "profiling options")
    profiling.add_argument(
        '--np-runs',
        default=10_000,
        dest='np_runs',
        type=int,
        metavar='N',
        help="number of runs of NodeProfiling for each node"
    )
    profiling.add_argument(
        '-o',
        '--output-dir',
        default='./output',
        dest='outpath',
        metavar='/PATH/TO/DIR',
        help='output dir for profiling results '
    )

    main(parser.parse_args())
