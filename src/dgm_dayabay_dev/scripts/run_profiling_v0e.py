#!/usr/bin/env python
from __future__ import annotations

# disable numpy multithreading
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from argparse import Namespace
from datetime import datetime
from pathlib import Path
from time import time

from dag_modelling.tools.logger import DEBUG as INFO4
from dag_modelling.tools.logger import INFO1, INFO2, INFO3, set_level
from dag_modelling.tools.profiling import (
    FitSimulationProfiler,
    FrameworkProfiler,
    MemoryProfiler,
    NodeProfiler,
    gather_related_nodes,
)
from models import available_models, load_model

set_level(INFO1)

# nodes that are executed more than 1 time during the fit process
NODE_PROFILE_NODES = [
    "MatrixProductDDt",
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
    "ArraySum",
    "Monotonize",
    "EnergyResolutionSigmaRelABC",
    "InverseSquareLaw",
    "ElSumSq",
    "Chi2",
    "LogProdDiag",
    "WeightedSumArgs",
    "LogPoissonRatio",
    "InterpolatorCore",
    "Exp",
    "ProductShiftedScaled",
    "AxisDistortionMatrix",
    "Difference",
    "IntegratorCore",
]

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

    outpath = Path(opts.outpath) / opts.version
    outpath.mkdir(parents=True, exist_ok=True)
    cur_time = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")

    storage = model.storage
    all_nodes = model.graph._nodes

    params_variable = [
        s._value_node for s in storage["parameters.variable"].walkvalues()
    ]
    params_free = [
        s.output.node for s in storage["parameters.free"].walkvalues()
    ]

    assert len(set(params_variable)) == len(params_variable), "duplicates"
    assert len(set(params_free)) == len(params_free), "duplicates"

    covmat_chi2cnp = storage["outputs.statistic.full.covmat.chi2cnp"]
    pull_chi2cnp = storage["outputs.statistic.full.pull.chi2cnp"]

    covmat_chi2cnp.data
    pull_chi2cnp.data

    graph_setups = {
        "covmat_chi2cnp": (params_free, covmat_chi2cnp.node),
        "pull_chi2cnp": (params_variable, pull_chi2cnp.node),
    }

    for stat_name, graph_setup in graph_setups.items():
        params, stat = graph_setup
        subgraph = list(gather_related_nodes(sources=params, sinks=[stat]))

        d_points = opts.fit_d_points

        print("=" * 60)
        print(f"Running profiling for subgraph with '{stat_name}' stat!")
        print("=" * 60)

        node_profiler = NodeProfiler(
            target_nodes=subgraph,
            filter_types=NODE_PROFILE_NODES,
            n_runs=opts.node_prof_runs,
        )
        st = time()
        node_profiler.estimate_target_nodes()
        report = node_profiler.print_report(sort_by="%_of_total", rows=100)
        report.to_csv(
            outpath / f"node_{stat_name}_{cur_time}.tsv", sep="\t", index=False
        )
        print(f"\nNode profiling took {time() - st:.2f} seconds.")

        framework_profiler = FrameworkProfiler(
            target_nodes=subgraph,
            n_runs=opts.framework_runs,
        )
        st = time()
        framework_profiler.estimate_framework_time()
        framework_profiler.print_report(rows=100).to_csv(
            outpath / f"framework_{stat_name}_{cur_time}.tsv", sep="\t", index=False
        )
        print(f"\nFramework profiling took {time() - st:.2f} seconds.")

        fit_param_wise_profiler = FitSimulationProfiler(
            mode="parameter-wise",
            parameters=params,
            endpoints=[stat],
            n_runs=opts.fit_param_wise_runs,
            n_derivative_points=d_points,
        )
        st = time()
        fit_param_wise_profiler.estimate_fit()
        report = fit_param_wise_profiler.print_report(
            # aggregations=("count", "step", "single", "calls", "sum"),
            rows=100
        )
        report.to_csv(
            outpath / f"fit_param_wise_p{d_points}_{stat_name}_{cur_time}.tsv",
            sep="\t",
            index=False,
        )
        print(f"\nParameter-wise fit profiling took {time() - st:.2f} seconds.")

        fit_simultaneous_profiler = FitSimulationProfiler(
            mode="simultaneous",
            parameters=params,
            endpoints=[stat],
            n_runs=opts.fit_simultaneous_runs,
            n_derivative_points=d_points,
        )
        st = time()
        fit_simultaneous_profiler.estimate_fit()
        report = fit_simultaneous_profiler.print_report(
            # aggregations=("count", "step", "single", "calls", "sum"),
            rows=100
        )
        report.to_csv(
            outpath / f"fit_simultaneous_p{d_points}_{stat_name}_{cur_time}.tsv",
            sep="\t",
            index=False,
        )
        print(f"\nSimultaneous fit profiling took {time() - st:.2f} seconds.")

    memory_profiler = MemoryProfiler(all_nodes)
    st = time()
    memory_profiler.estimate_target_nodes()
    report = memory_profiler.print_report(sort_by="size_sum", rows=100)
    report.to_csv(outpath / f"memory_{cur_time}.tsv", sep="\t", index=False)
    print(f"\nMemory profiling took {time() - st:.2f} seconds.")


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
        default="latest",
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
        default=50,
        dest="node_prof_runs",
        type=int,
        metavar="N_RUNS",
        help="number of runs of NodeProfiling for each node",
    )
    profiling_specs.add_argument(
        "--fw-runs",
        default=50,
        dest="framework_runs",
        type=int,
        metavar="N_RUNS",
        help="number of runs of NodeProfiling for each node",
    )
    profiling_specs.add_argument(
        "--fit-param-wise-runs",
        dest="fit_param_wise_runs",
        default=50,
        type=int,
        metavar="N_RUNS",
        help="number of runs for parameter-wise fit simulation profiling",
    )
    profiling_specs.add_argument(
        "--fit-simultaneous-runs",
        dest="fit_simultaneous_runs",
        default=50,
        type=int,
        metavar="N_RUNS",
        help="number of runs for simultaneous fit simulation profiling",
    )
    profiling_specs.add_argument(
        "--derivative-points",
        dest="fit_d_points",
        default=2,
        type=int,
        metavar="N_POINTS",
        help="number of derivative points for parameter-wise fit simulation profiling",
    )
    main(parser.parse_args())
