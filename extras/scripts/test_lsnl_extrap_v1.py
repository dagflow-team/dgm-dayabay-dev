#!/usr/bin/env python

from __future__ import annotations

from argparse import Namespace

from dag_modelling.tools.logger import set_verbosity
from matplotlib import pyplot as plt

from dgm_dayabay_dev.models import available_models, load_model


def main(opts: Namespace) -> None:
    if opts.verbose:
        set_verbosity(opts.verbose)

    override_indices = {idxdef[0]: tuple(idxdef[1:]) for idxdef in opts.index}
    model = load_model(
        opts.version,
        model_options=opts.model_options,
        close=opts.close,
        strict=opts.strict,
        source_type=opts.source_type,
        override_indices=override_indices,
        parameter_values=opts.par,
        lsnl_extrapolation_mode=opts.lsnl_extrapolation_mode,
    )

    escint = model.storage["outputs.detector.lsnl.curves.escint"].data
    curves = model.storage["outputs.detector.lsnl.curves.evis_parts"]

    plt.figure()
    ax = plt.subplot(
        111,
        xlabel="E scint, MeV",
        ylabel="E vis, MeV",
        title=f"LSNL curves (abs): extrap {opts.lsnl_extrapolation_mode}",
    )
    ax.grid()
    for label, out in curves.items():
        ax.plot(escint, out.data, label=label)
    plt.legend()

    fname = f"output/test_lsnl_extrap_abs_extrap:{opts.lsnl_extrapolation_mode}.pdf"
    plt.savefig(fname)
    print(f"Write: {fname}")

    plt.figure()
    ax = plt.subplot(
        111,
        xlabel="E scint, MeV",
        ylabel="f",
        title=f"LSNL curves (rel): extrap {opts.lsnl_extrapolation_mode}",
    )
    ax.grid()
    for label, out in curves.items():
        ax.plot(escint, out.data / escint, label=label)
    plt.legend()

    fname = f"output/test_lsnl_extrap_rel_extrap:{opts.lsnl_extrapolation_mode}.pdf"
    plt.savefig(fname)
    print(f"Write: {fname}")

    plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", default=1, action="count", help="verbosity level")
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
    model.add_argument("--model-options", "--mo", default={}, help="Model options as yaml dict")
    model.add_argument("--method", help="Call model's method")
    model.add_argument(
        "--lsnl-extrapolation-mode",
        "--lem",
        required=True,
        choices=("absolute", "relative"),
        help="LSNL extrapolation mode",
    )

    pars = parser.add_argument_group("pars", "setup pars")
    pars.add_argument("--par", nargs=2, action="append", default=[], help="set parameter value")

    main(parser.parse_args())
