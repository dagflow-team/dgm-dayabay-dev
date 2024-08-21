#!/usr/bin/env python

from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING, Literal, Sequence

from h5py import File

from dagflow.lib import Jacobian, MatrixProductDDt, MatrixProductDVDt, SumMatOrDiag
from dagflow.logger import INFO1, INFO2, INFO3, set_level
from models.dayabay_v0 import model_dayabay_v0

if TYPE_CHECKING:
    from dagflow.node import Node
    from dagflow.output import Output

set_level(INFO1)


def main(opts: Namespace) -> None:
    if opts.verbose:
        opts.verbose = min(opts.verbose, 3)
        set_level(globals()[f"INFO{opts.verbose}"])

    model = model_dayabay_v0(
        source_type=opts.source_type,
        spectrum_correction_mode=opts.spec,
        fission_fraction_normalized=opts.fission_fraction_normalized,
        parameter_values=opts.setpar,
    )

    idx_detectors = model.index["detector"]
    idx_detectors_periods = model.combinations["detector.period"]

    prediction = model.storage["outputs.eventscount.final.concatenated.detector"]
    jacobians = model.storage["outputs.covariance.detector.jacobians"]
    covmats = model.storage["outputs.covariance.detector.covmat_syst"]

    # prediction = model.storage["outputs.eventscount.final.concatenated.detector"]
    # jacobians = model.storage["outputs.covariance.detector.jacobians"]
    # covmats = model.storage["outputs.covariance.detector.covmat_syst"]

    if opts.graph:
        plot_graph([vt1, vt2], opts.graph)
        print(f"Save graph: {opts.graph}")


def build_jacobian(
    stat_unc2: Node,
    model,
    prediction: Output,
    parnames: Sequence[str],
    *,
    mode: Literal["normalized", "group", "group_normalized"],
):
    allpars_norm = model.storage.get_dict("parameters.normalized")
    allpars_group = model.storage.get_dict("parameters_group.constrained")
    match mode:
        case "normalized":
            normalized = True
            group = False
        case "group_normalized":
            normalized = True
            group = True
        case "group":
            normalized = False
            group = True
        case _:
            raise RuntimeError(f"Invalid mode {mode}")

    vsyst_list = []
    for parname in parnames:
        pars_cov = None
        if group:
            pargroup = allpars_group.get_value(parname)
            assert pargroup.is_constrained

            if normalized:
                pars = pargroup.norm_parameters
            else:
                pars = pargroup.parameters
                pars_cov = pargroup.constraint._covariance_node
        else:
            pars = list(allpars_norm.walkvalues(parname))

        with stat_unc2.graph.open():
            jac = Jacobian(f"Jacobian: {parname}", parameters=pars)
            prediction >> jac

            if normalized:
                vsyst = MatrixProductDDt.from_args(f"V syst: {parname}", matrix=jac)
            else:
                vsyst = MatrixProductDVDt.from_args(
                    f"V syst: {parname}", left=jac, square=pars_cov
                )
        vsyst_list.append(vsyst)

    with stat_unc2.graph.open():
        vtot = SumMatOrDiag.from_args(f"V total", stat_unc2, *vsyst_list)

    return vtot


def plot_graph(nodes: list[Node], output: str) -> None:
    from dagflow.graphviz import GraphDot

    GraphDot.from_nodes(
        nodes, show="all", mindepth=-5, maxdepth=4, keep_direction=True
    ).savegraph(output)


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

    model = parser.add_argument_group("model", "model related options")
    model.add_argument(
        "--spec",
        choices=("linear", "exponential"),
        help="antineutrino spectrum correction mode",
    )
    model.add_argument(
        "--fission-fraction-normalized",
        action="store_true",
        help="fission fraction correction",
    )

    covariance = parser.add_argument_group("covariance", "covariance options")
    covariance.add_argument(
        "-p", "--pars", nargs="+", default=[], help="parameters for covariance matrix"
    )

    pars = parser.add_argument_group("pars", "setup pars")
    pars.add_argument(
        "--setpar", nargs=2, action="append", default=[], help="set parameter value"
    )

    dot = parser.add_argument_group("graphviz", "plotting graphs")
    dot.add_argument("--graph", help="plot the graph")

    main(parser.parse_args())
