#!/usr/bin/env python

from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING

from dag_modelling.tools.logger import set_verbosity
from matplotlib import pyplot as plt
from numpy import full_like, linspace, ma

from dgm_dayabay_dev.models import available_models, load_model

if TYPE_CHECKING:
    from typing import Any


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
    )

    # chi2 = model.storage["outputs.statistic.full.chi2p_iterative"]
    chi2 = model.storage["outputs.statistic.stat.chi2p"]

    matrix = model.storage["outputs.detector.lsnl.matrix.AD11"]
    curve = model.storage["outputs.detector.lsnl.curves.evis_coarse"]
    curve_mono = model.storage["outputs.detector.lsnl.curves.evis_coarse_monotonous"]
    escint = model.storage["outputs.detector.lsnl.curves.escint"].data

    pars = list(model.storage["parameters.constrained.detector.lsnl_scale_a"].walkvalues())
    # pars_offset = 3
    # pars = pars[pars_offset : len(pars)] + pars[:pars_offset]

    def reset_pars():
        for ipar in pars:
            ipar.value = 0
            # ipar.connectible.node.taint()

    reset_pars()
    chi2.data

    npoints = 5 + 1
    x = linspace(-1.0, 1.0, npoints)

    N_ring = min(npoints * 2, 20)
    i_ring = -1
    ring_lsnl: list[dict[str, Any] | None] = [None] * N_ring
    error_found = False

    for ipar, par in enumerate(pars):
        y = full_like(x, -1)
        reset_pars()

        for i, val in enumerate(x):
            par.value = val
            print(par.name, i, val)
            y[i] = chi2.data[0]

            if not error_found:
                i_ring = i % N_ring
                ring_lsnl[i_ring] = {
                    "name": par.name,
                    "matrix": matrix.data.copy(),
                    "index": i,
                    "curve": curve.data.copy(),
                    "curve_mono": curve_mono.data.copy(),
                }

            # if i > 1:
            #     diff = y[i] - y[i - 1]
            #     diff0 = y[i - 1] - y[i - 2]
            #     if (
            #         not error_found
            #         and (diff * diff0) < 0
            #         and fabs(diff) > (2 * fabs(diff0))
            #     ):
            #         error_found = True
            #         print("Found", par.name, i, val)

        plt.plot(x, y, ":", alpha=0.8, label=f"{par.name} fwd", color=f"C{ipar+1}")

        reset_pars()
        y = full_like(x, -1)
        for i, val in reversed(list(enumerate(x))):
            par.value = val
            print(par.name, i, val)
            y[i] = chi2.data[0]

        plt.plot(x, y, "--", alpha=0.5, label=f"{par.name} bkw", color=f"C{ipar+1}")

        if "pull1" in par.name:
            error_found = True

    plt.legend()

    fname = "output/test_lsnl.pdf"
    plt.savefig(fname)
    print(f"Write: {fname}")

    plt.show()

    if error_found:
        # Rotate the ring to have elements in order from earliest to latest
        ring_lsnl = list(
            el for el in ring_lsnl[i_ring + 1 : N_ring] + ring_lsnl[: i_ring + 1] if el is not None
        )
        for r, lsnl in enumerate(ring_lsnl):
            if lsnl is None:
                continue
            mat = lsnl["matrix"]
            i = lsnl["index"]
            name = lsnl["name"]
            e = float(x[i])
            plt.matshow(ma.array(mat, mask=(mat == 0.0)), vmin=0, vmax=1)
            plt.gca().set_title(f"{name} {i=}, {e=}")
            fname = f"output/test_lsnl_mat_{r}_mat.pdf"
            plt.savefig(fname)
            print(f"Write: {fname}")

            plt.figure()
            ax = plt.subplot(111, xlabel="", ylabel="", title=f"Curve {name} {i=}, {e=:.6g}")
            ax.plot(
                escint,
                lsnl["curve"] / escint,
                "--",
                alpha=0.4,
                label=f"input",
                color=f"C{r+1}",
            )
            ax.plot(
                escint,
                lsnl["curve_mono"] / escint,
                ":",
                alpha=0.8,
                label=f"mono",
                # color=f"C{r+1}",
            )
            ax.legend()

            fname = f"output/test_lsnl_curve_{r}_full.pdf"
            plt.savefig(fname)
            print(f"Write: {fname}")

            if r == 0 or (lsnl_prev := ring_lsnl[r - 1]) is None or lsnl_prev["name"] != name:
                continue

            diff = mat - lsnl_prev["matrix"]
            plt.matshow(ma.array(diff, mask=(diff == 0.0)), vmin=0, vmax=1)
            plt.gca().set_title(f"Diff {name} {i=}-prev, {e=}")
            fname = f"output/test_lsnl_mat_{r}_diff.pdf"
            plt.savefig(fname)
            print(f"Write: {fname}")


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

    pars = parser.add_argument_group("pars", "setup pars")
    pars.add_argument("--par", nargs=2, action="append", default=[], help="set parameter value")

    main(parser.parse_args())
