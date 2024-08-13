#!/usr/bin/env python

from __future__ import annotations
from argparse import Namespace
from typing import TYPE_CHECKING, Sequence

from dagflow.logger import INFO1, set_level
from models.dayabay_v0 import model_dayabay_v0

if TYPE_CHECKING:
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

    prediction = model.storage["outputs.eventscount.final.total_alltime_concatenated"]
    build_jacobian(model, prediction, opts.pars, normalized=True)


def build_jacobian(
    model, prediction: Output, parnames: Sequence[str], *, normalized: bool
):
    if normalized:
        allpars = model.storage.get_dict("parameter.normalized")
    else:
        allpars = model.storage.get_dict("parameter.constrained")

    from dagflow.lib.Jacobian import Jacobian

    for parname in parnames:
        pars = list(allpars.walkvalues(parname))

        jac = Jacobian("jac", parameters=pars)
        prediction >> jac

        import IPython

        IPython.embed(colors="neutral")


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

    main(parser.parse_args())
