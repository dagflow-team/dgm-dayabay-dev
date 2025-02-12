#!/usr/bin/env python

from __future__ import annotations

from argparse import Namespace

from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from models import available_models, load_model

set_level(INFO1)


def main(opts: Namespace) -> None:
    if opts.verbose:
        opts.verbose = min(opts.verbose, 3)
        set_level(globals()[f"INFO{opts.verbose}"])

    model = load_model(
        opts.version,
        model_options=opts.model_options,
        source_type=opts.source_type,
        parameter_values=opts.par,
    )

    graph = model.graph
    storage = model.storage

    if opts.method:
        method = getattr(model, opts.method)
        assert method

        method()


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

    model = parser.add_argument_group("model", "model related options")
    model.add_argument(
        "--version",
        default="recent",
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
