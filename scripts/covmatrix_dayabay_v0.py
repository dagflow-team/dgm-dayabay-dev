#!/usr/bin/env python

from __future__ import annotations

from argparse import Namespace

from h5py import File

from dagflow.tools.logger import INFO1, INFO2, INFO3, logger, set_level
from models import load_model, available_models

set_level(INFO1)


def main(opts: Namespace) -> None:
    if opts.verbose:
        opts.verbose = min(opts.verbose, 3)
        set_level(globals()[f"INFO{opts.verbose}"])

    model = load_model(
        opts.version,
        model_options=opts.model_options,
        source_type=opts.source_type,
        parameter_values=opts.setpar,
    )

    ofile = File(opts.output, "w")

    outputs = model.storage["outputs"]
    mode = model.concatenation_mode
    edges = outputs[f"edges.energy_final"]
    prediction = outputs[f"eventscount.final.concatenated.selected"]
    jacobians = outputs[f"covariance.jacobians"]
    covmats = outputs[f"covariance.covmat_syst"]

    idx_tuple = (
        model.index["detector"]
        if mode == "detector"
        else model.combinations["detector.period"]
    )
    # idx_str = tuple(".".join(idx) for idx in idx_tuple)

    group = ofile.create_group(mode)

    group.create_dataset("elements", data=idx_tuple)
    group.create_dataset("edges", data=edges.data)
    group.create_dataset("model", data=prediction.data)

    for name, jacobian in jacobians.items():
        logger.info(f"Compute {name} ({mode}), {jacobian.dd.shape[1]} pars")
        group.create_dataset(f"jacobians/{name}", data=jacobian.data)

    for name, covmat in covmats.items():
        group.create_dataset(f"covmat_syst/{name}", data=covmat.data)

    ofile.close()
    print(f"Save output file: {opts.output}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("output", help="output h5py file")
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
        "--version",
        default="v0",
        choices=available_models(),
        help="model version",
    )
    model.add_argument("--model-options", "--mo", default={}, help="Model options as yaml dict")

    pars = parser.add_argument_group("pars", "setup pars")
    pars.add_argument(
        "--setpar", nargs=2, action="append", default=[], help="set parameter value"
    )

    main(parser.parse_args())
