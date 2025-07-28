#!/usr/bin/env python

from __future__ import annotations

from argparse import Namespace

from h5py import File

from dag_modelling.lib.calculus.jacobian import compute_covariance_matrix
from dag_modelling.tools.logger import INFO1, INFO2, INFO3, logger, set_level
from models import available_models, load_model
from nested_mapping import walkvalues

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

    if opts.store_jacobian:
        for name, jacobian in jacobians.items():
            logger.info(f"Compute jacobian {name} ({mode}), {jacobian.dd.shape[1]} pars")
            group.create_dataset(f"jacobians/{name}", data=jacobian.data)

    if opts.store_main:
        for name, covmat in covmats.items():
            logger.info(f"Compute covariance {name} ({mode})")
            group.create_dataset(f"covmat_syst/{name}", data=covmat.data)

    for setname in opts.extra_sets:
        for name, parlocation in extra_uncertainty_sets[setname].items():

            storage = model.storage["parameters.normalized"][parlocation]
            parameters = list(walkvalues(storage))

            logger.info(f"Compute extra covariance {name} ({mode}), {len(parameters)} pars")
            covmatsyst = compute_covariance_matrix(prediction, parameters)
            group.create_dataset(f"covmat_syst/extra/{name}", data=covmatsyst)

    ofile.close()
    print(f"Save output file: {opts.output}")


extra_uncertainty_sets = {
    "bkg": {
        "acc": "bkg.rate_scale.acc",
        "alphan": "bkg.rate.alphan",
        "amc": "bkg.uncertainty_scale.amc",
        "lihe": "bkg.uncertainty_scale_by_site.lihe",
        "fastn": "bkg.uncertainty_scale_by_site.fastn",
        "muonx": "bkg.uncertainty_scale_by_site.muonx",
    }
}


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
    model.add_argument(
        "--model-options", "--mo", default={}, help="Model options as yaml dict"
    )

    pars = parser.add_argument_group("pars", "setup pars")
    pars.add_argument(
        "--setpar", nargs=2, action="append", default=[], help="set parameter value"
    )

    covariance = parser.add_argument_group("covariance", "setup covariance")
    covariance.add_argument(
        "--extra-sets",
        default=[],
        nargs="+",
        choices=extra_uncertainty_sets.keys(),
        help="Extra uncertainty sets to calculate",
    )
    covariance.add_argument(
        "--no-main",
        action="store_false",
        dest="store_main",
        help="Skip saving main covariance matrices",
    )
    covariance.add_argument(
        "-j",
        "--jacobian",
        action="store_true",
        dest="store_jacobian",
        help="Save also jacobian matrices",
    )

    main(parser.parse_args())
