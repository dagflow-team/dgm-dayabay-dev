#!/usr/bin/env python

from argparse import Namespace
from itertools import islice
from numpy import allclose, fabs
from numpy.typing import NDArray

from h5py import File

from dagflow.logger import INFO1, INFO2, INFO3, set_level, logger
from models.dayabay_v0 import model_dayabay_v0

set_level(INFO1)

# fmt: off
comparison_objects = {
    # dagflow: gna
    "edges.energy_edep": "evis_edges",
    "kinematics_sampler.mesh_edep": "evis_mesh",
    "ibd.enu": {
        "name": "enu",
        "slice": slice(102, None),
        "ignore": "Different formula below 2me. Not used as it is below the threshold.",
        "atol": 1e-14,
    },
    "ibd.jacobian": {
        "name": "jacobian",
        "atol": 1e-15,
    },
}
# fmt: on


def main(opts: Namespace) -> None:
    if opts.verbose:
        opts.verbose = min(opts.verbose, 3)
        set_level(globals()[f"INFO{opts.verbose}"])

    model = model_dayabay_v0(source_type=opts.source_type)
    outputs_dgf = model.storage("outputs")

    if opts.last:
        iterable = islice(reversed(comparison_objects.items()), 1)
    else:
        iterable = comparison_objects.items()
    for key_dgf, key_gna in iterable:
        if isinstance(key_gna, dict):
            key_gna, cmpopts = key_gna["name"], key_gna
        else:
            cmpopts = {}

        path_gna = key_gna.replace(".", "/")
        data_gna = opts.input[path_gna][:]

        data_dgf = outputs_dgf[key_dgf].data

        is_ok, maxdiff = data_consistent(data_gna, data_dgf, **cmpopts)
        if is_ok:
            logger.log(INFO1, f"OK: {key_dgf}↔{key_gna}")
            if (ignore:=cmpopts.get("ignore")) is not None:
                logger.log(INFO2, f"↑Ignore: {ignore}")
        else:
            logger.error(f"FAIL: {key_dgf}↔{key_gna}, max diff {maxdiff:.2g}")

            if opts.embed_on_fail:
                import IPython; IPython.embed(colors='neutral')

def data_consistent(
    gna: NDArray,
    dgf: NDArray,
    *,
    slice: tuple[slice | None,...] | None = None,
    slice_gna: tuple[slice | None,...] | None = None,
    atol = 0,
    rtol = 0,
    # not used:
    name: str = "",
    ignore: str | None = None
) -> tuple[bool, float]:
    if slice_gna is not None:
        gna = gna[slice_gna]
    elif slice is not None:
        gna = gna[slice]
        dgf = dgf[slice]
    status = allclose(dgf, gna, rtol=rtol, atol=atol)

    maxdiff = fabs(dgf-gna).max()

    if status:
        return True, maxdiff

    return False, maxdiff

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", default=0, action="count", help="verbosity level"
    )

    input = parser.add_argument_group("input", "input related options")
    input.add_argument(
        "input", type=File, help="input file to compare to"
    )

    parser.add_argument(
        "-s",
        "--source-type",
        "--source",
        choices=("tsv", "hdf5", "root", "npz"),
        default="npz",
        help="Data source type",
    )

    crosscheck = parser.add_argument_group("comparison", "comparison related options")
    crosscheck.add_argument("-l", "--last", action="store_true", help="process only the last item")
    crosscheck.add_argument("-e", "--embed-on-fail", action="store_true", help="embed on failure")

    main(parser.parse_args())
