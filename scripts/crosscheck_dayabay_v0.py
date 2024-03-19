#!/usr/bin/env python

from argparse import Namespace
from itertools import islice, permutations
from typing import Any

from h5py import Dataset, File, Group
from numpy import allclose, fabs, nanmax
from numpy.typing import NDArray

from dagflow.logger import INFO1, INFO2, INFO3, logger, set_level
from dagflow.output import Output
from models.dayabay_v0 import model_dayabay_v0
from multikeydict.nestedmkdict import NestedMKDict

set_level(INFO1)

# fmt: off
comparison_objects = {
    # dagflow: gna
    "edges.energy_edep": "evis_edges",
    "kinematics_sampler.mesh_edep": "evis_mesh",
    "ibd.enu": {
        "gnaname": "enu",
        "slice": slice(102, None),
        "ignore": "Different formula below 2me. Not used as it is below the threshold.",
        "atol": 1e-14,
    },
    "ibd.jacobian": {
        "gnaname": "jacobian",
        "atol": 1e-15,
    },
    "ibd.crosssection": {
        "gnaname": "ibd_xsec",
        "rtol": 1.e-14
        },
    "oscprob": {
        "gnaname": "osc_prob_rd"
        }
}
# fmt: on


class Comparator:
    opts: Namespace
    output_dgf: NestedMKDict

    _cmpopts: dict[str, Any]
    _maxdiff: float = 0.0
    _maxreldiff: float = 0.0

    _skey_gna: str
    _skey_dgf: str

    def __init__(self, opts: Namespace):
        self.model = model_dayabay_v0(source_type=opts.source_type)
        self.opts = opts

        if opts.verbose:
            opts.verbose = min(opts.verbose, 3)
            set_level(globals()[f"INFO{opts.verbose}"])

        self.outputs_dgf = self.model.storage("outputs")

        self.compare()

    def compare(self) -> None:
        if self.opts.last:
            iterable = islice(reversed(comparison_objects.items()), 1)
        else:
            iterable = comparison_objects.items()
        for self._skey_dgf, cmpopts in iterable:
            match cmpopts:
                case dict():
                    self._skey_gna= cmpopts["gnaname"]
                    self._cmpopts = cmpopts
                case str():
                    self._skey_gna = cmpopts
                    self._cmpopts = {}
                case _:
                    raise RuntimeError(f"Invalid {cmpopts=}")

            self.compare_source()

    def compare_source(self) -> None:
        path_gna = self._skey_gna.replace(".", "/")

        data_storage_gna = self.opts.input[path_gna]
        data_storage_dgf = self.outputs_dgf.any(self._skey_dgf)

        match data_storage_dgf, data_storage_gna:
            case Output(), Dataset():
                data_gna = data_storage_gna[:]
                data_dgf = data_storage_dgf.data
                self.compare_outputs(data_gna, data_dgf)
            case NestedMKDict(), Group():
                self.compare_nested(data_storage_gna, data_storage_dgf)
            case _:
                raise RuntimeError("Unexpected data types")

    def compare_outputs(self, data_gna: NDArray, data_dgf: NDArray):
        is_ok = self.data_consistent(data_gna, data_dgf)
        if is_ok:
            logger.log(INFO1, f"OK: {self._skey_dgf}↔{self._skey_gna}")
            if (ignore := self._cmpopts.get("ignore")) is not None:
                logger.log(INFO2, f"↑Ignore: {ignore}")
        else:
            logger.error(
                f"FAIL: {self._skey_dgf}↔{self._skey_gna}, "
                f"max diff {self._maxdiff:.2g}, "
                f"max rel diff {self._maxreldiff:.2g}"
            )

            if self.opts.embed_on_fail:
                diff = data_dgf - data_gna
                import IPython

                IPython.embed(colors="neutral")

    def compare_nested(self, storage_gna: Group, storage_dgf: NestedMKDict):
        for key_d, output_dgf in storage_dgf.walkitems():
            data_d = output_dgf.data
            for key_g in permutations(key_d):
                path_g = '/'.join(key_g)

                try:
                    data_g = storage_gna[path_g]
                except KeyError:
                    continue

                self.compare_outputs(data_g[:], data_d)

    def data_consistent(
        self,
        gna: NDArray,
        dgf: NDArray,
        *,
        slice: tuple[slice | None, ...] | None = None,
        slice_gna: tuple[slice | None, ...] | None = None,
        atol=0,
        rtol=0,
        # not used:
        gnaname: str = "",
        ignore: str | None = None,
    ) -> bool:
        if (slice_gna:=self._cmpopts.get("slice_gna")) is not None:
            gna = gna[slice_gna]
        elif (slice:=self._cmpopts.get("slice")) is not None:
            gna = gna[slice]
            dgf = dgf[slice]
        atol = float(self._cmpopts.get("atol", 0.0))
        rtol = float(self._cmpopts.get("rtol", 0.0))
        status = allclose(dgf, gna, rtol=rtol, atol=atol)

        fdiff = fabs(dgf - gna)
        self._maxdiff = float(fdiff.max())
        self._maxreldiff = float(nanmax(fdiff / gna))

        if status:
            return True

        return False


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", default=0, action="count", help="verbosity level"
    )

    input = parser.add_argument_group("input", "input related options")
    input.add_argument("input", type=File, help="input file to compare to")

    parser.add_argument(
        "-s",
        "--source-type",
        "--source",
        choices=("tsv", "hdf5", "root", "npz"),
        default="npz",
        help="Data source type",
    )

    crosscheck = parser.add_argument_group("comparison", "comparison related options")
    crosscheck.add_argument(
        "-l", "--last", action="store_true", help="process only the last item"
    )
    crosscheck.add_argument(
        "-e", "--embed-on-fail", action="store_true", help="embed on failure"
    )

    c = Comparator(parser.parse_args())
