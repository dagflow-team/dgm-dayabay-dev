#!/usr/bin/env python

from argparse import Namespace
from contextlib import suppress
from itertools import islice, permutations
from typing import Any, Literal

from h5py import Dataset, File, Group
from matplotlib import pyplot as plt
from numpy import allclose, fabs, nanmax
from numpy.typing import NDArray

from dagflow.logger import INFO1, INFO2, INFO3, logger, set_level
from dagflow.output import Output
from models.dayabay_v0 import model_dayabay_v0
from multikeydict.nestedmkdict import NestedMKDict

set_level(INFO1)

reactors = ("DB1", "DB2", "LA1", "LA2", "LA3", "LA4")
# fmt: off
comparison_objects = {
    # dagflow: gna
    "edges.energy_edep": "evis_edges",
    "kinematics_sampler.mesh_edep": "evis_mesh",
    "ibd.enu": {"gnaname": "enu", "atol": 1e-14},
    "ibd.jacobian": {"gnaname": "jacobian", "atol": 1e-15},
    "ibd.crosssection": {"gnaname": "ibd_xsec", "rtol": 1.e-14},
    "oscprob": {"gnaname": "osc_prob_rd", "atol": 1e-15},
    "reactor_anue.neutrino_perfission_perMeV_nominal_pre": {"gnaname": "anuspec_coarse", "atol": 1.e-15},
    "reactor_anue.neutrino_perfission_perMeV_nominal": {"gnaname": "anuspec", "atol": 3.e-15},
    "reactor_offequilibrium_anue.correction_input.enu": {
        "gnaname": "offeq_correction_input_enu.DB1.U235",
        },
    "reactor_offequilibrium_anue.correction_input.offequilibrium_correction": {"gnaname": [f"offeq_correction_input.{reac}" for reac in reactors], "atol": 1.e-14},
    "snf_anue.correction_input.snf_correction": {"gnaname": "snf_correction_scale_input", "atol": 5.e-15},
    "snf_anue.correction_input.enu": {
        "gnaname": "snf_correction_scale_input_enu.DB1"
        },
    "snf_anue.correction_interpolated": {
        "gnaname": "snf_correction_scale_interpolated",
        # "atol": 3.e-15,
        "skip": True
    },
}
# fmt: on


class Comparator:
    opts: Namespace
    output_dgf: NestedMKDict

    _cmpopts: dict[str, Any] = {}
    _maxdiff: float = 0.0
    _maxreldiff: float = 0.0

    _skey_gna: str = ""
    _skey_dgf: str = ""
    _skey2_gna: str = ""
    _skey2_dgf: str = ""

    _data_g: NDArray
    _data_d: NDArray
    _diff: NDArray | Literal[False]

    def __init__(self, opts: Namespace):
        self.model = model_dayabay_v0(source_type=opts.source_type)
        self.opts = opts

        if opts.verbose:
            opts.verbose = min(opts.verbose, 3)
            set_level(globals()[f"INFO{opts.verbose}"])

        self.outputs_dgf = self.model.storage("outputs")

        with suppress(StopIteration):
            self.compare()

    def compare(self) -> None:
        if self.opts.last:
            iterable = islice(reversed(comparison_objects.items()), 1)
        else:
            iterable = comparison_objects.items()
        for self._skey_dgf, cmpopts in iterable:
            match cmpopts:
                case dict():
                    self._skey_gna = cmpopts["gnaname"]
                    self._cmpopts = cmpopts
                case str():
                    self._skey_gna = cmpopts
                    self._cmpopts = {}
                case _:
                    raise RuntimeError(f"Invalid {cmpopts=}")

            if self._cmpopts.get("skip"):
                continue

            match self._skey_gna:
                case list() | tuple():
                    keys_gna = self._skey_gna
                    for self._skey_gna in keys_gna:
                        self.compare_source()
                case str():
                    self.compare_source()
                case _:
                    raise RuntimeError()

    def compare_source(self) -> None:
        path_gna = self._skey_gna.replace(".", "/")

        data_storage_gna = self.opts.input[path_gna]
        data_storage_dgf = self.outputs_dgf.any(self._skey_dgf)

        match data_storage_dgf, data_storage_gna:
            case Output(), Dataset():
                self._data_g = data_storage_gna[:]
                self._data_d = data_storage_dgf.data
                self._skey2_dgf = ""
                self._skey2_gna = ""
                self.compare_outputs()
            case NestedMKDict(), Group():
                self.compare_nested(data_storage_gna, data_storage_dgf)
            case _:
                raise RuntimeError("Unexpected data types")

    def compare_outputs(self):
        is_ok = self.data_consistent(self._data_g, self._data_d)
        if is_ok:
            logger.log(INFO1, f"OK: {self.cmpstring} {self.shapestring}")
            if (ignore := self._cmpopts.get("ignore")) is not None:
                logger.log(INFO2, f"↑Ignore: {ignore}")
        else:
            logger.error(
                f"FAIL: {self.cmpstring} {self.shapestrings} "
                f"max diff {self._maxdiff:.2g}, "
                f"max rel diff {self._maxreldiff:.2g}"
            )

            if self.opts.plot_on_failure:
                if self._data_g.shape[0]<100:
                    style = 'o-'
                else:
                    style = '-'
                pargs = {"markerfacecolor": "none"}

                plt.figure()
                ax = plt.subplot(111, xlabel="", ylabel="", title=self.key_dgf)
                ax.plot(self._data_g, style, label="GNA", **pargs)
                ax.plot(self._data_d, style, label="dagflow", **pargs)
                ax.legend()
                ax.grid()

                plt.figure()
                ax = plt.subplot(
                    111, xlabel="", ylabel="dagflow/GNA", title=self.key_dgf
                )
                with suppress(ValueError):
                    ax.plot(self._data_d / self._data_g, style, **pargs)
                ax.grid()

                plt.show()

            if self.opts.embed_on_failure:
                try:
                    self._diff = self._data_d - self._data_g
                except:
                    self._diff = False

                import IPython

                IPython.embed(colors="neutral")

            if self.opts.exit_on_failure:
                raise StopIteration()

    @property
    def key_dgf(self) -> str:
        return f"{self._skey_dgf}{self._skey2_dgf}"

    @property
    def cmpstring(self) -> str:
        return (
            f"{self._skey_dgf}{self._skey2_dgf}↔{self._skey_gna}{self._skey2_gna}"
            f" rtol={self.rtol}"
            f" atol={self.atol}"
        )

    @property
    def shapestring(self) -> str:
        return f"{self._data_g.shape}"

    @property
    def shapestrings(self) -> str:
        return f"{self._data_d.shape}, {self._data_g.shape}"

    def compare_nested(self, storage_gna: Group, storage_dgf: NestedMKDict):
        for key_d, output_dgf in storage_dgf.walkitems():
            self._data_d = output_dgf.data
            self._skey2_dgf = ".".join(("",) + key_d)
            for key_g in permutations(key_d):
                path_g = "/".join(key_g)

                try:
                    data_g = storage_gna[path_g]
                except KeyError:
                    continue
                self._data_g = data_g[:]

                self._skey2_gna = ".".join(("",) + key_g)
                self.compare_outputs()
                break
            else:
                raise RuntimeError(
                    f"Was not able to find a match for {self._skey2_dgf}"
                )

    @property
    def atol(self) -> float:
        return float(self._cmpopts.get("atol", 0.0))

    @property
    def rtol(self) -> float:
        return float(self._cmpopts.get("rtol", 0.0))

    def data_consistent(self, gna: NDArray, dgf: NDArray) -> bool:
        if (slice_gna := self._cmpopts.get("slice_gna")) is not None:
            gna = gna[slice_gna]
        elif (slice := self._cmpopts.get("slice")) is not None:
            gna = gna[slice]
            dgf = dgf[slice]
        try:
            status = allclose(dgf, gna, rtol=self.rtol, atol=self.atol)
        except ValueError:
            self._maxdiff = -1
            self._maxreldiff = -1
            return False

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
        "-e", "--embed-on-failure", action="store_true", help="embed on failure"
    )
    crosscheck.add_argument(
        "-p", "--plot-on-failure", action="store_true", help="plot on failure"
    )
    crosscheck.add_argument(
        "-x", "--exit-on-failure", action="store_true", help="exit on failure"
    )

    c = Comparator(parser.parse_args())
