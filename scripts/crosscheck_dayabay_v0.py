#!/usr/bin/env python

from argparse import Namespace
from contextlib import suppress
from itertools import islice, permutations
from typing import Any, Callable, Literal

from h5py import Dataset, File, Group
from matplotlib import pyplot as plt
from numpy import allclose, array, fabs, nanmax
from numpy.typing import NDArray

from dagflow.logger import INFO1, INFO2, INFO3, logger, set_level
from dagflow.output import Output
from models.dayabay_v0 import model_dayabay_v0
from multikeydict.nestedmkdict import NestedMKDict

set_level(INFO1)

def strip_last_day_periods_6_8(key: str, key2: str, data: NDArray):
    if '6AD' in key2 or '8AD' in key2:
        return data[:-1]

    return data

def strip_last_day_if_empty(key: str, key2: str, data: NDArray):
    if data[-1]!=0.0:
        return data
    return data[:-1]

reactors = ("DB1", "DB2", "LA1", "LA2", "LA3", "LA4")
# fmt: off
comparison_parameters = {
        "baseline": {"gnaname": "baseline", "gnascale": 1000, "rtol": 1.e-15},
        "detector.nprotons_nominal_ad": {"gnaname": "nprotons_nominal"},
        "conversion.reactorPowerConversion": {"gnaname": "conversion_factor", "rtol": 1.e-8 }
}
comparison_objects = {
    # dagflow: gna
    "edges.energy_edep": "evis_edges",
    "kinematics_sampler.mesh_edep": "evis_mesh",
    "ibd.enu": {"gnaname": "enu", "atol": 1e-14},
    "ibd.jacobian": {"gnaname": "jacobian", "atol": 1e-15},
    "ibd.crosssection": {"gnaname": "ibd_xsec", "rtol": 1.e-14},
    "oscprob": {"gnaname": "osc_prob_rd", "atol": 1e-15},
    "reactor_anue.neutrino_perfission_perMeV_nominal_pre": {"gnaname": "anuspec_coarse", "atol": 5.e-15},
    "reactor_anue.neutrino_perfission_perMeV_nominal": {"gnaname": "anuspec", "atol": 5.e-15},
    "reactor_offequilibrium_anue.correction_input.enu": {"gnaname": "offeq_correction_input_enu.DB1.U235", "rtol": 1e-15},
    "reactor_offequilibrium_anue.correction_input.offequilibrium_correction": {"gnaname": [f"offeq_correction_input.{reac}" for reac in reactors], "atol": 1.e-14},
    "reactor_offequilibrium_anue.correction_interpolated": {"gnaname": "offeq_correction_scale_interpolated.DB1", "rtol": 5e-12, "atol": 5e-15},
    "snf_anue.correction_input.snf_correction": {"gnaname": "snf_correction_scale_input", "atol": 5.e-15},
    "snf_anue.correction_input.enu": {"gnaname": "snf_correction_scale_input_enu.DB1", "rtol": 1e-15},
    "snf_anue.correction_interpolated": {"gnaname": "snf_correction_scale_interpolated", "rtol": 5.e-12},
    "baseline_factor_percm2": {"gnaname": "parameters.dayabay.baselineweight", "rtol": 1.e-15},
    "detector.nprotons": {"gnaname": "parameters.dayabay.nprotons_ad"},
    # "daily_data.detector.livetime": {"gnaname": "livetime_daily", "preprocess_gna": strip_last_day_periods_6_8}, # should be inconsistent as it is not rescaled in GNA
    "daily_data.detector.efflivetime": {"gnaname": "efflivetime_daily", "preprocess_gna": strip_last_day_periods_6_8},
    "daily_data.reactor.power": {"gnaname": "thermal_power", "preprocess_gna": strip_last_day_periods_6_8},
    "daily_data.reactor.fission_fraction": {"gnaname": "fission_fractions", "preprocess_gna": strip_last_day_periods_6_8},
    # "reactor.energy_per_fission_core_weighted_MeV": {"gnaname": "eper_fission_times_ff", "preprocess_gna": strip_last_day_periods_6_8}, # available only in cross-check version of the input hdf
    # "reactor.energy_per_fission_core_average_MeV": { "gnaname": "denom", "preprocess_gna": strip_last_day_periods_6_8 }, # available only in cross-check version of the input hdf
    # "reactor_detector.number_of_fissions_nprotons_percm2_core": {"gnaname": "parameters.dayabay.power_livetime_factor", "rtol": 1.e-8}, # available only in cross-check version of the input hdf
    # "eventscount.reactor_active_periods": {"gnaname": "kinint2", "rtol": 1.e-8}, # available only in cross-check version of the input hdf
    # "eventscount.snf_periods": {"gnaname": "kinint2_snf", "rtol": 1.e-8}, # Inconsistent! The input cross check model seem to be broken. Available only in cross-check version of the input hdf
    "eventscount.periods": {"gnaname": "kinint2", "rtol": 1.e-8}
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

    _n_success: int = 0
    _n_fail: int = 0

    @property
    def data_g(self) -> NDArray:
        return self._data_g

    @data_g.setter
    def data_g(self, data: NDArray):
        try:
            fcn = self._cmpopts["preprocess_gna"]
        except (TypeError, KeyError):
            self._data_g = data
        else:
            self._data_g = fcn(self._skey_gna, self._skey2_gna, data)

    def __init__(self, opts: Namespace):
        self.model = model_dayabay_v0(source_type=opts.source_type)
        self.opts = opts

        if opts.verbose:
            opts.verbose = min(opts.verbose, 3)
            set_level(globals()[f"INFO{opts.verbose}"])

        self.outputs_dgf = self.model.storage("outputs")
        self.parameters_dgf = self.model.storage("parameter.all")

        with suppress(StopIteration):
            self.compare(
                self.opts.input["parameters/dayabay"],
                comparison_parameters,
                self.parameters_dgf,
                self.compare_parameters
            )

        with suppress(StopIteration):
            self.compare(
                self.opts.input,
                comparison_objects,
                self.outputs_dgf,
                self.compare_outputs
            )

    def compare(
        self,
        gnasource: File | Group,
        comparison_objects: dict,
        outputs_dgf: NestedMKDict,
        compare: Callable,
    ) -> None:
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
                        self.compare_source(gnasource, compare, outputs_dgf)
                case str():
                    self.compare_source(gnasource, compare, outputs_dgf)
                case _:
                    raise RuntimeError()

        print(
            f"Cross check done {self._n_success+self._n_fail}: {self._n_success} success, {self._n_fail} fail"
        )

    def compare_source(
        self,
        gnasource: File | Group,
        compare: Callable,
        outputs_dgf: NestedMKDict
    ) -> None:
        from dagflow.parameters import Parameter
        path_gna = self._skey_gna.replace(".", "/")

        data_storage_gna = gnasource[path_gna]
        data_storage_dgf = outputs_dgf.any(self._skey_dgf)

        match data_storage_dgf, data_storage_gna:
            case Output(), Dataset():
                self.data_g = data_storage_gna[:]
                self._data_d = data_storage_dgf.data
                self._skey2_dgf = ""
                self._skey2_gna = ""
                compare()
            case Parameter(), Dataset():
                self.data_g = data_storage_gna[:]
                self._data_d = data_storage_dgf.to_dict()
                if self._data_g.dtype.names:
                    self.data_g = array([self._data_g[0]["value"]], dtype="d")
                self._skey2_dgf = ""
                self._skey2_gna = ""
                compare()
            case NestedMKDict(), Group():
                self.compare_nested(data_storage_gna, data_storage_dgf, compare)
            case _:
                raise RuntimeError("Unexpected data types")

    def compare_parameters(self):
        is_ok = True
        for key in ("value",): #, "central", "sigma"):
            try:
                vd = self._data_d[key]
            except KeyError:
                continue
            # vg = self._data_g[0][key]
            vg = self._data_g[0]

            if (scaleg:= self._cmpopts.get("gnascale")) is not None:
                vg*=scaleg

            is_ok = allclose(vd, vg, rtol=self.rtol, atol=self.atol)

            if is_ok:
                logger.log(INFO1, f"OK: {self.cmpstring} [{key}]")
                logger.log(INFO2, f"    {self.tolstring}")
                if (ignore := self._cmpopts.get("ignore")) is not None:
                    logger.log(INFO2, f"↑Ignore: {ignore}")
            else:
                self._maxdiff = float(fabs(vd-vg))
                self._maxreldiff = float(self._maxdiff/vg)

                logger.error(f"FAIL: {self.cmpstring} [{key}]")
                logger.error(f"      {self.tolstring}")
                logger.error(f"      max diff {self._maxdiff:.2g}, ")
                logger.error(f"      max rel diff {self._maxreldiff:.2g}")

                if self.opts.embed_on_failure:
                    try:
                        self._diff = self._data_d - self._data_g
                    except:
                        self._diff = False

                    import IPython

                    IPython.embed(colors="neutral")

                if self.opts.exit_on_failure:
                    raise StopIteration()

    def compare_outputs(self):
        is_ok = self.data_consistent(self._data_g, self._data_d)
        if is_ok:
            logger.log(INFO1, f"OK: {self.cmpstring}")
            logger.log(INFO2, f"    {self.tolstring}")
            logger.log(INFO2, f"    {self.shapestring}")
            if (ignore := self._cmpopts.get("ignore")) is not None:
                logger.log(INFO2, f"↑Ignore: {ignore}")
        else:
            logger.error(f"FAIL: {self.cmpstring}")
            logger.error(f"      {self.parstring}")
            logger.error(f"      {self.tolstring}")
            logger.error(f"      {self.shapestrings}")
            logger.error(f"      max diff {self._maxdiff:.2g}, ")
            logger.error(f"      max rel diff {self._maxreldiff:.2g}")

            if self.opts.plot_on_failure:
                if self._data_g.shape[0] < 100:
                    style = "o-"
                else:
                    style = "-"
                pargs = {"markerfacecolor": "none", "alpha": 0.8}

                plt.figure()
                ax = plt.subplot(111, xlabel="", ylabel="", title=self.key_dgf)
                ax.plot(self._data_g, style, label="GNA", **pargs)
                ax.plot(self._data_d, style, label="dagflow", **pargs)
                scale_factor = self._data_g.sum() / self._data_d.sum()
                ax.plot(
                    self._data_d * scale_factor,
                    f"{style}-",
                    label="dagflow scaled",
                    **pargs,
                )
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
        return f"dagflow:{self._skey_dgf}{self._skey2_dgf} ↔ gna:{self._skey_gna}{self._skey2_gna}"

    @property
    def tolstring(self) -> str:
        return f"rtol={self.rtol}" f" atol={self.atol}"

    @property
    def parstring(self) -> str:
        return f"dagflow[0]={self._data_d[0]}  gna[0]={self._data_g[0]}"

    @property
    def shapestring(self) -> str:
        return f"{self._data_g.shape}"

    @property
    def shapestrings(self) -> str:
        return f"dagflow: {self._data_d.shape}, gna: {self._data_g.shape}"

    def compare_nested(self, storage_gna: Group, storage_dgf: NestedMKDict, compare: Callable):
        for key_d, output_dgf in storage_dgf.walkitems():
            try:
                self._data_d = output_dgf.data
            except AttributeError:
                self._data_d = output_dgf.to_dict()
            self._skey2_dgf = ".".join(("",) + key_d)
            for key_g in permutations(key_d):
                path_g = "/".join(key_g)
                self._skey2_gna = ".".join(("",) + key_g)

                try:
                    data_g = storage_gna[path_g]
                except KeyError:
                    continue
                self.data_g = data_g[:]

                if self._data_g.dtype.names:
                    self.data_g = array([self._data_g[0]["value"]], dtype="d")

                compare()
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

            self._n_fail += 1
            return False

        fdiff = fabs(dgf - gna)
        self._maxdiff = float(fdiff.max())
        self._maxreldiff = float(nanmax(fdiff / gna))

        if status:
            self._n_success += 1
            return True

        self._n_fail += 1
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
