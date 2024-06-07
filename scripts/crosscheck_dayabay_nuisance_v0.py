#!/usr/bin/env python

from argparse import Namespace
from contextlib import suppress
from itertools import permutations
from typing import Any, Callable, Generator, Literal, Mapping, Sequence

from h5py import Dataset, File, Group
from matplotlib import pyplot as plt
from numpy import allclose, array, fabs, ma, nanmax
from numpy.typing import NDArray

from dagflow.logger import INFO1, INFO2, INFO3, logger, set_level
from dagflow.output import Output
from models.dayabay_v0 import model_dayabay_v0
from multikeydict.nestedmkdict import NestedMKDict

set_level(INFO1)

# fmt: on
comparison = {
    "OffdiagScale": "skip",
    "acc_norm": "skip",
    "bkg_rate_alphan": "skip",
    "bkg_rate_amc": "skip",
    "bkg_rate_fastn": "skip",
    "bkg_rate_lihe": "skip",
    "effunc_uncorr": "skip",
    "eper_fission": "skip",
    "eres": "skip",
    "escale": "skip",
    "fission_fractions_corr": "skip",
    "global_norm": "skip",
    "lsnl_weight": "skip",
    "nominal_thermal_power": "skip",
    "offeq_scale": "skip",
    "DeltaMSq12": "skip",
    "DeltaMSq13": "skip",
    "DeltaMSq23": "skip",
    "SinSqDouble12": "skip",
    "SinSqDouble13": "skip",
    "snf_scale": "skip",
    "spectral_weights": "skip",
}
# fmt: on


class NuisanceComparator:
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

        if (slice_gna := self._cmpopts.get("slice_gna")) is not None:
            self._data_g = self._data_g[slice_gna]
        elif (slice := self._cmpopts.get("slice")) is not None:
            self._data_g = self._data_g[slice]

    @property
    def data_d(self) -> NDArray:
        return self._data_d

    @data_d.setter
    def data_d(self, data: NDArray):
        self._data_d = data

        if (slice := self._cmpopts.get("slice")) is not None:
            self._data_d = data[slice]

    def __init__(self, opts: Namespace):
        self.model = model_dayabay_v0(source_type=opts.source_type)
        self.opts = opts

        if opts.verbose:
            opts.verbose = min(opts.verbose, 3)
            set_level(globals()[f"INFO{opts.verbose}"])

        self.outputs_dgf = self.model.storage("outputs.eventscount.erec")
        self.parameters_dgf = self.model.storage("parameter.all")

        with suppress(StopIteration):
            self.process()

    def process(self) -> None:
        source = self.opts.input["dayabay"]

        skipped = set()
        for parpath, results in iterate_mappings_till_key(source, "values"):
            par = parpath[1:].replace("/", ".")

            value_central, value_minus, value_plus = results["values"]
            results_minus = results["minus"]
            results_plus = results["plus"]

            paritems = par.split(".")
            parname, index = paritems[0], paritems[1:]
            if parname=="pmns":
                parname, index = index[0], index[1:]

            cfg = comparison[parname]
            if cfg=="skip":
                if parname not in skipped:
                    print(f"{parname} skip")
                    skipped.add(parname)
                continue

            print(f"{parname}: v={value_central}, v-={value_minus}, v+={value_plus}")

            self._process_par_offset(parname, index, value_minus, results_minus)
            self._process_par_offset(parname, index, value_plus, results_plus)

    def _process_par_offset(
        self,
        parname: str,
        index: Sequence[str],
        value: float,
        results: Mapping
    ):
        print(parname, index, value, results)

    def compare_source(
        self, gnasource: File | Group, compare: Callable, outputs_dgf: NestedMKDict
    ) -> None:
        from dagflow.parameters import Parameter

        path_gna = self._skey_gna.replace(".", "/")

        data_storage_gna = gnasource[path_gna]
        data_storage_dgf = outputs_dgf.any(self._skey_dgf)

        match data_storage_dgf, data_storage_gna:
            case Output(), Dataset():
                self.data_g = data_storage_gna[:]
                self.data_d = data_storage_dgf.data
                self._skey2_dgf = ""
                self._skey2_gna = ""
                compare()
            case Parameter(), Dataset():
                self.data_g = data_storage_gna[:]
                self.data_d = data_storage_dgf.to_dict()
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
        for key in ("value",):  # , "central", "sigma"):
            try:
                vd = self._data_d[key]
            except KeyError:
                continue
            # vg = self._data_g[0][key]
            vg = self._data_g[0]

            if (scaleg := self._cmpopts.get("gnascale")) is not None:
                vg *= scaleg

            is_ok = allclose(vd, vg, rtol=self.rtol, atol=self.atol)

            if is_ok:
                logger.log(INFO1, f"OK: {self.cmpstring} [{key}]")
                logger.log(INFO2, f"    {self.tolstring}")
                if (ignore := self._cmpopts.get("ignore")) is not None:
                    logger.log(INFO2, f"↑Ignore: {ignore}")
            else:
                self._maxdiff = float(fabs(vd - vg))
                self._maxreldiff = float(self._maxdiff / vg)

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
                self.plot()

            if self.opts.embed_on_failure:
                try:
                    self._diff = self._data_d - self._data_g
                except:
                    self._diff = False

                import IPython

                IPython.embed(colors="neutral")

            if self.opts.exit_on_failure:
                raise StopIteration()

    def plot(self):
        ndim = self._data_g.ndim
        if ndim == 1:
            return self.plot_1d()
        elif ndim == 2:
            return self.plot_mat()

    def plot_mat(self):
        data_g = ma.array(self._data_g, mask=(self._data_g == 0))
        data_d = ma.array(self._data_d, mask=(self._data_d == 0))
        plt.figure()
        ax = plt.subplot(111, xlabel="", ylabel="", title=f"GNA {self.key_gna}")
        ax.matshow(data_g)
        ax.grid()

        plt.figure()
        ax = plt.subplot(111, xlabel="", ylabel="", title=f"dagflow {self.key_dgf}")
        ax.matshow(data_d)
        ax.grid()

        # plt.figure()
        # ax = plt.subplot(111, xlabel="", ylabel="", title=f"both {self.key_dgf}")
        # ax.matshow(data_g, alpha=0.6, cmap="viridis")
        # ax.matshow(data_d, alpha=0.6, cmap="inferno")
        # ax.grid()

        plt.figure()
        ax = plt.subplot(111, xlabel="", ylabel="", title=f"diff {self.key_dgf}")
        ax.matshow(data_d - data_g, alpha=0.6)
        ax.grid()

        plt.show()

    def plot_1d(self):
        if self._data_g.shape[0] < 100:
            style = "o-"
        else:
            style = "-"
        pargs = {"markerfacecolor": "none", "alpha": 0.4}

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
        ax = plt.subplot(111, xlabel="", ylabel="dagflow/GNA", title=self.key_dgf)
        with suppress(ValueError):
            ax.plot(self._data_d / self._data_g, style, **pargs)
        ax.grid()

        plt.show()

    @property
    def key_dgf(self) -> str:
        return f"{self._skey_dgf}{self._skey2_dgf}"

    @property
    def key_gna(self) -> str:
        return f"{self._skey_gna}{self._skey2_gna}"

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

    def compare_nested(
        self, storage_gna: Group, storage_dgf: NestedMKDict, compare: Callable
    ):
        for key_d, output_dgf in storage_dgf.walkitems():
            try:
                self.data_d = output_dgf.data
            except AttributeError:
                self.data_d = output_dgf.to_dict()
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


def iterate_mappings_till_key(
    source: Mapping, target_key: str, *, head: str = ""
) -> Generator[tuple[str, Any], None, None]:
    for subkey, submapping in source.items():
        try:
            keys = submapping.keys()
        except AttributeError:
            continue

        retkey = ".".join((head, subkey))
        if target_key in keys:
            yield retkey, submapping
        else:
            yield from iterate_mappings_till_key(submapping, target_key, head=retkey)


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

    c = NuisanceComparator(parser.parse_args())
