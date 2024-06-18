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
    "default": {"rtol": 1.0e-8},
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
    "lsnl_weight": {"location": "detector.lsnl_scale_a", "rtol": 1.0e-8},
    "nominal_thermal_power": "skip",
    "offeq_scale": "skip",
    "DeltaMSq12": "skip",
    "DeltaMSq13": "skip",
    "DeltaMSq23": "skip",
    "SinSqDouble12": "skip",
    "SinSqDouble13": "skip",
    "snf_scale": {
        "location": "reactor.snf_scale",
        # "rtol": 1.0e-8
    },
    "spectral_weights": "skip",
}
# fmt: on


class NuisanceComparator:
    __slots__ = (
        "model",
        "parameters_dgf",
        "opts",
        "outputs_dgf",
        "outputs_dgf_default",
        "_cmpopts",
        "_maxdiff",
        "_maxreldiff",
        "_skey_gna",
        "_skey_dgf",
        "_skey2_gna",
        "_skey2_dgf",
        "_skey_par_gna",
        "_skey_par_dgf",
        "_skey2_par_gna",
        "_skey2_par_dgf",
        "_data_gna",
        "_data_dgf",
        "_diff",
        "_n_success",
        "_n_fail",
    )
    opts: Namespace
    outputs_dgf: NestedMKDict
    outputs_dgf_default: NestedMKDict

    _cmpopts: dict[str, Any]
    _maxdiff: float
    _maxreldiff: float

    _skey_gna: str
    _skey_dgf: str
    _skey2_gna: str
    _skey2_dgf: str

    _skey_par_gna: str
    _skey_par_dgf: str
    _skey2_par_gna: str
    _skey2_par_dgf: str

    _data_gna: NDArray
    _data_dgf: NDArray
    _diff: NDArray | Literal[False]

    _n_success: int
    _n_fail: int

    def __init__(self, opts: Namespace):
        self._cmpopts = {}

        self._maxdiff = 0.0
        self._maxreldiff = 0.0

        self._skey_gna = ""
        self._skey_dgf = ""
        self._skey2_gna = ""
        self._skey2_dgf = ""

        self._skey_par_gna = ""
        self._skey_par_dgf = ""
        self._skey2_par_gna = ""
        self._skey2_par_dgf = ""

        self._n_success = 0
        self._n_fail = 0

        self.model = model_dayabay_v0(source_type=opts.source_type)
        self.opts = opts

        if opts.verbose:
            opts.verbose = min(opts.verbose, 3)
            set_level(globals()[f"INFO{opts.verbose}"])

        self._skey_gna = "erec"
        self._skey_dgf = "eventscount.erec"
        self.outputs_dgf = self.model.storage("outputs.eventscount.erec")
        self.parameters_dgf = self.model.storage("parameter.all")

        with suppress(StopIteration):
            self.process()

    def process(self) -> None:
        default = self.opts.input["default"]
        self._skey_par_gna = "default"
        self._skey_par_dgf = "default"
        self._cmpopts = comparison["default"]
        self._compare_hists(default, save=True)

        source = self.opts.input["dayabay"]

        parameters = self.model.storage("parameter.all")
        skipped = set()
        for parpath, results in iterate_mappings_till_key(source, "values"):
            par = parpath[1:].replace("/", ".")

            value_central, value_minus, value_plus = results["values"]
            results_minus = results["minus"]
            results_plus = results["plus"]

            paritems = par.split(".")
            parname, index = paritems[0], paritems[1:]
            if parname == "pmns":
                parname, index = index[0], index[1:]

            self._cmpopts = comparison[parname]
            if self._cmpopts == "skip":
                if parname not in skipped:
                    logger.warning(f"{parname} skip")
                    skipped.add(parname)
                continue

            logger.log(
                INFO1,
                f"{parname}: v={value_central}, v-={value_minus}, v+={value_plus}",
            )

            parsloc = parameters.any(self._cmpopts["location"])
            par = parsloc[index]
            assert par.value == value_central

            self._skey_par_gna = parname
            self._skey2_par_gna = ".".join([""] + index)
            self._skey_par_dgf = self._cmpopts["location"]
            self._skey2_par_dgf = ""

            par.push(value_plus)
            self._process_par_offset(parname, index, value_minus, results_minus)

            par.value = value_minus
            self._process_par_offset(parname, index, value_plus, results_plus)

            par.pop()

    def _process_par_offset(
        self, parname: str, index: Sequence[str], value: float, results: Mapping
    ):
        if self._compare_hists(results):
            logger.log(INFO1, f"OK: {self.cmpstring_par}")
            logger.log(INFO2, f"    {self.tolstring}")
            logger.log(INFO2, f"    {self.shapestring}")
        else:
            logger.error(f"FAIL: {self.cmpstring_par}")

    def _compare_hists(self, results: Mapping, *, save: bool = False) -> bool:
        is_ok = True
        for ad, addir in results.items():
            for period, data in addir.items():
                if (
                    period == "6AD"
                    and ad in ("AD22", "AD34")
                    or period == "7AD"
                    and ad == "AD11"
                ):
                    continue
                self._skey2_gna = f".{ad}.{period}"
                dgf = self.outputs_dgf[ad, period]

                self._data_dgf = dgf.data
                self._data_gna = data[:]

                is_ok &= self._compare_data()
        return is_ok

    def _compare_data(self) -> bool:
        is_ok = self._data_consistent(self._data_gna, self._data_dgf)
        if is_ok:
            logger.log(INFO2, f"OK: {self.cmpstring}")
            # logger.log(INFO2, f"    {self.tolstring}")
            # logger.log(INFO2, f"    {self.shapestring}")
            # if (ignore := self._cmpopts.get("ignore")) is not None:
            #     logger.log(INFO2, f"↑Ignore: {ignore}")

            return True

        logger.log(INFO1, f"OK: {self.cmpstring_par}")
        logger.error(f"FAIL: {self.cmpstring}")
        logger.error(f"      {self.tolstring}")
        logger.error(f"      {self.shapestrings}")
        logger.error(f"      max diff {self._maxdiff:.2g}, ")
        logger.error(f"      max rel diff {self._maxreldiff:.2g}")

        if self.opts.plot_on_failure:
            self.plot_1d()

        if self.opts.embed_on_failure:
            try:
                self._diff = self._data_dgf - self._data_gna
            except:
                self._diff = False

            import IPython

            IPython.embed(colors="neutral")

        if self.opts.exit_on_failure:
            raise StopIteration()

        return False

    def plot_1d(self):
        if self._data_gna.shape[0] < 100:
            style = "o-"
        else:
            style = "-"
        pargs = {"markerfacecolor": "none", "alpha": 0.4}

        plt.figure()
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="",
            title=f"""{self.cmpstring_par}:
{self.cmpstring}""",
        )
        ax.plot(self._data_gna, style, label="GNA", **pargs)
        ax.plot(self._data_dgf, style, label="dagflow", **pargs)
        # scale_factor = self._data_gna.sum() / self._data_dgf.sum()
        # ax.plot(
        #     self._data_dgf * scale_factor,
        #     f"{style}-",
        #     label="dagflow scaled",
        #     **pargs,
        # )
        ax.legend()
        ax.grid()

        plt.figure()
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="dagflow/GNA-1",
            title=f"""{self.cmpstring_par}:
{self.cmpstring}""",
        )
        with suppress(ValueError):
            ax.plot(self._data_dgf / self._data_gna - 1, style, **pargs)
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
    def cmpstring_par(self) -> str:
        return f"dagflow:{self._skey_par_dgf}{self._skey2_par_dgf} ↔ gna:{self._skey_par_gna}{self._skey2_par_gna}"

    @property
    def tolstring(self) -> str:
        return f"rtol={self.rtol}" f" atol={self.atol}"

    @property
    def shapestring(self) -> str:
        return f"{self._data_gna.shape}"

    @property
    def shapestrings(self) -> str:
        return f"dagflow: {self._data_dgf.shape}, gna: {self._data_gna.shape}"

    @property
    def atol(self) -> float:
        return float(self._cmpopts.get("atol", 0.0))

    @property
    def rtol(self) -> float:
        return float(self._cmpopts.get("rtol", 0.0))

    def _data_consistent(self, gna: NDArray, dgf: NDArray) -> bool:
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
