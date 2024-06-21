#!/usr/bin/env python

from argparse import Namespace
from contextlib import suppress
from itertools import permutations
from typing import Any, Generator, Literal, Mapping

from h5py import File
from matplotlib import pyplot as plt
from numpy import allclose, fabs, nanmax
from numpy.typing import NDArray

from dagflow.logger import INFO1, INFO2, INFO3, logger, set_level
from models.dayabay_v0 import model_dayabay_v0
from multikeydict.nestedmkdict import NestedMKDict

set_level(INFO1)


def minus_one(*values):
    return tuple(v - 1.0 for v in values)


comparison = {
    "default": {"rtol": 1.0e-8},
    "OffdiagScale": {
        # TODO
        "skip": True
    },
    "acc_norm": {"location": "all.bkg.rate.acc", "rtol": 1.0e-8, "scale": True},
    "bkg_rate_alphan": {"location": "all.bkg.rate.alphan", "rtol": 1.0e-8},
    "bkg_rate_amc": {"location": "all.bkg.rate.amc", "rtol": 1.0e-8},
    "bkg_rate_fastn": {"location": "all.bkg.rate.fastn", "rtol": 1.0e-8},
    "bkg_rate_lihe": {"location": "all.bkg.rate.lihe", "rtol": 1.0e-8},
    "effunc_uncorr": {
        # TODO
        "skip": True
    },
    "escale": {
        # TODO
        "skip": True
    },
    "eres": {
        "location": "all.detector.eres",
        "keys_mapping": {
            ("a",): ("a_nonuniform",),
            ("b",): ("b_stat",),
            ("c",): ("c_noise",),
        },
        "rtol": 1.0e-8,
    },
    "global_norm": {"location": "all.detector.global_normalization", "rtol": 1.0e-8},
    "lsnl_weight": {"location": "all.detector.lsnl_scale_a", "rtol": 1.0e-8},
    "DeltaMSq12": {"location": "all.oscprob.DeltaMSq21", "rtol": 1.0e-8},
    "DeltaMSq23": {"location": "all.oscprob.DeltaMSq32", "rtol": 1.0e-8},
    "SinSqDouble12": {"location": "all.oscprob.SinSq2Theta12", "rtol": 1.0e-8},
    "SinSqDouble13": {"location": "all.oscprob.SinSq2Theta13", "rtol": 1.0e-8},
    "spectral_weights": {
        "location": "all.neutrino_perfission",
        "keys_mapping": lambda s: (s[0].replace("anue_weight", "spec_scale"),),
        "rtol": 1.0e-8,
    },
    # Reactor
    "nominal_thermal_power": {"location": "all.reactor.nominal_thermal_power", "rtol": 1.0e-8},
    "fission_fractions_corr": {"location": "all.reactor.fission_fraction_scale", "rtol": 1.0e-8},
    "eper_fission": {"location": "all.reactor.energy_per_fission", "rtol": 1.0e-8},
    "offeq_scale": {"location": "all.reactor.offequilibrium_scale", "rtol": 1.0e-8},
    "snf_scale": {"location": "all.reactor.snf_scale", "rtol": 1.0e-8},
}


class NuisanceComparator:
    __slots__ = (
        "model",
        "parameters_dgf",
        "opts",
        "outputs_dgf",
        "outputs_dgf_default",
        "outputs_gna_default",
        "cmpopts",
        #
        "value_central",
        "value_current",
        "value_left",
        "value_right",
        #
        "maxdiff",
        "maxreldiff",
        #
        "index",
        "skey_gna",
        "skey_dgf",
        "skey2_gna",
        "skey2_dgf",
        "skey_par_gna",
        "skey_par_dgf",
        "skey2_par_gna",
        "skey2_par_dgf",
        "data_gna",
        "data_dgf",
        "diff",
        "n_success",
        "n_fail",
    )
    opts: Namespace
    outputs_dgf: NestedMKDict
    outputs_dgf_default: NestedMKDict
    outputs_gna_default: NestedMKDict

    cmpopts: dict[str, Any]

    value_central: float
    value_current: float
    value_left: float
    value_right: float

    maxdiff: float
    maxreldiff: float

    index: tuple[str, ...]

    skey_gna: str
    skey_dgf: str
    skey2_gna: str
    skey2_dgf: str

    skey_par_gna: str
    skey_par_dgf: str
    skey2_par_gna: str
    skey2_par_dgf: str

    data_gna: NDArray
    data_dgf: NDArray
    diff: NDArray | Literal[False]

    n_success: int
    n_fail: int

    def __init__(self, opts: Namespace):
        self.cmpopts = {}

        self.maxdiff = 0.0
        self.maxreldiff = 0.0

        self.index = ()

        self.skey_gna = ""
        self.skey_dgf = ""
        self.skey2_gna = ""
        self.skey2_dgf = ""

        self.skey_par_gna = ""
        self.skey_par_dgf = ""
        self.skey2_par_gna = ""
        self.skey2_par_dgf = ""

        self.n_success = 0
        self.n_fail = 0

        self.value_central = -1111.1111
        self.value_current = -1111.1111
        self.value_left = -1111.1111
        self.value_right = -1111.1111

        self.outputs_dgf_default = NestedMKDict(sep=".")
        self.outputs_gna_default = NestedMKDict(sep=".")

        self.model = model_dayabay_v0(source_type=opts.source_type)
        self.opts = opts

        if opts.verbose:
            opts.verbose = min(opts.verbose, 3)
            set_level(globals()[f"INFO{opts.verbose}"])

        self.skey_gna = "fine"
        self.skey_dgf = opts.object
        self.outputs_dgf = self.model.storage(f"outputs.{self.skey_dgf}")
        self.parameters_dgf = self.model.storage("parameter")

        with suppress(StopIteration):
            self.process()

    def _skip_par(self, parname: str) -> bool:
        if not self.opts.pars:
            return False

        for mask in self.opts.pars:
            if mask in parname:
                return False

        return True

    def process(self) -> None:
        self.check_default(save=True)

        source = self.opts.input["dayabay"]
        inactive_detectors = set(frozenset(ia) for ia in self.model.inactive_detectors)

        skipped = set()
        for parpath, results in iterate_mappings_till_key(source, "values"):
            if self._skip_par(parpath):
                continue

            par = parpath[1:].replace("/", ".")

            paritems = par.split(".")
            parname, index = paritems[0], tuple(paritems[1:])
            if parname == "pmns":
                parname, index = index[0], index[1:]
            if set(index) in inactive_detectors:
                logger.log(INFO1, f"Skip {parpath}")
                continue

            self.cmpopts = comparison[parname]
            if self.cmpopts.get("skip"):
                if parname not in skipped:
                    logger.warning(f"{parname} skip")
                    skipped.add(parname)
                continue

            value_fcn = self.cmpopts.get("value_fcn", lambda *v: v)
            self.value_central, self.value_left, self.value_right = value_fcn(
                *results["values"]
            )

            parsloc = self.parameters_dgf.any(self.cmpopts["location"])
            keys_mapping = self.cmpopts.get("keys_mapping", lambda s: s)
            if isinstance(keys_mapping, dict):
                keys_fcn = lambda s: keys_mapping.get(s, s)
            else:
                keys_fcn = keys_mapping
            par = get_orderless(parsloc, keys_fcn(index))

            if self.cmpopts.get("scale"):
                self.value_central *= par.value
                self.value_left *= self.value_central
                self.value_right *= self.value_central

            if par.value != self.value_central:
                logger.error(
                    f"Parameters not consistent: dgf={par.value} gna={self.value_central}"
                )
                continue

            results_left = results["minus"]
            results_right = results["plus"]

            self.skey_par_gna = parname
            self.skey2_par_gna = ".".join(("",) + index)
            self.skey_par_dgf = self.cmpopts["location"]
            self.skey2_par_dgf = ""

            self.value_current = self.value_right
            par.push(self.value_current)
            logger.log(INFO1, f"{parname}: v={self.valuestring}")
            self.process_par_offset(results_right)

            self.value_current = self.value_left
            par.value = self.value_current
            logger.log(INFO1, f"{parname}: v={self.valuestring}")
            self.process_par_offset(results_left)

            par.pop()
            self.check_default("restore", check_change=False)

    def check_default(
        self, label="default", *, save: bool = False, check_change: bool = True
    ):
        default = self.opts.input["default"]
        self.skey_par_gna = "default"
        self.skey_par_dgf = label
        self.cmpopts = comparison["default"]
        self.compare_hists(default, save=save, check_change=check_change)

    def process_par_offset(self, results: Mapping):
        if self.compare_hists(results):
            logger.log(INFO1, f"OK: {self.cmpstring_par}")
            logger.log(INFO2, f"    {self.tolstring}")
            logger.log(INFO2, f"    {self.shapestring}")
        else:
            logger.error(f"FAIL: {self.cmpstring_par}")

    def compare_hists(
        self, results: Mapping, *, save: bool = False, check_change: bool = True
    ) -> bool:
        if save:
            check_change = False
        is_ok = True

        change2_gna = 0.0
        change2_dgf = 0.0
        for ad, addir in results.items():
            for period, data in addir.items():
                if (
                    period == "6AD"
                    and ad in ("AD22", "AD34")
                    or period == "7AD"
                    and ad == "AD11"
                ):
                    continue
                self.skey2_gna = f".{ad}.{period}"
                self.index = (ad, period)
                dgf = self.outputs_dgf[ad, period]

                self.data_dgf = dgf.data
                self.data_gna = data[:]

                if save:
                    self.outputs_dgf_default[ad, period] = self.data_dgf.copy()
                    self.outputs_gna_default[ad, period] = self.data_gna.copy()
                else:
                    change2_dgf += (
                        (self.data_dgf - self.outputs_dgf_default[ad, period]) ** 2
                    ).sum()
                    change2_gna += (
                        (self.data_gna - self.outputs_gna_default[ad, period]) ** 2
                    ).sum()

                is_ok &= self.compare_data()

        if check_change:
            if change2_dgf == 0.0 or change2_gna == 0.0:
                logger.error(
                    f"FAIL: data unchanged dgf²={change2_dgf} gna²={change2_gna}"
                )
                return False
        return is_ok

    def compare_data(self) -> bool:
        is_ok = self.data_consistent(self.data_gna, self.data_dgf)
        if is_ok:
            logger.log(INFO2, f"OK: {self.cmpstring}")
            # logger.log(INFO2, f"    {self.tolstring}")
            # logger.log(INFO2, f"    {self.shapestring}")
            # if (ignore := self.cmpopts.get("ignore")) is not None:
            #     logger.log(INFO2, f"↑Ignore: {ignore}")

            return True

        logger.error(f"FAIL: {self.cmpstring_par}")
        logger.error(f"      {self.cmpstring}")
        logger.error(f"      {self.tolstring}")
        logger.error(f"      {self.shapestrings}")
        logger.error(f"      max diff {self.maxdiff:.2g}, ")
        logger.error(f"      max rel diff {self.maxreldiff:.2g}")

        if self.opts.plot_on_failure:
            self.plot_1d()

        if self.opts.embed_on_failure:
            try:
                self.diff = self.data_dgf - self.data_gna
            except:
                self.diff = False

            import IPython

            IPython.embed(colors="neutral")

        if self.opts.exit_on_failure:
            raise StopIteration()

        return False

    def plot_1d(self):
        if self.data_gna.shape[0] < 100:
            style = "o-"
        else:
            style = "-"
        pargs = {"markerfacecolor": "none", "alpha": 0.4}

        plt.figure()
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="",
            title=f"""{self.cmpstring_par}:\n{self.cmpstring}\n{self.valuestring}""",
        )
        ax.plot(self.data_gna, style, label="GNA", **pargs)
        ax.plot(self.data_dgf, style, label="dagflow", **pargs)
        # scale_factor = self.data_gna.sum() / self.data_dgf.sum()
        # ax.plot(
        #     self.data_dgf * scale_factor,
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
            ylabel="Ratio-1",
            title=f"""{self.cmpstring_par}:\n{self.cmpstring}\n{self.valuestring}""",
        )
        with suppress(ValueError):
            ax.plot(self.data_dgf / self.data_gna - 1, style, label="dgf/GNA", **pargs)

        ax.legend()
        ax.grid()

        plt.figure()
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="Ratio-1",
            title=f"""{self.cmpstring_par}:\n{self.cmpstring}\n{self.valuestring}""",
        )
        with suppress(ValueError):
            ax.plot(self.data_dgf / self.data_gna - 1, style, label="dgf/GNA", **pargs)

        dgf_reference = self.outputs_dgf_default[self.index]
        with suppress(ValueError):
            ax.plot(
                self.data_dgf / dgf_reference - 1, style, label="dgf/default", **pargs
            )

        gna_reference = self.outputs_gna_default[self.index]
        with suppress(ValueError):
            ax.plot(
                self.data_gna / gna_reference - 1, style, label="gna/default", **pargs
            )
        ax.legend()
        ax.grid()

        plt.show()

    @property
    def key_dgf(self) -> str:
        return f"{self.skey_dgf}{self.skey2_dgf}"

    @property
    def key_gna(self) -> str:
        return f"{self.skey_gna}{self.skey2_gna}"

    @property
    def cmpstring(self) -> str:
        return f"dagflow:{self.skey_dgf}{self.skey2_dgf} ↔ gna:{self.skey_gna}{self.skey2_gna}"

    @property
    def cmpstring_par(self) -> str:
        return f"dagflow:{self.skey_par_dgf}{self.skey2_par_dgf} ↔ gna:{self.skey_par_gna}{self.skey2_par_gna}"

    @property
    def valuestring(self) -> str:
        return f"{self.value_central}→{self.value_current}"

    @property
    def tolstring(self) -> str:
        return f"rtol={self.rtol}" f" atol={self.atol}"

    @property
    def shapestring(self) -> str:
        return f"{self.data_gna.shape}"

    @property
    def shapestrings(self) -> str:
        return f"dagflow: {self.data_dgf.shape}, gna: {self.data_gna.shape}"

    @property
    def atol(self) -> float:
        return float(self.cmpopts.get("atol", 0.0))

    @property
    def rtol(self) -> float:
        return float(self.cmpopts.get("rtol", 0.0))

    def data_consistent(self, gna: NDArray, dgf: NDArray) -> bool:
        try:
            status = allclose(dgf, gna, rtol=self.rtol, atol=self.atol)
        except ValueError:
            self.maxdiff = -1
            self.maxreldiff = -1

            self.n_fail += 1
            return False

        fdiff = fabs(dgf - gna)
        self.maxdiff = float(fdiff.max())
        self.maxreldiff = float(nanmax(fdiff / gna))

        if status:
            self.n_success += 1
            return True

        self.n_fail += 1
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


def get_orderless(storage: NestedMKDict | Any, key: list[str]) -> Any:
    if not key:
        return storage
    for pkey in permutations(key):
        with suppress(KeyError):
            return storage[pkey]
    raise KeyError(key)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", default=0, action="count", help="verbosity level"
    )

    input = parser.add_argument_group("input", "input related options")
    input.add_argument("input", type=File, help="input file to compare to")

    input.add_argument(
        "-s",
        "--source-type",
        "--source",
        choices=("tsv", "hdf5", "root", "npz"),
        default="npz",
        help="Data source type",
    )

    input.add_argument(
        "--object",
        default="eventscount.fine.total",
        help="output(s) to read from the model",
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
    crosscheck.add_argument(
        "--pars", nargs="+", help="patterns to search in parameter names"
    )

    c = NuisanceComparator(parser.parse_args())
