#!/usr/bin/env python

from argparse import Namespace
from contextlib import suppress
from itertools import permutations
from typing import Any, Generator, Literal, Mapping

from h5py import File
from matplotlib import pyplot as plt
from numpy import allclose, fabs, nanmax
from numpy.typing import NDArray

from dag_modelling.logger import INFO1, INFO2, INFO3, logger, set_level
from models.dayabay_v0 import model_dayabay_v0
from nested_mapping import NestedMapping

set_level(INFO1)


def minus_one(parname: tuple[str,...], *values: float) -> tuple[float,...]:
    match parname:
        case ("spectrum_correction", "corr_unc"):
            return values

    return tuple(v - 1.0 for v in values)

def hmuncmapping(key: tuple[str,...]) -> tuple[str,...]:
    match key:
        case ("corr_unc",):
            return ("corr",)
        case ("uncorr_unc", isotope, energy):
            idx = int(energy.rsplit("_", 1)[-1])
            return ("uncorr", isotope, f"unc_scale_{idx:03d}")

    raise RuntimeError(f"Do not know how to map {key}")

def default_setter(parname: tuple[str,...], par, value: float, normvalue: float) -> float:
    par.value = value
    return value

def uncorr_normalized_setter(key: tuple[str,...], par, value: float, normvalue: float) -> float:
    match key:
        case ("spectrum_correction", "uncorr_unc", *_):
            par.normvalue = normvalue

            return normvalue

    par.value = value
    return value

comparison = {
    "default": {"rtol": 1.0e-8},
    "OffdiagScale": {"location": "all.detector.iav_offdiag_scale_factor", "rtol": 1.0e-8},
    "acc_norm": {"location": "all.bkg.rate.acc", "rtol": 1.0e-8, "scale": True},
    "bkg_rate_alphan": {"location": "all.bkg.rate.alphan", "rtol": 1.0e-8},
    "bkg_rate_amc": {"location": "all.bkg.rate.amc", "rtol": 1.0e-8},
    "bkg_rate_fastn": {"location": "all.bkg.rate.fastn", "rtol": 1.0e-8},
    "bkg_rate_lihe": {"location": "all.bkg.rate.lihe", "rtol": 1.0e-8},
    "effunc_uncorr": {"location": "all.detector.detector_relative", "keys_mapping": lambda t: (t+("efficiency_factor",)), "rtol": 1.0e-8},
    "escale": {"location": "all.detector.detector_relative", "keys_mapping": lambda t: (t+("energy_scale_factor",)), "rtol": 1.0e-8},
    "eres": {"location": "all.detector.eres", "keys_mapping": {("a",): ("a_nonuniform",), ("b",): ("b_stat",), ("c",): ("c_noise",), }, "rtol": 1.0e-8},
    "global_norm": {"location": "all.detector.global_normalization", "rtol": 1.0e-8},
    "lsnl_weight": {"location": "all.detector.lsnl_scale_a", "rtol": 1.0e-8},
    "DeltaMSq12": {"location": "all.oscprob.DeltaMSq21", "rtol": 1.0e-8},
    "DeltaMSq23": {"location": "all.oscprob.DeltaMSq32", "rtol": 1.0e-8},
    "SinSqDouble12": {"location": "all.oscprob.SinSq2Theta12", "rtol": 1.0e-8},
    "SinSqDouble13": {"location": "all.oscprob.SinSq2Theta13", "rtol": 1.0e-8},
    "spectral_weights": {"location": "all.neutrino_per_fission_factor", "keys_mapping": lambda s: (s[0].replace("anue_weight", "spec_scale"),), "rtol": 1.0e-8},
    # Reactor
    "nominal_thermal_power": {"location": "all.reactor.nominal_thermal_power", "rtol": 1.0e-8},
    "fission_fractions_corr": {"location": "all.reactor.fission_fraction_scale", "rtol": 1.0e-8},
    "eper_fission": {"location": "all.reactor.energy_per_fission", "rtol": 1.0e-8},
    "offeq_scale": {"location": "all.reactor.offequilibrium_scale", "rtol": 1.0e-8},
    "snf_scale": {"location": "all.reactor.snf_scale", "rtol": 1.0e-8},
    "spectrum_correction": {
        "skip": False,
        "location": "all.reactor_anue.spectrum_uncertainty",
        "keys_mapping": hmuncmapping,
        "atol": 0.3,
        "rtol": 1.e-4,
        "value_fcn": minus_one,
        "value_setter": uncorr_normalized_setter
        }
}


class NuisanceComparator:
    __slots__ = (
        "model",
        "parameters_dgm",
        "opts",
        "outputs_dgm",
        "outputs_dgm_default",
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
        "skey_dgm",
        "skey2_gna",
        "skey2_dgm",
        "skey_par_gna",
        "skey_par_dgm",
        "skey2_par_gna",
        "skey2_par_dgm",
        "data_gna",
        "data_dgm",
        "diff",
        "n_success",
        "n_fail",
        "n_pars",
        "n_values",
        "n_hists",
    )
    opts: Namespace
    outputs_dgm: NestedMapping
    outputs_dgm_default: NestedMapping
    outputs_gna_default: NestedMapping

    cmpopts: dict[str, Any]

    value_central: float
    value_current: float
    value_left: float
    value_right: float

    maxdiff: float
    maxreldiff: float

    index: tuple[str, ...]

    skey_gna: str
    skey_dgm: str
    skey2_gna: str
    skey2_dgm: str

    skey_par_gna: str
    skey_par_dgm: str
    skey2_par_gna: str
    skey2_par_dgm: str

    data_gna: NDArray
    data_dgm: NDArray
    diff: NDArray | Literal[False]

    n_success: int
    n_fail: int

    n_pars: int
    n_values: int
    n_hists: int

    def __init__(self, opts: Namespace):
        self.cmpopts = {}

        self.maxdiff = 0.0
        self.maxreldiff = 0.0

        self.index = ()

        self.skey_gna = ""
        self.skey_dgm = ""
        self.skey2_gna = ""
        self.skey2_dgm = ""

        self.skey_par_gna = ""
        self.skey_par_dgm = ""
        self.skey2_par_gna = ""
        self.skey2_par_dgm = ""

        self.n_success = 0
        self.n_fail = 0
        self.n_pars = 0
        self.n_values = 0
        self.n_hists = 0

        self.value_central = "default"
        self.value_current = "default"
        self.value_left = "default"
        self.value_right = "default"

        self.outputs_dgm_default = NestedMapping(sep=".")
        self.outputs_gna_default = NestedMapping(sep=".")

        self.model = model_dayabay_v0(source_type=opts.source_type)
        self.opts = opts

        if opts.verbose:
            opts.verbose = min(opts.verbose, 3)
            set_level(globals()[f"INFO{opts.verbose}"])

        self.skey_gna = "fine"
        self.skey_dgm = opts.object
        self.outputs_dgm = self.model.storage(f"outputs.{self.skey_dgm}")
        self.parameters_dgm = self.model.storage("parameters")

        with suppress(StopIteration):
            self.process()

        logger.info(f"Processed {self.n_pars} parameters ({self.n_values} values) and {self.n_hists} histograms: {self.n_success} sucess / {self.n_fail} fail")

    def _skip_par(self, parname: str) -> bool:
        parname = parname.lower()
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

            value_fcn = self.cmpopts.get("value_fcn", lambda s, *v: v)
            self.value_central, self.value_left, self.value_right = value_fcn(
                paritems,
                *results["values"]
            )
            _, normvalue_left, normvalue_right = results["normvalues"]

            parsloc = self.parameters_dgm.get_any(self.cmpopts["location"])
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

            setter = self.cmpopts.get("value_setter", default_setter)

            if par.value != self.value_central:
                logger.error(
                    f"Parameters not consistent: dgm={par.value} gna={self.value_central}"
                )
                continue

            results_left = results["minus"]
            results_right = results["plus"]

            self.skey_par_gna = parname
            self.skey2_par_gna = ".".join(("",) + index)
            self.skey_par_dgm = self.cmpopts["location"]
            self.skey2_par_dgm = ""

            par.push()

            self.value_current = setter(paritems, par, self.value_right, normvalue_right)
            logger.log(INFO1, f"{parname}: v={self.valuestring}")
            self.process_par_offset(results_right)

            self.value_current = setter(paritems, par, self.value_left, normvalue_left)
            logger.log(INFO1, f"{parname}: v={self.valuestring}")
            self.process_par_offset(results_left)

            par.pop()
            self.check_default("restore", check_change=False)

            self.n_pars += 1
            self.n_values += 3

    def check_default(
        self, label="default", *, save: bool = False, check_change: bool = True
    ):
        default = self.opts.input["default"]
        self.skey_par_gna = "default"
        self.skey_par_dgm = label
        self.cmpopts = comparison["default"]
        if self.compare_hists(default, save=save, check_change=check_change):
            logger.log(INFO2, f"OK: default {self.cmpstring_par}")
        else:
            logger.error(f"FAIL: default {self.cmpstring_par}")

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
        change2_dgm = 0.0
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
                dgm = self.outputs_dgm[ad, period]

                self.data_dgm = dgm.data
                self.data_gna = data[:]

                if save:
                    self.outputs_dgm_default[ad, period] = self.data_dgm.copy()
                    self.outputs_gna_default[ad, period] = self.data_gna.copy()
                else:
                    change2_dgm += (
                        (self.data_dgm - self.outputs_dgm_default[ad, period]) ** 2
                    ).sum()
                    change2_gna += (
                        (self.data_gna - self.outputs_gna_default[ad, period]) ** 2
                    ).sum()

                is_ok &= self.compare_data()

        if check_change:
            if change2_dgm == 0.0 or change2_gna == 0.0:
                logger.error(
                    f"FAIL: data unchanged dgm²={change2_dgm} gna²={change2_gna}"
                )
                return False
        return is_ok

    def compare_data(self) -> bool:
        is_ok = self.data_consistent(self.data_gna, self.data_dgm)
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
            ss = self.opts.plot_filter
            if not ss or ss in self.cmpstring:
                self.plot_1d()

        if self.opts.embed_on_failure:
            try:
                self.diff = self.data_dgm - self.data_gna
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

        subplots_opts = dict(top=0.85, left=0.15)

        plt.figure()
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="",
            title=f"""{self.cmpstring_par}:\n{self.cmpstring}\n{self.valuestring}""",
        )
        plt.subplots_adjust(**subplots_opts)
        ax.plot(self.data_gna, style, label="GNA", **pargs)
        ax.plot(self.data_dgm, style, label="dgm", **pargs)
        # scale_factor = self.data_gna.sum() / self.data_dgm.sum()
        # ax.plot(
        #     self.data_dgm * scale_factor,
        #     f"{style}-",
        #     label="dgm scaled",
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
        plt.subplots_adjust(**subplots_opts)
        with suppress(ValueError):
            ax.plot(self.data_dgm / self.data_gna - 1, style, label="dgm/GNA", **pargs)

        ax.legend()
        ax.grid()

        plt.figure()
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="diff",
            title=f"""{self.cmpstring_par}:\n{self.cmpstring}\n{self.valuestring}""",
        )
        plt.subplots_adjust(**subplots_opts)
        with suppress(ValueError):
            ax.plot(self.data_dgm - self.data_gna, style, label="dgm-GNA", **pargs)

        ax.legend()
        ax.grid()

        plt.figure()
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="Ratio-1",
            title=f"""{self.cmpstring_par}:\n{self.cmpstring}\n{self.valuestring}""",
        )
        plt.subplots_adjust(**subplots_opts)
        with suppress(ValueError):
            ax.plot(self.data_dgm / self.data_gna - 1, style, label="dgm/GNA", **pargs)

        dgm_reference = self.outputs_dgm_default[self.index]
        with suppress(ValueError):
            ax.plot(
                self.data_dgm / dgm_reference - 1, style, label="dgm/default", **pargs
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
    def key_dgm(self) -> str:
        return f"{self.skey_dgm}{self.skey2_dgm}"

    @property
    def key_gna(self) -> str:
        return f"{self.skey_gna}{self.skey2_gna}"

    @property
    def cmpstring(self) -> str:
        return f"dgm: {self.skey_dgm}{self.skey2_dgm} ↔ gna:{self.skey_gna}{self.skey2_gna}"

    @property
    def cmpstring_par(self) -> str:
        return f"dgm: {self.skey_par_dgm}{self.skey2_par_dgm} ↔ gna:{self.skey_par_gna}{self.skey2_par_gna}"

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
        return f"dgm: {self.data_dgm.shape}, gna: {self.data_gna.shape}"

    @property
    def atol(self) -> float:
        return float(self.cmpopts.get("atol", 0.0))

    @property
    def rtol(self) -> float:
        return float(self.cmpopts.get("rtol", 0.0))

    def data_consistent(self, gna: NDArray, dgm: NDArray) -> bool:
        self.n_hists += 1
        try:
            status = allclose(dgm, gna, rtol=self.rtol, atol=self.atol)
        except ValueError:
            self.maxdiff = -1
            self.maxreldiff = -1

            self.n_fail += 1
            return False

        fdiff = fabs(dgm - gna)
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


from nested_mapping.typing import KeyLike, properkey


def get_orderless(storage: NestedMapping | Any, key: KeyLike) -> Any:
    key = properkey(key)
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
        "--plot-filter", help="plot only comparisons with substring"
    )
    crosscheck.add_argument(
        "-x", "--exit-on-failure", action="store_true", help="exit on failure"
    )
    crosscheck.add_argument(
        "--pars", nargs="+", help="patterns to search in parameter names"
    )

    c = NuisanceComparator(parser.parse_args())
