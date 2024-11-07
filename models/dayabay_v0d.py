from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from contextlib import suppress
from itertools import product
from os.path import relpath
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

from numpy import ndarray
from numpy.random import Generator

from dagflow.bundles.file_reader import FileReader
from dagflow.bundles.load_array import load_array
from dagflow.bundles.load_graph import load_graph, load_graph_data
from dagflow.bundles.load_parameters import load_parameters
from dagflow.core import Graph, NodeStorage
from dagflow.tools.logger import logger
from dagflow.tools.schema import LoadYaml
from multikeydict.nestedmkdict import NestedMKDict

if TYPE_CHECKING:
    from dagflow.core.meta_node import MetaNode

FutureType = Literal[
    "all",  # enable all the options
    "reactor-28days",  # merge reactor data, each 4 weeks
    "reactor-35days",  # merge reactor data, each 4 weeks
]

# Define a dictionary of groups of nuisance parameters in a format `name: path`,
# where path denotes the location of the parameters in the storage.
_SYSTEMATIC_UNCERTAINTIES_GROUPS = {
    "oscprob": "oscprob",
    "eres": "detector.eres",
    "lsnl": "detector.lsnl_scale_a",
    "iav": "detector.iav_offdiag_scale_factor",
    "detector_relative": "detector.detector_relative",
    "energy_per_fission": "reactor.energy_per_fission",
    "nominal_thermal_power": "reactor.nominal_thermal_power",
    "snf": "reactor.snf_scale",
    "neq": "reactor.nonequilibrium_scale",
    "fission_fraction": "reactor.fission_fraction_scale",
    "bkg_rate": "bkg.rate",
    "hm_corr": "reactor_anue.spectrum_uncertainty.corr",
    "hm_uncorr": "reactor_anue.spectrum_uncertainty.uncorr",
}


class model_dayabay_v0d:
    """The Daya Bay analysis implementation version v0d.

    Purpose:
        - introduce alternative data inputs
        - provide configuration to switch between inputs and compare them

    Attributes
    ----------
    storage : NodeStorage
        nested dictionary with model elements: nodes, parameters, etc.

    graph : Graph
        graph instance

    index : dict[str, tuple[str, ...]]
        dictionary with all possible names for replicated items, e.g.
        "detector": ("AD11", "AD12", ...); reactor: ("DB1", ...); ...
        index is setup within the model

    combinations : dict[str, tuple[tuple[str, ...], ...]]
        lists of all combinations of values of 1 and more indices,
        e.g. detector, detector/period, reator/isotope, reactor/isotope/period, etc.

    spectrum_correction_mode : str, default="exponential"
        mode of how the parameters of the free spectrum model
        are treated:
            - "exponential": pᵢ=0 by default, S(Eᵢ) is
              multiplied by exp(pᵢ) the correction is always
              positive, but nonlinear
            - "linear": pᵢ=0 by default, S(Eᵢ) is multiplied by
              1+pᵢ the correction may be negative, but is always
              linear

    concatenation_mode : str, default="detector_period"
        choses the observation to be analyzed:
            - "detector_period" - concatenation of observations at
              each detector at each period
            - "detector" - concatenation of observations at each
              detector (combined for all period)

    monte_carlo_mode : str, default="asimov"
        the Monte-Carlo mode for pseudo-data:
            - "asimov" - Asimov, no fluctuations
            - "normal-stats" - normal fluctuations with statistical
              errors
            - "poisson" - Poisson fluctuations

    path_data : Path
        path to the data

    source_type : str, default="npz"
        type of the data to read ("tsv", "hdf5", "root" or "npz")

    Technical attributes
    --------------------
    _strict : bool, default=True
        strict mode. Stop execution if:
            - the model is not complete
            - any labels were not applied

    _close : bool, default=True
        if True the graph is closed and memory is allocated
        may be used to debug corrupt model

    _random_generator : Generator
        numpy random generator to be used for ToyMC

    _covariance_matrix : MetaNode
        covariance matrix, computed on this model

    _frozen_nodes : dict[str, tuple]
        storage with nodes, which are being fixed at their values and
        require manual intervention in order to be recalculated
    """

    __slots__ = (
        "storage",
        "graph",
        "index",
        "combinations",
        "path_data",
        "spectrum_correction_mode",
        "concatenation_mode",
        "monte_carlo_mode",
        "source_type",
        "_strict",
        "_close",
        "_future",
        "_covariance_matrix",
        "_frozen_nodes",
        "_random_generator",
    )

    storage: NodeStorage
    graph: Graph
    index: dict[str, tuple[str, ...]]
    combinations: dict[str, tuple[tuple[str, ...], ...]]
    path_data: Path
    spectrum_correction_mode: Literal["linear", "exponential"]
    concatenation_mode: Literal["detector", "detector_period"]
    monte_carlo_mode: Literal["asimov", "normal-stats", "poisson"]
    source_type: Literal["tsv", "hdf5", "root", "npz"]
    _strict: bool
    _close: bool
    _random_generator: Generator
    _future: Collection[FutureType]
    _covariance_matrix: MetaNode
    _frozen_nodes: dict[str, tuple]

    def __init__(
        self,
        *,
        source_type: Literal["tsv", "hdf5", "root", "npz"] = "npz",
        strict: bool = True,
        close: bool = True,
        override_indices: Mapping[str, Sequence[str]] = {},
        spectrum_correction_mode: Literal["linear", "exponential"] = "exponential",
        seed: int = 0,
        monte_carlo_mode: Literal["asimov", "normal-stats", "poisson"] = "asimov",
        concatenation_mode: Literal["detector", "detector_period"] = "detector_period",
        parameter_values: dict[str, float | str] = {},
        future: Collection[FutureType] = set(),
    ):
        """Model initialization.

        Parameters
        ----------
        seed: int
              random seed to be passed to random generator for ToyMC
        override_indices : dict[str, Sequence[str]]
                           dictionary with indices to override self.index.
                           may be used to reduce the number of detectors or reactors in the
                           model

        for the dscription of other parameters, see description of the class.
        """
        self._strict = strict
        self._close = close

        self.storage = NodeStorage()
        self.path_data = Path("data/dayabay-v0d")
        self.source_type = source_type
        self.spectrum_correction_mode = spectrum_correction_mode
        self.concatenation_mode = concatenation_mode
        self.monte_carlo_mode = monte_carlo_mode
        self._random_generator = self._create_random_generator(seed)

        self._future = set(future)
        future_variants = set(get_args(FutureType))
        assert all(f in future_variants for f in self._future)
        if "all" in self._future:
            self._future = future_variants
            self._future.remove("all")
            self._future.remove("reactor-35days")
        if self._future:
            logger.info(f"Future options: {', '.join(self._future)}")
        self._covariance_matrix = None
        self._frozen_nodes = {}

        self.combinations = {}

        override_indices = {k: tuple(v) for k, v in override_indices.items()}
        self.build(override_indices)

        if parameter_values:
            self.set_parameters(parameter_values)

    def build(self, override_indices: dict[str, tuple[str, ...]] = {}):
        """Actually build the model.

        Steps:
            - initialize indices to describe repeated components
            - read parameters
            - block by block initialize the nodes of the model and connect them
                - read the data whenever necessary

        Parameters
        ----------
        override_indices : dict[str, tuple[str, ...]]
                           dictionary with indices to override self.index.
                           may be used to reduce the number of detectors or reactors in the
                           model
        """
        #
        # Import necessary nodes and loaders
        #
        from numpy import arange, concatenate, linspace, ones

        from dagflow.bundles.load_hist import load_hist
        from dagflow.bundles.load_record import load_record_data
        from dagflow.bundles.make_y_parameters_for_x import make_y_parameters_for_x
        from dagflow.lib.arithmetic import Division, Product, Sum
        from dagflow.lib.common import Array, Concatenation, ParArrayInput, Proxy, View
        from dagflow.lib.exponential import Exp
        from dagflow.lib.integration import Integrator
        from dagflow.lib.interpolation import Interpolator
        from dagflow.lib.linalg import Cholesky, VectorMatrixProduct
        from dagflow.lib.normalization import RenormalizeDiag
        from dagflow.lib.statistics import CovarianceMatrixGroup, LogProdDiag
        from dagflow.lib.summation import ArraySum, SumMatOrDiag
        from dagflow.tools.schema import LoadPy
        from dgf_detector import (
            AxisDistortionMatrix,
            EnergyResolution,
            Monotonize,
            Rebin,
        )
        from dgf_detector.bundles.refine_lsnl_data import refine_lsnl_data
        from dgf_reactoranueosc import (
            IBDXsecVBO1Group,
            InverseSquareLaw,
            NueSurvivalProbability,
        )
        from dgf_statistics import Chi2, CNPStat, MonteCarlo
        from models.bundles.refine_detector_data import refine_detector_data
        from models.bundles.refine_reactor_data import refine_reactor_data
        from models.bundles.sync_reactor_detector_data import sync_reactor_detector_data
        from multikeydict.tools import remap_items

        # Initialize the storage and paths
        storage = self.storage
        path_data = self.path_data

        path_parameters = path_data / "parameters"
        path_arrays = path_data / self.source_type
        # TODO: use source_type for everything
        path_root = path_data / "root"

        # Read Eν edges for the parametrization of free antineutrino spectrum model
        # Loads the python file and returns variable "edges", which should be defined
        # in the file and has type `ndarray`.
        #
        antineutrino_model_edges = LoadPy(
            path_parameters / "reactor_antineutrino_spectrum_edges.py",
            variable="edges",
            type=ndarray,
        )

        # Provide some convenience substitutions for labels
        index_names = {
            "U235": "²³⁵U",
            "U238": "²³⁸U",
            "Pu239": "²³⁹Pu",
            "Pu241": "²⁴¹Pu",
        }

        #
        # Provide indices, names and lists of values in order to work with repeated
        # items
        #
        index = self.index = {
            # Data acquisition period
            "period": ("6AD", "8AD", "7AD"),
            # Detector names
            "detector": (
                "AD11",
                "AD12",
                "AD21",
                "AD22",
                "AD31",
                "AD32",
                "AD33",
                "AD34",
            ),
            # Source of background events:
            #     - acc: accidental coincidences
            #     - lihe: ⁹Li and ⁸He related events
            #     - fastn: fast neutrons and muon-x background
            #     - amc: ²⁴¹Am¹³C calibration source related background
            #     - alphan: ¹³C(α,n)¹⁶O background
            "bkg": ("acc", "lihe", "fastn", "amc", "alphan"),
            # Experimental sites
            "site": ("EH1", "EH2", "EH3"),
            # Fissile isotopes
            "isotope": ("U235", "U238", "Pu239", "Pu241"),
            # Fissile isotopes, which spectrum requires Non-Equilibrium correction to be
            # applied
            "isotope_neq": ("U235", "Pu239", "Pu241"),
            # Nuclear reactors
            "reactor": ("DB1", "DB2", "LA1", "LA2", "LA3", "LA4"),
            # Sources of antineutrinos:
            #     - "nu_main": for antineutrinos from reactor cores with no
            #                  Non-Equilibrium correction applied
            #     - "nu_neq": antineutrinos from Non-Equilibrium correction
            #     - "nu_snf": antineutrinos from Spent Nuclear Fuel
            "anue_source": ("nu_main", "nu_neq", "nu_snf"),
            # Model related antineutrino spectrum correction type:
            #     - uncorrelated
            #     - correlated
            "anue_unc": ("uncorr", "corr"),
            # Part of the Liquid scintillator non-linearity (LSNL) parametrization
            "lsnl": ("nominal", "pull0", "pull1", "pull2", "pull3"),
            # Nuisance related part of the Liquid scintillator non-linearity (LSNL)
            # parametrization
            "lsnl_nuisance": ("pull0", "pull1", "pull2", "pull3"),
            # Free antineutrino spectrum parameter names: one parameter for each edge
            # from `antineutrino_model_edges`
            "spec": tuple(
                f"spec_scale_{i:02d}" for i in range(len(antineutrino_model_edges))
            ),
        }
        # Define isotope names in lower case
        index["isotope_lower"] = tuple(isotope.lower() for isotope in index["isotope"])

        # Optionally override (reduce) indices
        index.update(override_indices)

        # Check there are now overlaps
        index_all = (
            index["isotope"] + index["detector"] + index["reactor"] + index["period"]
        )
        set_all = set(index_all)
        if len(index_all) != len(set_all):
            raise RuntimeError("Repeated indices")

        # Collection combinations between 2 and more indices. Ensure some combinations,
        # e.g. detectors not present at certain periods, are excluded.
        # For example, combinations["reactor.detector"] contains:
        # (("DB1", "AD11"), ("DB1", "AD12"), ..., ("DB2", "AD11"), ...)
        #
        # The dictionary combinations is one of the main elements to loop over and match
        # parts of the computational graph
        inactive_detectors = ({"6AD", "AD22"}, {"6AD", "AD34"}, {"7AD", "AD11"})
        required_combinations = tuple(index.keys()) + (
            "reactor.detector",
            "reactor.isotope",
            "reactor.isotope_neq",
            "reactor.period",
            "reactor.isotope.period",
            "reactor.isotope.detector",
            "reactor.isotope_neq.detector",
            "reactor.isotope.detector.period",
            "reactor.isotope_neq.detector.period",
            "reactor.detector.period",
            "detector.period",
            "anue_unc.isotope",
            "bkg.detector",
            "bkg.detector.period",
        )
        combinations = self.combinations
        for combname in required_combinations:
            combitems = combname.split(".")
            items = []
            for it in product(*(index[item] for item in combitems)):
                if any(inact.issubset(it) for inact in inactive_detectors):
                    continue
                items.append(it)
            combinations[combname] = tuple(items)

        # Special treatment is needed for combinations of anue_source and isotope as
        # nu_neq is related to only a fraction of isotops, while nu_snf does not index
        # isotopes at all
        combinations["anue_source.reactor.isotope.detector"] = (
            tuple(
                ("nu_main",) + cmb for cmb in combinations["reactor.isotope.detector"]
            )
            + tuple(
                ("nu_neq",) + cmb
                for cmb in combinations["reactor.isotope_neq.detector"]
            )
            + tuple(("nu_snf",) + cmb for cmb in combinations["reactor.detector"])
        )

        # Start building the computational graph within a dedicated context, which
        # includes:
        # - graph - the graph instance.
        #     + All the nodes are added to the graph while graph is open.
        #     + On the exit from the context the graph closes itself, which triggers
        #       allocation of memory for the calculations.
        # - storage - nested dictionary, which is used to store all the created
        #   elements: nodes, outputs, parameters, data items, etc.
        # - filereader - manages reading the files
        #     + ensures, that the input files are opened only once
        #     + closes the files upon the exit of the context
        self.graph = Graph(close_on_exit=self._close, strict=self._strict)

        with self.graph, storage, FileReader:
            # Load all the parameters, necessary for the model. The parameters are
            # divided into three lists:
            # - constant - parameters are not expected to be modified during the
            #   analysis and thus are not passed to the minimizer.
            # - free - parameters that should be minimized and have no constraints
            # - constrained - parameters that should be minimized and have constraints.
            #   The constraints are defined by:
            #   + central values and uncertainties
            #   + central vectors and covariance matrices
            #
            # additionally the following lists are provided
            # - all - all the parameters, including fixed, free and constrained
            # - variable - free and constrained parameters
            # - normalized - a shadow definition of the constrained parameters. Each
            #   normalized parameter has value=0 when the constrained parameter is at
            #   its central value, +1, when it is offset by 1σ. The correlations,
            #   defined by the covariance matrices are properly treated. The conversion
            #   works the both ways: when normalized parameter is modified, the related
            #   constrained parameters are changed as well and vice versa. The
            #   parameters from this list are used to build the nuisance part of the χ²
            #   function.
            #
            # All the parameters are collected in the storage - a nested dictionary,
            # which can handle path-like keys, with "folders" split by periods:
            # - storage["parameters.all"] - storage with all the parameters
            # - storage["parameters", "all"] - the same storage with all the parameters
            # - storage["parameters.all.oscprob.SinSq2Theta12"] - neutrino oscillation
            #   parameter sin²2θ₁₂
            # - storage["parameters.constrained.oscprob.SinSq2Theta12"] - same neutrino
            #   oscillation parameter sin²2θ₁₂ in the list of constrained parameters.
            # - storage["parameters.normalized.oscprob.SinSq2Theta12"] - shadow
            #   (nuisance) parameter for sin²2θ₁₂.
            #
            # The constrained parameter has fields `value`, `normvalue`, `central`, and
            # `sigma`, which could be read to get the current value of the parameter,
            # normalized value, central value, and uncertainty. The assignment to the
            # fields changes the values. Additionally fields `sigma_relative` and
            # `sigma_percent` may be used to get and set the relative uncertainty.
            # ```python
            # p = storage["parameters.all.oscprob.SinSq2Theta12"]
            # print(p)        # print the description
            # print(p.value)  # print the current value
            # p.value = 0.8   # set the value to 0.8 - affects the model
            # p.central = 0.7 # set the central value to 0.7 - affects the nuisance term
            # p.normvalue = 1 # set the value to centra+1sigma
            # ```
            #
            # The non-constrained parameter lacks `central`, `sigma`, `normvalue`, etc
            # fields and is controlled only by `value`. The normalized parameter does
            # have `central` and `sigma` fields, but they are read only. The effect of
            # changing `value` field of the normalized parameter is the same as changing
            # `normvalue` field of its corresponding parameter.
            #
            # ```python
            # np = storage["parameters.normalized.oscprob.SinSq2Theta12"]
            # print(np)        # print the description
            # print(np.value)  # print the current value -> 0
            # np.value = 1     # set the value to centra+1sigma
            # np.normvalue = 1 # set the value to centra+1sigma
            # # p is also affected
            # ```
            #
            # Load oscillation parameters from 3 configuration files:
            # - Free sin²2θ₁₃ and Δm²₃₂
            # - Constrained sin²2θ₁₃ and Δm²₃₂
            # - Fixed: Neutrino Mass Ordering
            load_parameters(path="oscprob", load=path_parameters / "oscprob.yaml")
            load_parameters(
                path="oscprob",
                load=path_parameters / "oscprob_solar.yaml",
                joint_nuisance=True,
            )
            load_parameters(
                path="oscprob", load=path_parameters / "oscprob_constants.yaml"
            )
            # The parameters are located in "parameters.oscprob" folder as defined by
            # the `path` argument.
            # The annotated table with values may be then printed for any storage as
            # ```python
            # print(storage["parameters.all.oscprob"].to_table())
            # print(storage.get_dict("parameters.all.oscprob").to_table())
            # ```
            # the second line does the same, but ensures that the object, obtained from
            # a storage is another nested dictionary, not a parameter.
            #
            # The `joint_nuisance` options instructs the loader to provide a combined
            # nuisance term for the both the parameters, rather then two of them. The
            # nuisance terms for created constrained parameters are located in
            # "outputs.statistic.nuisance.parts" and may be printed with:
            # ```python
            # print(storage["outputs.statistic.nuisance"].to_table())
            # ```
            # The outputs are typically read-only. They are affected when the parameters
            # are modified and the relevant values are calculated upon request. In this
            # case, when the table is printed.

            # Load fixed parameters for Inverse Beta Decay (IBD) cross section:
            # - particle masses and lifetimes
            # - constants for Vogel-Beacom IBD cross section
            load_parameters(path="ibd", load=path_parameters / "pdg2024.yaml")
            load_parameters(path="ibd.csc", load=path_parameters / "ibd_constants.yaml")

            # Load the conversion constants from metric to natural units:
            # - reactor thermal power
            # - the argument of oscillation proabability
            # `scipy.constants` are used to provide the numbers.
            # There are no constants, except maybe 1, 1/3 and π, defined within the
            # code. All the numbers are read based on the configuration files.
            load_parameters(
                path="conversion", load=path_parameters / "conversion_thermal_power.py"
            )
            load_parameters(
                path="conversion",
                load=path_parameters / "conversion_oscprob_argument.py",
            )

            # Load reactor-detector baselines
            load_parameters(load=path_parameters / "baselines.yaml")

            # IBD and detector normalization parameters:
            # - free global IBD normalization factor
            # - fixed detector efficiency (variation is managed by uncorrelated
            #   "detector_relative.efficiency_factor")
            # - fixed correction to the number of protons in each detector
            load_parameters(
                path="detector", load=path_parameters / "detector_normalization.yaml"
            )
            load_parameters(
                path="detector", load=path_parameters / "detector_efficiency.yaml"
            )
            load_parameters(
                path="detector",
                load=path_parameters / "detector_nprotons_correction.yaml",
            )

            # Detector energy scale parameters:
            # - constrained correlated between detectors energy resolution parameters
            # - constrained correlated between detectors Liquid Scnitillator
            #   Non-Linearity (LSNL) parameters
            # - constrained uncorrelated between detectors energy distortion related to
            #   Inner Acrylic Vessel
            load_parameters(
                path="detector", load=path_parameters / "detector_eres.yaml"
            )
            load_parameters(
                path="detector",
                load=path_parameters / "detector_lsnl.yaml",
                replicate=index["lsnl_nuisance"],
            )
            load_parameters(
                path="detector",
                load=path_parameters / "detector_iav_offdiag_scale.yaml",
                replicate=index["detector"],
            )
            # Here we use `replicate` argument and pass a list of values. The parameters
            # are replicated for each index value. So 4 parameters for LSNL are created
            # and 8 parameters of IAV are created. The index values are used to
            # construct the path to parameter. See:
            # ```python
            # print(storage["outputs.statistic.nuisance.parts"].to_table())
            # ```
            # which contains parameters "AD11", "AD12", etc.

            # Relative uncorrelated between detectors parameters:
            # - relative efficiency factor (constrained)
            # - relative energy scale factor (constrained)
            # the parameters of each detector are correlated between each other.
            load_parameters(
                path="detector",
                load=path_parameters / "detector_relative.yaml",
                replicate=index["detector"],
                keys_order=(
                    ("pargroup", "par", "detector"),
                    ("pargroup", "detector", "par"),
                ),
            )
            # By default extra index is appended at the end of the key (path). A
            # `keys_order` argument is used to change the order of the keys from
            # group.par.detector to group.detector.par so it is easier to access both
            # the parameters of a single detector.

            # Load reactor related parameters:
            # - constrained nominal thermal power
            # - constrained mean energy release per fission
            # - constrained Non-EQuilibrium (NEQ) correction scale
            # - cosntrained Spent Nuclear Fuel (SNF) scale
            # - fixed values of the fission fractions for the SNF calculation
            load_parameters(
                path="reactor",
                load=path_parameters / "reactor_thermal_power_nominal.yaml",
                replicate=index["reactor"],
            )
            load_parameters(
                path="reactor", load=path_parameters / "reactor_energy_per_fission.yaml"
            )
            load_parameters(
                path="reactor",
                load=path_parameters / "reactor_snf.yaml",
                replicate=index["reactor"],
            )
            load_parameters(
                path="reactor",
                load=path_parameters / "reactor_nonequilibrium_correction.yaml",
                replicate=combinations["reactor.isotope_neq"],
            )
            load_parameters(
                path="reactor",
                load=path_parameters / "reactor_snf_fission_fractions.yaml",
            )
            # The nominal thermal power is replicated for each reactor, making its
            # uncertainty uncorrelated. Energy per fission (and fission fraction) has
            # distinct value (and uncertainties) for each isotope, therefore the
            # configuration files have an entry for each index and `replicate` argument
            # is not required. SNF and NEQ corrections are made uncorrelated between the
            # reactors. As only fraction of isotopes are affected by NEQ a dedicated
            # index `isotope_neq` is used for it.

            # Read the constrained and correlated fission fractions. The fission
            # fractions are partially correlated within each reactor. Therefore the
            # configuration file provides the uncertainties and correlations for
            # isotopes. The parameters are then replicated for each reactor and the
            # index is modified to have `isotope` as the innermost part.
            load_parameters(
                path="reactor",
                load=path_parameters / "reactor_fission_fraction_scale.yaml",
                replicate=index["reactor"],
                keys_order=(
                    ("par", "isotope", "reactor"),
                    ("par", "reactor", "isotope"),
                ),
            )

            # Finally the constrained background rates are loaded. They include the
            # rates and uncertainties for 5 sources of background events for 6-8
            # detectors during 3 periods of data taking.
            load_parameters(path="bkg.rate", load=path_parameters / "bkg_rate_acc.yaml")
            load_parameters(path="bkg.rate", load=path_parameters / "bkg_rates.yaml")

            # Additionally a few constants are provided.
            # A constant to convert seconds to days for the backgrounds estimation
            load_parameters(
                format="value",
                state="fixed",
                parameters={
                    "conversion": {
                        "seconds_in_day_inverse": 1 / (60 * 60 * 24),
                    }
                },
                labels={
                    "conversion": {
                        "seconds_in_day_inverse": "Fraction of a day in a second",
                    }
                },
            )

            # 1/3 and 2/3 needed to construct Combined Neyman-Pearson χ²
            load_parameters(
                format="value",
                state="fixed",
                parameters={
                    "stats": {
                        "pearson": 2 / 3,
                        "neyman": 1 / 3,
                    }
                },
                labels={
                    "stats": {
                        "pearson": "Coefficient for Pearson's part of CNP χ²",
                        "neyman": "Coefficient for Neyman's part of CNP χ²",
                    }
                },
            )

            # Provide a few variable for handy read/write access of the model objects,
            # including:
            # - `nodes` - nested dictionary with nodes. Node is an instantiated function
            #   and is a main building block of the model. Nodes have inputs (function
            #   arguments) and outputs (return values). The model is built by connecting
            #   the outputs of the nodes to inputs of the following nodes.
            # - `inputs` - storage for not yet connected inputs. The inputs are removed
            #   from the storage after connection and the storage is expected to be
            #   empty by the end of the model construction
            # - `outputs` - the return values of the functions used in the model. A
            #   single output contains a single numpy array. **All** the final and
            #   intermediate data may be accessed via outputs. Note: the function
            #   evaluation is triggered by reading the output.
            # - `data` - storage with raw (input) data arrays. It is used as an
            #   intermediate storage, populated with `load_graph_data` and
            #   `load_record_data` methods.
            # - `parameters` - already populated storage with parameters.
            nodes = storage.child("nodes")
            inputs = storage.child("inputs")
            outputs = storage.child("outputs")
            data = storage.child("data")
            parameters = storage("parameters")
            parameters_nuisance_normalized = storage("parameters.normalized")

            # In this section the actual parts of the calculation are created as nodes.
            # First of all the binning is defined for the histograms.
            # - internal binning for the integration: 240 bins of 50 keV from 0 to 241.
            # - final binning for the statistical analysis: 20 keV from 1.2 MeV to 2 MeV
            #   with two wide bins below from 0.7 MeV and above up to 12 MeV.
            # - cosθ (positron angle) edges [-1,1] are defined explicitly for the
            #   integration of the Inverse Beta Decay (IBD) cross section.
            in_edges_fine = linspace(0, 12, 241)
            in_edges_final = concatenate(([0.7], arange(1.2, 8.01, 0.20), [12.0]))
            in_edges_costheta = [-1, 1]

            # Instantiate the storage nodes for bin edges. In what follows all the
            # nodes, outputs and inputs are automatically added to the relevant storage
            # locations. This is done via usage of the `Node.replicate()` class method.
            # The method is also responsible for creating indexed copies of the classes,
            # hence the name.
            edges_costheta, _ = Array.replicate(
                name="edges.costheta", array=in_edges_costheta
            )
            edges_energy_common, _ = Array.replicate(
                name="edges.energy_common", array=in_edges_fine
            )
            edges_energy_final, _ = Array.replicate(
                name="edges.energy_final", array=in_edges_final
            )
            # For example the final energy node is stored as "nodes.edges.energy_final".
            # The output may be accessed via the node itself, of via the storage of
            # outputs as "outputs.edges.energy_final".
            # ```python
            # node = storage["nodes.edges.energy_final"] # Obtain the node from the
            #                                            # storage
            # output = storage["outputs.edges.energy_final"] # Obtain the outputs from
            #                                                # the storage
            # output = node.outputs["array"] # Obtain the outputs from the node by name
            # output = node.outputs[0] # Obtain the outputs from the node by position
            # print(output.data) # Get the data
            # ```
            # The access to the `output.data` triggers calculation recursively and
            # returns numpy array afterwards. The returned array is read only so the
            # user has no way to overwrite internal data accidentally.

            # For the fine binning we provide a few views, each of which is associated
            # to a distinct energy in the energy conversion process:
            # - Enu - neutrino energy.
            # - Edep - deposited energy of a positron..
            # - Escint - energy, converted to the scintillation.
            # - Evis - visible energy: scintillation energy after non-linearity
            #   correction.
            # - Erec - reconstructed energy: after smearing.
            View.replicate(name="edges.energy_enu", output=edges_energy_common)
            edges_energy_edep, _ = View.replicate(
                name="edges.energy_edep", output=edges_energy_common
            )
            edges_energy_escint, _ = View.replicate(
                name="edges.energy_escint", output=edges_energy_common
            )
            edges_energy_evis, _ = View.replicate(
                name="edges.energy_evis", output=edges_energy_common
            )
            edges_energy_erec, _ = View.replicate(
                name="edges.energy_erec", output=edges_energy_common
            )
            # While all these nodes refer to the same array, they will have different
            # labels, which is needed for making proper plots.

            # Finally, create a node with segment edges for modelling the reactor
            # electron antineutrino spectra.
            Array.replicate(
                name="reactor_anue.spectrum_free_correction.spec_model_edges",
                array=antineutrino_model_edges,
            )

            # Initialize the integration nodes. The product of reactor electron
            # antineutrino spectrum, IBD cross section and electron antineutrino
            # survival probability is integrated in each bin by a two-fold integral over
            # deposited energy and positron angle. The precision of integration is
            # defined beforehand for each Edep bin and independently for each cosθ bin.
            # As soon as integration precision and bin edges are defined all the values
            # of Edep and cosθ we need to compute the target functions on are defined as
            # well.
            #
            # Initialize the orders of integration (Gauss-Legendre quadratures) for Edep
            # and cosθ. The `Array.from_value` method is used to initialize an array
            # from a single number. The definition of bin edges is used in order to
            # specify the shape. `store=True` is set so the created nodes are added to
            # the storage.
            # In partucular using order 5 for Edep and 3 for cosθ means 15=5×3 points
            # will be used to integrate each 2d bin.
            Array.from_value(
                "kinematics.integration.orders_edep",
                5,
                edges=edges_energy_edep,
                store=True,
            )
            Array.from_value(
                "kinematics.integration.orders_costheta",
                3,
                edges=edges_costheta,
                store=True,
            )

            # Instantiate integration nodes. The integration consist of a single
            # sampling node, which based on bin edges and integration orders provides
            # samples (meshes) of points to compute the integrable function on. In the
            # case of 2d integrtion each mesh is 2d array, similar to one, produced by
            # numpy.meshgred function. A dedicated integrator node, which does the
            # actual integration, is created for each integrable function. In the
            # Daya Bay case the integrator part is replicated: an instance created for
            # each combination of "anue_source.reactor.isotope.detector" indices. Note,
            # that NEQ part (anue_source) has no contribution from ²³⁸U and SNF part has
            # not isotope index at all. In particular 384 integration nodes are created.
            Integrator.replicate(
                "gl2d",
                path="kinematics",
                names={
                    "sampler": "sampler",
                    "integrator": "integral",
                    "mesh_x": "sampler.mesh_edep",
                    "mesh_y": "sampler.mesh_costheta",
                    "orders_x": "sampler.orders_edep",
                    "orders_y": "sampler.orders_costheta",
                },
                replicate_outputs=combinations["anue_source.reactor.isotope.detector"],
            )
            # Pass the integration orders to the sampler inputs. The operator `>>` is
            # used to make a connection `input >> output` or batch connection
            # `input(s) >> outputs`. The operator connects all the objects on the left
            # side to all the corresponding objects on the right side. Missing pairs
            # will cause an exception.
            outputs.get_value("kinematics.integration.orders_edep") >> inputs.get_value(
                "kinematics.sampler.orders_edep"
            )
            outputs.get_value(
                "kinematics.integration.orders_costheta"
            ) >> inputs.get_value("kinematics.sampler.orders_costheta")
            # Regular way of accessing dictionaries via `[]` operator may be used. Here
            # we use `storage.get_value(key)` function to ensure the value is an object,
            # but not a nested dictionary. Similarly `storage.get_dict(key)` may be used
            # to ensure the value is a nested dictionary.
            #
            # There are a few ways to find and access the created inputs and outputs.
            # 1. get a node as `Node.replicate()` return value.
            #   - access node's inputs and outputs.
            # 2. get a node from node storage.
            #   - access node's inputs and outputs.
            # 3. access inputs and outputs via the storage.
            #
            # If replicate creates a single (main) node, as Integrator does, it is
            # returned as a first return value. Then print may be used to print
            # available inputs and outputs.
            # ```python
            # integrator, integrator_storage = Integrator.replicate(...)
            # orders_x >> integrator.inputs["orders_x"] # connect orders X to sampler's
            #                                           # input
            # integrator.outputs["x"] >> function_input # connect mesh X to function's
            #                                           # input
            # ```
            # Alternatively, the inputs may be accessed from the storage, as it is done
            # above. The list of created inputs and outputs may be found by passing
            # `verbose=True` flat to the replicate function as
            # `Node.replicate(verbose=True)`. The second return value of the function is
            # always a created storage with all the inputs and outputs, which can be
            # printed to the terminal:
            # ```python
            # integrator, integrator_storage = Integrator.replicate(...)
            # integrator_storage.print() # print local storage
            # storage.print() # print global storage
            # integrator_storage["inputs"].print() # print inputs from a local storage
            # ```
            # The local storage is always merged to the common (context) storage. It is
            # ensured that there is no overlap in the keys.

            # As of now we know all the Edep and cosθ points to compute the target
            # functions on. The next step is to initialize the functions themselves.
            # Here we create an instance of Inverse Beta Decay cross section, which also
            # includes conversion from deposited energy Edep to neutrino energy Enu and
            # corresponding dEnu/dEdep jacobian. The IBD nodes may operate with either
            # positron energy Ee as input, or deposited energy Edep=Ee+m(e) as input,
            # which is specified via an argument.
            ibd, _ = IBDXsecVBO1Group.replicate(
                path="kinematics.ibd", input_energy="edep"
            )
            # IBD cross section depends on a set of parameters, including neutron
            # liftime, proton and neutron masses, vector coupling constant, etc. The
            # values of these parameters were previously loaded and are located in the
            # 'parameters.constant.ibd' namespace. The IBD node(s) have an input for
            # each parameter. In order to connect the parameters the `<<` operator is
            # used as `node << parameters_storage`. It will loop over all the inputs of
            # the node and find parameters of the same name in the right hand side
            # namespace. Missing parameters are skipped, extra parameters are ignored.
            ibd << storage("parameters.constant.ibd")
            ibd << storage("parameters.constant.ibd.csc")
            # Connect the integration meshes for Edep and cosθ to the inputs of the
            # IBD node.
            outputs.get_value("kinematics.sampler.mesh_edep") >> ibd.inputs["edep"]
            (
                outputs.get_value("kinematics.sampler.mesh_costheta")
                >> ibd.inputs["costheta"]
            )
            # There is an output, which yields neutrino energy Enu (mesh), corresponding
            # to the Edep, cosθ meshes. As it will be used quite often, let us save it
            # to a variable.
            kinematic_integrator_enu = ibd.outputs["enu"]

            # Initialize survival probability for reactor electron antineutrinos. As it
            # is affected by the distance, we replicate it for each combination of
            # "reactor.detector" indices of count of 48. It is defined for energies in
            # MeV, while the unit for distance may be choosen between "m" and "km".
            NueSurvivalProbability.replicate(
                name="oscprob",
                distance_unit="m",
                replicate_outputs=combinations["reactor.detector"],
                surprobArgConversion=True,
            )
            # If created in the verbose mode one can see, that the following items are
            # created:
            # - nodes.oscprob.DB1.AD11
            # - nodes.oscprob.DB1.AD12
            # - ...
            # - inputs.oscprob.enu.DB1.AD11
            # - inputs.oscprob.enu.DB1.AD12
            # - ...
            # - inputs.oscprob.L.DB1.AD11
            # - inputs.oscprob.L.DB1.AD12
            # - ...
            # - inputs.oscprob.surprobArgConversion.DB1.AD11
            # - inputs.oscprob.surprobArgConversion.DB1.AD12
            # - ...
            # - outputs.oscprob.DB1.AD11
            # - outputs.oscprob.DB1.AD12
            # - ...
            # On one hand each node with its inputs and outputs may be accessed via
            # "nodes.oscprob.<reactor>.<detector>" address. On the other hand all the
            # inputs, corresponding to the baselines and input energyies may be accessed
            # via "inputs.oscprob.L" and "inputs.oscprob.enu" respectively. It is then
            # under user control whether he wants to provide similar or different data
            # for them.
            # Connect the same mesh of neutrino energy to all the 48 inputs:
            kinematic_integrator_enu >> inputs.get_dict("oscprob.enu")
            # Connect the corresponding baselines:
            parameters.get_dict("constant.baseline") >> inputs.get_dict("oscprob.L")
            # The matching is done based on the index with order being ignored. Thus
            # baselines stored as "DB1.AD11" or "AD11.DB1" both may be connected to the
            # input "DB1.AD11". Moreover, if the left part has fewer indices, the
            # connection will be broadcasted, e.g. "DB1" on the left will be connected
            # to all the indices on the right, containing "DB1".
            #
            # Provide a conversion constant to convert the argument of sin²(...Δm²L/E)
            # from chosen units to natural ones.
            parameters.get_value("all.conversion.oscprobArgConversion") >> inputs(
                "oscprob.surprobArgConversion"
            )
            # Also connect free, constrained and constant oscillation parameters to each
            # instance of the oscillation probability.
            nodes.get_dict("oscprob") << parameters("free.oscprob")
            nodes.get_dict("oscprob") << parameters("constrained.oscprob")
            nodes.get_dict("oscprob") << parameters("constant.oscprob")

            # fmt: off


            #
            # Nominal antineutrino spectrum
            #
            load_graph(
                name = "reactor_anue.neutrino_per_fission_per_MeV_input",
                filenames = path_arrays / f"reactor_anue_spectrum_interp_scaled_approx_50keV.{self.source_type}",
                x = "enu",
                y = "spec",
                merge_x = True,
                replicate_outputs = index["isotope"],
            )

            #
            # Interpolate for the integration mesh
            #
            Interpolator.replicate(
                method = "exp",
                names = {
                    "indexer": "reactor_anue.spec_indexer",
                    "interpolator": "reactor_anue.neutrino_per_fission_per_MeV_nominal",
                    },
                replicate_outputs = index["isotope"],
            )
            outputs.get_value("reactor_anue.neutrino_per_fission_per_MeV_input.enu") >> inputs.get_value("reactor_anue.neutrino_per_fission_per_MeV_nominal.xcoarse")
            outputs("reactor_anue.neutrino_per_fission_per_MeV_input.spec") >> inputs("reactor_anue.neutrino_per_fission_per_MeV_nominal.ycoarse")
            kinematic_integrator_enu >> inputs.get_value("reactor_anue.neutrino_per_fission_per_MeV_nominal.xfine")

            #
            # SNF and NEQ normalization factors
            #
            load_parameters(
                format="value",
                state="fixed",
                parameters={
                    "reactor": {
                        "snf_factor": 1.0,
                        "neq_factor": 1.0,
                    }
                },
                labels={
                    "reactor": {
                        "snf_factor": "Common Spent Nuclear Fuel (SNF) factor",
                        "neq_factor": "Common Non-Equilibrium (NEQ) factor",
                    }
                },
            )

            #
            # Non-Equilibrium correction
            #
            load_graph(
                name = "reactor_nonequilibrium_anue.correction_input",
                x = "enu",
                y = "nonequilibrium_correction",
                merge_x = True,
                filenames = path_arrays / f"nonequilibrium_correction.{self.source_type}",
                replicate_outputs = index["isotope_neq"],
                dtype = "d"
            )

            Interpolator.replicate(
                method = "linear",
                names = {
                    "indexer": "reactor_nonequilibrium_anue.correction_indexer",
                    "interpolator": "reactor_nonequilibrium_anue.correction_interpolated",
                    },
                replicate_outputs = index["isotope_neq"],
                underflow = "constant",
                overflow = "constant",
            )
            outputs.get_value("reactor_nonequilibrium_anue.correction_input.enu") >> inputs.get_value("reactor_nonequilibrium_anue.correction_interpolated.xcoarse")
            outputs("reactor_nonequilibrium_anue.correction_input.nonequilibrium_correction") >> inputs("reactor_nonequilibrium_anue.correction_interpolated.ycoarse")
            kinematic_integrator_enu >> inputs.get_value("reactor_nonequilibrium_anue.correction_interpolated.xfine")

            #
            # SNF correction
            #
            load_graph(
                name = "snf_anue.correction_input",
                x = "enu",
                y = "snf_correction",
                merge_x = True,
                filenames = path_arrays / f"snf_correction.{self.source_type}",
                replicate_outputs = index["reactor"],
                dtype = "d"
            )
            Interpolator.replicate(
                method = "linear",
                names = {
                    "indexer": "snf_anue.correction_indexer",
                    "interpolator": "snf_anue.correction_interpolated",
                    },
                replicate_outputs = index["reactor"],
                underflow = "constant",
                overflow = "constant",
            )
            outputs.get_value("snf_anue.correction_input.enu") >> inputs.get_value("snf_anue.correction_interpolated.xcoarse")
            outputs("snf_anue.correction_input.snf_correction") >> inputs("snf_anue.correction_interpolated.ycoarse")
            kinematic_integrator_enu >> inputs.get_value("snf_anue.correction_interpolated.xfine")

            #
            # Reactor antineutrino spectral correction:
            #   - not constrained
            #   - correlated between isotopes
            #   - uncorrelated between energy intervals
            #
            make_y_parameters_for_x(
                    outputs.get_value("reactor_anue.spectrum_free_correction.spec_model_edges"),
                    namefmt = "spec_scale_{:02d}",
                    format = "value",
                    state = "variable",
                    key = "neutrino_per_fission_factor",
                    values = 0.0,
                    labels = "Edge {i:02d} ({value:.2f} MeV) reactor antineutrino spectrum correction" \
                           + (" (exp)" if self.spectrum_correction_mode == "exponential" else " (linear)"),
                    hide_nodes = True
                    )

            Concatenation.replicate(
                    parameters("all.neutrino_per_fission_factor"),
                    name = "reactor_anue.spectrum_free_correction.input"
                    )
            outputs.get_value("reactor_anue.spectrum_free_correction.input").dd.axes_meshes = (outputs.get_value("reactor_anue.spectrum_free_correction.spec_model_edges"),)
            if self.spectrum_correction_mode == "exponential":
                Exp.replicate(
                        outputs.get_value("reactor_anue.spectrum_free_correction.input"),
                        name = "reactor_anue.spectrum_free_correction.correction"
                        )
            else:
                Array.from_value("reactor_anue.spectrum_free_correction.unity", 1.0, dtype="d", mark="1", label="Array of 1 element =1", shape=1, store=True)
                Sum.replicate(
                        outputs.get_value("reactor_anue.spectrum_free_correction.unity"),
                        outputs.get_value("reactor_anue.spectrum_free_correction.input"),
                        name = "reactor_anue.spectrum_free_correction.correction"
                        )
                outputs.get_value("reactor_anue.spectrum_free_correction.correction").dd.axes_meshes = (outputs.get_value("reactor_anue.spectrum_free_correction.spec_model_edges"),)

            Interpolator.replicate(
                method = "exp",
                names = {
                    "indexer": "reactor_anue.spectrum_free_correction.indexer",
                    "interpolator": "reactor_anue.spectrum_free_correction.interpolated"
                    },
            )
            outputs.get_value("reactor_anue.spectrum_free_correction.spec_model_edges") >> inputs.get_value("reactor_anue.spectrum_free_correction.interpolated.xcoarse")
            outputs.get_value("reactor_anue.spectrum_free_correction.correction") >> inputs.get_value("reactor_anue.spectrum_free_correction.interpolated.ycoarse")
            kinematic_integrator_enu >> inputs.get_value("reactor_anue.spectrum_free_correction.interpolated.xfine")

            #
            # Huber+Mueller spectrum shape uncertainties
            #   - constrained
            #   - two parts:
            #       - uncorrelated between isotopes and energy intervals
            #       - correlated between isotopes and energy intervals
            #
            load_graph(
                name = "reactor_anue.spectrum_uncertainty",
                filenames = path_arrays / f"reactor_anue_spectrum_unc_interp_scaled_approx_50keV.{self.source_type}",
                x = "enu_centers",
                y = "uncertainty",
                merge_x = True,
                replicate_outputs = combinations["anue_unc.isotope"],
            )

            for isotope in index["isotope"]:
                make_y_parameters_for_x(
                        outputs.get_value("reactor_anue.spectrum_uncertainty.enu_centers"),
                        namefmt = "unc_scale_{:03d}",
                        format = ("value", "sigma_absolute"),
                        state = "variable",
                        key = f"reactor_anue.spectrum_uncertainty.uncorr.{isotope}",
                        values = (0.0, 1.0),
                        labels = f"Edge {{i:02d}} ({{value:.2f}} MeV) uncorrelated {index_names[isotope]} spectrum correction",
                        disable_last_one = False, # True for the constant interpolation, last edge is unused
                        hide_nodes = True
                        )

            load_parameters(
                    path = "reactor_anue.spectrum_uncertainty",
                    format=("value", "sigma_absolute"),
                    state="variable",
                    parameters={
                        "corr": (0.0, 1.0)
                        },
                    labels={
                        "corr": "Correlated ν̅ spectrum shape correction"
                        },
                    joint_nuisance = False
                    )

            Concatenation.replicate(
                    parameters("constrained.reactor_anue.spectrum_uncertainty.uncorr"),
                    name = "reactor_anue.spectrum_uncertainty.scale.uncorr",
                    replicate_outputs = index["isotope"]
                    )

            Product.replicate(
                    outputs("reactor_anue.spectrum_uncertainty.scale.uncorr"),
                    outputs("reactor_anue.spectrum_uncertainty.uncertainty.uncorr"),
                    name = "reactor_anue.spectrum_uncertainty.correction.uncorr",
                    replicate_outputs = index["isotope"]
                    )

            Product.replicate(
                    parameters.get_value("constrained.reactor_anue.spectrum_uncertainty.corr"),
                    outputs("reactor_anue.spectrum_uncertainty.uncertainty.corr"),
                    name = "reactor_anue.spectrum_uncertainty.correction.corr",
                    replicate_outputs = index["isotope"]
                    )

            single_unity = Array("single_unity", [1.0], dtype="d", mark="1", label="Array of 1 element =1")
            Sum.replicate(
                    outputs("reactor_anue.spectrum_uncertainty.correction.uncorr"),
                    single_unity,
                    name = "reactor_anue.spectrum_uncertainty.correction.uncorr_factor",
                    replicate_outputs = index["isotope"]
                    )
            Sum.replicate(
                    outputs("reactor_anue.spectrum_uncertainty.correction.corr"),
                    single_unity,
                    name = "reactor_anue.spectrum_uncertainty.correction.corr_factor",
                    replicate_outputs = index["isotope"]
                    )

            Product.replicate(
                    outputs("reactor_anue.spectrum_uncertainty.correction.uncorr_factor"),
                    outputs("reactor_anue.spectrum_uncertainty.correction.corr_factor"),
                    name = "reactor_anue.spectrum_uncertainty.correction.full",
                    replicate_outputs = index["isotope"]
                    )

            Interpolator.replicate(
                method = "linear",
                names = {
                    "indexer": "reactor_anue.spectrum_uncertainty.correction_index",
                    "interpolator": "reactor_anue.spectrum_uncertainty.correction_interpolated"
                    },
                replicate_outputs=index["isotope"]
            )
            outputs.get_value("reactor_anue.spectrum_uncertainty.enu_centers") >> inputs.get_value("reactor_anue.spectrum_uncertainty.correction_interpolated.xcoarse")
            outputs("reactor_anue.spectrum_uncertainty.correction.full") >> inputs("reactor_anue.spectrum_uncertainty.correction_interpolated.ycoarse")
            kinematic_integrator_enu >> inputs.get_value("reactor_anue.spectrum_uncertainty.correction_interpolated.xfine")

            #
            # Antineutrino spectrum with corrections
            #
            Product.replicate(
                    outputs("reactor_anue.neutrino_per_fission_per_MeV_nominal"),
                    outputs.get_value("reactor_anue.spectrum_free_correction.interpolated"),
                    outputs("reactor_anue.spectrum_uncertainty.correction_interpolated"),
                    name = "reactor_anue.part.neutrino_per_fission_per_MeV_main",
                    replicate_outputs=index["isotope"],
                    )

            Product.replicate(
                    outputs("reactor_anue.neutrino_per_fission_per_MeV_nominal"),
                    outputs("reactor_nonequilibrium_anue.correction_interpolated"),
                    name = "reactor_anue.part.neutrino_per_fission_per_MeV_neq_nominal",
                    allow_skip_inputs = True,
                    skippable_inputs_should_contain = ("U238",),
                    replicate_outputs=index["isotope_neq"],
                    )

            #
            # Livetime
            #
            load_record_data(
                name = "daily_data.detector_all",
                filenames = path_arrays/f"livetimes_Dubna_AdSimpleNL_all.{self.source_type}",
                replicate_outputs = index["detector"],
                objects = lambda idx, _: f"EH{idx[-2]}AD{idx[-1]}",
                columns = ("day", "ndet", "livetime", "eff", "efflivetime"),
                skip = inactive_detectors
            )
            refine_detector_data(
                data("daily_data.detector_all"),
                data.child("daily_data.detector"),
                detectors = index["detector"]
            )

            if "reactor-28days" in self._future:
                logger.warning("Future: use merged reactor data, period: 28 days")
                load_record_data(
                    name = "daily_data.reactor_all",
                    filenames = path_arrays/f"reactor_power_28days.{self.source_type}",
                    replicate_outputs = index["reactor"],
                    columns = ("period", "day", "ndet", "ndays", "power") + index["isotope_lower"],
                )
                assert "reactor-35days" not in self._future, "Mutually exclusive options"
            elif "reactor-35days" in self._future:
                logger.warning("Future: use merged reactor data, period: 35 days")
                load_record_data(
                    name = "daily_data.reactor_all",
                    filenames = path_arrays/f"reactor_power_35days.{self.source_type}",
                    replicate_outputs = index["reactor"],
                    columns = ("period", "day", "ndet", "ndays", "power") + index["isotope_lower"],
                )
            else:
                load_record_data(
                    name = "daily_data.reactor_all",
                    filenames = path_arrays/f"reactor_thermal_power_weekly.{self.source_type}",
                    replicate_outputs = index["reactor"],
                    columns = ("period", "day", "ndet", "ndays", "power") + index["isotope_lower"],
                )
            refine_reactor_data(
                data("daily_data.reactor_all"),
                data.child("daily_data.reactor"),
                reactors = index["reactor"],
                isotopes = index["isotope"],
            )

            sync_reactor_detector_data(
                    data("daily_data.reactor"),
                    data("daily_data.detector"),
                    )

            Array.from_storage(
                "daily_data.detector.livetime",
                storage("data"),
                remove_used_arrays = True,
                dtype = "d"
            )

            Array.from_storage(
                "daily_data.detector.eff",
                storage("data"),
                remove_used_arrays = True,
                dtype = "d"
            )

            Array.from_storage(
                "daily_data.detector.efflivetime",
                storage("data"),
                remove_used_arrays = True,
                dtype = "d"
            )

            Array.from_storage(
                "daily_data.reactor.power",
                storage("data"),
                remove_used_arrays = True,
                dtype = "d"
            )

            Array.from_storage(
                "daily_data.reactor.fission_fraction",
                storage("data"),
                remove_used_arrays = True,
                dtype = "d"
            )
            del storage["data.daily_data"]

            #
            # Neutrino rate
            #
            Product.replicate(
                    parameters("all.reactor.nominal_thermal_power"),
                    parameters.get_value("all.conversion.reactorPowerConversion"),
                    name = "reactor.thermal_power_nominal_MeVs",
                    replicate_outputs = index["reactor"]
                    )

            Product.replicate(
                    parameters("central.reactor.nominal_thermal_power"),
                    parameters.get_value("all.conversion.reactorPowerConversion"),
                    name = "reactor.thermal_power_nominal_MeVs_central",
                    replicate_outputs = index["reactor"]
                    )

            # Time dependent, fit dependent (non-nominal) for reactor core
            Product.replicate(
                    parameters("all.reactor.fission_fraction_scale"),
                    outputs("daily_data.reactor.fission_fraction"),
                    name = "daily_data.reactor.fission_fraction_scaled",
                    replicate_outputs=combinations["reactor.isotope.period"],
                    )

            Product.replicate(
                    parameters("all.reactor.energy_per_fission"),
                    outputs("daily_data.reactor.fission_fraction_scaled"),
                    name = "reactor.energy_per_fission_weighted_MeV",
                    replicate_outputs=combinations["reactor.isotope.period"],
                    )

            Sum.replicate(
                    outputs("reactor.energy_per_fission_weighted_MeV"),
                    name = "reactor.energy_per_fission_average_MeV",
                    replicate_outputs=combinations["reactor.period"],
                    )

            Product.replicate(
                    outputs("daily_data.reactor.power"),
                    outputs("daily_data.reactor.fission_fraction_scaled"),
                    outputs("reactor.thermal_power_nominal_MeVs"),
                    name = "reactor.thermal_power_isotope_MeV_per_second",
                    replicate_outputs=combinations["reactor.isotope.period"],
                    )

            Division.replicate(
                    outputs("reactor.thermal_power_isotope_MeV_per_second"),
                    outputs("reactor.energy_per_fission_average_MeV"),
                    name = "reactor.fissions_per_second",
                    replicate_outputs=combinations["reactor.isotope.period"],
                    )

            # Nominal, time and reactor independent power and fission fractions for SNF
            # NOTE: central values are used for energy_per_fission
            Product.replicate(
                    parameters("central.reactor.energy_per_fission"),
                    parameters("all.reactor.fission_fraction_snf"),
                    name = "reactor.energy_per_fission_snf_weighted_MeV",
                    replicate_outputs=index["isotope"],
                    )

            Sum.replicate(
                    outputs("reactor.energy_per_fission_snf_weighted_MeV"),
                    name = "reactor.energy_per_fission_snf_average_MeV",
                    )

            # NOTE: central values are used for the thermal power
            Product.replicate(
                    parameters("all.reactor.fission_fraction_snf"),
                    outputs("reactor.thermal_power_nominal_MeVs_central"),
                    name = "reactor.thermal_power_snf_isotope_MeV_per_second",
                    replicate_outputs=combinations["reactor.isotope"],
                    )

            Division.replicate(
                    outputs("reactor.thermal_power_snf_isotope_MeV_per_second"),
                    outputs.get_value("reactor.energy_per_fission_snf_average_MeV"),
                    name = "reactor.fissions_per_second_snf",
                    replicate_outputs=combinations["reactor.isotope"],
                    )

            # Effective number of fissions seen in Detector from Reactor from Isotope during Period
            Product.replicate(
                    outputs("reactor.fissions_per_second"),
                    outputs("daily_data.detector.efflivetime"),
                    name = "reactor_detector.nfissions_daily",
                    replicate_outputs=combinations["reactor.isotope.detector.period"],
                    allow_skip_inputs = True,
                    skippable_inputs_should_contain = inactive_detectors
                    )

            # Total effective number of fissions from a Reactor seen in the Detector during Period
            ArraySum.replicate(
                    outputs("reactor_detector.nfissions_daily"),
                    name = "reactor_detector.nfissions",
                    )

            # Baseline factor from Reactor to Detector: 1/(4πL²)
            InverseSquareLaw.replicate(
                name="reactor_detector.baseline_factor_per_cm2",
                scale="m_to_cm",
                replicate_outputs=combinations["reactor.detector"]
            )
            parameters("constant.baseline") >> inputs("reactor_detector.baseline_factor_per_cm2")

            # Number of protons per detector
            Product.replicate(
                    parameters.get_value("all.detector.nprotons_nominal_ad"),
                    parameters("all.detector.nprotons_correction"),
                    name = "detector.nprotons",
                    replicate_outputs = index["detector"]
            )

            # Number of fissions × N protons × ε / (4πL²)  (main)
            Product.replicate(
                    outputs("reactor_detector.nfissions"),
                    outputs("detector.nprotons"),
                    outputs("reactor_detector.baseline_factor_per_cm2"),
                    parameters.get_value("all.detector.efficiency"),
                    name = "reactor_detector.nfissions_nprotons_per_cm2",
                    replicate_outputs=combinations["reactor.isotope.detector.period"],
                    )

            Product.replicate(
                    outputs("reactor_detector.nfissions_nprotons_per_cm2"),
                    parameters("all.reactor.nonequilibrium_scale"),
                    parameters.get_value("all.reactor.neq_factor"),
                    name = "reactor_detector.nfissions_nprotons_per_cm2_neq",
                    replicate_outputs=combinations["reactor.isotope.detector.period"],
                    )

            # Detector live time
            ArraySum.replicate(
                    outputs("daily_data.detector.livetime"),
                    name = "detector.livetime",
                    )

            ArraySum.replicate(
                    outputs("daily_data.detector.efflivetime"),
                    name = "detector.efflivetime",
                    )

            Product.replicate(
                    outputs("detector.efflivetime"),
                    parameters.get_value("all.conversion.seconds_in_day_inverse"),
                    name="detector.efflivetime_days",
                    replicate_outputs=combinations["detector.period"],
                    allow_skip_inputs=True,
                    skippable_inputs_should_contain=inactive_detectors,
                    )

            # Effective live time × N protons × ε / (4πL²)  (SNF)
            Product.replicate(
                    outputs("detector.efflivetime"),
                    outputs("detector.nprotons"),
                    outputs("reactor_detector.baseline_factor_per_cm2"),
                    parameters("all.reactor.snf_scale"),
                    parameters.get_value("all.reactor.snf_factor"),
                    parameters.get_value("all.detector.efficiency"),
                    name = "reactor_detector.livetime_nprotons_per_cm2_snf",
                    replicate_outputs=combinations["reactor.detector.period"],
                    allow_skip_inputs = True,
                    skippable_inputs_should_contain = inactive_detectors
                    )

            #
            # Average SNF Spectrum
            #
            Product.replicate(
                    outputs("reactor_anue.neutrino_per_fission_per_MeV_nominal"),
                    outputs("reactor.fissions_per_second_snf"),
                    name = "snf_anue.neutrino_per_second_isotope",
                    replicate_outputs=combinations["reactor.isotope"],
                    )

            Sum.replicate(
                    outputs("snf_anue.neutrino_per_second_isotope"),
                    name = "snf_anue.neutrino_per_second",
                    replicate_outputs=index["reactor"],
                    )

            Product.replicate(
                    outputs("snf_anue.neutrino_per_second"),
                    outputs("snf_anue.correction_interpolated"),
                    name = "snf_anue.neutrino_per_second_snf",
                    replicate_outputs = index["reactor"]
                    )

            #
            # Integrand: flux × oscillation probability × cross section
            # [Nν·cm²/fission/proton]
            #
            Product.replicate(
                    outputs.get_value("kinematics.ibd.crosssection"),
                    outputs.get_value("kinematics.ibd.jacobian"),
                    name="kinematics.ibd.crosssection_jacobian",
            )

            Product.replicate(
                    outputs.get_value("kinematics.ibd.crosssection_jacobian"),
                    outputs("oscprob"),
                    name="kinematics.ibd.crosssection_jacobian_oscillations",
                    replicate_outputs=combinations["reactor.detector"]
            )

            Product.replicate(
                    outputs("kinematics.ibd.crosssection_jacobian_oscillations"),
                    outputs("reactor_anue.part.neutrino_per_fission_per_MeV_main"),
                    name="kinematics.neutrino_cm2_per_MeV_per_fission_per_proton.part.nu_main",
                    replicate_outputs=combinations["reactor.isotope.detector"]
            )

            Product.replicate(
                    outputs("kinematics.ibd.crosssection_jacobian_oscillations"),
                    outputs("reactor_anue.part.neutrino_per_fission_per_MeV_neq_nominal"),
                    name="kinematics.neutrino_cm2_per_MeV_per_fission_per_proton.part.nu_neq",
                    replicate_outputs=combinations["reactor.isotope_neq.detector"]
            )

            Product.replicate(
                    outputs("kinematics.ibd.crosssection_jacobian_oscillations"),
                    outputs("snf_anue.neutrino_per_second_snf"),
                    name="kinematics.neutrino_cm2_per_MeV_per_fission_per_proton.part.nu_snf",
                    replicate_outputs=combinations["reactor.detector"]
            )
            outputs("kinematics.neutrino_cm2_per_MeV_per_fission_per_proton.part.nu_main") >> inputs("kinematics.integral.nu_main")
            outputs("kinematics.neutrino_cm2_per_MeV_per_fission_per_proton.part.nu_neq") >> inputs("kinematics.integral.nu_neq")
            outputs("kinematics.neutrino_cm2_per_MeV_per_fission_per_proton.part.nu_snf") >> inputs("kinematics.integral.nu_snf")

            #
            # Multiply by the scaling factors:
            #  - nu_main: fissions_per_second[p,r,i] × effective live time[p,d] × N protons[d] × efficiency[d]
            #  - nu_neq:  fissions_per_second[p,r,i] × effective live time[p,d] × N protons[d] × efficiency[d] × nonequilibrium scale[r,i] × neq_factor(=1)
            #  - nu_snf:                               effective live time[p,d] × N protons[d] × efficiency[d] × SNF scale[r]              × snf_factor(=1)
            #
            Product.replicate(
                    outputs("kinematics.integral.nu_main"),
                    outputs("reactor_detector.nfissions_nprotons_per_cm2"),
                    name = "eventscount.parts.nu_main",
                    replicate_outputs = combinations["reactor.isotope.detector.period"]
                    )

            Product.replicate(
                    outputs("kinematics.integral.nu_neq"),
                    outputs("reactor_detector.nfissions_nprotons_per_cm2_neq"),
                    name = "eventscount.parts.nu_neq",
                    replicate_outputs = combinations["reactor.isotope_neq.detector.period"],
                    allow_skip_inputs = True,
                    skippable_inputs_should_contain = ("U238",)
                    )

            Product.replicate(
                    outputs("kinematics.integral.nu_snf"),
                    outputs("reactor_detector.livetime_nprotons_per_cm2_snf"),
                    name = "eventscount.parts.nu_snf",
                    replicate_outputs = combinations["reactor.detector.period"]
                    )

            Sum.replicate(
                outputs("eventscount.parts"),
                name="eventscount.raw",
                replicate_outputs=combinations["detector.period"]
            )

            #
            # Detector effects
            #
            load_array(
                name = "detector.iav",
                filenames = path_arrays/f"detector_IAV_matrix_P14A_LS.{self.source_type}",
                replicate_outputs = ("matrix_raw",),
                name_function = {"matrix_raw": "iav_matrix"},
                array_kwargs = {
                    'edges': (edges_energy_escint, edges_energy_edep)
                    }
            )

            RenormalizeDiag.replicate(mode="offdiag", name="detector.iav.matrix_rescaled", replicate_outputs=index["detector"])
            parameters("all.detector.iav_offdiag_scale_factor") >> inputs("detector.iav.matrix_rescaled.scale")
            outputs.get_value("detector.iav.matrix_raw") >> inputs("detector.iav.matrix_rescaled.matrix")

            VectorMatrixProduct.replicate(name="eventscount.iav", replicate_outputs=combinations["detector.period"])
            outputs("detector.iav.matrix_rescaled") >> inputs("eventscount.iav.matrix")
            outputs("eventscount.raw") >> inputs("eventscount.iav.vector")

            load_graph_data(
                name = "detector.lsnl.curves",
                x = "edep",
                y = "evis_parts",
                merge_x = True,
                filenames = path_arrays/f"detector_LSNL_curves_Jan2022_newE_v1.{self.source_type}",
                replicate_outputs = index["lsnl"],
            )

            # Refine LSNL curves: interpolate with smaller step
            refine_lsnl_data(
                storage("data.detector.lsnl.curves"),
                edepname = 'edep',
                nominalname = 'evis_parts.nominal',
                refine_times = 4,
                newmin = 0.5,
                newmax = 12.1
            )

            Array.from_storage(
                "detector.lsnl.curves",
                storage("data"),
                meshname = "edep",
                remove_used_arrays = True
            )

            Product.replicate(
                outputs("detector.lsnl.curves.evis_parts"),
                parameters("constrained.detector.lsnl_scale_a"),
                name = "detector.lsnl.curves.evis_parts_scaled",
                allow_skip_inputs = True,
                skippable_inputs_should_contain = ("nominal",),
                replicate_outputs=index["lsnl_nuisance"]
            )

            Sum.replicate(
                outputs.get_value("detector.lsnl.curves.evis_parts.nominal"),
                outputs("detector.lsnl.curves.evis_parts_scaled"),
                name="detector.lsnl.curves.evis_coarse"
            )

            #
            # Force Evis(Edep) to grow monotonously
            # - Required by matrix calculation algorithm
            # - Introduced to achieve stable minimization
            # - Non-monotonous behavior happens for extreme systematic values and is not expected to affect the analysis
            Monotonize.replicate(
                    name="detector.lsnl.curves.evis_coarse_monotonous",
                    index_fraction = 0.5,
                    gradient = 1.0,
                    with_x = True
                    )
            outputs.get_value("detector.lsnl.curves.edep") >> inputs.get_value("detector.lsnl.curves.evis_coarse_monotonous.x")
            outputs.get_value("detector.lsnl.curves.evis_coarse") >> inputs.get_value("detector.lsnl.curves.evis_coarse_monotonous.y")

            remap_items(
                parameters("all.detector.detector_relative"),
                outputs.child("detector.parameters_relative"),
                reorder_indices=[
                    ["detector", "parameters"],
                    ["parameters", "detector"],
                ],
            )

            # Interpolate Evis(Edep)
            Interpolator.replicate(
                method = "linear",
                names = {
                    "indexer": "detector.lsnl.indexer_fwd",
                    "interpolator": "detector.lsnl.interpolated_fwd",
                    },
            )
            outputs.get_value("detector.lsnl.curves.edep") >> inputs.get_value("detector.lsnl.interpolated_fwd.xcoarse")
            outputs.get_value("detector.lsnl.curves.evis_coarse_monotonous") >> inputs.get_value("detector.lsnl.interpolated_fwd.ycoarse")
            edges_energy_edep >> inputs.get_value("detector.lsnl.interpolated_fwd.xfine")

            # Introduce uncorrelated between detectors energy scale for interpolated Evis[detector]=s[detector]*Evis(Edep)
            Product.replicate(
                outputs.get_value("detector.lsnl.interpolated_fwd"),
                outputs("detector.parameters_relative.energy_scale_factor"),
                name="detector.lsnl.curves.evis",
                replicate_outputs = index["detector"]
            )

            # Introduce uncorrelated between detectors energy scale for coarse Evis[detector]=s[detector]*Evis(Edep)
            Product.replicate(
                outputs.get_value("detector.lsnl.curves.evis_coarse_monotonous"),
                outputs("detector.parameters_relative.energy_scale_factor"),
                name="detector.lsnl.curves.evis_coarse_monotonous_scaled",
                replicate_outputs = index["detector"]
            )

            # Interpolate Edep(Evis[detector])
            Interpolator.replicate(
                method = "linear",
                names = {
                    "indexer": "detector.lsnl.indexer_bwd",
                    "interpolator": "detector.lsnl.interpolated_bwd",
                    },
                replicate_xcoarse = True,
                replicate_outputs = index["detector"]
            )
            outputs.get_dict("detector.lsnl.curves.evis_coarse_monotonous_scaled") >> inputs.get_dict("detector.lsnl.interpolated_bwd.xcoarse")
            outputs.get_value("detector.lsnl.curves.edep")  >> inputs.get_dict("detector.lsnl.interpolated_bwd.ycoarse")
            edges_energy_evis.outputs[0] >> inputs.get_dict("detector.lsnl.interpolated_bwd.xfine")

            # Build LSNL matrix
            AxisDistortionMatrix.replicate(name="detector.lsnl.matrix", replicate_outputs=index["detector"])
            edges_energy_edep.outputs[0] >> inputs("detector.lsnl.matrix.EdgesOriginal")
            outputs.get_value("detector.lsnl.interpolated_fwd") >> inputs.get_dict("detector.lsnl.matrix.EdgesModified")
            outputs.get_dict("detector.lsnl.interpolated_bwd") >> inputs.get_dict("detector.lsnl.matrix.EdgesModifiedBackwards")
            VectorMatrixProduct.replicate(name="eventscount.evis", replicate_outputs=combinations["detector.period"])
            outputs("detector.lsnl.matrix") >> inputs("eventscount.evis.matrix")
            outputs("eventscount.iav") >> inputs("eventscount.evis.vector")

            EnergyResolution.replicate(path="detector.eres")
            nodes.get_value("detector.eres.sigma_rel") << parameters("constrained.detector.eres")
            outputs.get_value("edges.energy_evis") >> inputs.get_value("detector.eres.matrix")
            outputs.get_value("edges.energy_evis") >> inputs.get_value("detector.eres.e_edges")

            VectorMatrixProduct.replicate(name="eventscount.erec", replicate_outputs=combinations["detector.period"])
            outputs.get_value("detector.eres.matrix") >> inputs("eventscount.erec.matrix")
            outputs("eventscount.evis") >> inputs("eventscount.erec.vector")

            Product.replicate(
                parameters.get_value("all.detector.global_normalization"),
                outputs("detector.parameters_relative.efficiency_factor"),
                name = "detector.normalization",
                replicate_outputs=index["detector"],
            )

            Product.replicate(
                outputs("detector.normalization"),
                outputs("eventscount.erec"),
                name = "eventscount.fine.ibd_normalized",
                replicate_outputs=combinations["detector.period"],
            )

            Sum.replicate(
                outputs("eventscount.fine.ibd_normalized"),
                name = "eventscount.fine.ibd_normalized_detector",
                replicate_outputs=combinations["detector"],
            )

            Rebin.replicate(
                names={"matrix": "detector.rebin.matrix_ibd", "product": "eventscount.final.ibd"},
                replicate_outputs=combinations["detector.period"],
            )
            edges_energy_erec >> inputs.get_value("detector.rebin.matrix_ibd.edges_old")
            edges_energy_final >> inputs.get_value("detector.rebin.matrix_ibd.edges_new")
            outputs("eventscount.fine.ibd_normalized") >> inputs("eventscount.final.ibd")

            #
            # Backgrounds
            #
            bkg_names = {
                'acc': "accidental",
                'lihe': "lithium9",
                'fastn': "fastNeutron",
                'amc': "amCSource",
                'alphan': "carbonAlpha",
                'muon': "muonRelated"
            }
            load_hist(
                name = "bkg",
                x = "erec",
                y = "spectrum_shape",
                merge_x = True,
                normalize = True,
                filenames = path_root/"bkg_SYSU_input_by_period_{}.root",
                replicate_files = index["period"],
                replicate_outputs = combinations["bkg.detector"],
                skip = inactive_detectors,
                key_order = (1, 2, 0),
                objects = lambda _, idx: f"DYB_{bkg_names[idx[0]]}_expected_spectrum_EH{idx[-2][-2]}_AD{idx[-2][-1]}"
            )

            # TODO:
            # GNA upload fast-n as array from 0 to 12 MeV (50 keV), and it normalized to 1.
            # So, every bin contain 0.00416667.
            # TODO: remove in dayabay-v1
            fastn_data = ones(240) / 240
            for key, spectrum in storage("outputs.bkg.spectrum_shape.fastn").walkitems():
                spectrum._data[:] = fastn_data

            Product.replicate(
                    parameters("all.bkg.rate.acc"),
                    outputs("bkg.spectrum_shape.acc"),
                    name="bkg.spectrum_per_day.acc",
                    replicate_outputs=combinations["detector.period"],
                    )

            Product.replicate(
                    # outputs("bkg.rate.lihe"),
                    parameters("all.bkg.rate.lihe"),
                    outputs("bkg.spectrum_shape.lihe"),
                    name="bkg.spectrum_per_day.lihe",
                    replicate_outputs=combinations["detector.period"],
                    )

            Product.replicate(
                    # outputs("bkg.rate.fastn"),
                    parameters("all.bkg.rate.fastn"),
                    outputs("bkg.spectrum_shape.fastn"),
                    name="bkg.spectrum_per_day.fastn",
                    replicate_outputs=combinations["detector.period"],
                    )

            Product.replicate(
                    parameters("all.bkg.rate.alphan"),
                    outputs("bkg.spectrum_shape.alphan"),
                    name="bkg.spectrum_per_day.alphan",
                    replicate_outputs=combinations["detector.period"],
                    )

            Product.replicate(
                    parameters("all.bkg.rate.amc"),
                    outputs("bkg.spectrum_shape.amc"),
                    name="bkg.spectrum_per_day.amc",
                    replicate_outputs=combinations["detector.period"],
                    )

            # Total spectrum of Background in Detector during Period
            # spectrum_per_day [N / day] * efflivetime [sec] * seconds_in_day_inverse [day / sec] -> [N]
            Product.replicate(
                    outputs("detector.efflivetime_days"),
                    outputs("bkg.spectrum_per_day"),
                    name="bkg.spectrum",
                    replicate_outputs=combinations["bkg.detector.period"],
                    )

            Sum.replicate(
                    outputs("bkg.spectrum"),
                    name="eventscount.fine.bkg",
                    replicate_outputs=combinations["detector.period"],
                    )

            Sum.replicate(
                    outputs("eventscount.fine.ibd_normalized"),
                    outputs("eventscount.fine.bkg"),
                    name="eventscount.fine.total",
                    replicate_outputs=combinations["detector.period"],
                    check_edges_contents=True,
                    )

            Rebin.replicate(
                    names={"matrix": "detector.rebin.matrix_bkg", "product": "eventscount.final.bkg"},
                    replicate_outputs=combinations["detector.period"],
            )
            edges_energy_erec >> inputs.get_value("detector.rebin.matrix_bkg.edges_old")
            edges_energy_final >> inputs.get_value("detector.rebin.matrix_bkg.edges_new")
            outputs("eventscount.fine.bkg") >> inputs("eventscount.final.bkg")

            Sum.replicate(
                outputs("eventscount.final.ibd"),
                outputs("eventscount.final.bkg"),
                name="eventscount.final.detector_period",
                replicate_outputs=combinations["detector.period"],
            )

            Concatenation.replicate(
                outputs("eventscount.final.detector_period"),
                name="eventscount.final.concatenated.detector_period",
            )

            Sum.replicate(
                outputs("eventscount.final.detector_period"),
                name="eventscount.final.detector",
                replicate_outputs=index["detector"],
            )

            Concatenation.replicate(
                outputs("eventscount.final.detector"),
                name="eventscount.final.concatenated.detector"
            )

            outputs["eventscount.final.concatenated.selected"] = outputs[f"eventscount.final.concatenated.{self.concatenation_mode}"]

            #
            # Covariance matrices
            #
            self._covariance_matrix = CovarianceMatrixGroup(store_to="covariance")

            for name, parameters_source in self.systematic_uncertainties_groups().items():
                self._covariance_matrix.add_covariance_for(name, parameters_nuisance_normalized[parameters_source])
            self._covariance_matrix.add_covariance_sum()

            outputs.get_value("eventscount.final.concatenated.selected") >> self._covariance_matrix

            npars_cov = self._covariance_matrix.get_parameters_count()
            list_parameters_nuisance_normalized = list(parameters_nuisance_normalized.walkvalues())
            npars_nuisance = len(list_parameters_nuisance_normalized)
            if npars_cov!=npars_nuisance:
                raise RuntimeError("Some parameters are missing from covariance matrix")

            parinp_mc = ParArrayInput(
                name="mc.parameters.inputs",
                parameters=list_parameters_nuisance_normalized,
            )

            #
            # Statistic
            #
            # Create Nuisance parameters
            Sum.replicate(outputs("statistic.nuisance.parts"), name="statistic.nuisance.all")

            MonteCarlo.replicate(
                name="data.pseudo.self",
                mode=self.monte_carlo_mode,
                generator=self._random_generator,
            )
            outputs.get_value("eventscount.final.concatenated.selected") >> inputs.get_value("data.pseudo.self.data")
            self._frozen_nodes["pseudodata"] = (nodes.get_value("data.pseudo.self"),)

            Proxy.replicate(
                name="data.pseudo.proxy",
            )
            outputs.get_value("data.pseudo.self") >> inputs.get_value("data.pseudo.proxy.input")

            MonteCarlo.replicate(
                name="covariance.data.fixed",
                mode="asimov",
                generator=self._random_generator,
            )
            outputs.get_value("eventscount.final.concatenated.selected") >> inputs.get_value("covariance.data.fixed.data")
            self._frozen_nodes["covariance_data_fixed"] = (nodes.get_value("covariance.data.fixed"),)

            MonteCarlo.replicate(
                name="mc.parameters.toymc",
                mode="normal-unit",
                shape=(npars_nuisance,),
                generator=self._random_generator,
            )
            outputs.get_value("mc.parameters.toymc") >> parinp_mc
            nodes["mc.parameters.inputs"] = parinp_mc

            Cholesky.replicate(name="cholesky.stat.variable")
            outputs.get_value("eventscount.final.concatenated.selected") >> inputs.get_value("cholesky.stat.variable")

            Cholesky.replicate(name="cholesky.stat.fixed")
            outputs.get_value("covariance.data.fixed") >> inputs.get_value("cholesky.stat.fixed")

            Cholesky.replicate(name="cholesky.stat.data.fixed")
            outputs.get_value("data.pseudo.proxy") >> inputs.get_value("cholesky.stat.data.fixed")

            SumMatOrDiag.replicate(name="covariance.covmat_full_p.stat_fixed")
            outputs.get_value("covariance.data.fixed") >> nodes.get_value("covariance.covmat_full_p.stat_fixed")
            outputs.get_value("covariance.covmat_syst.sum") >> nodes.get_value("covariance.covmat_full_p.stat_fixed")

            Cholesky.replicate(name="cholesky.covmat_full_p.stat_fixed")
            outputs.get_value("covariance.covmat_full_p.stat_fixed") >> inputs.get_value("cholesky.covmat_full_p.stat_fixed")

            SumMatOrDiag.replicate(name="covariance.covmat_full_p.stat_variable")
            outputs.get_value("eventscount.final.concatenated.selected") >> nodes.get_value("covariance.covmat_full_p.stat_variable")
            outputs.get_value("covariance.covmat_syst.sum") >> nodes.get_value("covariance.covmat_full_p.stat_variable")

            Cholesky.replicate(name="cholesky.covmat_full_p.stat_variable")
            outputs.get_value("covariance.covmat_full_p.stat_variable") >> inputs.get_value("cholesky.covmat_full_p.stat_variable")

            SumMatOrDiag.replicate(name="covariance.covmat_full_n")
            outputs.get_value("data.pseudo.proxy") >> nodes.get_value("covariance.covmat_full_n")
            outputs.get_value("covariance.covmat_syst.sum") >> nodes.get_value("covariance.covmat_full_n")

            Cholesky.replicate(name="cholesky.covmat_full_n")
            outputs.get_value("covariance.covmat_full_n") >> inputs.get_value("cholesky.covmat_full_n")


            # (1) chi-squared Pearson stat (fixed Pearson errors)
            Chi2.replicate(name="statistic.stat.chi2p_iterative")
            outputs.get_value("eventscount.final.concatenated.selected") >> inputs.get_value("statistic.stat.chi2p_iterative.theory")
            outputs.get_value("cholesky.stat.fixed") >> inputs.get_value("statistic.stat.chi2p_iterative.errors")
            outputs.get_value("data.pseudo.proxy") >> inputs.get_value("statistic.stat.chi2p_iterative.data")

            # (2-2) chi-squared Neyman stat
            Chi2.replicate(name="statistic.stat.chi2n")
            outputs.get_value("eventscount.final.concatenated.selected") >> inputs.get_value("statistic.stat.chi2n.theory")
            outputs.get_value("cholesky.stat.data.fixed") >> inputs.get_value("statistic.stat.chi2n.errors")
            outputs.get_value("data.pseudo.proxy") >> inputs.get_value("statistic.stat.chi2n.data")

            # (2-1)
            Chi2.replicate(name="statistic.stat.chi2p")
            outputs.get_value("eventscount.final.concatenated.selected") >> inputs.get_value("statistic.stat.chi2p.theory")
            outputs.get_value("cholesky.stat.variable") >> inputs.get_value("statistic.stat.chi2p.errors")
            outputs.get_value("data.pseudo.proxy") >> inputs.get_value("statistic.stat.chi2p.data")

            # (5) chi-squared Pearson syst (fixed Pearson errors)
            Chi2.replicate(name="statistic.full.chi2p_covmat_fixed")
            outputs.get_value("data.pseudo.proxy") >> inputs.get_value("statistic.full.chi2p_covmat_fixed.data")
            outputs.get_value("eventscount.final.concatenated.selected") >> inputs.get_value("statistic.full.chi2p_covmat_fixed.theory")
            outputs.get_value("cholesky.covmat_full_p.stat_fixed") >> inputs.get_value("statistic.full.chi2p_covmat_fixed.errors")

            # (2-3) chi-squared Neyman syst
            Chi2.replicate(name="statistic.full.chi2n_covmat")
            outputs.get_value("data.pseudo.proxy") >> inputs.get_value("statistic.full.chi2n_covmat.data")
            outputs.get_value("eventscount.final.concatenated.selected") >> inputs.get_value("statistic.full.chi2n_covmat.theory")
            outputs.get_value("cholesky.covmat_full_n") >> inputs.get_value("statistic.full.chi2n_covmat.errors")

            # (2-4) Pearson variable stat errors
            Chi2.replicate(name="statistic.full.chi2p_covmat_variable")
            outputs.get_value("data.pseudo.proxy") >> inputs.get_value("statistic.full.chi2p_covmat_variable.data")
            outputs.get_value("eventscount.final.concatenated.selected") >> inputs.get_value("statistic.full.chi2p_covmat_variable.theory")
            outputs.get_value("cholesky.covmat_full_p.stat_variable") >> inputs.get_value("statistic.full.chi2p_covmat_variable.errors")

            CNPStat.replicate(name="statistic.staterr.cnp")
            outputs.get_value("data.pseudo.proxy") >> inputs.get_value("statistic.staterr.cnp.data")
            outputs.get_value("eventscount.final.concatenated.selected") >> inputs.get_value("statistic.staterr.cnp.theory")

            # (3) chi-squared CNP stat
            Chi2.replicate(name="statistic.stat.chi2cnp")
            outputs.get_value("data.pseudo.proxy") >> inputs.get_value("statistic.stat.chi2cnp.data")
            outputs.get_value("eventscount.final.concatenated.selected") >> inputs.get_value("statistic.stat.chi2cnp.theory")
            outputs.get_value("statistic.staterr.cnp") >> inputs.get_value("statistic.stat.chi2cnp.errors")

            # (2) chi-squared Pearson stat + pull (fixed Pearson errors)
            Sum.replicate(
                outputs.get_value("statistic.stat.chi2p_iterative"),
                outputs.get_value("statistic.nuisance.all"),
                name="statistic.full.chi2p_iterative",
            )
            # (4) chi-squared CNP stat + pull (fixed Pearson errors)
            Sum.replicate(
                outputs.get_value("statistic.stat.chi2cnp"),
                outputs.get_value("statistic.nuisance.all"),
                name="statistic.full.chi2cnp",
            )

            LogProdDiag.replicate(name="statistic.log_prod_diag")
            outputs.get_value("cholesky.covmat_full_p.stat_variable") >> inputs.get_value("statistic.log_prod_diag")

            # (7) chi-squared Pearson stat + log|V| (unfixed Pearson errors)
            Sum.replicate(
                outputs.get_value("statistic.stat.chi2p"),
                outputs.get_value("statistic.log_prod_diag"),
                name="statistic.stat.chi2p_unbiased",
            )

            # (8) chi-squared Pearson stat + log|V| + pull (unfixed Pearson errors)
            Sum.replicate(
                outputs.get_value("statistic.stat.chi2p_unbiased"),
                outputs.get_value("statistic.nuisance.all"),
                name="statistic.full.chi2p_unbiased",
            )

            Product.replicate(
                parameters.get_value("all.stats.pearson"),
                outputs.get_value("statistic.full.chi2p_covmat_variable"),
                name="statistic.helper.pearson",
            )
            Product.replicate(
                parameters.get_value("all.stats.neyman"),
                outputs.get_value("statistic.full.chi2n_covmat"),
                name="statistic.helper.neyman",
            )
            # (2-4) CNP covmat
            Sum.replicate(
                outputs.get_value("statistic.helper.pearson"),
                outputs.get_value("statistic.helper.neyman"),
                name="statistic.full.chi2cnp_covmat",
            )
            # fmt: on

        self._setup_labels()

        # Ensure stem nodes are calculated
        self._touch()

    @staticmethod
    def _create_random_generator(seed: int) -> Generator:
        from numpy.random import MT19937, SeedSequence

        (sequence,) = SeedSequence(seed).spawn(1)
        algo = MT19937(seed=sequence.spawn(1)[0])
        return Generator(algo)

    def _touch(self):
        for output in (
            self.storage["outputs"].get_dict("eventscount.final.detector").walkvalues()
        ):
            output.touch()

    def update_frozen_nodes(self):
        for nodes in self._frozen_nodes.values():
            for node in nodes:
                node.unfreeze()
                node.touch()

    def update_covariance_matrix(self):
        self._covariance_matrix.update_matrices()

    def set_parameters(
        self,
        parameter_values: (
            Mapping[str, float | str] | Sequence[tuple[str, float | int]]
        ) = (),
    ):
        parameters_storage = self.storage("parameters.all")
        if isinstance(parameter_values, Mapping):
            iterable = parameter_values.items()
        else:
            iterable = parameter_values

        for parname, svalue in iterable:
            value = float(svalue)
            par = parameters_storage[parname]
            par.push(value)
            print(f"Set {parname}={svalue}")

    def next_sample(
        self, *, mc_parameters: bool = True, mc_statistics: bool = True
    ) -> None:
        if mc_parameters:
            self.storage.get_value("nodes.mc.parameters.toymc").next_sample()
            self.storage.get_value("nodes.mc.parameters.inputs").touch()

        if mc_statistics:
            self.storage.get_value("nodes.data.pseudo.self").next_sample()

        if mc_parameters:
            self.storage.get_value("nodes.mc.parameters.toymc").reset()
            self.storage.get_value("nodes.mc.parameters.inputs").touch()

    @staticmethod
    def systematic_uncertainties_groups() -> dict[str, str]:
        return dict(_SYSTEMATIC_UNCERTAINTIES_GROUPS)

    def _setup_labels(self):
        labels = LoadYaml(relpath(__file__.replace(".py", "_labels.yaml")))

        processed_keys_set = set()
        self.storage("nodes").read_labels(labels, processed_keys_set=processed_keys_set)
        self.storage("outputs").read_labels(
            labels, processed_keys_set=processed_keys_set
        )
        self.storage("inputs").remove_connected_inputs()
        self.storage.read_paths(index=self.index)
        self.graph.build_index_dict(self.index)

        labels_mk = NestedMKDict(labels, sep=".")
        if not self._strict:
            return

        for key in processed_keys_set:
            labels_mk.delete_with_parents(key)

        if not labels_mk:
            return

        unused_keys = list(labels_mk.walkjoinedkeys())
        may_ignore = {
            "detector.lsnl.curves.evis_coarse_monotonous_scaled.group.text",
            "detector.lsnl.indexer_bwd.group.text",
            "detector.lsnl.interpolated_bwd.group.text",
        }
        for key in may_ignore:
            with suppress(ValueError):
                unused_keys.remove(key)

        if not unused_keys:
            return

        raise RuntimeError(
            f"The following label groups were not used: {', '.join(unused_keys)}"
        )


# TODO: remove
# vim: tw=88 fo-=t
