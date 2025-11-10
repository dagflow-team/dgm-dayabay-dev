from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from os.path import relpath
from pathlib import Path
from typing import TYPE_CHECKING

from dag_modelling.core import Graph, NodeStorage
from dag_modelling.tools.logger import INFO, logger
from nested_mapping import NestedMapping
from numpy import ndarray
from numpy.random import Generator
from pandas import DataFrame

# pyright: reportUnusedExpression=false

if TYPE_CHECKING:
    from typing import KeysView, Literal

    from dag_modelling.core.meta_node import MetaNode
    from numpy.typing import NDArray

# Define a dictionary of groups of nuisance parameters in a format `name: path`,
# where path denotes the location of the parameters in the storage.
_SYSTEMATIC_UNCERTAINTIES_GROUPS = {
    "survival_probability": "survival_probability",
    "eres": "detector.eres",
    "lsnl": "detector.lsnl_scale_a",
    "iav": "detector.iav_offdiag_scale_factor",
    "detector_relative": "detector.detector_relative",
    "energy_per_fission": "reactor.energy_per_fission",
    "thermal_power": "reactor.thermal_power_scale",
    "snf": "reactor.snf_scale",
    "neq": "reactor.nonequilibrium_scale",
    "fission_fraction": "reactor.fission_fraction_scale",
    "background_rate": "background",
    "hm_corr": "reactor_antineutrino.spectrum_uncertainty.corr",
    "hm_uncorr": "reactor_antineutrino.spectrum_uncertainty.uncorr",
    "absolute_efficiency": "detector.detector_absolute",
}


class model_dayabay:
    """The Daya Bay model implementation version v1. A candidate for the public release.

    Purpose:
        - Replace power/fision fractions input with neutrino rate per reactor.

    Attributes
    ----------
    storage : NodeStorage
        Nested dictionary with model elements: nodes, parameters, etc.
    graph : Graph
        Graph instance.
    index : dict[str, tuple[str, ...]]
        Dictionary with all possible names for replicated items, e.g.
        "detector": ("AD11", "AD12", ...); reactor: ("R1", ...); ...
        index is setup within the model.
    combinations : dict[str, tuple[tuple[str, ...], ...]]
        Lists of all combinations of values of 1 and more indices,
        e.g. detector, detector/period, reator/isotope, reactor/isotope/period, etc.
    spectrum_correction_interpolation_mode : str, default="exponential"
        Mode of how the parameters of the free spectrum model
        are treated:
            - "exponential": pᵢ=0 by default, S(Eᵢ) is
              multiplied by exp(pᵢ) the correction is always
              positive, but nonlinear.
            - "linear": pᵢ=0 by default, S(Eᵢ) is multiplied by
              1+pᵢ the correction may be negative, but is always
              linear.
    spectrum_correction_location : str, default="before-integration"
        Place, where the spectrum correction is applied:
            - "before-integration": the antineutrino spectrum of each isotope is
              corrected, domain — neutrino energy.
            - "after-integration": the expected spectrum of each detector during each
              period is corrected (before detector effects), domain: deposited energy
              (Edep). The conversion from Eν to Edep is done approximately by a constant
              shift.
    concatenation_mode : str, default="detector_period"
        Choses the observation to be analyzed:
            - "detector_period" - concatenation of observations at
              each detector at each period,
            - "detector" - concatenation of observations at each
              detector (combined for all period).
    monte_carlo_mode : str, default="asimov".
        The Monte-Carlo mode for pseudo-data:
            - "asimov" - Asimov, no fluctuations,
            - "normal-stats" - normal fluctuations with statistical,
              errors,
            - "poisson" - Poisson fluctuations.
    covariance_groups: list[Literal["survival_probability", "eres", "lsnl", "iav", "detector_relative",
        "energy_per_fission", "thermal_power", "snf", "neq", "fission_fraction", "background_rate",
        "hm_corr", "hm_uncorr"]], default=[]
        List of nuicance groups to be added to covariance matrix. If no parameters passed,
        full covariance matrix will be created.
    pull_groups: list[Literal["survival_probability", "eres", "lsnl", "iav", "detector_relative",
        "energy_per_fission", "thermal_power", "snf", "neq", "fission_fraction", "background_rate",
        "hm_corr", "hm_uncorr"]], default=[]
        List of nuicance groups to be added to `nuisance.extra_pull`. If no parameters passed, it will add all nuisance parameters.
    final_erec_bin_edges : Path | Sequence[int | float] | NDArray | None, default=None
        Text file with bin edges for the final binning or the edges themselves, which is relevant for the χ² calculation.
    is_absolute_efficiency_fixed : bool, default=True
        Switch detector absolute correlated efficiency from fixed to constrained parameter.
    path_data : Path
        Path to the data.
    leading_mass_splitting_3l_name: Literal["DeltaMSq32", "DeltaMSq31"], default="DeltaMSq32"
        Leading mass splitting.

    Technical attributes
    --------------------
    _source_type : str, default="hdf5"
        Type of the data to read ("tsv", "hdf5", "root" or "npz").
    _strict : bool, default=True
        Strict mode. Stop execution if:
            - the model is not complete,
            - any labels were not applied.
    _close : bool, default=True
        if True the graph is closed and memory is allocated
        may be used to debug corrupt model.
    _random_generator : Generator
        numpy random generator to be used for ToyMC.
    _covariance_matrix : MetaNode
        covariance matrix, computed on this model.
    _frozen_nodes : dict[str, tuple]
        storage with nodes, which are being fixed at their values and
        require manual intervention in order to be recalculated.
    """

    __slots__ = (
        "storage",
        "graph",
        "index",
        "combinations",
        "_path_data",
        "_leading_mass_splitting_3l_name",
        "spectrum_correction_interpolation_mode",
        "spectrum_correction_location",
        "concatenation_mode",
        "monte_carlo_mode",
        "_covariance_groups",
        "_pull_groups",
        "_is_absolute_efficiency_fixed",
        "_arrays_dict",
        "_source_type",
        "_strict",
        "_close",
        "_process_labels",
        "_covariance_matrix",
        "_frozen_nodes",
        "_random_generator",
    )

    storage: NodeStorage
    graph: Graph
    index: dict[str, tuple[str, ...]]
    combinations: dict[str, tuple[tuple[str, ...], ...]]
    _path_data: Path
    _leading_mass_splitting_3l_name: Literal["DeltaMSq32", "DeltaMSq31"]
    spectrum_correction_interpolation_mode: Literal["linear", "exponential"]
    spectrum_correction_location: Literal["before-integration", "after-integration"]
    concatenation_mode: Literal["detector", "detector_period"]
    monte_carlo_mode: Literal["asimov", "normal-stats", "poisson"]
    _arrays_dict: dict[str, Path | NDArray | None]
    _covariance_groups: (
        Sequence[
            Literal[
                "survival_probability",
                "eres",
                "lsnl",
                "iav",
                "detector_relative",
                "energy_per_fission",
                "thermal_power",
                "snf",
                "neq",
                "fission_fraction",
                "background_rate",
                "hm_corr",
                "hm_uncorr",
            ]
        ]
        | KeysView
    )
    _pull_groups: Sequence[
        Literal[
            "survival_probability",
            "eres",
            "lsnl",
            "iav",
            "detector_relative",
            "energy_per_fission",
            "thermal_power",
            "snf",
            "neq",
            "fission_fraction",
            "background_rate",
            "hm_corr",
            "hm_uncorr",
        ]
    ]
    _arrays_dict: dict[str, Path | NDArray | None]
    _is_absolute_efficiency_fixed: bool
    _source_type: Literal["tsv", "hdf5", "root", "npz"]
    _strict: bool
    _close: bool
    _process_labels: bool
    _random_generator: Generator
    _covariance_matrix: MetaNode
    _frozen_nodes: dict[str, tuple]

    def __init__(
        self,
        *,
        strict: bool = True,
        close: bool = True,
        process_labels: bool = True,
        override_indices: Mapping[str, Sequence[str]] = {},
        override_cfg_files: Mapping[str, str] = {},
        leading_mass_splitting_3l_name: Literal["DeltaMSq32", "DeltaMSq31"] = "DeltaMSq32",
        spectrum_correction_interpolation_mode: Literal["linear", "exponential"] = "exponential",
        spectrum_correction_location: Literal[
            "before-integration", "after-integration"
        ] = "before-integration",
        seed: int = 0,
        monte_carlo_mode: Literal["asimov", "normal-stats", "poisson"] = "asimov",
        concatenation_mode: Literal["detector", "detector_period"] = "detector_period",
        parameter_values: dict[str, float | str] = {},
        path_data: str | Path | None = None,
        antineutrino_spectrum_segment_edges: str | Path | None = None,
        final_erec_bin_edges: str | Path | Sequence[int | float] | NDArray | None = None,
        covariance_groups: (
            Sequence[
                Literal[
                    "survival_probability",
                    "eres",
                    "lsnl",
                    "iav",
                    "detector_relative",
                    "energy_per_fission",
                    "thermal_power",
                    "snf",
                    "neq",
                    "fission_fraction",
                    "background_rate",
                    "hm_corr",
                    "hm_uncorr",
                ]
            ]
            | KeysView
        ) = [],
        pull_groups: Sequence[
            Literal[
                "survival_probability",
                "eres",
                "lsnl",
                "iav",
                "detector_relative",
                "energy_per_fission",
                "thermal_power",
                "snf",
                "neq",
                "fission_fraction",
                "background_rate",
                "hm_corr",
                "hm_uncorr",
            ]
        ] = [],
        is_absolute_efficiency_fixed: bool = True,
    ):
        """Model initialization.

        Parameters
        ----------
        seed: int
              random seed to be passed to random generator for ToyMC
        override_indices : dict[str, Sequence[str]]
                           dictionary with indices to override self.index.
                           may be used to reduce the number of detectors or reactors in
                           the model

        for the description of other parameters, see description of the class.
        """
        self._strict = strict
        self._close = close
        self._process_labels = process_labels

        assert spectrum_correction_interpolation_mode in {"linear", "exponential"}
        assert spectrum_correction_location in {
            "before-integration",
            "after-integration",
        }
        assert monte_carlo_mode in {"asimov", "normal-stats", "poisson"}
        assert concatenation_mode in {"detector", "detector_period"}

        self._is_absolute_efficiency_fixed = is_absolute_efficiency_fixed

        for covariance_group in covariance_groups:
            assert covariance_group in self.systematic_uncertainties_groups

        if not covariance_groups:
            covariance_groups = self.systematic_uncertainties_groups.keys()

        covariance_groups_set = set(covariance_groups)
        pull_groups_set = set(pull_groups)
        pull_covariance_intersect = pull_groups_set.intersection(covariance_groups_set)
        if pull_covariance_intersect:
            logger.log(
                INFO,
                "Pull groups intersect with covariance groups: " f"{pull_covariance_intersect}",
            )

        systematic_groups_pull_covariance_intersect = (
            set(self.systematic_uncertainties_groups.keys())
            .difference(covariance_groups_set)
            .difference(pull_groups_set)
        )
        if systematic_groups_pull_covariance_intersect:
            logger.log(
                INFO,
                "Several systematic groups are missing from `pull_groups` and `covariance_groups`: "
                f"{systematic_groups_pull_covariance_intersect}",
            )

        from .tools.validate_load_array import validate_load_array

        self._arrays_dict = {
            "antineutrino_spectrum_segment_edges": validate_load_array(
                antineutrino_spectrum_segment_edges
            ),
            "final_erec_bin_edges": validate_load_array(final_erec_bin_edges),
        }

        if antineutrino_spectrum_segment_edges is not None and override_cfg_files.get(
            "antineutrino_spectrum_segment_edges"
        ):
            raise RuntimeError(
                "Antineutrino bin edges couldn't be overloaded via `antineutrino_spectrum_segment_edges` and `override_cfg_files` simultaneously"
            )

        if final_erec_bin_edges is not None and override_cfg_files.get("final_erec_bin_edges"):
            raise RuntimeError(
                "Final Erec bin edges couldn't be overloaded via `final_erec_bin_edges` and `override_cfg_files` simultaneously"
            )

        match path_data:
            case str() | Path():
                self._path_data = Path(path_data)
            case None:
                self._path_data = Path("data/dayabay-v1a_neutrino_rate_total/hdf5")
            case _:
                raise RuntimeError(f"Unsupported path option: {path_data}")

        from .tools.validate_dataset import validate_dataset_get_source_type

        self._source_type = validate_dataset_get_source_type(
            self._path_data, "dataset_info.yaml", version_min="1.0.0", version_max="2.0.0"
        )

        self.storage = NodeStorage()
        self._leading_mass_splitting_3l_name = leading_mass_splitting_3l_name
        self.spectrum_correction_interpolation_mode = spectrum_correction_interpolation_mode
        self.spectrum_correction_location = spectrum_correction_location
        self.concatenation_mode = concatenation_mode
        self.monte_carlo_mode = monte_carlo_mode
        self._covariance_groups = covariance_groups
        self._pull_groups = pull_groups

        from .tools.validate_load_array import validate_load_array

        self._arrays_dict = {
            "antineutrino_spectrum_segment_edges": validate_load_array(
                antineutrino_spectrum_segment_edges
            ),
            "final_erec_bin_edges": validate_load_array(final_erec_bin_edges),
        }
        self._random_generator = self._create_random_generator(seed)

        logger.log(INFO, f"Model version: {type(self).__name__}")
        logger.log(INFO, f"Data path: {self.path_data!s}")
        logger.log(INFO, f"Concatenation mode: {self.concatenation_mode}")
        logger.log(
            INFO,
            f"Spectrum correction mode: {self.spectrum_correction_interpolation_mode}",
        )
        logger.log(
            INFO,
            f"Spectrum correction location: {self.spectrum_correction_location.replace('-', ' ')}",
        )
        assert self.spectrum_correction_interpolation_mode in {"linear", "exponential"}
        assert self.spectrum_correction_location in {
            "before-integration",
            "after-integration",
        }

        self._frozen_nodes = {}
        self.combinations = {}

        override_indices = {k: tuple(v) for k, v in override_indices.items()}

        cfg_file_mapping = self._build_cfg_file_mapping(override_cfg_files)

        self.build(cfg_file_mapping, override_indices)

        if parameter_values:
            self.set_parameters(parameter_values)

    def _build_cfg_file_mapping(self, override_cfg_files: Mapping[str, str]) -> dict[str, Path]:
        path_data = self.path_data
        path_parameters = path_data / "parameters"

        # Dataset items
        cfg_file_mapping = {
            "antineutrino_spectrum_segment_edges": path_parameters
            / "reactor_antineutrino_spectrum_edges.tsv",
            "final_erec_bin_edges": path_parameters / "final_erec_bin_edges.tsv",
            "parameters.survival_probability": path_parameters / "survival_probability.yaml",
            "parameters.survival_probability_solar": path_parameters
            / "survival_probability_solar.yaml",
            "parameters.survival_probability_constants": path_parameters
            / "survival_probability_constants.yaml",
            "parameters.pdg_constants": path_parameters / "pdg2024.yaml",
            "parameters.ibd_constants": path_parameters / "ibd_constants.yaml",
            "parameters.conversion_thermal_power": path_parameters / "conversion_thermal_power.py",
            "parameters.conversion_survival_probability": path_parameters
            / "conversion_survival_probability_argument.py",
            "parameters.baselines": path_parameters / "baselines.yaml",
            "parameters.detector_normalization": path_parameters / "detector_normalization.yaml",
            "parameters.detector_efficiency": path_parameters / "detector_efficiency.yaml",
            "parameters.detector_n_protons_nominal": path_parameters
            / "detector_n_protons_nominal.yaml",
            "parameters.detector_n_protons_correction": path_parameters
            / "detector_n_protons_correction.yaml",
            "parameters.detector_eres": path_parameters / "detector_eres.yaml",
            "parameters.detector_lsnl": path_parameters / "detector_lsnl.yaml",
            "parameters.detector_iav_offdiag_scale": path_parameters
            / "detector_iav_offdiag_scale.yaml",
            "parameters.detector_relative": path_parameters / "detector_relative.yaml",
            "parameters.detector_absolute": path_parameters / "extra/detector_absolute.yaml",
            "parameters.neutrinos_per_fission": path_parameters
            / "neutrinos_per_fission_huber_mueller.yaml",
            "parameters.reactor_thermal_power_nominal": path_parameters
            / "reactor_thermal_power_nominal.yaml",
            "parameters.reactor_thermal_power_uncertainty": path_parameters
            / "reactor_thermal_power_uncertainty.yaml",
            "parameters.reactor_energy_per_fission": path_parameters
            / "reactor_energy_per_fission.yaml",
            "parameters.reactor_snf": path_parameters / "reactor_snf.yaml",
            "parameters.reactor_nonequilibrium_correction": path_parameters
            / "reactor_nonequilibrium_correction.yaml",
            "parameters.reactor_fission_fractions": path_parameters
            / "reactor_fission_fractions.yaml",
            "parameters.reactor_fission_fractions_scale": path_parameters
            / "reactor_fission_fraction_scale.yaml",
            "parameters.background_rate_scale_accidentals": path_parameters
            / "background_rate_scale_accidentals.yaml",
            "parameters.background_rates_uncorrelated": path_parameters
            / f"background_rates_uncorrelated.yaml",
            "parameters.background_rates_correlated": path_parameters
            / f"background_rates_correlated.yaml",
            "parameters.background_rate_uncertainty_scale_amc": path_parameters
            / "background_rate_uncertainty_scale_amc.yaml",
            "parameters.background_rate_uncertainty_scale_site": path_parameters
            / f"background_rate_uncertainty_scale_site.yaml",
            "reactor_antineutrino_spectra": path_data
            / f"reactor_antineutrino_spectra_hm.{self.source_type}",
            "reactor_antineutrino_spectra_uncertainties": path_data
            / f"reactor_antineutrino_spectra_hm_uncertainties.{self.source_type}",
            "nonequilibrium_correction": path_data
            / f"nonequilibrium_correction.{self.source_type}",
            "snf_correction": path_data / f"snf_correction.{self.source_type}",
            "daily_detector_data": path_data
            / f"dayabay_dataset/dayabay_daily_detector_data.{self.source_type}",
            "daily_antineutrino_rate_data": path_data / f"neutrino_rate.{self.source_type}",
            "iav_matrix": path_data / f"detector_iav_matrix.{self.source_type}",
            "lsnl_curves": path_data / f"detector_lsnl_curves.{self.source_type}",
            "background_spectra": path_data / "dayabay_dataset/dayabay_background_spectra_{}."
            f"{self.source_type}",
            "dataset": path_data / "dayabay_dataset/dayabay_ibd_spectra_{}." f"{self.source_type}",
        }
        for cfg_name, path in override_cfg_files.items():
            cfg_file_mapping.update({cfg_name: Path(path)})

        for array_name, array in self._arrays_dict.items():
            match array:
                case ndarray():
                    del cfg_file_mapping[array_name]
                case Path():
                    cfg_file_mapping[array_name] = array

        return cfg_file_mapping

    @property
    def source_type(self) -> Literal["tsv", "hdf5", "npz", "root"]:
        return self._source_type

    @property
    def path_data(self) -> Path:
        return self._path_data

    @property
    def nbins(self) -> int:
        return self.storage["outputs.eventscount.final.concatenated.selected"].data.shape[0]

    def build(
        self, cfg_file_mapping: dict[str, Path], override_indices: dict[str, tuple[str, ...]] = {}
    ):
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
        from dag_modelling.bundles.file_reader import FileReader
        from dag_modelling.bundles.load_array import load_array
        from dag_modelling.bundles.load_graph import load_graph, load_graph_data
        from dag_modelling.bundles.load_hist import load_hist
        from dag_modelling.bundles.load_parameters import load_parameters
        from dag_modelling.bundles.load_record import load_record_data
        from dag_modelling.bundles.make_y_parameters_for_x import make_y_parameters_for_x
        from dag_modelling.lib.arithmetic import (
            Abs,
            Difference,
            Division,
            Product,
            ProductShiftedScaled,
            Sum,
        )
        from dag_modelling.lib.axis import BinCenter, BinWidth
        from dag_modelling.lib.common import Array, Concatenation, Proxy, View
        from dag_modelling.lib.exponential import Exp
        from dag_modelling.lib.hist import AxisDistortionMatrixPointwise, Rebin
        from dag_modelling.lib.integration import Integrator
        from dag_modelling.lib.interpolation import Interpolator
        from dag_modelling.lib.linalg import Cholesky, VectorMatrixProduct
        from dag_modelling.lib.normalization import RenormalizeDiag
        from dag_modelling.lib.parameters import ParArrayInput
        from dag_modelling.lib.physics import EnergyResolution
        from dag_modelling.lib.statistics import (
            Chi2,
            CNPStat,
            CovarianceMatrixGroup,
            LogPoissonRatio,
            LogProdDiag,
            MonteCarlo,
        )
        from dag_modelling.lib.summation import ArraySum, SumMatOrDiag
        from dgm_reactor_neutrino import (
            IBDXsecVBO1Group,
            InverseSquareLaw,
            NueSurvivalProbability,
        )
        from nested_mapping.tools import remap_items
        from numpy import linspace

        from ...bundles.refine_reactor_data_variable_periods import refine_reactor_data
        from ...bundles.sync_reactor_detector_data import sync_reactor_detector_data
        from .bundles.refine_detector_data import refine_detector_data
        from .bundles.refine_lsnl_data import refine_lsnl_data
        from .bundles.refine_neutrino_rate_data import refine_neutrino_rate_data
        from .bundles.sync_neutrino_rate_detector_data import sync_neutrino_rate_detector_data

        storage = self.storage

        # Read Eν edges for the parametrization of free antineutrino spectrum model
        # Loads the python file and returns variable "edges", which should be defined
        # in the file and has type `ndarray`.
        if isinstance(self._arrays_dict["antineutrino_spectrum_segment_edges"], ndarray):
            antineutrino_model_edges = self._arrays_dict["antineutrino_spectrum_segment_edges"]
            logger.info(
                f"Antineutrino model bin edges passed via argument: {antineutrino_model_edges!s}"
            )
        else:
            antineutrino_model_edges = FileReader.record[
                cfg_file_mapping["antineutrino_spectrum_segment_edges"]
            ]["E_neutrino_MeV"]

        # Provide some convenience substitutions for labels
        index_names = {
            "U235": "²³⁵U",
            "U238": "²³⁸U",
            "Pu239": "²³⁹Pu",
            "Pu241": "²⁴¹Pu",
        }
        site_arrangement = {
            "EH1": ("AD11", "AD12"),
            "EH2": ("AD21", "AD22"),
            "EH3": ("AD31", "AD32", "AD33", "AD34"),
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
            # A subset of detector names, which are considered for the χ² calculation.
            # Will be applied for selection of the histograms for the model prediction and real data.
            "detector_selected": (
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
            #     - accidentals: accidental coincidences
            #     - lithium_helium: ⁹Li and ⁸He related events
            #     - fast_neutrons: fast neutrons, includes also and muon decay background
            #     - amc: ²⁴¹Am¹³C calibration source related background
            #     - alpha_neutron: ¹³C(α,n)¹⁶O background
            "background": (
                "accidentals",
                "lithium_helium",
                "fast_neutrons",
                "amc",
                "alpha_neutron",
            ),
            "background_stable": (
                "lithium_helium",
                "fast_neutrons",
                "amc",
                "alpha_neutron",
            ),  # TODO: doc
            "background_site_correlated": ("lithium_helium", "fast_neutrons"),  # TODO: doc
            "background_not_site_correlated": ("accidentals", "amc", "alpha_neutron"),  # TODO: doc
            "background_not_correlated": ("accidentals", "alpha_neutron"),  # TODO: doc
            # Experimental sites
            "site": ("EH1", "EH2", "EH3"),
            # Fissile isotopes
            "isotope": ("U235", "U238", "Pu239", "Pu241"),
            # Fissile isotopes, which spectrum requires Non-Equilibrium correction to be
            # applied
            "isotope_neq": ("U235", "Pu239", "Pu241"),
            # Nuclear reactors
            "reactor": ("R1", "R2", "R3", "R4", "R5", "R6"),
            # Sources of antineutrinos:
            #     - "nu_main": for antineutrinos from reactor cores with no
            #                  Non-Equilibrium correction applied
            #     - "nu_neq": antineutrinos from Non-Equilibrium correction
            #     - "nu_snf": antineutrinos from Spent Nuclear Fuel
            "antineutrino_source": ("nu_main", "nu_neq", "nu_snf"),
            # Model related antineutrino spectrum correction type:
            #     - uncorrelated
            #     - correlated
            "antineutrino_unc": ("uncorr", "corr"),
            # Part of the Liquid scintillator non-linearity (LSNL) parametrization
            "lsnl": ("nominal", "pull0", "pull1", "pull2", "pull3"),
            # Nuisance related part of the Liquid scintillator non-linearity (LSNL)
            # parametrization
            "lsnl_nuisance": ("pull0", "pull1", "pull2", "pull3"),
            # Free antineutrino spectrum parameter names: one parameter for each edge
            # from `antineutrino_model_edges`
            "spec": tuple(f"spec_scale_{i:02d}" for i in range(len(antineutrino_model_edges))),
        }

        # Define isotope names in lower case
        index["isotope_lower"] = tuple(isotope.lower() for isotope in index["isotope"])

        # Optionally override (reduce) indices
        for index_name, index_values in override_indices.items():
            if index_name not in index:
                raise RuntimeError(f"Invalide index {index_name} found when overriding indices")
            index[index_name] = index_values

        # Check that the detector indices are consistent.
        detectors = index["detector"]
        detectors_selected = set(index["detector_selected"])
        assert detectors_selected.issubset(
            detectors
        ), f"index['detector_selected'] is not consistent with index['detector']: {detectors_selected} ⊈ {detectors}"
        index["detector_excluded"] = tuple(d for d in detectors if not d in detectors_selected)

        # Check there are now overlaps
        index_all = index["isotope"] + index["detector"] + index["reactor"] + index["period"]
        set_all = set(index_all)
        if len(index_all) != len(set_all):
            raise RuntimeError("Repeated indices")

        # Collection combinations between 2 and more indices. Ensure some combinations,
        # e.g. detectors not present at certain periods, are excluded.
        # For example, combinations["reactor.detector"] contains:
        # (("R1", "AD11"), ("R1", "AD12"), ..., ("R2", "AD11"), ...)
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
            "site.period",
            "period.detector",
            "antineutrino_unc.isotope",
            "background.detector",
            "background_stable.detector",
            "background.detector.period",
            "background.period.detector",
            "background_stable.detector.period",
            "background_site_correlated.detector.period",
            "background_not_site_correlated.detector.period",
            "background_not_correlated.detector.period",
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

        # Special treatment is needed for combinations of antineutrino_source and isotope as
        # nu_neq is related to only a fraction of isotopes, while nu_snf does not index
        # isotopes at all
        combinations["antineutrino_source.reactor.isotope.detector"] = (
            tuple(("nu_main",) + cmb for cmb in combinations["reactor.isotope.detector"])
            + tuple(("nu_neq",) + cmb for cmb in combinations["reactor.isotope_neq.detector"])
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
            # - storage["parameters.all.survival_probability.SinSq2Theta12"] - neutrino oscillation
            #   parameter sin²2θ₁₂
            # - storage["parameters.constrained.survival_probability.SinSq2Theta12"] - same neutrino
            #   oscillation parameter sin²2θ₁₂ in the list of constrained parameters.
            # - storage["parameters.normalized.survival_probability.SinSq2Theta12"] - shadow
            #   (nuisance) parameter for sin²2θ₁₂.
            #
            # The constrained parameter has fields `value`, `normvalue`, `central`, and
            # `sigma`, which could be read to get the current value of the parameter,
            # normalized value, central value, and uncertainty. The assignment to the
            # fields changes the values. Additionally fields `sigma_relative` and
            # `sigma_percent` may be used to get and set the relative uncertainty.
            # ```python
            # p = storage["parameters.all.survival_probability.SinSq2Theta12"]
            # print(p)        # print the description
            # print(p.value)  # print the current value
            # p.value = 0.8   # set the value to 0.8 - affects the model
            # p.central = 0.7 # set the central value to 0.7 - affects the nuisance term
            # p.normvalue = 1 # set the value to central+1sigma
            # ```
            #
            # The non-constrained parameter lacks `central`, `sigma`, `normvalue`, etc
            # fields and is controlled only by `value`. The normalized parameter does
            # have `central` and `sigma` fields, but they are read only. The effect of
            # changing `value` field of the normalized parameter is the same as changing
            # `normvalue` field of its corresponding parameter.
            #
            # ```python
            # np = storage["parameters.normalized.survival_probability.SinSq2Theta12"]
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
            load_parameters(
                path="survival_probability",
                load=cfg_file_mapping["parameters.survival_probability"],
            )

            load_parameters(
                path="survival_probability",
                load=cfg_file_mapping["parameters.survival_probability_solar"],
                joint_nuisance=True,
            )
            load_parameters(
                path="survival_probability",
                load=cfg_file_mapping["parameters.survival_probability_constants"],
            )

            # The parameters are located in "parameters.survival_probability" folder as defined by
            # the `path` argument.
            # The annotated table with values may be then printed for any storage as
            # ```python
            # print(storage["parameters.all.survival_probability"].to_table())
            # print(storage.get_dict("parameters.all.survival_probability").to_table())
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
            load_parameters(path="ibd", load=cfg_file_mapping["parameters.pdg_constants"])
            load_parameters(path="ibd.csc", load=cfg_file_mapping["parameters.ibd_constants"])

            # Load the conversion constants from metric to natural units:
            # - reactor thermal power
            # - the argument of oscillation probability
            # `scipy.constants` are used to provide the numbers.
            # There are no constants, except maybe 1, 1/3 and π, defined within the
            # code. All the numbers are read based on the configuration files.
            load_parameters(
                path="conversion", load=cfg_file_mapping["parameters.conversion_thermal_power"]
            )

            load_parameters(
                path="conversion",
                load=cfg_file_mapping["parameters.conversion_survival_probability"],
            )

            # TODO
            load_parameters(
                path="reactor", load=cfg_file_mapping["parameters.neutrinos_per_fission"]
            )

            # Load reactor-detector baselines
            load_parameters(load=cfg_file_mapping["parameters.baselines"])

            # IBD and detector normalization parameters:
            # - free global IBD normalization factor
            # - fixed detector efficiency (variation is managed by uncorrelated
            #   "detector_relative.efficiency_factor")
            # - fixed correction to the number of protons in each detector
            load_parameters(
                path="detector", load=cfg_file_mapping["parameters.detector_normalization"]
            )
            load_parameters(
                path="detector", load=cfg_file_mapping["parameters.detector_efficiency"]
            )
            load_parameters(
                path="detector", load=cfg_file_mapping["parameters.detector_n_protons_nominal"]
            )
            load_parameters(
                path="detector", load=cfg_file_mapping["parameters.detector_n_protons_correction"]
            )

            # Detector energy scale parameters:
            # - constrained correlated between detectors energy resolution parameters
            # - constrained correlated between detectors Liquid Scintillator
            #   Non-Linearity (LSNL) parameters
            # - constrained uncorrelated between detectors energy distortion related to
            #   Inner Acrylic Vessel
            load_parameters(path="detector", load=cfg_file_mapping["parameters.detector_eres"])
            load_parameters(
                path="detector",
                load=cfg_file_mapping["parameters.detector_lsnl"],
                replicate=index["lsnl_nuisance"],
            )
            load_parameters(
                path="detector",
                load=cfg_file_mapping["parameters.detector_iav_offdiag_scale"],
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
                load=cfg_file_mapping["parameters.detector_relative"],
                replicate=index["detector"],
                keys_order=(
                    ("pargroup", "par", "detector"),
                    ("pargroup", "detector", "par"),
                ),
            )

            # Absolute correlated between detectors efficiency factor
            load_parameters(
                path="detector",
                load=cfg_file_mapping["parameters.detector_absolute"],
                state="fixed" if self._is_absolute_efficiency_fixed else "variable",
            )
            # By default extra index is appended at the end of the key (path). A
            # `keys_order` argument is used to change the order of the keys from
            # group.par.detector to group.detector.par so it is easier to access both
            # the parameters of a single detector.

            # Load reactor related parameters:
            # - constrained nominal thermal power
            # - constrained thermal power uncertainty, uncorrelated between reactors
            # - constrained mean energy release per fission
            # - fixed values of mean fission fractions
            # - constrained Non-EQuilibrium (NEQ) correction scale
            # - constrained Spent Nuclear Fuel (SNF) scale
            load_parameters(
                path="reactor",
                load=cfg_file_mapping["parameters.reactor_thermal_power_nominal"],
                replicate=index["reactor"],
            )
            load_parameters(
                path="reactor",
                load=cfg_file_mapping["parameters.reactor_thermal_power_uncertainty"],
                replicate=index["reactor"],
            )
            load_parameters(
                path="reactor", load=cfg_file_mapping["parameters.reactor_energy_per_fission"]
            )
            load_parameters(
                path="reactor",
                load=cfg_file_mapping["parameters.reactor_fission_fractions"],
            )
            load_parameters(
                path="reactor",
                load=cfg_file_mapping["parameters.reactor_snf"],
                replicate=index["reactor"],
            )
            load_parameters(
                path="reactor",
                load=cfg_file_mapping["parameters.reactor_nonequilibrium_correction"],
                replicate=combinations["reactor.isotope_neq"],
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
                load=cfg_file_mapping["parameters.reactor_fission_fractions_scale"],
                replicate=index["reactor"],
                keys_order=(
                    ("par", "isotope", "reactor"),
                    ("par", "reactor", "isotope"),
                ),
            )

            # Finally the constrained background rates are loaded. They include the
            # rates and uncertainties for 5 sources of background events for 6-8
            # detectors during 3 periods of data taking.
            load_parameters(
                path="background.rate_scale",
                load=cfg_file_mapping["parameters.background_rate_scale_accidentals"],
                replicate=combinations["period.detector"],
            )
            load_parameters(
                path="background.rate",
                load=cfg_file_mapping["parameters.background_rates_uncorrelated"],
            )
            load_parameters(
                path="background.rate",
                load=cfg_file_mapping["parameters.background_rates_correlated"],
                sigma_visible=True,
            )
            load_parameters(
                path="background.uncertainty_scale",
                load=cfg_file_mapping["parameters.background_rate_uncertainty_scale_amc"],
            )
            load_parameters(
                path="background.uncertainty_scale_by_site",
                load=cfg_file_mapping["parameters.background_rate_uncertainty_scale_site"],
                replicate=combinations["site.period"],
            )

            # Additionally a few constants are provided.
            # A constant to convert seconds to days for the backgrounds estimation
            load_parameters(
                format="value",
                state="fixed",
                parameters={
                    "conversion": {
                        "seconds_in_day": (60 * 60 * 24),
                        "seconds_in_day_inverse": 1 / (60 * 60 * 24),
                    }
                },
                labels={
                    "conversion": {
                        "seconds_in_day": "Number of seconds in a day",
                        "seconds_in_day_inverse": "Fraction of a day in a second",
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
            nodes = storage.create_child("nodes")
            inputs = storage.create_child("inputs")
            outputs = storage.create_child("outputs")
            data = storage.create_child("data")
            parameters = storage.get_dict("parameters")
            parameters_nuisance_normalized = storage.get_dict("parameters.normalized")

            # In this section the actual parts of the calculation are created as nodes.
            # First of all the binning is defined for the histograms.
            # - internal binning for the integration: 240 bins of 50 keV from 0 to 241.
            # - final binning for the statistical analysis: 20 keV from 1.3 MeV to 2 MeV
            #   with two wide bins below from 0.7 MeV and above up to 12 MeV.
            # - cosθ (positron angle) edges [-1,1] are defined explicitly for the
            #   integration of the Inverse Beta Decay (IBD) cross section.
            in_edges_fine = linspace(0, 12, 241)
            in_edges_costheta = [-1, 1]

            if isinstance(self._arrays_dict["final_erec_bin_edges"], ndarray):
                in_edges_final = self._arrays_dict["final_erec_bin_edges"]
                logger.info(f"Final Erec bin edges passed via argument: {in_edges_final!s}")
            else:
                in_edges_final = FileReader.record[cfg_file_mapping["final_erec_bin_edges"]][
                    "E_rec_MeV"
                ]

            # Instantiate the storage nodes for bin edges. In what follows all the
            # nodes, outputs and inputs are automatically added to the relevant storage
            # locations. This is done via usage of the `Node.replicate()` class method.
            # The method is also responsible for creating indexed copies of the classes,
            # hence the name.
            edges_costheta, _ = Array.replicate(name="edges.costheta", array=in_edges_costheta)
            edges_energy_common, _ = Array.replicate(
                name="edges.energy_common", array=in_edges_fine
            )
            edges_energy_final, _ = Array.replicate(name="edges.energy_final", array=in_edges_final)
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
            # - Edep - deposited energy of a positron.
            # - Escint - energy, converted to the scintillation light.
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

            # For deposited energy bins provide also bin centers, to be used for
            # antineutrino spectrum correction (free).
            BinCenter.replicate(
                edges_energy_edep,
                name="edges.centers.energy_edep",
            )

            # Finally, create a node with segment edges for modelling the reactor
            # electron antineutrino spectra.
            Array.replicate(
                name="reactor_antineutrino.spectrum_free_correction.spec_model_edges",
                array=antineutrino_model_edges,
            )

            # Define supplementary width of the model of spectra. May be used for
            # plotting.
            BinWidth.replicate(
                outputs.get_value("reactor_antineutrino.spectrum_free_correction.spec_model_edges"),
                name="reactor_antineutrino.spectrum_free_correction.spec_model_widths",
            )

            # Introduce Δ=Eν-Edep=m(n)-m(p)-m(e) approximately connecting neutrino and
            # deposited energy.
            Difference.replicate(
                parameters.get_value("constant.ibd.NeutronMass"),
                parameters.get_value("constant.ibd.ProtonMass"),
                parameters.get_value("constant.ibd.ElectronMass"),
                name="constants.Delta_Enu_Edep",
            )

            # Convert antineutrino energy of the edges of antineutrino spectrum
            # parametrization approximately to deposited energy.
            Difference.replicate(
                outputs.get_value("reactor_antineutrino.spectrum_free_correction.spec_model_edges"),
                outputs.get_value("constants.Delta_Enu_Edep"),
                name="reactor_antineutrino.spectrum_free_correction_post.spec_model_edges_edep_approx",
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
            # In particular using order 5 for Edep and 3 for cosθ means 15=5×3 points
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
            # case of 2d integration each mesh is 2d array, similar to one, produced by
            # numpy.meshgrid function. A dedicated integrator node, which does the
            # actual integration, is created for each integrable function. In the
            # Daya Bay case the integrator part is replicated: an instance created for
            # each combination of "antineutrino_source.reactor.isotope.detector" indices. Note,
            # that NEQ part (antineutrino_source) has no contribution from ²³⁸U and SNF part has
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
                replicate_outputs=combinations["antineutrino_source.reactor.isotope.detector"],
            )
            # Pass the integration orders to the sampler inputs. The operator `>>` is
            # used to make a connection `input >> output` or batch connection
            # `input(s) >> outputs`. The operator connects all the objects on the left
            # side to all the corresponding objects on the right side. Missing pairs
            # will cause an exception.
            outputs.get_value("kinematics.integration.orders_edep") >> inputs.get_value(
                "kinematics.sampler.orders_edep"
            )
            outputs.get_value("kinematics.integration.orders_costheta") >> inputs.get_value(
                "kinematics.sampler.orders_costheta"
            )
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
            # functions on. The next step is to initialize the functions themselves,
            # which include: Inverse Beta Decay cross section (IBD), electron
            # antineutrino survival probability and antineutrino spectrum.

            # Here we create an instance of Inverse Beta Decay cross section, which also
            # includes conversion from deposited energy Edep to neutrino energy Enu and
            # corresponding dEnu/dEdep jacobian. The IBD nodes may operate with either
            # positron energy Ee as input, or deposited energy Edep=Ee+m(e) as input,
            # which is specified via an argument.
            ibd, _ = IBDXsecVBO1Group.replicate(path="kinematics.ibd", input_energy="edep")
            # IBD cross section depends on a set of parameters, including neutron
            # lifetime, proton and neutron masses, vector coupling constant, etc. The
            # values of these parameters were previously loaded and are located in the
            # 'parameters.constant.ibd' namespace. The IBD node(s) have an input for
            # each parameter. In order to connect the parameters the `<<` operator is
            # used as `node << parameters_storage`. It will loop over all the inputs of
            # the node and find parameters of the same name in the right hand side
            # namespace. Missing parameters are skipped, extra parameters are ignored.
            ibd << storage.get_dict("parameters.constant.ibd")
            ibd << storage.get_dict("parameters.constant.ibd.csc")
            # Connect the integration meshes for Edep and cosθ to the inputs of the
            # IBD node.
            outputs.get_value("kinematics.sampler.mesh_edep") >> ibd.inputs["edep"]
            (outputs.get_value("kinematics.sampler.mesh_costheta") >> ibd.inputs["costheta"])
            # There is an output, which yields neutrino energy Enu (mesh), corresponding
            # to the Edep, cosθ meshes. As it will be used quite often, let us save it
            # to a variable.
            kinematic_integrator_enu = ibd.outputs["enu"]

            # Initialize survival probability for reactor electron antineutrinos. As it
            # is affected by the distance, we replicate it for each combination of
            # "reactor.detector" indices of count of 48. It is defined for energies in
            # MeV, while the unit for distance may be chosen between "m" and "km".
            NueSurvivalProbability.replicate(
                name="survival_probability",
                leading_mass_splitting_3l_name=self._leading_mass_splitting_3l_name,
                distance_unit="m",
                replicate_outputs=combinations["reactor.detector"],
                surprobArgConversion=True,
            )
            # If created in the verbose mode one can see, that the following items are
            # created:
            # - nodes.survival_probability.R1.AD11
            # - nodes.survival_probability.R1.AD12
            # - ...
            # - inputs.survival_probability.enu.R1.AD11
            # - inputs.survival_probability.enu.R1.AD12
            # - ...
            # - inputs.survival_probability.L.R1.AD11
            # - inputs.survival_probability.L.R1.AD12
            # - ...
            # - inputs.survival_probability.surprobArgConversion.R1.AD11
            # - inputs.survival_probability.surprobArgConversion.R1.AD12
            # - ...
            # - outputs.survival_probability.R1.AD11
            # - outputs.survival_probability.R1.AD12
            # - ...
            # On one hand each node with its inputs and outputs may be accessed via
            # "nodes.survival_probability.<reactor>.<detector>" address. On the other hand all the
            # inputs, corresponding to the baselines and input energies may be accessed
            # via "inputs.survival_probability.L" and "inputs.survival_probability.enu" respectively. It is then
            # under user control whether he wants to provide similar or different data
            # for them.
            # Connect the same mesh of neutrino energy to all the 48 inputs:
            kinematic_integrator_enu >> inputs.get_dict("survival_probability.enu")
            # Connect the corresponding baselines:
            parameters.get_dict("constant.baseline") >> inputs.get_dict("survival_probability.L")
            # The matching is done based on the index with order being ignored. Thus
            # baselines stored as "R1.AD11" or "AD11.R1" both may be connected to the
            # input "R1.AD11". Moreover, if the left part has fewer indices, the
            # connection will be broad casted, e.g. "R1" on the left will be connected
            # to all the indices on the right, containing "R1".
            #
            # Provide a conversion constant to convert the argument of sin²(...Δm²L/E)
            # from chosen units to natural ones.
            parameters.get_value(
                "all.conversion.survival_probability_argument_factor"
            ) >> inputs.get_dict("survival_probability.surprobArgConversion")
            # Also connect free, constrained and constant oscillation parameters to each
            # instance of the oscillation probability.
            nodes.get_dict("survival_probability") << parameters.get_dict(
                "free.survival_probability"
            )
            nodes.get_dict("survival_probability") << parameters.get_dict(
                "constrained.survival_probability"
            )
            nodes.get_dict("survival_probability") << parameters.get_dict(
                "constant.survival_probability"
            )

            # The third component is the antineutrino spectrum as dN/dE per fission. We
            # start from loading the reference antineutrino spectrum (Huber-Mueller)
            # from input files. There are four spectra for four active isotopes. The
            # loading is done with the command `load_graph`, which supports hdf5, npz,
            # root, tsv (files or folder) or compressed tsv.bz2. The command will read
            # items with names "U235", "U238", "Pu239" and "Pu241" (from
            # index["isotope"]) as follows:
            # - hdf5: open with filename, request (X,Y) dataset by name.
            # - npz: open with filename, get (X,Y) array from a dictionary by name.
            # - root: open with filename, get TH1D object by name. Build graph by taking
            #         **left edges** of the bins and their heights. `uproot` is used to
            #         load ROOT files by default. If `$ROOTSYS` is defined, then ROOT is
            #         used directly.
            # - tsv: different arrays are kept in distinct files. Therefore for the tsv
            #        some logic is implemented to find the files. Given 'filename.tsv'
            #        and 'key', the following files are checked:
            #        + filename.tsv/key.tsv
            #        + filename.tsv/filename_key.tsv
            #        + filename_key.tsv
            #        + filename.tsv/key.tsv.bz2
            #        + filename.tsv/filename_key.tsv.bz2
            #        + filename_key.tsv.bz2
            #        The graph is expected to be written in 2 columns: X, Y.
            # A complementary method `load_graph_data` with same signature loads the
            # data, but keeps it as numpy arrays in the storage and does not create
            # nodes, so the user can modify the data before feeding it into the graph.
            #
            # The appropriate loader is chosen based on extension. The objects are
            # loaded and stored in the "reactor_antineutrino.neutrino_per_fission_per_MeV_input"
            # location. As `merge_x` flag is specified, only on X array is stored with
            # no index. A dedicated check is performed to ensure the graphs have
            # consistent X axes.
            # Note, that each Y node (called spec) will have an reference to the X node,
            # so it could be used when plotting.
            load_graph(
                name="reactor_antineutrino.neutrino_per_fission_per_MeV_input",
                filenames=cfg_file_mapping["reactor_antineutrino_spectra"],
                x="enu",
                y="spec",
                merge_x=True,
                replicate_outputs=index["isotope"],
            )

            # The input antineutrino spectra have step of 50 keV. They now should be
            # interpolated to the integration mesh. Similarly to integration nodes,
            # interpolation is implemented in two steps by two nodes: `indexer` node
            # identifies the indexes of segments, which should be used to interpolate
            # each mesh point. The `interpolator` does the actual interpolation, using
            # the input coarse data and indices from indexer. We instruct interpolator
            # to create a distinct node for each isotope by setting `replicate_outputs`
            # argument. We use exponential interpolation (`method="exp"`) and provide
            # names for the nodes and outputs. By default interpolator does
            # extrapolation in both directions outside of the domain.
            Interpolator.replicate(
                method="exp",
                names={
                    "indexer": "reactor_antineutrino.spec_indexer",
                    "interpolator": "reactor_antineutrino.neutrino_per_fission_per_MeV_nominal",
                },
                replicate_outputs=index["isotope"],
            )
            # Connect the common neutrino energy mesh as coarse input of the
            # interpolator.
            outputs.get_value(
                "reactor_antineutrino.neutrino_per_fission_per_MeV_input.enu"
            ) >> inputs.get_value(
                "reactor_antineutrino.neutrino_per_fission_per_MeV_nominal.xcoarse"
            )
            # Connect the input antineutrino spectra as coarse Y inputs of the
            # interpolator. This is performed for each of the 4 isotopes.
            outputs.get_dict(
                "reactor_antineutrino.neutrino_per_fission_per_MeV_input.spec"
            ) >> inputs.get_dict(
                "reactor_antineutrino.neutrino_per_fission_per_MeV_nominal.ycoarse"
            )
            # The interpolators are using the same target mesh for all the same target
            # mesh. Use the neutrino energy mesh provided by interpolator as an input to
            # fine X of the interpolation.
            kinematic_integrator_enu >> inputs.get_value(
                "reactor_antineutrino.neutrino_per_fission_per_MeV_nominal.xfine"
            )

            # The antineutrino spectrum in this analysis is a subject of five
            # independent corrections:
            # 1. Non-EQuilibrium correction (NEQ). A dedicated correction to ILL (Huber)
            #    antineutrino spectra from ²³⁵U, ²³⁹Pu, ²⁴¹Pu. Note: that ²³⁸U as no NEQ
            #    applied. In order to handle this an alternative index `isotope_neq` is
            #    used.
            # 2. Spent Nuclear Fuel (SNF) correction to account for the existence of
            #    antineutrino flux from spent nuclear fuel.
            # 3. Average antineutrino spectrum correction — not constrained correction
            #    to the shape of average reactor antineutrino spectrum. Having its
            #    parameters free during the fit effectively implements the relative
            #    Far-to-Near measurement. The correction curve is single and is applied
            #    to all the isotopes (correlated between the isotopes).
            # 4. Input antineutrino spectrum (Huber-Mueller) related:
            #    a. constrained spectrum shape correction due to
            #       model uncertainties. Uncorrelated between energy intervals and
            #       uncorrelated between isotopes.
            #    b. constrained spectrum shape correction due to model uncertainties.
            #       Correlated between energy intervals and correlated between isotopes.
            #
            # For convenience reasons let us introduce two constants to enable/disable
            # SNF and NEQ contributions. The `load_parameters()` method is used. The
            # syntax is similar to the one in yaml files.
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

            # Similarly to the case of electron antineutrino spectrum load the
            # corresponding NEQ (1.) corrections to 3 out of 4 isotopes. The correction
            # C should be applied to spectrum as follows: S'(Eν)=S(Eν)(1+C(Eν))
            load_graph(
                name="reactor_antineutrino.nonequilibrium_antineutrino.correction_input",
                x="enu",
                y="nonequilibrium_correction",
                merge_x=True,
                filenames=cfg_file_mapping["nonequilibrium_correction"],
                replicate_outputs=index["isotope_neq"],
            )

            # Create interpolators for NEQ correction. Use linear interpolation
            # (`method="linear"`). The regions outside the domains will be filled with a
            # constant (0 by default).
            Interpolator.replicate(
                method="linear",
                names={
                    "indexer": "reactor_antineutrino.nonequilibrium_antineutrino.correction_indexer",
                    "interpolator": "reactor_antineutrino.nonequilibrium_antineutrino.correction_interpolated",
                },
                replicate_outputs=index["isotope_neq"],
                underflow="constant",
                overflow="constant",
            )
            # Similarly to the case of antineutrino spectrum connect coarse X, a few
            # coarse Y and target mesh to the interpolator nodes.
            outputs.get_value(
                "reactor_antineutrino.nonequilibrium_antineutrino.correction_input.enu"
            ) >> inputs.get_value(
                "reactor_antineutrino.nonequilibrium_antineutrino.correction_interpolated.xcoarse"
            )
            outputs.get_dict(
                "reactor_antineutrino.nonequilibrium_antineutrino.correction_input.nonequilibrium_correction"
            ) >> inputs.get_dict(
                "reactor_antineutrino.nonequilibrium_antineutrino.correction_interpolated.ycoarse"
            )
            kinematic_integrator_enu >> inputs.get_value(
                "reactor_antineutrino.nonequilibrium_antineutrino.correction_interpolated.xfine"
            )

            # Now load the SNF (2.) correction. The SNF correction is different from NEQ
            # in a sense that it is computed for each reactor, not isotope. Thus we will
            # use reactor index for it. Aside from index the loading and interpolation
            # procedure is similar to that of NEQ correction.
            load_graph(
                name="reactor_antineutrino.snf_antineutrino.correction_input",
                x="enu",
                y="snf_correction",
                merge_x=True,
                filenames=cfg_file_mapping["snf_correction"],
                replicate_outputs=index["reactor"],
            )
            Interpolator.replicate(
                method="linear",
                names={
                    "indexer": "reactor_antineutrino.snf_antineutrino.correction_indexer",
                    "interpolator": "reactor_antineutrino.snf_antineutrino.correction_interpolated",
                },
                replicate_outputs=index["reactor"],
                underflow="constant",
                overflow="constant",
            )
            outputs.get_value(
                "reactor_antineutrino.snf_antineutrino.correction_input.enu"
            ) >> inputs.get_value(
                "reactor_antineutrino.snf_antineutrino.correction_interpolated.xcoarse"
            )
            outputs.get_dict(
                "reactor_antineutrino.snf_antineutrino.correction_input.snf_correction"
            ) >> inputs.get_dict(
                "reactor_antineutrino.snf_antineutrino.correction_interpolated.ycoarse"
            )
            kinematic_integrator_enu >> inputs.get_value(
                "reactor_antineutrino.snf_antineutrino.correction_interpolated.xfine"
            )

            # Finally create the parametrization of the correction to the shape of
            # average reactor electron antineutrino spectrum (3.). The `spec_scale`
            # correction is defined on a user defined segments `spec_model_edges`. The
            # values of the correction scale Fᵢ are 0 by default at each edge. There exit
            # two options of operation:
            # - exponential: Sᵢ=exp(F₁) are used for each edge. The result is always
            #                positive, although non-linear.
            # - linear: Sᵢ=1+Fᵢ is used. The behavior is always linear, but this approach
            #           may yield negative results.
            # The parameters Fᵢ are free parameters of the fit. The behavior of the
            # correction Sᵢ is interpolated exponentially within the segments ensuring
            # the overall correction is continuous for the whole spectrum.
            #
            # To initialize the correction we use a convenience function
            # `make_y_parameters_for_x`, which is using `load_parameters()` to create
            # 'parameters.free.neutrino_per_fission_factor.spec_scale_00` and other
            # parameters with proper labels.
            # An option `hide_nodes` is used to ensure the nodes are not shown on the
            # graph to keep it less busy. It does not affect the computation mechanism.
            # Note: in order to create the parameters, it will access the edges.
            # Therefore the node with edges should be closed. This is typically the case
            # for the Array nodes, which already have data, but may be not the case for
            # other nodes.
            make_y_parameters_for_x(
                outputs.get_value("reactor_antineutrino.spectrum_free_correction.spec_model_edges"),
                namefmt="spec_scale_{:02d}",
                format="value",
                state="variable",
                key="neutrino_per_fission_factor",
                values=0.0,
                labels="Edge {i:02d} ({value:.2f} MeV) reactor antineutrino spectrum correction"
                + (
                    " (exp)"
                    if self.spectrum_correction_interpolation_mode == "exponential"
                    else " (linear)"
                ),
                hide_nodes=True,
            )

            # The created parameters are now available to be used for the minimizer, but
            # in order to use them conveniently they should be kept as an array.
            # Concatenation node is used to organize an array. The result of
            # concatenation will be updated lazily as the minimizer modifies the
            # parameters.
            Concatenation.replicate(
                parameters.get_dict("all.neutrino_per_fission_factor"),
                name="reactor_antineutrino.spectrum_free_correction.input",
            )
            # For convenience purposes let us assign `spec_model_edges` as X axis for
            # the array of parameters.
            outputs.get_value(
                "reactor_antineutrino.spectrum_free_correction.input"
            ).dd.axes_meshes = (
                outputs.get_value("reactor_antineutrino.spectrum_free_correction.spec_model_edges"),
            )

            # Depending on chosen method, convert the parameters to the correction
            # on a scale.
            if self.spectrum_correction_interpolation_mode == "exponential":
                # Exponentiate the array of values. No `>>` is used as the array is
                # passed as an argument and the connection is done internally.
                Exp.replicate(
                    outputs.get_value("reactor_antineutrino.spectrum_free_correction.input"),
                    name="reactor_antineutrino.spectrum_free_correction.correction",
                )
            else:
                # Instead of exponent use linear `1+x` approach. First, create an array
                # with [1].
                Array.from_value(
                    "reactor_antineutrino.spectrum_free_correction.unity",
                    1.0,
                    dtype="d",
                    mark="1",
                    label="Array of 1 element =1",
                    shape=1,
                    store=True,
                )
                # Calculate the sum of [1] and array of spectral parameters. The
                # broadcasting is done similarly to numpy, i.e. the result of the
                # operation has the shape of the spectral parameters and 1 is added to
                # each element.
                Sum.replicate(
                    outputs.get_value("reactor_antineutrino.spectrum_free_correction.unity"),
                    outputs.get_value("reactor_antineutrino.spectrum_free_correction.input"),
                    name="reactor_antineutrino.spectrum_free_correction.correction",
                )
                # For convenience purposes assign `spec_model_edges` as X axis for the
                # array of scale factors.
                outputs.get_value(
                    "reactor_antineutrino.spectrum_free_correction.correction"
                ).dd.axes_meshes = (
                    outputs.get_value(
                        "reactor_antineutrino.spectrum_free_correction.spec_model_edges"
                    ),
                )

            # Interpolate the spectral correction exponentially. The extrapolation will
            # be applied to the points outside of the domain. The description of the
            # interpolation procedure is given above
            Interpolator.replicate(
                method="exp",
                names={
                    "indexer": "reactor_antineutrino.spectrum_free_correction.indexer",
                    "interpolator": "reactor_antineutrino.spectrum_free_correction.interpolated",
                },
            )
            outputs.get_value(
                "reactor_antineutrino.spectrum_free_correction.spec_model_edges"
            ) >> inputs.get_value(
                "reactor_antineutrino.spectrum_free_correction.interpolated.xcoarse"
            )
            outputs.get_value(
                "reactor_antineutrino.spectrum_free_correction.correction"
            ) >> inputs.get_value(
                "reactor_antineutrino.spectrum_free_correction.interpolated.ycoarse"
            )
            kinematic_integrator_enu >> inputs.get_value(
                "reactor_antineutrino.spectrum_free_correction.interpolated.xfine"
            )

            # Alternative post-fit spectrum correction.
            # TODO: doc
            Interpolator.replicate(
                method="exp",
                names={
                    "indexer": "reactor_antineutrino.spectrum_free_correction_post.indexer",
                    "interpolator": "reactor_antineutrino.spectrum_free_correction_post.interpolated",
                },
            )
            outputs.get_value(
                "reactor_antineutrino.spectrum_free_correction_post.spec_model_edges_edep_approx"
            ) >> inputs.get_value(
                "reactor_antineutrino.spectrum_free_correction_post.interpolated.xcoarse"
            )
            outputs.get_value(
                "reactor_antineutrino.spectrum_free_correction.correction"
            ) >> inputs.get_value(
                "reactor_antineutrino.spectrum_free_correction_post.interpolated.ycoarse"
            )
            outputs.get_value("edges.centers.energy_edep") >> inputs.get_value(
                "reactor_antineutrino.spectrum_free_correction_post.interpolated.xfine"
            )

            # Load the uncertainties, related to Huber+Mueller antineutrino spectrum
            # shape. The uncertainties are individual for each of the isotopes. There
            # are two parts:
            #   - uncorrelated between isotopes and energy intervals. Will add an extra
            #     parameter for each energy interval for each isotope (4a).
            #   - correlated between isotopes and energy intervals. Controlled by a
            #     single parameter (4b).
            # Note, that on average the antineutrino spectrum shape is free and
            # controlled by antineutrino spectrum correction, which will be the dominant
            # one, but is applied to all the isotopes. Therefore during the fit the
            # spectrum shape distortions which may be introduced by both the free
            # average shape correction and constrained isotope spectrum shape
            # correction, the former will take precedence. Only distortions, which may
            # not be introduced by average correction will be introduced by isotope
            # (un)correlated correction and will cause an extra punishment to the χ²
            # function.
            #
            # The total correction to antineutrino spectrum of isotope i Sᵢ(Eₖ) is
            # defined as Cₖ(Eᵢ)=(1+ηᵢₖσᵢₖ)·(1+ζΣᵢₖ), where σᵢₖ is a value of
            # uncorrelated uncertainty of isotope i at energy k and ηᵢₖ is the value of
            # corresponding nuisance parameter (central value=0, uncertainty=1). Σᵢₖ is
            # a value of correlated uncertainty of isotope i at energy k and ζ is the
            # value of corresponding nuisance parameter, common for all bins and
            # isotopes. As with previous corrections, the Cₖ(Eᵢ) define the correction
            # and the nodes, while the behaviour between the nodes is defined by the
            # interpolation.
            #
            # Load uncorrelated uncertainties as graphs from the input files. Original
            # Huber+Mueller uncertainties provided for 250 keV bins are scaled down to
            # 50 keV bins assuming the fine bins have no correlations between each
            # other. Both correlated and uncorrelated uncertainties are read as
            # controlled by index `antineutrino_unc` with values `corr` and `uncorr`
            # respectively.
            load_graph(
                name="reactor_antineutrino.spectrum_uncertainty",
                filenames=cfg_file_mapping["reactor_antineutrino_spectra_uncertainties"],
                x="enu_centers",
                y="uncertainty",
                merge_x=True,
                replicate_outputs=combinations["antineutrino_unc.isotope"],
            )

            # Create a set of parameters ηᵢₖ using a convenience method
            # `make_y_parameters_for_x`. For each isotope for each point of neutrino
            # energy of correction parametrization create a variable parameter
            # `reactor_antineutrino.spectrum_uncertainty.uncorr.{isotope}.unc_scale_{index}`
            # with central value 0 and uncertainty 1. Labels will reflect the value of
            # energy. The last point may be disabled with `disable_last_one` in case the
            # constant interpolation is used.
            #
            # A `hide_nodes` options is used to mark all of the nodes except the leading
            # and trailing ones to be hidden when plotting the graph as the bulk of
            # nodes overload the graph too much.
            for isotope in index["isotope"]:
                make_y_parameters_for_x(
                    outputs.get_value("reactor_antineutrino.spectrum_uncertainty.enu_centers"),
                    namefmt="unc_scale_{:03d}",
                    format=("value", "sigma_absolute"),
                    state="variable",
                    key=f"reactor_antineutrino.spectrum_uncertainty.uncorr.{isotope}",
                    values=(0.0, 1.0),
                    labels=f"Edge {{i:02d}} ({{value:.2f}} MeV) uncorrelated {index_names[isotope]} spectrum correction",
                    disable_last_one=False,  # True for the constant interpolation, last edge is unused
                    hide_nodes=True,
                )

            # Create a single nuisance parameter for the correlated uncertainty
            # correction.
            load_parameters(
                path="reactor_antineutrino.spectrum_uncertainty",
                format=("value", "sigma_absolute"),
                state="variable",
                parameters={"corr": (0.0, 1.0)},
                labels={"corr": "Correlated ν̅ spectrum shape correction"},
            )

            # Concatenate a set of variables into an array for each isotope. When the
            # nuisance parameter is modified it also affects the corresponding array
            # element and thus is propagated to the calculation.
            Concatenation.replicate(
                parameters.get_dict("constrained.reactor_antineutrino.spectrum_uncertainty.uncorr"),
                name="reactor_antineutrino.spectrum_uncertainty.scale.uncorr",
                replicate_outputs=index["isotope"],
            )

            # For each isotope compute an element-wise product of array of nuisance
            # parameters and corresponding uncorrelated uncertainty. The result is
            # ηᵢₖσᵢₖ.
            Product.replicate(
                outputs.get_dict("reactor_antineutrino.spectrum_uncertainty.scale.uncorr"),
                outputs.get_dict("reactor_antineutrino.spectrum_uncertainty.uncertainty.uncorr"),
                name="reactor_antineutrino.spectrum_uncertainty.correction.uncorr",
                replicate_outputs=index["isotope"],
            )

            # For each isotope compute an element-wise product of nuisance parameter and
            # corresponding correlated uncertainty. The result is ζΣᵢₖ.
            Product.replicate(
                parameters.get_value("constrained.reactor_antineutrino.spectrum_uncertainty.corr"),
                outputs.get_dict("reactor_antineutrino.spectrum_uncertainty.uncertainty.corr"),
                name="reactor_antineutrino.spectrum_uncertainty.correction.corr",
                replicate_outputs=index["isotope"],
            )

            # Now we need to compute 1+ηᵢₖσᵢₖ  and 1+ζΣᵢₖ. For this we create an array
            # with 1.
            single_unity = Array(
                "single_unity",
                [1.0],
                dtype="d",
                mark="1",
                label="Array of 1 element =1",
            )
            # Add it to uncorrelated correction...
            Sum.replicate(
                outputs.get_dict("reactor_antineutrino.spectrum_uncertainty.correction.uncorr"),
                single_unity,
                name="reactor_antineutrino.spectrum_uncertainty.correction.uncorr_factor",
                replicate_outputs=index["isotope"],
            )
            # And to correlated correction...
            Sum.replicate(
                outputs.get_dict("reactor_antineutrino.spectrum_uncertainty.correction.corr"),
                single_unity,
                name="reactor_antineutrino.spectrum_uncertainty.correction.corr_factor",
                replicate_outputs=index["isotope"],
            )
            # Multiply results together.
            Product.replicate(
                outputs.get_dict(
                    "reactor_antineutrino.spectrum_uncertainty.correction.uncorr_factor"
                ),
                outputs.get_dict(
                    "reactor_antineutrino.spectrum_uncertainty.correction.corr_factor"
                ),
                name="reactor_antineutrino.spectrum_uncertainty.correction.full",
                replicate_outputs=index["isotope"],
            )

            # Interpolate the result on to the integration points similarly to how it is
            # done in the previous cases.
            Interpolator.replicate(
                method="linear",
                names={
                    "indexer": "reactor_antineutrino.spectrum_uncertainty.correction_index",
                    "interpolator": "reactor_antineutrino.spectrum_uncertainty.correction_interpolated",
                },
                replicate_outputs=index["isotope"],
            )
            outputs.get_value(
                "reactor_antineutrino.spectrum_uncertainty.enu_centers"
            ) >> inputs.get_value(
                "reactor_antineutrino.spectrum_uncertainty.correction_interpolated.xcoarse"
            )
            outputs.get_dict(
                "reactor_antineutrino.spectrum_uncertainty.correction.full"
            ) >> inputs.get_dict(
                "reactor_antineutrino.spectrum_uncertainty.correction_interpolated.ycoarse"
            )
            kinematic_integrator_enu >> inputs.get_value(
                "reactor_antineutrino.spectrum_uncertainty.correction_interpolated.xfine"
            )

            # Finally apply all the corrections 1, 3 and 4 to the antineutrino spectra.
            # The SNF will be treated later.
            # The free average antineutrino spectrum correction (3.) and constrained
            # corrections to antineutrino spectra from each isotopes (4.) are multiplied
            # to the nominal antineutrino spectrum.
            if self.spectrum_correction_location == "before-integration":
                Product.replicate(
                    outputs.get_dict("reactor_antineutrino.neutrino_per_fission_per_MeV_nominal"),
                    outputs.get_value("reactor_antineutrino.spectrum_free_correction.interpolated"),
                    outputs.get_dict(
                        "reactor_antineutrino.spectrum_uncertainty.correction_interpolated"
                    ),
                    name="reactor_antineutrino.part.neutrino_per_fission_per_MeV_main",
                    replicate_outputs=index["isotope"],
                )
            else:
                Product.replicate(
                    outputs.get_dict("reactor_antineutrino.neutrino_per_fission_per_MeV_nominal"),
                    outputs.get_dict(
                        "reactor_antineutrino.spectrum_uncertainty.correction_interpolated"
                    ),
                    name="reactor_antineutrino.part.neutrino_per_fission_per_MeV_main",
                    replicate_outputs=index["isotope"],
                )

            # The NEQ correction (1.) is applied to the nominal antineutrino spectra as
            # well and is not affected by the free and constrained spectra distortions.
            # As long as ²³⁸U spectrum has no NEQ correction we use a truncated index
            # `isotope_neq` and explicitly allow to skip ²³⁸U from the nominal spectra.
            Product.replicate(
                outputs.get_dict("reactor_antineutrino.neutrino_per_fission_per_MeV_nominal"),
                outputs.get_dict(
                    "reactor_antineutrino.nonequilibrium_antineutrino.correction_interpolated"
                ),
                name="reactor_antineutrino.part.neutrino_per_fission_per_MeV_neq_nominal",
                allow_skip_inputs=True,
                skippable_inputs_should_contain=("U238",),
                replicate_outputs=index["isotope_neq"],
            )
            # The application of SNF (2.) and further usage of the antineutrino flux
            # will happen later. At this point it is time to load the necessary data to
            # calculate the expected number of neutrino interactions.

            # In the following we read time dependent detector and reactor performance
            # data. For detectors the data is daily. It includes:
            # - Detector live time in seconds.
            # - Efficiency, related to muon veto and multiplicity cut.
            # - Rate of accidental events in inverse seconds.
            # For reactors the data is ~monthly (TODO: specify). It includes:
            # - Average thermal power, relative to the nominal value.
            # - Average fission fractions for each isotope.
            #
            # A function `load_record_data` to load table data, which in general works
            # similarly to `load_graph` and `load_graph_data`. The function reads
            # tabular data from the input file and stores its columns as array in the
            # `storage["data"]`. It support the same formats:
            # - hdf5/npz: reads numpy record array.
            # - root: reads TTree.
            # - tsv: reads text file with columns.
            #
            # Here we read the input file. The data is read from the file based on
            # detector name, passed via `replicate_output` argument. Note, a function
            # `name_function` may be provided to generate key to read based on index
            # value. The columns, which should be stored are passed via the `columns`
            # argument. The default key order is
            # `[column_name, index1_value, index2_value, ...]`. The data provided
            # includes:
            # - day - number of day since start of data taking, 0-based.
            # - n_det - number of active detectors (6, 8, or 7).
            # - livetime, eff, rate_accidentals - daily detector data according to the
            #                                     description above.
            # - eff_livetime - effective livetime = livetime*eff.
            load_record_data(
                name="daily_data.detector_all",
                filenames=cfg_file_mapping["daily_detector_data"],
                replicate_outputs=index["detector"],
                columns=("day", "n_det", "livetime", "eff", "eff_livetime", "rate_accidentals"),
                skip=inactive_detectors,
            )

            # The data of each detector is stored for the whole period of data taking.
            # For this particular analysis the data should be split into arrays for each
            # particular period. This is done by the `refine_detector_data` function,
            # which reads columns and splits them based on `n_det` value. The function
            # processes `days` and columns, specified in the `columns` argument.
            # The data is read from "daily_data.detector_all" and stored in
            # "daily_data.detector".
            refine_detector_data(
                data("daily_data.detector_all"),
                data.create_child("daily_data.detector"),
                detectors=index["detector"],
                skip=inactive_detectors,
                columns=("livetime", "eff", "eff_livetime", "rate_accidentals"),
            )

            # The reactor data is stored and read in a similar way and contains the
            # following columns:
            # - period - number of period for which data is presented, 0-based.
            # - day - number of the first day of the period relative to the start of the
            #         data taking, 0-based.
            # - n_days - length of the period in days.
            # - power - average thermal power, relative to nominal.
            # - u235, u238, pu239, pu241 - fission fractions of corresponding isotope.
            # TODO
            load_record_data(
                name="daily_data.antineutrino_rate_all",
                filenames=cfg_file_mapping["daily_antineutrino_rate_data"],
                replicate_outputs=index["reactor"],
                columns=("period", "day", "n_det_mask", "n_days", "neutrino_rate_per_s"),
            )

            # TODO
            refine_neutrino_rate_data(
                data("daily_data.antineutrino_rate_all"),
                data.create_child("daily_data.reactor"),
                reactors=index["reactor"],
            )

            # Reactor data is then converted from monthly (TODO: specify) to daily (no
            # interpolation) and split them into data taking periods. The data is read
            # from "daily_data.reactor_all" and stored in "daily_data.reactor".
            # TODO

            # The detector and reactor data have different minimal period, therefore the
            # arrays are not matching. With the following procedure we produce matching
            # arrays for detector properties and reactor data based on the `day`. The
            # procedure also checks that the data ranges are consistent.

            # TODO
            sync_neutrino_rate_detector_data(
                data("daily_data.reactor"),
                data("daily_data.detector"),
            )

            # Finally for convenience we change the nesting order making the data taking
            # period the innermost index. This does not affect matching the indices,
            # however is more convenient for plotting.
            # TODO
            data["daily_data.reactor.antineutrino_rate_per_s"] = remap_items(
                data.get_dict("daily_data.reactor.neutrino_rate_per_s"),
                reorder_indices={
                    "from": ["period", "reactor"],
                    "to": ["reactor", "period"],
                },
            )

            # After the data is split into arrays and synchronized we create array nodes
            # for each input using `Array.from_storage` class method.
            # `remove_processed_arrays` instructs the constructor to remove the data from
            # storage after node is created in order not to keep duplicated data. We
            # specifically set the data type to ensure the model does not depend on the
            # datatype of the input.
            #
            # The following arrays are created:
            # - days — an array of day numbers. A dedicated array for each period is
            # stored.
            # - detector data for each detector and period:
            #   - livetime
            #   - eff
            #   - eff_livetime
            #   - rate_accidentals
            # - reactor data for each reactor and period:
            #   - power
            #   - fission_fractions
            Array.from_storage(
                "daily_data.detector.days",
                storage.get_dict("data"),
                remove_processed_arrays=True,
                dtype="i",
            )

            # For convenience the array with days is moved one level up the structure.
            # It may be used for both reactor and detector data.
            outputs["daily_data.days"] = outputs.pop(
                "daily_data.detector.days", delete_parents=True
            )

            Array.from_storage(
                "daily_data.detector.livetime",
                storage.get_dict("data"),
                remove_processed_arrays=True,
                dtype="d",
            )

            Array.from_storage(
                "daily_data.detector.eff",
                storage.get_dict("data"),
                remove_processed_arrays=True,
                dtype="d",
            )

            Array.from_storage(
                "daily_data.detector.eff_livetime",
                storage.get_dict("data"),
                remove_processed_arrays=True,
                dtype="d",
            )

            Array.from_storage(
                "daily_data.detector.rate_accidentals",
                storage.get_dict("data"),
                remove_processed_arrays=True,
                dtype="d",
            )

            Array.from_storage(
                "daily_data.reactor.antineutrino_rate_per_s",
                storage.get_dict("data"),
                remove_processed_arrays=True,
                dtype="d",
            )
            del storage["data.daily_data"]

            # Compute a total (effective) livetime for each detector.
            # ArraySum operation does not to combine different array: and produces an
            # output for each input. Therefore there is no need to provide
            # `replicate_outputs` argument this time.
            ArraySum.replicate(
                outputs.get_dict("daily_data.detector.livetime"),
                name="detector.livetime",
            )

            ArraySum.replicate(
                outputs.get_dict("daily_data.detector.eff_livetime"),
                name="detector.eff_livetime",
            )

            # At this point we have the information to compute the antineutrino
            # flux.
            # TODO
            # - nominal thermal power [MeV/s], fit-dependent
            # - nominal thermal power [MeV/s], fit-independent (central values)
            # - fission fraction, corrected based on nuisance parameters [fraction]
            # - average energy per fission

            # Thermal power [MeV/s] for each reactor is defined as multiplication of
            # parameters `nominal_thermal_power` [GW] for each reactor and conversion
            # constant. While storages may be accessed with `[]` we explicitly use
            # `get_dict` and `get_value` methods to indicate whether we expect a single
            # object or a nested storage.
            Product.replicate(
                parameters.get_dict("all.reactor.nominal_thermal_power"),
                parameters.get_value("all.conversion.conversion_reactor_power"),
                name="reactor.thermal_power_nominal_MeVs",
                replicate_outputs=index["reactor"],
            )

            # We repeat the same procedure for the central value. While the previous
            # "thermal_power_nominal_MeVs" depends on the minimization parameters
            # "all.reactor.nominal_thermal_power", for the following product we use
            # "central.reactor.nominal_thermal_power", which do not depend on them.
            Product.replicate(
                parameters.get_dict("constant.reactor.nominal_thermal_power"),
                parameters.get_value("all.conversion.conversion_reactor_power"),
                name="reactor.thermal_power_nominal_MeVs_central",
                replicate_outputs=index["reactor"],
            )

            # Apply the variable scale (nuisance) to the fiction fractions. Fission
            # fractions are time dependent and the scale is applied to each day. The
            # result is an array for each reactor, isotope, period triplet.
            # TODO
            Product.replicate(
                parameters.get_dict("all.reactor.fission_fraction_scale"),
                parameters.get_dict("all.reactor.fission_fractions"),
                name="reactor.fission_fractions_scaled",
                replicate_outputs=combinations["reactor.isotope"],
            )

            # Compute absollute value of previous transformation. It is needed because
            # sometime minimization procedure goes to the non-physical values of
            # fission fraction. This transforamtion limits possible variations.
            # TODO: code
            Abs.replicate(
                name="reactor.fission_fractions_scaled_abs",
                replicate_outputs=combinations["reactor.isotope"],
            )
            outputs.get_dict("reactor.fission_fractions_scaled") >> inputs.get_dict(
                "reactor.fission_fractions_scaled_abs"
            )

            # Using daily fission fractions compute weighted energy per fission in each
            # isotope in each reactor during each period. This is an intermediate step
            # to obtain average energy per fission in each reactor.
            # TODO: check
            Product.replicate(
                parameters.get_dict("all.reactor.energy_per_fission"),
                outputs.get_dict("reactor.fission_fractions_scaled_abs"),
                name="reactor.energy_per_fission_weighted_MeV",
                replicate_outputs=combinations["reactor.isotope"],
            )

            Product.replicate(
                parameters.get_dict("central.reactor.energy_per_fission"),
                parameters.get_dict("all.reactor.fission_fractions"),
                name="reactor.energy_per_fission_nominal_weighted_MeV",
                replicate_outputs=combinations["isotope"],
            )

            # TODO
            Product.replicate(
                parameters.get_dict("all.reactor.neutrinos_per_fission"),
                parameters.get_dict("all.reactor.fission_fractions"),
                name="reactor.neutrinos_per_fission_nominal_weighted_MeV",
                replicate_outputs=combinations["isotope"],
            )

            # Sum weighted energy per fission within each reactor (isotope index
            # removed) to compute average energy per fission in each reactor during each
            # period.
            # TODO: check
            Sum.replicate(
                outputs.get_dict("reactor.energy_per_fission_weighted_MeV"),
                name="reactor.energy_per_fission_average_MeV",
                replicate_outputs=combinations["reactor"],
            )

            Sum.replicate(
                outputs.get_dict("reactor.energy_per_fission_nominal_weighted_MeV"),
                name="reactor.energy_per_fission_nominal_average_MeV",
            )

            # TODO
            Sum.replicate(
                outputs.get_dict("reactor.neutrinos_per_fission_nominal_weighted_MeV"),
                name="reactor.neutrinos_per_fission_nominal_average_MeV",
            )

            # TODO
            Division.replicate(
                outputs.get_value("reactor.neutrinos_per_fission_nominal_average_MeV"),
                outputs.get_value("reactor.energy_per_fission_nominal_average_MeV"),
                name="reactor.neutrinos_per_MeV_nominal_average",
            )

            # Compute daily contribution of each isotope to reactor's thermal power by
            # multiplying fission fractions, nominal thermal power [MeV/s] and fractional
            # thermal power.
            # TODO: code?

            # Compute number of fissions per second related to each isotope in each
            # reactor and each period: divide partial thermal power by average energy
            # per fission.
            Division.replicate(
                outputs.get_dict("reactor.fission_fractions_scaled_abs"),
                outputs.get_dict("reactor.energy_per_fission_average_MeV"),
                name="reactor.fission_fractions_per_MeV",
                replicate_outputs=combinations["reactor.isotope"],
            )

            # TODO
            Division.replicate(
                outputs.get_dict("daily_data.reactor.antineutrino_rate_per_s"),
                outputs.get_value("reactor.neutrinos_per_MeV_nominal_average"),
                name="daily_data.reactor.thermal_power_average_MeV_per_s",
                replicate_outputs=combinations["reactor.period"],
            )

            # In the few following operations repeat the calculation of fissions per
            # second for SNF. This time we use fixed average fission fractions. The SNF
            # is defined as a fraction of nominal antineutrino spectrum from reactor.
            # Therefore the isotope index is used.

            # For SNF contribution use central values for the nominal thermal power.
            Product.replicate(
                parameters.get_dict("all.reactor.fission_fractions"),
                outputs.get_dict("reactor.thermal_power_nominal_MeVs_central"),
                name="reactor.thermal_power_snf_isotope_MeV_per_second",
                replicate_outputs=combinations["reactor.isotope"],
            )

            # Compute fissions per second for SNF calculation.
            Division.replicate(
                outputs.get_dict("reactor.thermal_power_snf_isotope_MeV_per_second"),
                outputs.get_value("reactor.energy_per_fission_nominal_average_MeV"),
                name="reactor.fissions_per_second_snf",
                replicate_outputs=combinations["reactor.isotope"],
            )

            # Now we need to incorporate the knowledge on the detector operation.
            # Compute the number of fissions as seen from each detector: multiply
            # fissions per second [#/s] by detector livetime [s]. This is done on a
            # daily basis. The result is for each isotope in each reactor seen at
            # each detector during each period.
            #
            # Not all detectors are available during all the periods. Still
            # corresponding combination of indices may arise during iteration. Therefore
            # we provide a list of indices, which should not trigger an exception.
            # TODO
            Product.replicate(
                outputs.get_dict("daily_data.reactor.thermal_power_average_MeV_per_s"),
                outputs.get_dict("daily_data.detector.eff_livetime"),
                name="daily_data.reactor_detector.thermal_energy_MeV",
                replicate_outputs=combinations["reactor.detector.period"],
                allow_skip_inputs=True,
                skippable_inputs_should_contain=inactive_detectors,
            )

            # Sum up each array of daily data to obtain number of fissions as seen by
            # each detector from each isotope from each reactor during each period.
            ArraySum.replicate(
                outputs.get_dict("daily_data.reactor_detector.thermal_energy_MeV"),
                name="reactor_detector.thermal_energy_MeV",
            )

            # Based on the distances compute baseline factors (1/[4πL²]) for
            # reactor-detector combinations using `InverseSquareLaw` node. A scale
            # factor is applied internally to convert meters to centimeters to make
            # factor consistent with cross section [cm⁻²].
            InverseSquareLaw.replicate(
                name="reactor_detector.baseline_factor_per_cm2",
                scale="m_to_cm",
                replicate_outputs=combinations["reactor.detector"],
            )
            # The baselines are passed to the corresponding inputs.
            parameters.get_dict("constant.baseline") >> inputs.get_dict(
                "reactor_detector.baseline_factor_per_cm2"
            )

            # Apply fit related correction for each detector to common nominal number of
            # target protons.
            Product.replicate(
                parameters.get_value("all.detector.n_protons_nominal_ad"),
                parameters.get_dict("all.detector.n_protons_correction"),
                name="detector.n_protons",
                replicate_outputs=index["detector"],
            )

            # Now we can combine total number of fissions (producing neutrinos) in each
            # reactor from each isotope, number of target protons in each detector,
            # corresponding baseline factor and efficiency:
            # - Number of fissions × N protons × ε / (4πL²) (main)
            # The result bears four indices: reactor, isotope, detector and period.
            # TODO: change units since here
            Product.replicate(
                outputs.get_dict("reactor_detector.thermal_energy_MeV"),
                outputs.get_dict("detector.n_protons"),
                outputs.get_dict("reactor_detector.baseline_factor_per_cm2"),
                parameters.get_value("all.detector.efficiency"),
                name="reactor_detector.n_fissions_n_protons_per_cm2",
                replicate_outputs=combinations["reactor.detector.period"],
            )

            # TODO
            Product.replicate(
                outputs.get_dict("reactor_detector.n_fissions_n_protons_per_cm2"),
                outputs.get_dict("reactor.fission_fractions_per_MeV"),
                parameters.get_dict("all.reactor.thermal_power_scale"),
                name="reactor_detector.n_fissions_n_protons_per_cm2_scaled",
                replicate_outputs=combinations["reactor.isotope.detector.period"],
            )

            # A parallel branch will be used for NEQ correction, with previous value
            # multiplied by `neq_factor=1` (simple switch) and fit defined
            # `nonequilibrium_scale`.
            Product.replicate(
                outputs.get_dict("reactor_detector.n_fissions_n_protons_per_cm2_scaled"),
                parameters.get_dict("all.reactor.nonequilibrium_scale"),
                parameters.get_value("all.reactor.neq_factor"),
                name="reactor_detector.n_fissions_n_protons_per_cm2_neq",
                replicate_outputs=combinations["reactor.isotope.detector.period"],
            )

            # Compute similar values for SNF. Note, that it should be also multiplied by
            # effective livetime:
            # - Effective live time × N protons × ε / (4πL²)  (SNF)
            Product.replicate(
                outputs.get_dict("detector.eff_livetime"),
                outputs.get_dict("detector.n_protons"),
                outputs.get_dict("reactor_detector.baseline_factor_per_cm2"),
                parameters.get_dict("all.reactor.snf_scale"),
                parameters.get_value("all.reactor.snf_factor"),
                parameters.get_value("all.detector.efficiency"),
                name="reactor_detector.livetime_n_protons_per_cm2_snf",
                replicate_outputs=combinations["reactor.detector.period"],
                allow_skip_inputs=True,
                skippable_inputs_should_contain=inactive_detectors,
            )

            Product.replicate(
                outputs.get_dict("reactor_antineutrino.neutrino_per_fission_per_MeV_nominal"),
                outputs.get_dict("reactor.fissions_per_second_snf"),
                name="reactor_antineutrino.snf_antineutrino.neutrino_per_second_isotope",
                replicate_outputs=combinations["reactor.isotope"],
            )

            Sum.replicate(
                outputs.get_dict(
                    "reactor_antineutrino.snf_antineutrino.neutrino_per_second_isotope"
                ),
                name="reactor_antineutrino.snf_antineutrino.neutrino_per_second",
                replicate_outputs=index["reactor"],
            )

            Product.replicate(
                outputs.get_dict("reactor_antineutrino.snf_antineutrino.neutrino_per_second"),
                outputs.get_dict("reactor_antineutrino.snf_antineutrino.correction_interpolated"),
                name="reactor_antineutrino.snf_antineutrino.neutrino_per_second_snf",
                replicate_outputs=index["reactor"],
            )

            # The three quantities, calculated above will be used to produce
            # antineutrino spectrum from nuclear reactors:
            # - main — raw antineutrino flux based on the input spectra.
            # - neq — extra antineutrino flux due to NEQ correction to input spectra.
            # - snf - extra antineutrino flux from SNF, assumed to be located at the
            #         same position as reactors.
            # Later the relevant numbers will be organized in storage with keys
            # `nu_main`, `nu_neq` and `nu_snf`, which represent the `antineutrino_source` index.

            # The following part is related to the calculation of a product of
            # flux × oscillation probability × cross section [Nν·cm²/fission/proton]
            # This functions will be further integrated into the bins of the histogram
            # to build the expected observation.

            # The integration is done versus deposited positron energy (Edep) and thus
            # requires an energy conversion Jacobian (dEν/dEdep). As both the cross
            # section and Jacobian do not depend on any nuisance parameters directly,
            # they are multiplied first.
            Product.replicate(
                outputs.get_value("kinematics.ibd.crosssection"),
                outputs.get_value("kinematics.ibd.jacobian"),
                name="kinematics.ibd.crosssection_jacobian",
            )

            # For each reactor-detector pair make a product of survival probability,
            # cross section and Jacobian.
            Product.replicate(
                outputs.get_value("kinematics.ibd.crosssection_jacobian"),
                outputs.get_dict("survival_probability"),
                name="kinematics.ibd.crosssection_jacobian_oscillations",
                replicate_outputs=combinations["reactor.detector"],
            )

            # Finally, multiply it by the antineutrino spectrum from each isotope.
            # The result has three indices: isotope, reactor, detector.
            Product.replicate(
                outputs.get_dict("kinematics.ibd.crosssection_jacobian_oscillations"),
                outputs.get_dict("reactor_antineutrino.part.neutrino_per_fission_per_MeV_main"),
                name="kinematics.neutrino_cm2_per_MeV_per_fission_per_proton.part.nu_main",
                replicate_outputs=combinations["reactor.isotope.detector"],
            )

            # Do the same the antineutrino spectrum, related to the NEQ correction
            # (applies to 3 isotopes out of 4).
            Product.replicate(
                outputs.get_dict("kinematics.ibd.crosssection_jacobian_oscillations"),
                outputs.get_dict(
                    "reactor_antineutrino.part.neutrino_per_fission_per_MeV_neq_nominal"
                ),
                name="kinematics.neutrino_cm2_per_MeV_per_fission_per_proton.part.nu_neq",
                replicate_outputs=combinations["reactor.isotope_neq.detector"],
            )

            # And for SNF.
            Product.replicate(
                outputs.get_dict("kinematics.ibd.crosssection_jacobian_oscillations"),
                outputs.get_dict("reactor_antineutrino.snf_antineutrino.neutrino_per_second_snf"),
                name="kinematics.neutrino_cm2_per_MeV_per_fission_per_proton.part.nu_snf",
                replicate_outputs=combinations["reactor.detector"],
            )

            # Main, NEQ and SNF contributions are now stored in nearby with indices
            # `nu_main`, `nu_neq` and `nu_snf` and may be connected to the relevant
            # inputs of the 2d integrators.
            outputs.get_dict(
                "kinematics.neutrino_cm2_per_MeV_per_fission_per_proton.part"
            ) >> inputs.get_dict("kinematics.integral")

            # Multiply by the integrated functions by relevant scaling factors, which
            # together consist of:
            #  - nu_main:   fissions_per_second[p,r,i] ×
            #             × effective live time[p,d] ×
            #             × N protons[d] ×
            #             × efficiency[d]
            Product.replicate(
                outputs.get_dict("kinematics.integral.nu_main"),
                outputs.get_dict("reactor_detector.n_fissions_n_protons_per_cm2_scaled"),
                name="eventscount.parts.nu_main",
                replicate_outputs=combinations["reactor.isotope.detector.period"],
            )

            #  - nu_neq:    fissions_per_second[p,r,i] ×
            #             × effective live time[p,d] ×
            #             × N protons[d] ×
            #             × efficiency[d] ×
            #             × NEQ scale[r,i] ×
            #             × neq_factor(=1)
            # As NEQ is not applied to ²³⁸U, allow related inputs to be left
            # unprocessed.
            Product.replicate(
                outputs.get_dict("kinematics.integral.nu_neq"),
                outputs.get_dict("reactor_detector.n_fissions_n_protons_per_cm2_neq"),
                name="eventscount.parts.nu_neq",
                replicate_outputs=combinations["reactor.isotope_neq.detector.period"],
                allow_skip_inputs=True,
                skippable_inputs_should_contain=("U238",),
            )

            #  - nu_snf:    effective live time[p,d] ×
            #             × N protons[d] ×
            #             × efficiency[d] ×
            #             × SNF scale[r] ×
            #             × snf_factor(=1)
            Product.replicate(
                outputs.get_dict("kinematics.integral.nu_snf"),
                outputs.get_dict("reactor_detector.livetime_n_protons_per_cm2_snf"),
                name="eventscount.parts.nu_snf",
                replicate_outputs=combinations["reactor.detector.period"],
            )

            # Finally sum together the contributions from reactors and from antineutrino
            # sources (main, NEQ, SNF) obtaining an expected spectrum in each detector
            # during each period.
            Sum.replicate(
                outputs.get_dict("eventscount.parts"),
                name="eventscount.stages.raw",
                replicate_outputs=combinations["detector.period"],
            )

            # TODO: doc
            if self.spectrum_correction_location == "after-integration":
                Product.replicate(
                    outputs.get_dict("eventscount.stages.raw"),
                    outputs.get_value(
                        "reactor_antineutrino.spectrum_free_correction_post.interpolated"
                    ),
                    name="eventscount.stages.raw_antineutrino_spectrum_corrected",
                    replicate_outputs=combinations["detector.period"],
                )

            # At this points the IBD spectra at each detector are available assuming the
            # ideal detector response. 4 transformations will be applied in the
            # following order:
            # - IAV effect — Energy loss due to particle passage through the wall of
            # Inner Acrylic Vessel (IAV).
            # - Energy scale distortion:
            #   + LSNL effect — common non-linear energy scale distortion.
            #   + relative energy scale — uncorrelated between detectors linear energy
            #                             scale.
            # - Energy smearing due to finite energy resolution
            # - Rebinning to the final binning.

            # Load the IAV matrix from the input file.
            # The `load_array` method works in a similar way to `load_graph` and
            # `load_record`. It expects the file (folder for tsv) to contain an array or
            # a set of arrays. The `name_function` makes the method to read "iav_matrix"
            # from file and store as "matrx_raw".
            #
            # An extra argument `edges` is passed to the constructor of an Array object
            # setting edges. Due to this, it will check that the histogram the effect is
            # applied to has consistent edges. It will also define the edges for the
            # output histogram. The edges are used for automated plotting for X axis and
            # its label.
            load_array(
                name="detector.iav",
                filenames=cfg_file_mapping["iav_matrix"],
                replicate_outputs=("matrix_raw",),
                name_function={"matrix_raw": "iav_matrix"},
                array_kwargs={"edges": (edges_energy_escint, edges_energy_edep)},
            )

            # The IAV distortion has an uncorrelated between detector uncertainty,
            # introduced as a factor, which scales the off-diagonal elements of the IAV
            # matrix.
            RenormalizeDiag.replicate(
                mode="offdiag",
                name="detector.iav.matrix_rescaled",
                replicate_outputs=index["detector"],
            )
            # An IAV distortion node has two inputs "scale" for the scale factor and
            # "matrix" for the matrix itself.
            # Match and connect 8 IAV scales to the appropriate inputs.
            parameters.get_dict("all.detector.iav_offdiag_scale_factor") >> inputs.get_dict(
                "detector.iav.matrix_rescaled.scale"
            )
            # Connect a single IAV matrix to 8 rescaling nodes.
            outputs.get_value("detector.iav.matrix_raw") >> inputs.get_dict(
                "detector.iav.matrix_rescaled.matrix"
            )

            # The correction is applied as matrix multiplication of smearing matrix over
            # column for each detector during each period..
            VectorMatrixProduct.replicate(
                name="eventscount.stages.iav",
                mode="column",
                replicate_outputs=combinations["detector.period"],
            )
            # Match and connect rescaled IAV distortion matrix for each detector to the
            # smearing node of each detector during each period.
            outputs.get_dict("detector.iav.matrix_rescaled") >> inputs.get_dict(
                "eventscount.stages.iav.matrix"
            )
            # Match and connect IBD histogram each detector during each period to the
            # relevant IAV smearing input.
            if self.spectrum_correction_location == "after-integration":
                outputs.get_dict(
                    "eventscount.stages.raw_antineutrino_spectrum_corrected"
                ) >> inputs.get_dict("eventscount.stages.iav.vector")
            else:
                outputs.get_dict("eventscount.stages.raw") >> inputs.get_dict(
                    "eventscount.stages.iav.vector"
                )

            # The LSNL distortion matrix is created based on a relative energy scale
            # distortion curve and a few nuisance curves, loaded with `load_graph_data`
            # method. The graphs will be modified before the nodes are created.
            #
            # Within our definition LSNL converts the deposited within scintillator
            # energy (Escint) into visible energy (Evis).
            load_graph_data(
                name="detector.lsnl.curves",
                x="escint",
                y="evis_parts",
                merge_x=True,
                filenames=cfg_file_mapping["lsnl_curves"],
                replicate_outputs=index["lsnl"],
            )

            # Pre-process LSNL curves in the following order:
            # - convert relative curves to absolute ones
            # - interpolate with 4 times smaller step using `cubic` interpolation
            #   (`refine_times` argument)
            # - extrapolate linearly absolute curves to an extended range
            #   (`newmin` and `newmax`)
            # - compute (nominal-pullᵢ) difference curves to be used as corrections
            #
            # The new fine Escint will be stored to `xname`. The argument `nominalname`
            # selects the nominal curve. The curves will be overwritten.
            refine_lsnl_data(
                storage.get_dict("data.detector.lsnl.curves"),
                xname="escint",
                nominalname="evis_parts.nominal",
                refine_times=4,
                newmin=0.5,
                newmax=12.1,
                # savgol_filter_smoothen = (10, 4)
            )

            # Create (graph) arrays for the LSNL curves. A dedicated array for X axis,
            # based on meshname="escint" will be created. Each curve will have a
            # reference to the Edep node as its X axis.
            # The processed arrays from `storage["data"]` will be removed with parent
            # dictionaries.
            Array.from_storage(
                "detector.lsnl.curves",
                storage.get_dict("data"),
                meshname="escint",
                remove_processed_arrays=True,
            )

            # Multiply nuisance LSNL curves by nuisance parameters (central=0). Allow to
            # skip nominal curve.
            Product.replicate(
                outputs.get_dict("detector.lsnl.curves.evis_parts"),
                parameters.get_dict("constrained.detector.lsnl_scale_a"),
                name="detector.lsnl.curves.evis_parts_scaled",
                allow_skip_inputs=True,
                skippable_inputs_should_contain=("nominal",),
                replicate_outputs=index["lsnl_nuisance"],
            )

            # Sum the curves together.
            Sum.replicate(
                outputs.get_value("detector.lsnl.curves.evis_parts.nominal"),
                outputs.get_dict("detector.lsnl.curves.evis_parts_scaled"),
                name="detector.lsnl.curves.evis_coarse",
            )

            # Calculate relative versions of the curves exclusively for plotting
            # reasons. The relative curves will not be used for the analysis.
            # First, compute relative curves as f(Escint)/Escint.
            Division.replicate(
                outputs.get_dict("detector.lsnl.curves.evis_parts"),
                outputs.get_value("detector.lsnl.curves.escint"),
                name="detector.lsnl.curves.relative.evis_parts",
                replicate_outputs=index["lsnl"],
            )
            # TODO
            nodes["detector.lsnl.curves.relative.evis_parts_individual.nominal"] = nodes.get_value(
                "detector.lsnl.curves.relative.evis_parts.nominal"
            )
            outputs["detector.lsnl.curves.relative.evis_parts_individual.nominal"] = (
                outputs.get_value("detector.lsnl.curves.relative.evis_parts.nominal")
            )
            Sum.replicate(
                outputs.get_dict("detector.lsnl.curves.relative.evis_parts"),
                outputs.get_value("detector.lsnl.curves.relative.evis_parts.nominal"),
                name="detector.lsnl.curves.relative.evis_parts_individual",
                replicate_outputs=index["lsnl_nuisance"],
                allow_skip_inputs=True,
                skippable_inputs_should_contain=["nominal"],
            )

            Product.replicate(
                outputs.get_dict("detector.lsnl.curves.relative.evis_parts"),
                parameters.get_dict("constrained.detector.lsnl_scale_a"),
                name="detector.lsnl.curves.relative.evis_parts_scaled",
                allow_skip_inputs=True,
                skippable_inputs_should_contain=("nominal",),
                replicate_outputs=index["lsnl_nuisance"],
            )

            Sum.replicate(
                outputs.get_value("detector.lsnl.curves.relative.evis_parts.nominal"),
                outputs.get_dict("detector.lsnl.curves.relative.evis_parts_scaled"),
                name="detector.lsnl.curves.relative.evis_coarse",
            )

            # Uncorrelated between detector energy scale correction is applied together
            # with LSNL with a single matrix. In order to do this the common LSNL curve
            # will be multiplied by a factor for each detector.
            #
            # The relative energy scale parameters are partially correlated with
            # relative efficiencies and thus are provided in pairs. Before using them,
            # we need to first extract a map of energy scale parameters for each
            # detector, which is done by `remap_items` function.
            remap_items(
                parameters.get_dict("all.detector.detector_relative"),
                parameters.create_child("selected.detector.parameters_relative"),
                reorder_indices={
                    "from": ["detector", "parameters"],
                    "to": ["parameters", "detector"],
                },
            )
            # Now there is a storage for energy scale parameters
            # "selected.detector.parameters_relative.energy_scale_factor".

            # In order to build the LSNL+energy scale conversion matrix a most precise
            # way requires two branches of interpolations:
            # - forward — modify Escint bin edges with energy scale conversion.
            # - backward — modify Evis bin edges with inverse energy scale conversion.
            # Interpolate Evis(Escint)

            # TODO: documentation
            Product.replicate(
                outputs.get_value("detector.lsnl.curves.evis_coarse"),
                parameters.get_dict("selected.detector.parameters_relative.energy_scale_factor"),
                name="detector.lsnl.curves.evis_coarse_scaled",
                replicate_outputs=index["detector"],
            )

            AxisDistortionMatrixPointwise.replicate(
                name="detector.lsnl.matrix",
                replicate_outputs=index["detector"],
            )
            edges_energy_escint.outputs[0] >> inputs.get_dict("detector.lsnl.matrix.EdgesOriginal")
            edges_energy_evis.outputs[0] >> inputs.get_dict("detector.lsnl.matrix.EdgesTarget")

            outputs.get_value("detector.lsnl.curves.escint") >> inputs.get_dict(
                "detector.lsnl.matrix.DistortionOriginal",
            )
            outputs.get_dict("detector.lsnl.curves.evis_coarse_scaled") >> inputs.get_dict(
                "detector.lsnl.matrix.DistortionTarget",
            )

            # Finally as in the case with IAV apply distortions to the spectra for each
            # detector and period.
            VectorMatrixProduct.replicate(
                name="eventscount.stages.evis",
                mode="column",
                replicate_outputs=combinations["detector.period"],
            )
            outputs.get_dict("detector.lsnl.matrix") >> inputs.get_dict(
                "eventscount.stages.evis.matrix"
            )
            # Use outputs after IAV correction to serve as inputs.
            outputs.get_dict("eventscount.stages.iav") >> inputs.get_dict(
                "eventscount.stages.evis.vector"
            )

            # The smearing due to finite energy resolution is also defined by three
            # nodes:
            # - σ-node — the node which computes the width of the resolution.
            # - matrix node — the node, which creates a smearing matrix based on σ(E).
            # - VectorMatrixProduct node — the one, which applies the smearing matrix.
            #
            # The first two nodes are managed via meta node EnergyResolution class.
            # These kind of classes do create multiple nodes inside, interconnect them
            # together and pass the inputs and outputs for external use.
            # In this particular case the instance will manage have
            # - EnergyResolutionSigmaRelABC to compute σ(E)/E = sqrt(a² + b²/E + c²/E²)
            # - EnergyResolutionMatrixBC to compute the matrix based on smearing at Bin
            #   Centers (BC)
            # - BinCenter — tiny node to compute bin centers.
            #
            # TODO: target edges (to output_edges)
            EnergyResolution.replicate(path="detector.eres")

            # Pass energy resolution parameters a_nonuniform, b_stat, c_noise to the
            # common σ-node.
            nodes.get_value("detector.eres.sigma_rel") << parameters.get_dict(
                "constrained.detector.eres"
            )
            # Pass bin edges for visible energy σ(E) to compute bin centers to pass to
            # the σ(E).
            outputs.get_value("edges.energy_evis") >> inputs.get_value("detector.eres.e_edges")
            # Pass bin edges for visible energy (input) to the matrix.
            outputs.get_value("edges.energy_evis") >> inputs.get_value(
                "detector.eres.matrix.e_edges"
            )
            # Pass bin edges for reconstructed energy (input) to the matrix.
            outputs.get_value("edges.energy_erec") >> inputs.get_value(
                "detector.eres.matrix.e_edges_out"
            )

            # Finally as on previous steps compute a product of a common energy
            # resolution matrix and input spectrum (after LSNL) for each detector during
            # period.
            VectorMatrixProduct.replicate(
                name="eventscount.stages.erec",
                mode="column",
                replicate_outputs=combinations["detector.period"],
            )
            outputs.get_value("detector.eres.matrix") >> inputs.get_dict(
                "eventscount.stages.erec.matrix"
            )
            outputs.get_dict("eventscount.stages.evis") >> inputs.get_dict(
                "eventscount.stages.erec.vector"
            )

            # Compute a product of global normalization and per-detector efficiency
            # factor, to be used to scale the IBD spectrum.
            Product.replicate(
                parameters.get_value("all.detector.global_normalization"),
                parameters.get_dict("selected.detector.parameters_relative.efficiency_factor"),
                parameters.get_value("all.detector.detector_absolute.efficiency_factor"),
                name="detector.normalization",
                replicate_outputs=index["detector"],
            )

            # Apply individual normalization (per detector) to each detectors prediction
            # during each period.
            Product.replicate(
                outputs.get_dict("detector.normalization"),
                outputs.get_dict("eventscount.stages.erec"),
                name="eventscount.fine.ibd_normalized",
                replicate_outputs=combinations["detector.period"],
            )

            # Sum together the spectra of each detector from different periods.
            # The corresponding outputs are to be used simply to provide the spectra for
            # user for plotting/saving. They are not used further in the calculations.
            Sum.replicate(
                outputs.get_dict("eventscount.fine.ibd_normalized"),
                name="eventscount.fine.ibd_normalized_detector",
                replicate_outputs=combinations["detector"],
            )

            # Rebin the expected histograms for each detector during each period into
            # final binning. It is done by MetaNode Rebin, which combine the computation
            # of the rebin matrix and its application via VectorMatrixProduct.
            Rebin.replicate(
                names={
                    "matrix": "detector.rebin.matrix_ibd",
                    "product": "eventscount.final.ibd",
                },
                replicate_outputs=combinations["detector.period"],
            )
            # Connect old (Erec) and new (final) energy edges.
            edges_energy_erec >> inputs.get_value("detector.rebin.matrix_ibd.edges_old")
            edges_energy_final >> inputs.get_value("detector.rebin.matrix_ibd.edges_new")
            # Pass the fine-bin spectra into inputs.
            outputs.get_dict("eventscount.fine.ibd_normalized") >> inputs.get_dict(
                "eventscount.final.ibd"
            )

            # The following block is related to the computation and application of the
            # background spectra. The background rates are given per day, therefore
            # first we need to convert the effective livetime in s to day. This will be
            # used for most of the sources of backgrounds, except for accidentals as
            # accidental rates are given on a daily basis.
            Product.replicate(
                outputs.get_dict("detector.eff_livetime"),
                parameters.get_value("constant.conversion.seconds_in_day_inverse"),
                name="detector.eff_livetime_days",
                replicate_outputs=combinations["detector.period"],
                allow_skip_inputs=True,
                skippable_inputs_should_contain=inactive_detectors,
            )

            # Compute the daily number accidental background events by multiplying daliy
            # rate of accidentals and daily effective livetime. The temporary unit is
            # [#·s/day].
            Product.replicate(
                outputs.get_dict("daily_data.detector.eff_livetime"),
                outputs.get_dict("daily_data.detector.rate_accidentals"),
                name="daily_data.detector.num_acc_s_day",
                replicate_outputs=combinations["detector.period"],
            )

            # Sum the contents of each array to obtain the total number of accidental
            # events in each detector during each period. Still [#·s/day].
            ArraySum.replicate(
                outputs.get_dict("daily_data.detector.num_acc_s_day"),
                name="background.count_acc_fixed_s_day",
            )

            # Finally, normalize the unit by dividing by number of seconds in day and
            # obtain total number of accidentals.
            Product.replicate(
                outputs.get_dict("background.count_acc_fixed_s_day"),
                parameters.get_value("constant.conversion.seconds_in_day_inverse"),
                name="background.count_fixed.accidentals",
                replicate_outputs=combinations["detector.period"],
            )

            # Now we need to load the normalized spectra for each backgrounds source.
            # There are different files for each period. Within each file the spectra
            # for each background and each detector are located.
            # HERE
            load_hist(
                name="background",
                x="erec",
                y="spectrum_shape",
                merge_x=True,
                normalize=True,
                filenames=cfg_file_mapping["background_spectra"],
                replicate_files=index["period"],
                replicate_outputs=combinations["background.detector"],
                skip=inactive_detectors,
                key_order=(
                    ("period", "background", "detector"),
                    ("background", "detector", "period"),
                ),
                name_function=lambda _, idx: f"spectrum_shape_{idx[0]}_{idx[1]}",
            )

            # HERE get_value/get_dict
            # fmt: off
            # TODO: labels
            Product.replicate(
                parameters("all.background.rate"),
                outputs("detector.eff_livetime_days"),
                name="background.count_fixed",
                replicate_outputs=combinations["background_stable.detector.period"],
            )

            # TODO: labels
            Product.replicate(
                parameters("all.background.rate_scale.accidentals"),
                outputs("background.count_fixed.accidentals"),
                name="background.count.accidentals",
                replicate_outputs=combinations["detector.period"],
            )

            remap_items(
                parameters.get_dict("constrained.background.uncertainty_scale_by_site"),
                parameters.create_child("selected.background.uncertainty_scale"),
                rename_indices=site_arrangement,
                skip_indices_target=inactive_detectors,
            )

            # TODO: labels
            ProductShiftedScaled.replicate(
                outputs("background.count_fixed"),
                parameters("sigma.background.rate"),
                parameters.get_dict("selected.background.uncertainty_scale"),
                name="background.count",
                shift=1.0,
                replicate_outputs=combinations["background_site_correlated.detector.period"],
                allow_skip_inputs=True,
                skippable_inputs_should_contain=combinations[
                    "background_not_site_correlated.detector.period"
                ],
            )

            # TODO: labels
            ProductShiftedScaled.replicate(
                outputs("background.count_fixed.amc"),
                parameters("sigma.background.rate.amc"),
                parameters.get_value("all.background.uncertainty_scale.amc"),
                name="background.count.amc",
                shift=1.0,
                replicate_outputs=combinations["detector.period"],
            )

            outputs["background.count.alpha_neutron"] = outputs.get_dict("background.count_fixed.alpha_neutron")

            # TODO: labels
            Product.replicate(
                outputs("background.count"),
                outputs("background.spectrum_shape"),
                name="background.spectrum",
                replicate_outputs=combinations["background.detector.period"],
            )

            Sum.replicate(
                outputs("background.spectrum"),
                name="eventscount.fine.background",
                replicate_outputs=combinations["detector.period"],
            )

            Sum.replicate(
                outputs("background.spectrum"),
                name="eventscount.fine.background_by_source",
                replicate_outputs=combinations["background.detector"],
            )

            Rebin.replicate(
                names={
                    "matrix": "detector.rebin.matrix_background_by_source",
                    "product": "eventscount.final.background_by_source",
                },
                replicate_outputs=combinations["background.detector"],
            )
            edges_energy_erec >> inputs.get_value("detector.rebin.matrix_background_by_source.edges_old")
            edges_energy_final >> inputs.get_value(
                "detector.rebin.matrix_background_by_source.edges_new"
            )
            outputs("eventscount.fine.background_by_source") >> inputs("eventscount.final.background_by_source")

            Sum.replicate(
                outputs("eventscount.fine.ibd_normalized"),
                outputs("eventscount.fine.background"),
                name="eventscount.fine.total",
                replicate_outputs=combinations["detector.period"],
                check_edges_contents=True,
            )

            Rebin.replicate(
                names={
                    "matrix": "detector.rebin.matrix_background",
                    "product": "eventscount.final.background",
                },
                replicate_outputs=combinations["detector.period"],
            )
            edges_energy_erec >> inputs.get_value("detector.rebin.matrix_background.edges_old")
            edges_energy_final >> inputs.get_value(
                "detector.rebin.matrix_background.edges_new"
            )
            outputs("eventscount.fine.background") >> inputs("eventscount.final.background")

            Sum.replicate(
                outputs("eventscount.final.ibd"),
                outputs("eventscount.final.background"),
                name="eventscount.final.detector_period",
                replicate_outputs=combinations["detector.period"],
            )

            remap_items(
                outputs.get_dict("eventscount.final.detector_period"),
                outputs.create_child("eventscount.final.detector_period_selected"),
                skip_indices_target=index["detector_excluded"],
            )

            Concatenation.replicate(
                outputs("eventscount.final.detector_period_selected"),
                name="eventscount.final.concatenated.detector_period",
            )

            Sum.replicate(
                outputs("eventscount.final.detector_period_selected"),
                name="eventscount.final.detector",
                replicate_outputs=index["detector_selected"],
            )

            Concatenation.replicate(
                outputs("eventscount.final.detector"),
                name="eventscount.final.concatenated.detector",
            )

            outputs["eventscount.final.concatenated.selected"] = outputs.get_value(
                f"eventscount.final.concatenated.{self.concatenation_mode}"
                )

            #
            # Covariance matrices
            #
            self._covariance_matrix = CovarianceMatrixGroup(store_to="covariance")

            for group in self._covariance_groups:
                self._covariance_matrix.add_covariance_for(
                    group, parameters_nuisance_normalized[
                    self.systematic_uncertainties_groups[group]
                ])
            self._covariance_matrix.add_covariance_sum()

            (
                outputs.get_value("eventscount.final.concatenated.selected")
                >> self._covariance_matrix
            )

            list_parameters_nuisance_normalized = list(
                parameters_nuisance_normalized.walkvalues()
            )
            npars_nuisance = len(list_parameters_nuisance_normalized)

            parinp_mc = ParArrayInput(
                name="mc.parameters.inputs",
                parameters=list_parameters_nuisance_normalized,
                tainted=False
            )

            #
            # Real data
            #
            load_hist(
                name="data.real",
                x="erec",
                y="fine",
                merge_x=True,
                filenames=cfg_file_mapping["dataset"],
                replicate_files=index["period"],
                replicate_outputs=combinations["detector"],
                skip=inactive_detectors,
                dtype="d",
                name_function=lambda _, idx: f"ibd_spectrum_{idx[1]}",
            )

            Rebin.replicate(
                names={
                    "matrix": "detector.rebin_matrix.real",
                    "product": "data.real.final.detector_period",
                },
                replicate_outputs=combinations["detector.period"],
            )
            edges_energy_erec >> inputs.get_value(
                "detector.rebin_matrix.real.edges_old"
            )
            edges_energy_final >> inputs.get_value(
                "detector.rebin_matrix.real.edges_new"
            )
            outputs["data.real.fine"] >> inputs.get_dict("data.real.final.detector_period")

            remap_items(
                outputs.get_dict("data.real.final.detector_period"),
                outputs.create_child("data.real.final.detector_period_selected"),
                skip_indices_target=index["detector_excluded"],
            )

            Concatenation.replicate(
                outputs("data.real.final.detector_period_selected"),
                name="data.real.concatenated.detector_period",
            )

            Sum.replicate(
                outputs("data.real.final.detector_period_selected"),
                name="data.real.final.detector",
                replicate_outputs=index["detector_selected"],
            )

            Concatenation.replicate(
                outputs.get_dict("data.real.final.detector"),
                name="data.real.concatenated.detector",
            )

            outputs["data.real.concatenated.selected"] = outputs.get_value(
                f"data.real.concatenated.{self.concatenation_mode}"
                )

            #
            # Summary
            # Collect some summary data for output tables
            #
            ArraySum.replicate(
                outputs("data.real.final.detector"),
                name="summary.total.ibd_candidates",
            )

            ArraySum.replicate(
                outputs("data.real.final.detector_period"),
                name="summary.periods.ibd_candidates",
            )
            outputs["summary.periods.ibd_candidates"] = remap_items(
                outputs.get_dict("summary.periods.ibd_candidates"),
                reorder_indices={
                    "from": ["detector", "period"],
                    "to": ["period", "detector"],
                },
            )

            Sum.replicate(
                outputs("detector.livetime"),
                name="summary.total.livetime",
                replicate_outputs=index["detector"],
            )

            Sum.replicate(
                outputs("detector.livetime"),
                name="summary.periods.livetime",
                replicate_outputs=combinations["period.detector"],
            )

            Sum.replicate(
                outputs("detector.eff_livetime"),
                name="summary.total.eff_livetime",
                replicate_outputs=index["detector"],
            )

            Sum.replicate(
                outputs("detector.eff_livetime"),
                name="summary.periods.eff_livetime",
                replicate_outputs=combinations["period.detector"],
            )

            Division.replicate(
                outputs("summary.total.eff_livetime"),
                outputs("summary.total.livetime"),
                name="summary.total.eff",
                replicate_outputs=index["detector"],
            )

            Division.replicate(
                outputs("summary.periods.eff_livetime"),
                outputs("summary.periods.livetime"),
                name="summary.periods.eff",
                replicate_outputs=combinations["period.detector"],
            )

            Sum.replicate(
                outputs("background.count"),
                name="summary.total.background_count",
                replicate_outputs=combinations["background.detector"],
            )

            remap_items(
                outputs("background.count"),
                outputs.create_child("summary.periods.background_count"),
                reorder_indices={
                    "from": ["background", "detector", "period"],
                    "to": ["background", "period", "detector"],
                },
            )

            Division.replicate(
                outputs("summary.total.background_count"),
                outputs("summary.total.eff_livetime"),
                name="summary.total.background_rate_s",
                replicate_outputs=combinations["background.detector"],
            )

            Division.replicate(
                outputs("summary.periods.background_count"),
                outputs("summary.periods.eff_livetime"),
                name="summary.periods.background_rate_s",
                replicate_outputs=combinations["background.period.detector"],
            )

            Product.replicate(
                outputs("summary.total.background_rate_s"),
                parameters["constant.conversion.seconds_in_day"],
                name="summary.total.background_rate",
                replicate_outputs=combinations["background.detector"],
            )

            Product.replicate(
                outputs("summary.periods.background_rate_s"),
                parameters["constant.conversion.seconds_in_day"],
                name="summary.periods.background_rate",
                replicate_outputs=combinations["background.period.detector"],
            )

            Sum.replicate(
                outputs("summary.total.background_rate"),
                name="summary.total.background_rate_total",
                replicate_outputs=index["detector"],
            )

            Sum.replicate(
                outputs("summary.periods.background_rate"),
                name="summary.periods.background_rate_total",
                replicate_outputs=combinations["period.detector"],
            )

            Division.replicate(
                outputs("summary.total.ibd_candidates"),
                outputs("summary.total.eff_livetime"),
                name="summary.total.rate_ibd_candidates_s",
                replicate_outputs=combinations["detector"],
            )

            Division.replicate(
                outputs("summary.periods.ibd_candidates"),
                outputs("summary.periods.eff_livetime"),
                name="summary.periods.rate_ibd_candidates_s",
                replicate_outputs=combinations["period.detector"],
            )

            Product.replicate(
                outputs("summary.total.rate_ibd_candidates_s"),
                parameters["constant.conversion.seconds_in_day"],
                name="summary.total.rate_ibd_candidates",
                replicate_outputs=combinations["detector"],
            )

            Product.replicate(
                outputs("summary.periods.rate_ibd_candidates_s"),
                parameters["constant.conversion.seconds_in_day"],
                name="summary.periods.rate_ibd_candidates",
                replicate_outputs=combinations["period.detector"],
            )

            Difference.replicate(
                outputs("summary.total.rate_ibd_candidates"),
                outputs("summary.total.background_rate"),
                name="summary.total.rate_ibd",
                replicate_outputs=combinations["detector"],
            )

            Difference.replicate(
                outputs("summary.periods.rate_ibd_candidates"),
                outputs("summary.periods.background_rate"),
                name="summary.periods.rate_ibd",
                replicate_outputs=combinations["period.detector"],
            )

            #
            # Statistic
            #
            # Create Nuisance parameters
            Sum.replicate(
                outputs("statistic.nuisance.parts"), name="statistic.nuisance.all"
            )
            if self._pull_groups:
                Sum.replicate(
                    *[
                        outputs[f"statistic.nuisance.parts.{self.systematic_uncertainties_groups[group]}"] for group
                        in self._pull_groups
                    ],
                    name="statistic.nuisance.pull_extra"
                )
            else:
                Array.replicate(name="statistic.nuisance.pull_extra", array=[0])

            MonteCarlo.replicate(
                name="data.pseudo.self",
                mode=self.monte_carlo_mode,
                generator=self._random_generator,
            )
            outputs.get_value(
                "eventscount.final.concatenated.selected"
            ) >> inputs.get_value("data.pseudo.self.data")
            self._frozen_nodes["pseudodata"] = (nodes.get_value("data.pseudo.self"),)

            Proxy.replicate(
                name="data.proxy",
            )
            outputs.get_value("data.pseudo.self") >> inputs.get_value(
                "data.proxy.input"
            )
            outputs.get_value("data.real.concatenated.selected") >> nodes["data.proxy"]

            MonteCarlo.replicate(
                name="covariance.data.fixed",
                mode="asimov",
                generator=self._random_generator,
            )
            outputs.get_value(
                "eventscount.final.concatenated.selected"
            ) >> inputs.get_value("covariance.data.fixed.data")
            self._frozen_nodes["covariance_data_fixed"] = (
                nodes.get_value("covariance.data.fixed"),
            )

            MonteCarlo.replicate(
                name="mc.parameters.toymc",
                mode="normal-unit",
                shape=(npars_nuisance,),
                generator=self._random_generator,
                tainted=False,
                frozen=True
            )
            outputs.get_value("mc.parameters.toymc") >> parinp_mc
            nodes["mc.parameters.inputs"] = parinp_mc

            #
            # Covariance matrices and Cholesky decomposition
            #
            Cholesky.replicate(name="cholesky.stat.variable")
            outputs.get_value(
                "eventscount.final.concatenated.selected"
            ) >> inputs.get_value("cholesky.stat.variable")

            Cholesky.replicate(name="cholesky.stat.fixed")
            outputs.get_value("covariance.data.fixed") >> inputs.get_value(
                "cholesky.stat.fixed"
            )

            Cholesky.replicate(name="cholesky.stat.data.fixed")
            outputs.get_value("data.proxy") >> inputs.get_value(
                "cholesky.stat.data.fixed"
            )

            SumMatOrDiag.replicate(name="covariance.covmat_full_p.fixed_stat")
            outputs.get_value("covariance.data.fixed") >> nodes.get_value(
                "covariance.covmat_full_p.fixed_stat"
            )
            outputs.get_value("covariance.covmat_syst.sum") >> nodes.get_value(
                "covariance.covmat_full_p.fixed_stat"
            )

            Cholesky.replicate(name="cholesky.covmat_full_p.fixed_stat")
            outputs.get_value(
                "covariance.covmat_full_p.fixed_stat"
            ) >> inputs.get_value("cholesky.covmat_full_p.fixed_stat")

            SumMatOrDiag.replicate(name="covariance.covmat_full_p.variable_stat")
            outputs.get_value(
                "eventscount.final.concatenated.selected"
            ) >> nodes.get_value("covariance.covmat_full_p.variable_stat")
            outputs.get_value("covariance.covmat_syst.sum") >> nodes.get_value(
                "covariance.covmat_full_p.variable_stat"
            )

            Cholesky.replicate(name="cholesky.covmat_full_p.variable_stat")
            outputs.get_value(
                "covariance.covmat_full_p.variable_stat"
            ) >> inputs.get_value("cholesky.covmat_full_p.variable_stat")


            SumMatOrDiag.replicate(name="covariance.covmat_full_n")
            outputs.get_value("data.proxy") >> nodes.get_value(
                "covariance.covmat_full_n"
            )
            outputs.get_value("covariance.covmat_syst.sum") >> nodes.get_value(
                "covariance.covmat_full_n"
            )

            Cholesky.replicate(name="cholesky.covmat_full_n")
            outputs.get_value("covariance.covmat_full_n") >> inputs.get_value(
                "cholesky.covmat_full_n"
            )

            #
            # Chi-squared functions
            #

            # Chi-squared Pearson, stat (fixed stat errors)
            Chi2.replicate(name="statistic.stat.chi2p_iterative")
            outputs.get_value(
                "eventscount.final.concatenated.selected"
            ) >> inputs.get_value("statistic.stat.chi2p_iterative.theory")
            outputs.get_value("cholesky.stat.fixed") >> inputs.get_value(
                "statistic.stat.chi2p_iterative.errors"
            )
            outputs.get_value("data.proxy") >> inputs.get_value(
                "statistic.stat.chi2p_iterative.data"
            )

            # Chi-squared Neyman, stat
            Chi2.replicate(name="statistic.stat.chi2n")
            outputs.get_value(
                "eventscount.final.concatenated.selected"
            ) >> inputs.get_value("statistic.stat.chi2n.theory")
            outputs.get_value("cholesky.stat.data.fixed") >> inputs.get_value(
                "statistic.stat.chi2n.errors"
            )
            outputs.get_value("data.proxy") >> inputs.get_value(
                "statistic.stat.chi2n.data"
            )

            # Chi-squared Pearson, stat (variable stat errors)
            Chi2.replicate(name="statistic.stat.chi2p")
            outputs.get_value(
                "eventscount.final.concatenated.selected"
            ) >> inputs.get_value("statistic.stat.chi2p.theory")
            outputs.get_value("cholesky.stat.variable") >> inputs.get_value(
                "statistic.stat.chi2p.errors"
            )
            outputs.get_value("data.proxy") >> inputs.get_value(
                "statistic.stat.chi2p.data"
            )

            # Chi-squared Pearson, stat+syst, cov. matrix (fixed stat errors)
            Chi2.replicate(name="statistic.full.covmat.chi2p_iterative")
            outputs.get_value("data.proxy") >> inputs.get_value(
                "statistic.full.covmat.chi2p_iterative.data"
            )
            outputs.get_value(
                "eventscount.final.concatenated.selected"
            ) >> inputs.get_value("statistic.full.covmat.chi2p_iterative.theory")
            outputs.get_value("cholesky.covmat_full_p.fixed_stat") >> inputs.get_value(
                "statistic.full.covmat.chi2p_iterative.errors"
            )

            # Chi-squared Neyman, stat+syst, cov. matrix
            Chi2.replicate(name="statistic.full.covmat.chi2n")
            outputs.get_value("data.proxy") >> inputs.get_value(
                "statistic.full.covmat.chi2n.data"
            )
            outputs.get_value(
                "eventscount.final.concatenated.selected"
            ) >> inputs.get_value("statistic.full.covmat.chi2n.theory")
            outputs.get_value("cholesky.covmat_full_n") >> inputs.get_value(
                "statistic.full.covmat.chi2n.errors"
            )

            # Chi-squared Pearson, stat+syst, cov. matrix (variable stat errors)
            Chi2.replicate(name="statistic.full.covmat.chi2p")
            outputs.get_value("data.proxy") >> inputs.get_value(
                "statistic.full.covmat.chi2p.data"
            )
            outputs.get_value(
                "eventscount.final.concatenated.selected"
            ) >> inputs.get_value("statistic.full.covmat.chi2p.theory")
            outputs.get_value(
                "cholesky.covmat_full_p.variable_stat"
            ) >> inputs.get_value("statistic.full.covmat.chi2p.errors")

            LogProdDiag.replicate(name="statistic.log_prod_diag.full")
            outputs.get_value(
                "cholesky.covmat_full_p.variable_stat"
            ) >> inputs.get_value("statistic.log_prod_diag.full")

            # Chi-squared Pearson, stat+syst, cov. matrix (variable stat errors)
            Sum.replicate(
                outputs.get_value("statistic.full.covmat.chi2p"),
                outputs.get_value("statistic.log_prod_diag.full"),
                name="statistic.full.covmat.chi2p_unbiased",
            )

            # CNP stat error
            CNPStat.replicate(name="statistic.staterr.cnp")
            outputs.get_value("data.proxy") >> inputs.get_value(
                "statistic.staterr.cnp.data"
            )
            outputs.get_value(
                "eventscount.final.concatenated.selected"
            ) >> inputs.get_value("statistic.staterr.cnp.theory")

            # Chi-squared CNP, stat
            Chi2.replicate(name="statistic.stat.chi2cnp")
            outputs.get_value("data.proxy") >> inputs.get_value(
                "statistic.stat.chi2cnp.data"
            )
            outputs.get_value(
                "eventscount.final.concatenated.selected"
            ) >> inputs.get_value("statistic.stat.chi2cnp.theory")
            outputs.get_value("statistic.staterr.cnp") >> inputs.get_value(
                "statistic.stat.chi2cnp.errors"
            )

            # Log Poisson Ratio
            LogPoissonRatio.replicate(name="statistic.stat.chi2poisson")
            outputs.get_value("data.proxy") >> inputs.get_value(
                "statistic.stat.chi2poisson.data"
            )
            outputs.get_value(
                "eventscount.final.concatenated.selected"
            ) >> inputs.get_value("statistic.stat.chi2poisson.theory")

            # Chi-squared Pearson, stat+syst, pull (fixed stat errors)
            Sum.replicate(
                outputs.get_value("statistic.stat.chi2p_iterative"),
                outputs.get_value("statistic.nuisance.all"),
                name="statistic.full.pull.chi2p_iterative",
            )

            # Chi-squared Pearson, stat+syst, pull (variable stat errors)
            Sum.replicate(
                outputs.get_value("statistic.stat.chi2p"),
                outputs.get_value("statistic.nuisance.all"),
                name="statistic.full.pull.chi2p",
            )

            # Chi-squared CNP, stat+syst, pull
            Sum.replicate(
                outputs.get_value("statistic.stat.chi2cnp"),
                outputs.get_value("statistic.nuisance.all"),
                name="statistic.full.pull.chi2cnp",
            )

            LogProdDiag.replicate(name="statistic.log_prod_diag.stat")
            outputs.get_value(
                "cholesky.stat.variable"
            ) >> inputs.get_value("statistic.log_prod_diag.stat")

            # Chi-squared Pearson, stat, +log|Vstat| (variable stat errors)
            Sum.replicate(
                outputs.get_value("statistic.stat.chi2p"),
                outputs.get_value("statistic.log_prod_diag.stat"),
                name="statistic.stat.chi2p_unbiased",
            )

            # Chi-squared Pearson, stat+syst, pull, +log|V| (variable stat errors)
            Sum.replicate(
                outputs.get_value("statistic.stat.chi2p_unbiased"),
                outputs.get_value("statistic.nuisance.all"),
                name="statistic.full.pull.chi2p_unbiased",
            )

            # CNP stat variance
            CNPStat.replicate(name="statistic.staterr.cnp_variance", mode="variance")
            outputs.get_value("data.proxy") >> inputs.get_value(
                "statistic.staterr.cnp_variance.data"
            )
            outputs.get_value(
                "eventscount.final.concatenated.selected"
            ) >> inputs.get_value("statistic.staterr.cnp_variance.theory")

            # CNP, stat+syst, cov. matrix (linear cobination)
            SumMatOrDiag.replicate(
                    outputs.get_value("statistic.staterr.cnp_variance"),
                    outputs.get_value("covariance.covmat_syst.sum"),
                    name = "covariance.covmat_full_cnp"
                    )

            # CNP Cholesky
            Cholesky.replicate(name="cholesky.covmat_full_cnp")
            outputs.get_value(
                "covariance.covmat_full_cnp"
            ) >> inputs.get_value("cholesky.covmat_full_cnp")

            # CNP, stat+syst, cov. matrix (as in the paper)
            Chi2.replicate(name="statistic.full.covmat.chi2cnp")
            outputs.get_value("data.proxy") >> inputs.get_value(
                "statistic.full.covmat.chi2cnp.data"
            )
            outputs.get_value(
                "eventscount.final.concatenated.selected"
            ) >> inputs.get_value("statistic.full.covmat.chi2cnp.theory")
            outputs.get_value(
                "cholesky.covmat_full_cnp"
            ) >> inputs.get_value("statistic.full.covmat.chi2cnp.errors")

            # Log Poisson Ratio, stat+syst, pull
            Sum.replicate(
                outputs.get_value("statistic.stat.chi2poisson"),
                outputs.get_value("statistic.nuisance.all"),
                name="statistic.full.pull.chi2poisson",
            )
            # fmt: on

        self._setup_labels()

        # Model will load real data
        self.switch_data("real")

        # Ensure stem nodes are calculated
        self._touch()

    @staticmethod
    def _create_random_generator(seed: int) -> Generator:
        from numpy.random import MT19937, SeedSequence

        (sequence,) = SeedSequence(seed).spawn(1)
        algo = MT19937(seed=sequence.spawn(1)[0])
        return Generator(algo)

    def _touch(self):
        for output in self.storage["outputs"].get_dict("eventscount.final.detector").walkvalues():
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
        parameter_values: Mapping[str, float | str] | Sequence[tuple[str, float | int]] = (),
        *,
        mode: Literal["value", "normvalue"] = "value",
    ):
        parameters_storage = self.storage("parameters.all")
        if isinstance(parameter_values, Mapping):
            iterable = parameter_values.items()
        else:
            iterable = parameter_values

        match mode:
            case "value":

                def setter(par, value):
                    par.push(value)
                    print(f"Push {parname}={svalue}")

            case "normvalue":

                def setter(par, value):
                    par.normvalue = value
                    print(f"Set norm {parname}={svalue}")

            case _:
                raise ValueError(mode)

        for parname, svalue in iterable:
            value = float(svalue)
            par = parameters_storage[parname]
            setter(par, value)

    def switch_data(self, key: Literal["asimov", "real"]) -> None:
        """Switch data.proxy output.

        Parameters
        ----------
        type : Literal["asimov", "real"]
            Choice for switching, Asimov or real data observation

        Returns
        -------
        None
        """
        if key not in {"asimov", "real"}:
            raise KeyError(f"Switch to `{key}` is not supported, `asimov`, `real` supported only")
        self.storage["nodes.data.proxy"].switch_input({"asimov": 0, "real": 1}[key])

    def next_sample(self, *, mc_parameters: bool = True, mc_statistics: bool = True) -> None:
        if mc_parameters:
            self.storage.get_value("nodes.mc.parameters.toymc").next_sample()
            self.storage.get_value("nodes.mc.parameters.inputs").touch()

        if mc_statistics:
            self.storage.get_value("nodes.data.pseudo.self").next_sample()

        if mc_parameters:
            self.storage.get_value("nodes.mc.parameters.toymc").reset()
            self.storage.get_value("nodes.mc.parameters.inputs").touch()

    @property
    def systematic_uncertainties_groups(self) -> dict[str, str]:
        # TODO: update logic
        if self._is_absolute_efficiency_fixed:
            return {
                group: parname
                for group, parname in _SYSTEMATIC_UNCERTAINTIES_GROUPS.items()
                if group != "absolute_efficiency"
            }
        else:
            return {group: parname for group, parname in _SYSTEMATIC_UNCERTAINTIES_GROUPS.items()}

    def _setup_labels(self):
        from dag_modelling.tools.schema import LoadYaml

        processed_keys_set = set()
        if self._process_labels:
            logger.log(INFO, "Processing labels")
            labels = LoadYaml(relpath(__file__.replace(".py", ".yaml")))
            self.storage("nodes").read_labels(labels, processed_keys_set=processed_keys_set)
            self.storage("outputs").read_labels(labels, processed_keys_set=processed_keys_set)

        self.storage("inputs").remove_connected_inputs()
        self.storage.read_paths(index=self.index)
        self.graph.build_index_dict(self.index)

        if not self._process_labels:
            return

        labels_mk = NestedMapping(labels, sep=".")  # pyright: ignore [reportPossiblyUnboundVariable]
        if not self._strict:
            return

        for key in processed_keys_set:
            labels_mk.delete_with_parents(key)

        if not labels_mk:
            return

        unused_keys = list(labels_mk.walkjoinedkeys())
        may_ignore = ["__common_definitions__"]

        for key_may_ignore in list(may_ignore):
            cleanup = False
            for i, key_unused in reversed(tuple(enumerate(unused_keys))):
                if key_unused.startswith(key_may_ignore):
                    del unused_keys[i]
                    cleanup = True
            if cleanup:
                may_ignore.remove(key_may_ignore)

        if may_ignore:
            raise RuntimeError(
                "The following items to ignore were not used, update the model._setup_labels:\n"
                f"{may_ignore}"
            )

        if not unused_keys:
            return

        raise RuntimeError(f"The following label groups were not used: {', '.join(unused_keys)}")

    def make_summary_table(
        self, period: Literal["total", "6AD", "8AD", "7AD"] = "total"
    ) -> DataFrame:
        match period:
            case "total":
                source_fmt = f"summary.{period}.{{name}}"
            case "6AD" | "8AD" | "7AD":
                source_fmt = f"summary.periods.{{name}}.{period}"
            case _:
                raise ValueError(period)

        column_sources = {
            "ibd_candidates": source_fmt.format(name="ibd_candidates"),
            "daq_time_day": source_fmt.format(name="livetime"),
            "daq_time_day_eff": source_fmt.format(name="eff_livetime"),
            "eff": source_fmt.format(name="eff"),
            "rate_accidentals": source_fmt.format(name="background_rate.accidentals"),
            "rate_fast_neutrons": source_fmt.format(name="background_rate.fast_neutrons"),
            "rate_lithium_helium": source_fmt.format(name="background_rate.lithium_helium"),
            "rate_amc": source_fmt.format(name="background_rate.amc"),
            "rate_alpha_neutron": source_fmt.format(name="background_rate.alpha_neutron"),
            "rate_background_total": source_fmt.format(name="background_rate_total"),
            "rate_ibd": source_fmt.format(name="rate_ibd"),
        }

        rows = list(self.index["detector"])
        columns = list(column_sources)
        df = DataFrame(index=rows, columns=columns, dtype="f8")

        for key, path in column_sources.items():
            try:
                source = self.storage["outputs"].get_dict(path)
            except KeyError:
                print("error", key)
                continue
            for k, output in source.walkitems():
                data = output.data
                assert data.size == 1
                value = output.data[0]

                df.loc[k, key] = value
        df[df.isna()] = 0.0

        df["daq_time_day"] /= 60.0 * 60.0 * 24.0

        df.reset_index(inplace=True, names=["name"])
        df = df.astype(
            {
                "name": str,
                "ibd_candidates": int,
            }
        )

        return df

    def print_summary_table(self):
        df = self.make_summary_table()
        print(df.to_string())
