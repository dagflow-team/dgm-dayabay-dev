from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Literal

from numpy import ndarray

from dagflow.bundles.file_reader import FileReader
from dagflow.bundles.load_array import load_array
from dagflow.bundles.load_graph import load_graph, load_graph_data
from dagflow.bundles.load_parameters import load_parameters
from dagflow.graph import Graph
from dagflow.lib.arithmetic import Division, Product, Sum
from dagflow.lib.InterpolatorGroup import InterpolatorGroup
from dagflow.storage import NodeStorage
from dagflow.tools.schema import LoadYaml
from multikeydict.nestedmkdict import NestedMKDict
from multikeydict.tools import remap_items

SourceTypes = Literal["tsv", "hdf5", "root", "npz"]


class model_dayabay_v0:
    __slots__ = (
        "storage",
        "graph",
        "inactive_detectors",
        "_override_indices",
        "_path_data",
        "_source_type",
        "_strict",
        "_close",
        "_spectrum_correction_mode",
        "_fission_fraction_normalized",
    )

    storage: NodeStorage
    graph: Graph | None
    inactive_detectors: tuple[set[str], ...]
    _path_data: Path
    _override_indices: Mapping[str, Sequence[str]]
    _source_type: SourceTypes
    _strict: bool
    _close: bool
    _spectrum_correction_mode: Literal["linear", "exponential"]
    _fission_fraction_normalized: bool

    def __init__(
        self,
        *,
        source_type: SourceTypes = "npz",
        strict: bool = True,
        close: bool = True,
        override_indices: Mapping[str, Sequence[str]] = {},
        spectrum_correction_mode: Literal["linear", "exponential"] = "exponential",
        fission_fraction_normalized: bool = False,
    ):
        self._strict = strict
        self._close = close

        self.graph = None
        self.storage = NodeStorage()
        self._path_data = Path("data/dayabay-v0")
        self._source_type = source_type
        self._override_indices = override_indices
        self._spectrum_correction_mode = spectrum_correction_mode
        self._fission_fraction_normalized = fission_fraction_normalized

        self.inactive_detectors = ({"6AD", "AD22"}, {"6AD", "AD34"}, {"7AD", "AD11"})

        self.build()

    def build(self):
        storage = self.storage
        path_data = self._path_data

        path_parameters = path_data / "parameters"
        path_arrays = path_data / self._source_type
        path_root = path_data / "root"

        from dagflow.tools.schema import LoadPy

        antineutrino_model_edges = LoadPy(
            path_parameters / "reactor_antineutrino_spectrum_edges.py",
            variable="edges",
            type=ndarray,
        )

        # Provide a list of indices and their values. Values should be globally unique
        index = {}
        index["isotope"] = ("U235", "U238", "Pu239", "Pu241")
        index["isotope_lower"] = tuple(i.lower() for i in index["isotope"])
        index["isotope_offeq"] = ("U235", "Pu239", "Pu241")
        index["detector"] = (
            "AD11",
            "AD12",
            "AD21",
            "AD22",
            "AD31",
            "AD32",
            "AD33",
            "AD34",
        )
        index["site"] = ("EH1", "EH2", "EH3")
        index["reactor"] = ("DB1", "DB2", "LA1", "LA2", "LA3", "LA4")
        index["anue_source"] = ("main", "offeq", "snf")
        index["period"] = ("6AD", "8AD", "7AD")
        index["lsnl"] = ("nominal", "pull0", "pull1", "pull2", "pull3")
        index["lsnl_nuisance"] = ("pull0", "pull1", "pull2", "pull3")
        # index["bkg"] = ('acc', 'lihe', 'fastn', 'amc', 'alphan', 'muon')
        index["bkg"] = ("acc", "lihe", "fastn", "amc", "alphan")
        index["spec"] = tuple(
            f"spec_scale_{i:02d}" for i in range(len(antineutrino_model_edges))
        )

        index.update(self._override_indices)

        index_all = (
            index["isotope"] + index["detector"] + index["reactor"] + index["period"]
        )
        set_all = set(index_all)
        if len(index_all) != len(set_all):
            raise RuntimeError("Repeated indices")

        required_combinations = (
            "reactor.detector",
            "reactor.isotope",
            "reactor.isotope_offeq",
            "reactor.period",
            "reactor.isotope.period",
            "reactor.isotope.detector",
            "reactor.isotope_offeq.detector",
            "reactor.isotope.detector.period",
            "reactor.isotope_offeq.detector.period",
            "reactor.detector.period",
            "detector.period",
            "bkg.detector",
            "bkg.detector.period",
        )
        # Provide the combinations of indices
        combinations = {}
        for combname in required_combinations:
            combitems = combname.split(".")
            items = []
            for it in product(*(index[item] for item in combitems)):
                if any(inact.issubset(it) for inact in self.inactive_detectors):
                    continue
                items.append(it)
            combinations[combname] = tuple(items)

        combinations["anue_source.reactor.isotope.detector"] = (
            tuple(("main",) + cmb for cmb in combinations["reactor.isotope.detector"])
            + tuple(
                ("offeq",) + cmb
                for cmb in combinations["reactor.isotope_offeq.detector"]
            )
            + tuple(("snf",) + cmb for cmb in combinations["reactor.detector"])
        )

        spectrum_correction_is_exponential = (
            self._spectrum_correction_mode == "exponential"
        )

        with Graph(
            close=self._close, strict=self._strict
        ) as graph, storage, FileReader:
            # fmt: off
            self.graph = graph
            #
            # Load parameters
            #
            load_parameters(path="oscprob",    load=path_parameters/"oscprob.yaml")
            load_parameters(path="oscprob",    load=path_parameters/"oscprob_solar.yaml", joint_nuisance=True)
            load_parameters(path="oscprob",    load=path_parameters/"oscprob_constants.yaml"
            )

            load_parameters(path="ibd",        load=path_parameters/"pdg2012.yaml")
            load_parameters(path="ibd.csc",    load=path_parameters/"ibd_constants.yaml"
            )
            load_parameters(path="conversion", load=path_parameters/"conversion_thermal_power.yaml")
            load_parameters(path="conversion", load=path_parameters/"conversion_oscprob_argument.yaml")

            load_parameters(                   load=path_parameters/"baselines.yaml")

            load_parameters(path="detector",   load=path_parameters/"detector_efficiency.yaml")
            load_parameters(path="detector",   load=path_parameters/"detector_normalization.yaml")
            load_parameters(path="detector",   load=path_parameters/"detector_nprotons_correction.yaml")
            load_parameters(path="detector",   load=path_parameters/"detector_eres.yaml")
            load_parameters(path="detector",   load=path_parameters/"detector_lsnl.yaml",
                            replicate=index["lsnl_nuisance"])
            load_parameters(path="detector",   load=path_parameters/"detector_relative_energy_scale.yaml",
                            replicate=index["detector"], replica_key_offset=1)

            load_parameters(path="reactor",    load=path_parameters/"reactor_energy_per_fission.yaml")
            load_parameters(path="reactor",    load=path_parameters/"reactor_thermal_power_nominal.yaml",
                            replicate=index["reactor"])
            load_parameters(path="reactor",    load=path_parameters/"reactor_snf.yaml",
                            replicate=index["reactor"])
            load_parameters(path="reactor",    load=path_parameters/"reactor_offequilibrium_correction.yaml",
                            replicate=combinations["reactor.isotope_offeq"])
            load_parameters(path="reactor",    load=path_parameters/"reactor_snf_fission_fractions.yaml")
            load_parameters(path="reactor",    load=path_parameters/"reactor_fission_fraction_scale.yaml",
                            replicate=index["reactor"], replica_key_offset=1)

            load_parameters(path="bkg.rate",   load=path_parameters/"bkg_rates.yaml")
            # fmt: on

            labels = {  # TODO, not propagated
                "neutrino_perfission": {
                    name: f"Edge {i:02d} ({edge:.2f} MeV) Reactor antineutrino spectrum correction"
                    for i, (name, edge) in enumerate(
                        zip(index["spec"], antineutrino_model_edges)
                    )
                }
            }
            if spectrum_correction_is_exponential:
                neutrino_perfission_correction_central_value = 0.0
                labels = {
                    "neutrino_perfission": "Reactor antineutrino spectrum correction (exp)"
                }
            else:
                neutrino_perfission_correction_central_value = 1.0
                labels = {
                    "neutrino_perfission": "Reactor antineutrino spectrum correction (linear)"
                }

            load_parameters(
                format="value",
                state="variable",
                parameters={
                    "neutrino_perfission": neutrino_perfission_correction_central_value
                },
                labels=labels,
                replicate=index["spec"],
            )

            # Some normalization factors
            load_parameters(
                format="value",
                state="fixed",
                parameters={
                    "reactor": {
                        "snf_factor": 1.0,
                        "offeq_factor": 1.0,
                    }
                },
                labels={
                    "reactor": {
                        "snf_factor": "Common SNF factor",
                        "offeq_factor": "Common offequilibrium factor",
                    }
                },
            )

            # Normalization constants
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
                        "seconds_in_day_inverse": "One divided by seconds in day",
                    }
                },
            )

            nodes = storage.child("nodes")
            inputs = storage.child("inputs")
            outputs = storage.child("outputs")
            data = storage.child("data")
            parameters = storage("parameter")

            # fmt: off
            # Create Nuisance parameters
            Sum.replicate(outputs("statistic.nuisance.parts"), name="statistic.nuisance.all")

            #
            # Create nodes
            #
            labels = LoadYaml(__file__.replace(".py", "_labels.yaml"))

            from numpy import arange, concatenate, linspace

            #
            # Define binning
            #
            in_edges_fine = linspace(0, 12, 241)
            in_edges_final = concatenate(([0.7], arange(1.2, 8.01, 0.20), [12.0]))

            from dagflow.lib.Array import Array
            from dagflow.lib.View import View
            edges_costheta, _ = Array.make_stored("edges.costheta", [-1, 1])
            edges_energy_common, _ = Array.make_stored(
                "edges.energy_common", in_edges_fine
            )
            edges_energy_final, _ = Array.make_stored(
                "edges.energy_final", in_edges_final
            )
            View.make_stored("edges.energy_enu", edges_energy_common)
            edges_energy_edep, _ = View.make_stored("edges.energy_edep", edges_energy_common)
            edges_energy_escint, _ = View.make_stored("edges.energy_escint", edges_energy_common)
            edges_energy_evis, _ = View.make_stored("edges.energy_evis", edges_energy_common)
            edges_energy_erec, _ = View.make_stored("edges.energy_erec", edges_energy_common)

            Array.make_stored("reactor_anue.spec_model_edges", antineutrino_model_edges)

            #
            # Integration, kinematics
            #
            integration_orders_edep, _ = Array.from_value( "kinematics_sampler.ordersx", 5, edges=edges_energy_edep)
            integration_orders_costheta, _ = Array.from_value("kinematics_sampler.ordersy", 3, edges=edges_costheta)

            from dagflow.lib.IntegratorGroup import IntegratorGroup
            integrator, _ = IntegratorGroup.replicate(
                "2d",
                names = {
                    "sampler": "kinematics_sampler",
                    "integrator": "kinematics_integral",
                    "x": "mesh_edep",
                    "y": "mesh_costheta"
                },
                replicate_outputs = combinations["anue_source.reactor.isotope.detector"]
            )
            integration_orders_edep >> integrator("ordersX")
            integration_orders_costheta >> integrator("ordersY")

            from dgf_reactoranueosc.IBDXsecVBO1Group import IBDXsecVBO1Group
            ibd, _ = IBDXsecVBO1Group.make_stored(use_edep=True)
            ibd << storage("parameter.constant.ibd")
            ibd << storage("parameter.constant.ibd.csc")
            outputs["kinematics_sampler.mesh_edep"] >> ibd.inputs["edep"]
            outputs["kinematics_sampler.mesh_costheta"] >> ibd.inputs["costheta"]
            kinematic_integrator_enu = ibd.outputs["enu"]

            #
            # Oscillations
            #
            from dgf_reactoranueosc.NueSurvivalProbability import \
                NueSurvivalProbability
            NueSurvivalProbability.replicate(
                name="oscprob",
                distance_unit="m",
                replicate_outputs=combinations["reactor.detector"],
                oscprobArgConversion = True
            )
            kinematic_integrator_enu >> inputs("oscprob.enu")
            parameters("constant.baseline") >> inputs("oscprob.L")
            parameters["all.conversion.oscprobArgConversion"] >> inputs("oscprob.oscprobArgConversion")
            nodes("oscprob") << parameters("free.oscprob")
            nodes("oscprob") << parameters("constrained.oscprob")
            nodes("oscprob") << parameters("constant.oscprob")

            #
            # Nominal antineutrino spectrum
            #
            load_graph(
                name = "reactor_anue.neutrino_perfission_perMeV_input",
                filenames = path_arrays / f"reactor_anue_spectra_50kev.{self._source_type}",
                x = "enu",
                y = "spec",
                merge_x = True,
                replicate_outputs = index["isotope"],
            )

            #
            # Pre-interpolate input spectrum on coarser grid
            # NOTE:
            #     - not needed with the current scheme:
            #         - spectrum correction applied by multiplication
            #     - introduced for the consistency with GNA
            #     - to be removed in v1 TODO
            #
            InterpolatorGroup.replicate(
                method = "exp",
                names = {
                    "indexer": "reactor_anue.spec_indexer_pre",
                    "interpolator": "reactor_anue.neutrino_perfission_perMeV_nominal_pre",
                    },
                replicate_outputs = index["isotope"],
            )
            outputs["reactor_anue.neutrino_perfission_perMeV_input.enu"] >> inputs["reactor_anue.neutrino_perfission_perMeV_nominal_pre.xcoarse"]
            outputs("reactor_anue.neutrino_perfission_perMeV_input.spec") >> inputs("reactor_anue.neutrino_perfission_perMeV_nominal_pre.ycoarse")
            outputs["reactor_anue.spec_model_edges"] >> inputs["reactor_anue.neutrino_perfission_perMeV_nominal_pre.xfine"]

            #
            # Interpolate for the integration mesh
            #
            InterpolatorGroup.replicate(
                method = "exp",
                names = {
                    "indexer": "reactor_anue.spec_indexer",
                    "interpolator": "reactor_anue.neutrino_perfission_perMeV_nominal",
                    },
                replicate_outputs = index["isotope"],
            )
            # Commented in favor of pre-interpolated part (below)
            # outputs["reactor_anue.neutrino_perfission_perMeV_input.enu"] >> inputs["reactor_anue.neutrino_perfission_perMeV_nominal.xcoarse"]
            # outputs("reactor_anue.neutrino_perfission_perMeV_input.spec") >> inputs("reactor_anue.neutrino_perfission_perMeV_nominal.ycoarse")
            outputs["reactor_anue.spec_model_edges"] >> inputs["reactor_anue.neutrino_perfission_perMeV_nominal.xcoarse"]
            outputs("reactor_anue.neutrino_perfission_perMeV_nominal_pre") >> inputs("reactor_anue.neutrino_perfission_perMeV_nominal.ycoarse")
            kinematic_integrator_enu >> inputs["reactor_anue.neutrino_perfission_perMeV_nominal.xfine"]

            #
            # Offequilibrium correction
            #
            load_graph(
                name = "reactor_offequilibrium_anue.correction_input",
                x = "enu",
                y = "offequilibrium_correction",
                merge_x = True,
                filenames = path_arrays / f"offequilibrium_correction.{self._source_type}",
                replicate_outputs = index["isotope_offeq"],
                dtype = "d"
            )

            InterpolatorGroup.replicate(
                method = "linear",
                names = {
                    "indexer": "reactor_offequilibrium_anue.correction_indexer",
                    "interpolator": "reactor_offequilibrium_anue.correction_interpolated",
                    },
                replicate_outputs = index["isotope_offeq"],
                underflow = "constant",
                overflow = "constant",
            )
            outputs["reactor_offequilibrium_anue.correction_input.enu"] >> inputs["reactor_offequilibrium_anue.correction_interpolated.xcoarse"]
            outputs("reactor_offequilibrium_anue.correction_input.offequilibrium_correction") >> inputs("reactor_offequilibrium_anue.correction_interpolated.ycoarse")
            kinematic_integrator_enu >> inputs["reactor_offequilibrium_anue.correction_interpolated.xfine"]

            #
            # SNF correction
            #
            load_graph(
                name = "snf_anue.correction_input",
                x = "enu",
                y = "snf_correction",
                merge_x = True,
                filenames = path_arrays / f"snf_correction.{self._source_type}",
                replicate_outputs = index["reactor"],
                dtype = "d"
            )
            InterpolatorGroup.replicate(
                method = "linear",
                names = {
                    "indexer": "snf_anue.correction_indexer",
                    "interpolator": "snf_anue.correction_interpolated",
                    },
                replicate_outputs = index["reactor"],
                underflow = "constant",
                overflow = "constant",
            )
            outputs["snf_anue.correction_input.enu"] >> inputs["snf_anue.correction_interpolated.xcoarse"]
            outputs("snf_anue.correction_input.snf_correction") >> inputs("snf_anue.correction_interpolated.ycoarse")
            kinematic_integrator_enu >> inputs["snf_anue.correction_interpolated.xfine"]

            #
            # Free antineutrino spectrum correction: spectrum model
            #
            from dagflow.lib import Exp
            from dagflow.lib.Concatenation import Concatenation

            if spectrum_correction_is_exponential:
                Concatenation.replicate(
                        parameters("all.neutrino_perfission"),
                        name = "reactor_anue.spec_free_correction_input"
                        )
                Exp.replicate(
                        outputs["reactor_anue.spec_free_correction_input"],
                        name = "reactor_anue.spec_free_correction"
                        )
                outputs["reactor_anue.spec_free_correction_input"].dd.axes_meshes = (outputs["reactor_anue.spec_model_edges"],)
            else:
                Concatenation.replicate(
                        parameters("all.neutrino_perfission"),
                        name = "reactor_anue.spec_free_correction"
                        )
                outputs["reactor_anue.spec_free_correction"].dd.axes_meshes = (outputs["reactor_anue.spec_model_edges"],)

            InterpolatorGroup.replicate(
                method = "exp",
                names = {
                    "indexer": "reactor_anue.spec_free_correction_indexer",
                    "interpolator": "reactor_anue.spec_free_correction_interpolated"
                    },
            )
            outputs["reactor_anue.spec_model_edges"] >> inputs["reactor_anue.spec_free_correction_interpolated.xcoarse"]
            outputs["reactor_anue.spec_free_correction"] >> inputs["reactor_anue.spec_free_correction_interpolated.ycoarse"]
            kinematic_integrator_enu >> inputs["reactor_anue.spec_free_correction_interpolated.xfine"]

            #
            # Antineutrino spectrum with corrections
            #
            Product.replicate(
                    outputs("reactor_anue.neutrino_perfission_perMeV_nominal"),
                    outputs["reactor_anue.spec_free_correction_interpolated"],
                    name = "reactor_anue.part.neutrino_perfission_perMeV_main",
                    replicate_outputs=index["isotope"],
                    )

            Product.replicate(
                    outputs("reactor_anue.neutrino_perfission_perMeV_nominal"),
                    outputs("reactor_offequilibrium_anue.correction_interpolated"),
                    name = "reactor_anue.part.neutrino_perfission_perMeV_offeq_nominal",
                    allow_skip_inputs = True,
                    skippable_inputs_should_contain = ("U238",),
                    replicate_outputs=index["isotope_offeq"],
                    )


            #
            # Livetime
            #
            from dagflow.bundles.load_record import load_record_data
            load_record_data(
                name = "daily_data.detector_all",
                filenames = path_arrays/f"livetimes_Dubna_AdSimpleNL_all.{self._source_type}",
                replicate_outputs = index["detector"],
                objects = lambda idx, _: f"EH{idx[-2]}AD{idx[-1]}",
                columns = ("day", "ndet", "livetime", "eff", "efflivetime"),
                skip = self.inactive_detectors
            )
            from models.bundles.refine_detector_data import \
                refine_detector_data
            refine_detector_data(
                data("daily_data.detector_all"),
                data.child("daily_data.detector"),
                detectors = index["detector"]
            )

            load_record_data(
                name = "daily_data.reactor_all",
                filenames = path_arrays/f"weekly_power_fulldata_release_v2.{self._source_type}",
                replicate_outputs = ("core_data",),
                columns = ("week", "day", "ndet", "core", "power") + index["isotope_lower"],
                key_order = (0,)
            )
            from models.bundles.refine_reactor_data import refine_reactor_data
            refine_reactor_data(
                data("daily_data.reactor_all"),
                data.child("daily_data.reactor"),
                reactors = index["reactor"],
                isotopes = index["isotope"],
            )

            from models.bundles.sync_reactor_detector_data import \
                sync_reactor_detector_data
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
                    parameters["all.conversion.reactorPowerConversion"],
                    name = "reactor.thermal_power_nominal_MeVs",
                    replicate_outputs = index["reactor"]
                    )

            # Time dependent, fit dependent (non-nominal) for reactor core
            Product.replicate(
                    parameters("all.reactor.fission_fraction_scale"),
                    outputs("daily_data.reactor.fission_fraction"),
                    name = "daily_data.reactor.fission_fraction_scaled",
                    replicate_outputs=combinations["reactor.isotope.period"],
                    )

            #
            # Fission fraction normalized
            #
            if self._fission_fraction_normalized:
                Sum.replicate(
                    outputs("daily_data.reactor.fission_fraction_scaled"),
                    name="daily_data.reactor.fission_fraction_scaled_normalization_factor",
                    replicate_outputs=combinations["reactor.period"],
                )

                Division.replicate(
                    outputs("daily_data.reactor.fission_fraction_scaled"),
                    outputs("daily_data.reactor.fission_fraction_scaled_normalization_factor"),
                    name="daily_data.reactor.fission_fraction_scaled_normalized",
                    replicate_outputs=combinations["reactor.isotope.period"],
                )

                Product.replicate(
                        parameters("all.reactor.energy_per_fission"),
                        outputs("daily_data.reactor.fission_fraction_scaled_normalized"),
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
                        outputs("daily_data.reactor.fission_fraction_scaled_normalized"),
                        outputs("reactor.thermal_power_nominal_MeVs"),
                        name = "reactor.thermal_power_isotope_MeV_persecond",
                        replicate_outputs=combinations["reactor.isotope.period"],
                        )
            else:
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
                        name = "reactor.thermal_power_isotope_MeV_persecond",
                        replicate_outputs=combinations["reactor.isotope.period"],
                        )

            Division.replicate(
                    outputs("reactor.thermal_power_isotope_MeV_persecond"),
                    outputs("reactor.energy_per_fission_average_MeV"),
                    name = "reactor.fissions_persecond",
                    replicate_outputs=combinations["reactor.isotope.period"],
                    )

            # Nominal, time and reactor independent power and fission fractions for SNF
            Product.replicate(
                    parameters("all.reactor.energy_per_fission"),
                    parameters("all.reactor.fission_fraction_snf"),
                    name = "reactor.energy_per_fission_snf_weighted_MeV",
                    replicate_outputs=index["isotope"],
                    )

            Sum.replicate(
                    outputs("reactor.energy_per_fission_snf_weighted_MeV"),
                    name = "reactor.energy_per_fission_snf_average_MeV",
                    )

            Product.replicate(
                    parameters("all.reactor.fission_fraction_snf"),
                    outputs("reactor.thermal_power_nominal_MeVs"),
                    name = "reactor.thermal_power_snf_isotope_MeV_persecond",
                    replicate_outputs=combinations["reactor.isotope"],
                    )

            Division.replicate(
                    outputs("reactor.thermal_power_snf_isotope_MeV_persecond"),
                    outputs["reactor.energy_per_fission_snf_average_MeV"],
                    name = "reactor.fissions_persecond_snf",
                    replicate_outputs=combinations["reactor.isotope"],
                    )

            # Effective number of fissions seen in Detector from Reactor from Isotope during Period
            Product.replicate(
                    outputs("reactor.fissions_persecond"),
                    outputs("daily_data.detector.efflivetime"),
                    name = "reactor_detector.number_of_fissions_daily",
                    replicate_outputs=combinations["reactor.isotope.detector.period"],
                    allow_skip_inputs = True,
                    skippable_inputs_should_contain = self.inactive_detectors
                    )

            # Total effective number of fissions from a Reactor seen in the Detector during Period
            from dagflow.lib import ArraySum
            ArraySum.replicate(
                    outputs("reactor_detector.number_of_fissions_daily"),
                    name = "reactor_detector.number_of_fissions",
                    )

            # Baseline factor from Reactor to Detector: 1/(4πL²)
            from dgf_reactoranueosc.InverseSquareLaw import InverseSquareLaw
            InverseSquareLaw.replicate(
                name="baseline_factor_percm2",
                scale="m_to_cm",
                replicate_outputs=combinations["reactor.detector"]
            )
            parameters("constant.baseline") >> inputs("baseline_factor_percm2")

            # Number of protons per detector
            Product.replicate(
                    parameters["all.detector.nprotons_nominal_ad"],
                    parameters("all.detector.nprotons_correction"),
                    name = "detector.nprotons",
                    replicate_outputs = index["detector"]
            )

            # Number of fissions × N protons × ε / (4πL²)  (main)
            Product.replicate(
                    outputs("reactor_detector.number_of_fissions"),
                    outputs("detector.nprotons"),
                    outputs("baseline_factor_percm2"),
                    parameters["all.detector.efficiency"],
                    name = "reactor_detector.number_of_fissions_nprotons_percm2",
                    replicate_outputs=combinations["reactor.isotope.detector.period"],
                    )

            Product.replicate(
                    outputs("reactor_detector.number_of_fissions_nprotons_percm2"),
                    parameters("all.reactor.offequilibrium_scale"),
                    parameters["all.reactor.offeq_factor"],
                    name = "reactor_detector.number_of_fissions_nprotons_percm2_offeq",
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
                    parameters["all.conversion.seconds_in_day_inverse"],
                    name="detector.efflivetime_days",
                    replicate_outputs=combinations["detector.period"],
                    allow_skip_inputs=True,
                    skippable_inputs_should_contain=self.inactive_detectors,
                    )

            # Effective live time × N protons × ε / (4πL²)  (SNF)
            Product.replicate(
                    outputs("detector.efflivetime"),
                    outputs("detector.nprotons"),
                    outputs("baseline_factor_percm2"),
                    parameters("all.reactor.snf_scale"),
                    parameters[ "all.reactor.snf_factor" ],
                    parameters["all.detector.efficiency"],
                    name = "reactor_detector.livetime_nprotons_percm2_snf",
                    replicate_outputs=combinations["reactor.detector.period"],
                    allow_skip_inputs = True,
                    skippable_inputs_should_contain = self.inactive_detectors
                    )

            #
            # Average SNF Spectrum
            #
            Product.replicate(
                    outputs("reactor_anue.neutrino_perfission_perMeV_nominal"),
                    outputs("reactor.fissions_persecond_snf"),
                    name = "snf_anue.neutrino_persecond_isotope",
                    replicate_outputs=combinations["reactor.isotope"],
                    )

            Sum.replicate(
                    outputs("snf_anue.neutrino_persecond_isotope"),
                    name = "snf_anue.neutrino_persecond",
                    replicate_outputs=index["reactor"],
                    )

            Product.replicate(
                    outputs("snf_anue.neutrino_persecond"),
                    outputs("snf_anue.correction_interpolated"),
                    name = "snf_anue.neutrino_persecond_snf",
                    replicate_outputs = index["reactor"]
                    )

            #
            # Integrand: flux × oscillation probability × cross section
            # [Nν·cm²/fission/proton]
            #
            Product.replicate(
                    outputs["ibd.crosssection"],
                    outputs["ibd.jacobian"],
                    name="ibd.crosssection_jacobian",
            )

            Product.replicate(
                    outputs["ibd.crosssection_jacobian"],
                    outputs("oscprob"),
                    name="ibd.crosssection_jacobian_oscillations",
                    replicate_outputs=combinations["reactor.detector"]
            )

            Product.replicate(
                    outputs("ibd.crosssection_jacobian_oscillations"),
                    outputs("reactor_anue.part.neutrino_perfission_perMeV_main"),
                    name="neutrino_cm2_perMeV_perfission_perproton.part.main",
                    replicate_outputs=combinations["reactor.isotope.detector"]
            )

            Product.replicate(
                    outputs("ibd.crosssection_jacobian_oscillations"),
                    outputs("reactor_anue.part.neutrino_perfission_perMeV_offeq_nominal"),
                    name="neutrino_cm2_perMeV_perfission_perproton.part.offeq",
                    replicate_outputs=combinations["reactor.isotope_offeq.detector"]
            )

            Product.replicate(
                    outputs("ibd.crosssection_jacobian_oscillations"),
                    outputs("snf_anue.neutrino_persecond_snf"),
                    name="neutrino_cm2_perMeV_perfission_perproton.part.snf",
                    replicate_outputs=combinations["reactor.detector"]
            )
            outputs("neutrino_cm2_perMeV_perfission_perproton.part.main") >> inputs("kinematics_integral.main")
            outputs("neutrino_cm2_perMeV_perfission_perproton.part.offeq") >> inputs("kinematics_integral.offeq")
            outputs("neutrino_cm2_perMeV_perfission_perproton.part.snf") >> inputs("kinematics_integral.snf")

            #
            # Multiply by the scaling factors:
            #  - main:  fissions_per_second[p,r,i] × effective live time[p,d] × N protons[d] × efficiency[d]
            #  - offeq: fissions_per_second[p,r,i] × effective live time[p,d] × N protons[d] × efficiency[d] × offequilibrium scale[r,i] × offeq_factor(=1)
            #  - snf:                                effective live time[p,d] × N protons[d] × efficiency[d] × SNF scale[r]              × snf_factor(=1)
            #
            Product.replicate(
                    outputs("kinematics_integral.main"),
                    outputs("reactor_detector.number_of_fissions_nprotons_percm2"),
                    name = "eventscount.parts.main",
                    replicate_outputs = combinations["reactor.isotope.detector.period"]
                    )

            Product.replicate(
                    outputs("kinematics_integral.offeq"),
                    outputs("reactor_detector.number_of_fissions_nprotons_percm2_offeq"),
                    name = "eventscount.parts.offeq",
                    replicate_outputs = combinations["reactor.isotope_offeq.detector.period"],
                    allow_skip_inputs = True,
                    skippable_inputs_should_contain = ("U238",)
                    )

            Product.replicate(
                    outputs("kinematics_integral.snf"),
                    outputs("reactor_detector.livetime_nprotons_percm2_snf"),
                    name = "eventscount.parts.snf",
                    replicate_outputs = combinations["reactor.detector.period"]
                    )

            # Debug node: eventscount.reactor_active_periods
            Sum.replicate(
                outputs("eventscount.parts.main"),
                outputs("eventscount.parts.offeq"),
                name="eventscount.reactor_active_periods",
                replicate_outputs=combinations["detector.period"]
            )
            # Debug node: eventscount.reactor_active_periods
            Sum.replicate(
                outputs("eventscount.parts.snf"),
                name="eventscount.snf_periods",
                replicate_outputs=combinations["detector.period"]
            )

            Sum.replicate(
                outputs("eventscount.parts"),
                name="eventscount.raw",
                replicate_outputs=combinations["detector.period"]
            )

            # Sum.replicate(
            #     outputs("eventscount.periods.raw"),
            #     name="eventscount.raw",
            #     replicate_outputs=index["detector"]
            # )
            #
            # Detector effects
            #
            load_array(
                name = "detector.iav",
                filenames = path_arrays/f"detector_IAV_matrix_P14A_LS.{self._source_type}",
                replicate_outputs = ("matrix_raw",),
                objects = {"matrix_raw": "iav_matrix"},
                array_kwargs = {
                    'edges': (edges_energy_escint, edges_energy_edep)
                    }
            )

            from dagflow.lib.NormalizeMatrix import NormalizeMatrix
            NormalizeMatrix.replicate(name="detector.iav.matrix")
            outputs["detector.iav.matrix_raw"] >> nodes["detector.iav.matrix"]

            from dagflow.lib.VectorMatrixProduct import VectorMatrixProduct
            VectorMatrixProduct.replicate(name="eventscount.iav", replicate_outputs=combinations["detector.period"])
            outputs["detector.iav.matrix"] >> inputs("eventscount.iav.matrix")
            outputs("eventscount.raw") >> inputs("eventscount.iav.vector")

            load_graph_data(
                name = "detector.lsnl.curves",
                x = "edep",
                y = "evis_parts",
                merge_x = True,
                filenames = path_arrays/f"detector_LSNL_curves_Jan2022_newE_v1.{self._source_type}",
                replicate_outputs = index["lsnl"],
            )

            # Coarse LSNL model, consistent with GNA implementation
            from dgf_detector.bundles.cross_check_refine_lsnl_data import \
                cross_check_refine_lsnl_data
            cross_check_refine_lsnl_data(
                storage("data.detector.lsnl.curves"),
                edepname = 'edep',
                nominalname = 'evis_parts.nominal',
                newmin = 0.5,
                newmax = 12.1
            )

            # TODO: proper refinement for v1
            # from dgf_detector.bundles.refine_lsnl_data import refine_lsnl_data
            # refine_lsnl_data(
            #     storage("data.detector.lsnl.curves"),
            #     edepname = 'edep',
            #     nominalname = 'evis_parts.nominal',
            #     refine_times = 4,
            #     newmin = 0.5,
            #     newmax = 12.1
            # )

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
                outputs["detector.lsnl.curves.evis_parts.nominal"],
                outputs("detector.lsnl.curves.evis_parts_scaled"),
                name="detector.lsnl.curves.evis_common"
            )

            from dgf_detector.Monotonize import Monotonize
            Monotonize.replicate(
                    name="detector.lsnl.curves.evis_common_monotonic",
                    index_fraction = 0.5,
                    gradient = 1.0,
                    with_x = True
                    )
            outputs["detector.lsnl.curves.edep"] >> inputs["detector.lsnl.curves.evis_common_monotonic.x"]
            outputs["detector.lsnl.curves.evis_common"] >> inputs["detector.lsnl.curves.evis_common_monotonic.y"]

            detector_parameters_secodary = storage("outputs.detector").child("parameters_secondary")
            remap_items(
                storage("parameter.all.detector.energy_scale"),
                detector_parameters_secodary,
                reorder_indices=[
                    ["detector", "parameter"],
                    ["parameter", "detector"],
                ],
            )

            Product.replicate(
                outputs["detector.lsnl.curves.evis_common_monotonic"],
                detector_parameters_secodary("energy_scale_factor_relative"),
                name="detector.lsnl.curves.evis",
                replicate_outputs = index["detector"]
            )
            InterpolatorGroup.replicate(
                method = "linear",
                names = {
                    "indexer": "detector.lsnl.indexer_fwd",
                    "interpolator": "detector.lsnl.interpolated_fwd",
                    },
                replicate_outputs = index["detector"]
            )
            outputs["detector.lsnl.curves.edep"] >> inputs["detector.lsnl.interpolated_fwd.xcoarse"]
            outputs("detector.lsnl.curves.evis") >> inputs("detector.lsnl.interpolated_fwd.ycoarse")
            edges_energy_edep >> inputs["detector.lsnl.interpolated_fwd.xfine"]

            ## TODO:
            ## - for backward interpolation need multiple X definitions (detectors)
            ## - thus need to replicate the indexer
            # InterpolatorGroup.replicate(
            #     method = "linear",
            #     names = {
            #         "indexer": "detector.lsnl.indexer_bwd",
            #         "interpolator": "detector.lsnl.interpolated_bwd",
            #         },
            #     replicate_outputs = index["detector"]
            # )
            # outputs("detector.lsnl.curves.evis") >> inputs["detector.lsnl.interpolated_bwd.xcoarse"]
            # outputs["detector.lsnl.curves.edep"]  >> inputs("detector.lsnl.interpolated_bwd.ycoarse")
            # edges_energy_evis >> inputs["detector.lsnl.interpolated_bwd.xfine"]

            # from dgf_detector.AxisDistortionMatrix import AxisDistortionMatrix
            # AxisDistortionMatrix.replicate(name="detector.lsnl.matrix", replicate_outputs=index["detector"])
            # edges_energy_edep.outputs[0] >> inputs("detector.lsnl.matrix.EdgesOriginal")
            # outputs("detector.lsnl.interpolated_fwd") >> inputs("detector.lsnl.matrix.EdgesModified")
            # outputs("detector.lsnl.interpolated_bwd") >> inputs("detector.lsnl.matrix.EdgesModifiedBackwards")

            # TODO: Outdated LSNL matrix (cross check)
            from dgf_detector.AxisDistortionMatrixLinear import \
                AxisDistortionMatrixLinear
            AxisDistortionMatrixLinear.replicate(name="detector.lsnl.matrix_linear", replicate_outputs=index["detector"])
            edges_energy_edep.outputs[0] >> inputs("detector.lsnl.matrix_linear.EdgesOriginal")
            outputs("detector.lsnl.interpolated_fwd") >> inputs("detector.lsnl.matrix_linear.EdgesModified")

            # TODO: masked LSNL matrix (cross check)
            from numpy import ones
            lsnl_mask = ones((240, 240), dtype="d")
            lsnl_mask[:14,:] = 0.0
            lsnl_mask[:,:16] = 0.0
            lsnl_mask[:,232:] = 0.0
            Array.make_stored("detector.lsnl.gna_mask", lsnl_mask)
            Product.replicate(
                outputs("detector.lsnl.matrix_linear"),
                outputs["detector.lsnl.gna_mask"],
                name="detector.lsnl.matrix_linear_masked",
                replicate_outputs=index["detector"]
            )

            VectorMatrixProduct.replicate(name="eventscount.evis", replicate_outputs=combinations["detector.period"])
            # outputs("detector.lsnl.matrix") >> inputs("eventscount.evis.matrix")
            outputs("detector.lsnl.matrix_linear_masked") >> inputs("eventscount.evis.matrix")
            outputs("eventscount.iav") >> inputs("eventscount.evis.vector")

            from dgf_detector.EnergyResolution import EnergyResolution
            EnergyResolution.replicate(path="detector.eres")
            nodes["detector.eres.sigma_rel"] << parameters("constrained.detector.eres")
            outputs["edges.energy_evis"] >> inputs["detector.eres.matrix"]
            outputs["edges.energy_evis"] >> inputs["detector.eres.e_edges"]

            VectorMatrixProduct.replicate(name="eventscount.erec", replicate_outputs=combinations["detector.period"])
            outputs["detector.eres.matrix"] >> inputs("eventscount.erec.matrix")
            outputs("eventscount.evis") >> inputs("eventscount.erec.vector")

            from dgf_detector.Rebin import Rebin
            Rebin.replicate(
                names={"matrix": "detector.rebin_matrix_ibd", "product": "eventscount.final.ibd"},
                replicate_outputs=combinations["detector.period"],
            )
            edges_energy_erec >> inputs["detector.rebin_matrix_ibd.edges_old"]
            edges_energy_final >> inputs["detector.rebin_matrix_ibd.edges_new"]
            outputs("eventscount.erec") >> inputs("eventscount.final.ibd")

            #
            # Backgrounds
            #
            from dagflow.bundles.load_hist import load_hist
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
                skip = self.inactive_detectors,
                key_order = (1, 2, 0),
                objects = lambda _, idx: f"DYB_{bkg_names[idx[0]]}_expected_spectrum_EH{idx[-2][-2]}_AD{idx[-2][-1]}"
            )

            ads_at_sites = {
                    "EH1": ("AD11", "AD12"),
                    "EH2": ("AD21", "AD22"),
                    "EH3": ("AD31", "AD32", "AD33", "AD34"),
                    }
            remap_items(
                parameters("all.bkg.rate.fastn"),
                outputs.child("bkg.rate.fastn"),
                rename_indices = ads_at_sites,
                skip_indices_target = self.inactive_detectors,
                fcn = lambda par: par.output
            )
            remap_items(
                parameters("all.bkg.rate.lihe"),
                outputs.child("bkg.rate.lihe"),
                rename_indices = ads_at_sites,
                skip_indices_target = self.inactive_detectors,
                fcn = lambda par: par.output
            )

            # NOTE:
            # GNA upload fast-n as array from 0 to 12 MeV (50 keV), and it normalized to 1.
            # So, every bin contain 0.00416667.
            # TODO: remove in dayabay-v1
            fastn_data = ones(240) / 240
            for key, spectrum in storage("outputs.bkg.spectrum_shape.fastn").walkitems():
                spectrum.data[:] = fastn_data

            Product.replicate(
                    parameters("all.bkg.rate.acc"),
                    outputs("bkg.spectrum_shape.acc"),
                    name="bkg.spectrum_per_day.acc",
                    replicate_outputs=combinations["detector.period"],
                    )

            Product.replicate(
                    outputs("bkg.rate.lihe"),
                    outputs("bkg.spectrum_shape.lihe"),
                    name="bkg.spectrum_per_day.lihe",
                    replicate_outputs=combinations["detector.period"],
                    )

            Product.replicate(
                    outputs("bkg.rate.fastn"),
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
                    outputs("eventscount.fine.bkg"),
                    outputs("eventscount.erec"),
                    name="eventscount.fine.total",
                    replicate_outputs=combinations["detector.period"],
                    check_edges_contents=True,
                    )

            Rebin.replicate(
                    names={"matrix": "detector.rebin_bkg_matrix", "product": "eventscount.final.bkg"},
                    replicate_outputs=combinations["detector.period"],
            )
            edges_energy_erec >> inputs["detector.rebin_bkg_matrix.edges_old"]
            edges_energy_final >> inputs["detector.rebin_bkg_matrix.edges_new"]
            outputs("eventscount.fine.bkg") >> inputs("eventscount.final.bkg")

            Sum.replicate(
                outputs("eventscount.final.bkg"),
                outputs("eventscount.final.ibd"),
                name="eventscount.final.total",
                replicate_outputs=combinations["detector.period"],
            )

            #
            # Statistic
            #
            from dgf_statistics.MonteCarlo import MonteCarlo
            MonteCarlo.replicate(
                name="pseudo.data",
                mode="asimov",
                replicate_outputs=combinations["detector.period"],
                replicate_inputs=combinations["detector.period"]
            )
            outputs("eventscount.final.total") >> inputs("pseudo.data.input")

            from dgf_statistics.Chi2 import Chi2
            Chi2.replicate(
                replicate_inputs=combinations["detector.period"],
                name="statistic.stat.chi2p"
            )
            outputs("pseudo.data") >> inputs("statistic.stat.chi2p.data")
            outputs("eventscount.final.total") >> inputs("statistic.stat.chi2p.theory")
            outputs("pseudo.data") >> inputs("statistic.stat.chi2p.errors")

            from dgf_statistics.CNPStat import CNPStat
            CNPStat.replicate(
                replicate_inputs=combinations["detector.period"],
                replicate_outputs=combinations["detector.period"],
                name="statistic.staterr.cnp"
            )
            outputs("pseudo.data") >> inputs("statistic.staterr.cnp.data")
            outputs("eventscount.final.total") >> inputs("statistic.staterr.cnp.theory")

            Chi2.replicate(replicate_inputs=combinations["detector.period"], name="statistic.stat.chi2cnp")
            outputs("pseudo.data") >> inputs("statistic.stat.chi2cnp.data")
            outputs("eventscount.final.total") >> inputs("statistic.stat.chi2cnp.theory")
            outputs("statistic.staterr.cnp") >> inputs("statistic.stat.chi2cnp.errors")

            Sum.replicate(outputs["statistic.stat.chi2p"], outputs["statistic.nuisance.all"], name="statistic.full.chi2p")
            Sum.replicate(outputs["statistic.stat.chi2cnp"], outputs["statistic.nuisance.all"], name="statistic.full.chi2cnp")
            # fmt: on

        processed_keys_set = set()
        storage("nodes").read_labels(labels, processed_keys_set=processed_keys_set)
        storage("outputs").read_labels(labels, processed_keys_set=processed_keys_set)
        storage("inputs").remove_connected_inputs()
        storage.read_paths(index=index)
        graph.build_index_dict(index)

        labels_mk = NestedMKDict(labels, sep=".")
        if self._strict:
            for key in processed_keys_set:
                labels_mk.delete_with_parents(key)
            if labels_mk:
                raise RuntimeError(
                    f"The following label groups were not used: {tuple(labels_mk.walkkeys())}"
                )
