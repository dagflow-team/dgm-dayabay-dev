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
from dagflow.storage import NodeStorage
from dagflow.tools.schema import LoadYaml
from multikeydict.nestedmkdict import NestedMKDict
from multikeydict.tools import remap_items

SourceTypes = Literal["tsv", "hdf5", "root", "npz"]


class model_dayabay_v0:
    __slots__ = (
        "storage",
        "graph",
        "_override_indices",
        "_path_data",
        "_source_type",
        "_strict",
        "_close",
        "_spectrum_correction_mode",
    )

    storage: NodeStorage
    graph: Graph | None
    _path_data: Path
    _override_indices: Mapping[str, Sequence[str]]
    _source_type: SourceTypes
    _strict: bool
    _close: bool
    _spectrum_correction_mode: Literal["linear", "exponential"]

    def __init__(
        self,
        *,
        source_type: SourceTypes = "tsv",
        strict: bool = True,
        close: bool = True,
        override_indices: Mapping[str, Sequence[str]] = {},
        spectrum_correction_mode: Literal["linear", "exponential"] = "exponential",
    ):
        self._strict = strict
        self._close = close

        self.graph = None
        self.storage = NodeStorage()
        self._path_data = Path("data/dayabay-v0")
        self._source_type = source_type
        self._override_indices = override_indices
        self._spectrum_correction_mode = spectrum_correction_mode

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

        # fmt: off

        # Provide a list of indices and their values. Values should be globally unique
        index = {}
        index["isotope"] = ("U235", "U238", "Pu239", "Pu241")
        index["isotope_lower"] = tuple(i.lower() for i in index["isotope"])
        index["isotope_offeq"] = ("U235", "Pu239", "Pu241")
        index["detector"] = ("AD11", "AD12", "AD21", "AD22", "AD31", "AD32", "AD33", "AD34")
        index["site"] = ("EH1", "EH2", "EH3")
        index["reactor"] = ("DB1", "DB2", "LA1", "LA2", "LA3", "LA4")
        index["period"] = ("6AD", "8AD", "7AD")
        index["lsnl"] = ("nominal", "pull0", "pull1", "pull2", "pull3")
        index["lsnl_nuisance"] = ("pull0", "pull1", "pull2", "pull3")
        # index["bkg"] = ('acc', 'lihe', 'fastn', 'amc', 'alphan', 'muon')
        index["bkg"] = ('acc', 'lihe', 'fastn', 'amc', 'alphan')
        index["spec"] = tuple(f"spec_scale_{i:02d}" for i in range(len(antineutrino_model_edges)))

        index.update(self._override_indices)

        index_all = (index["isotope"] + index["detector"] + index["reactor"] + index["period"])
        set_all = set(index_all)
        if len(index_all) != len(set_all):
            raise RuntimeError("Repeated indices")

        # Provide the combinations of indices
        combinations = {
                comb: tuple(product(*(index[item] for item in comb.split("."))))
                for comb in
                (
                    "reactor.detector",
                    "reactor.isotope",
                    "reactor.isotope_offeq",
                    "reactor.period",
                    "reactor.isotope.period",
                    "reactor.isotope.detector",
                    "reactor.isotope.detector.period",
                    "bkg.detector"
                    )
                }

        inactive_detectors = ({"6AD", "AD22"}, {"6AD", "AD34"}, {"7AD", "AD11"})
        # unused_backgrounds = (("6AD", "muon"), ("8AD", "muon"))
        combinations["period.detector"] = tuple(
            pair
            for pair in product(index["period"], index["detector"])
            if set(pair) not in inactive_detectors
        )
        # fmt: on

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
            load_parameters(                   load=path_parameters/"baselines_snf.yaml")

            load_parameters(path="detector",   load=path_parameters/"detector_efficiency.yaml")
            load_parameters(path="detector",   load=path_parameters/"detector_normalization.yaml")
            load_parameters(path="detector",   load=path_parameters/"detector_nprotons_correction.yaml")
            load_parameters(path="detector",   load=path_parameters/"detector_eres.yaml")
            load_parameters(path="detector",   load=path_parameters/"detector_lsnl.yaml",
                            replicate=index["lsnl_nuisance"])
            load_parameters(path="detector",   load=path_parameters/"detector_relative_energy_scale.yaml",
                            replicate=index["detector"])

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
                "reactor_anue_spectrum": {
                    name: f"Edge {i:02d} ({edge:.2f} MeV) Reactor antineutrino spectrum correction"
                    for i, (name, edge) in enumerate(
                        zip(index["spec"], antineutrino_model_edges)
                    )
                }
            }
            if spectrum_correction_is_exponential:
                reactor_anue_spectrum_correction_central_value = 0.0
                labels = {
                    "reactor_anue_spectrum": "Reactor antineutrino spectrum correction (exp)"
                }
            else:
                reactor_anue_spectrum_correction_central_value = 1.0
                labels = {
                    "reactor_anue_spectrum": "Reactor antineutrino spectrum correction (linear)"
                }

            load_parameters(
                format="value",
                state="variable",
                parameters={
                    "reactor_anue_spectrum": reactor_anue_spectrum_correction_central_value
                },
                labels=labels,
                replicate=index["spec"],
            )
            # fmt: off

            nodes = storage.child("nodes")
            inputs = storage.child("inputs")
            outputs = storage.child("outputs")
            data = storage.child("data")
            parameters = storage("parameter")

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
            edges_energy_evis, _ = View.make_stored("edges.energy_evis", edges_energy_common)
            edges_energy_erec, _ = View.make_stored("edges.energy_erec", edges_energy_common)

            #
            # Integration, kinematics
            #
            integration_orders_edep, _ = Array.from_value( "kinematics_sampler.ordersx", 5, edges=edges_energy_edep)
            integration_orders_costheta, _ = Array.from_value("kinematics_sampler.ordersy", 4, edges=edges_costheta)

            from dagflow.lib.IntegratorGroup import IntegratorGroup
            integrator, _ = IntegratorGroup.replicate(
                "2d",
                names = {
                    "sampler": "kinematics_sampler",
                    "integrator": "kinematics_integral",
                    "x": "mesh_edep",
                    "y": "mesh_costheta"
                },
                replicate = combinations["reactor.isotope.detector"]
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
            NueSurvivalProbability.replicate(name="oscprob", distance_unit="m", replicate=combinations["reactor.detector"])
            kinematic_integrator_enu >> inputs("oscprob.enu")
            parameters("constant.baseline") >> inputs("oscprob.L")
            nodes("oscprob") << parameters("free.oscprob")
            nodes("oscprob") << parameters("constrained.oscprob")
            nodes("oscprob") << parameters("constant.oscprob")

            #
            # Nominal antineutrino spectrum
            #
            load_graph(
                name = "reactor_anue.input_spectrum",
                filenames = path_arrays / f"reactor_anue_spectra_50kev.{self._source_type}",
                x = "enu",
                y = "spec",
                merge_x = True,
                replicate = index["isotope"],
            )
            from dagflow.lib.InterpolatorGroup import InterpolatorGroup
            InterpolatorGroup.replicate(
                method = "exp",
                names = {
                    "indexer": "reactor_anue.spec_indexer",
                    "interpolator": "reactor_anue.spec_nominal",
                    },
                replicate = index["isotope"],
            )
            outputs["reactor_anue.input_spectrum.enu"] >> inputs["reactor_anue.spec_nominal.xcoarse"]
            outputs("reactor_anue.input_spectrum.spec") >> inputs("reactor_anue.spec_nominal.ycoarse")
            kinematic_integrator_enu >> inputs["reactor_anue.spec_nominal.xfine"]

            #
            # Offequilibrium correction
            #
            load_graph(
                name = "reactor_offequilibrium_anue.correction_input",
                x = "enu",
                y = "offequilibrium_correction",
                merge_x = True,
                filenames = path_arrays / f"offequilibrium_correction.{self._source_type}",
                replicate = index["isotope_offeq"],
                dtype = "d"
            )

            InterpolatorGroup.replicate(
                method = "linear",
                names = {
                    "indexer": "reactor_offequilibrium_anue.correction_indexer",
                    "interpolator": "reactor_offequilibrium_anue.correction_interpolated",
                    },
                replicate = index["isotope_offeq"],
            )
            outputs["reactor_offequilibrium_anue.correction_input.enu"] >> inputs["reactor_offequilibrium_anue.correction_interpolated.xcoarse"]
            outputs("reactor_offequilibrium_anue.correction_input.offequilibrium_correction") >> inputs("reactor_offequilibrium_anue.correction_interpolated.ycoarse")
            kinematic_integrator_enu >> inputs["reactor_offequilibrium_anue.correction_interpolated.xfine"]

            #
            # SNF correction
            #
            load_graph(
                name = "reactor_snf_anue.correction_input",
                x = "enu",
                y = "snf_correction",
                merge_x = True,
                filenames = path_arrays / f"snf_correction.{self._source_type}",
                replicate = index["reactor"],
                dtype = "d"
            )
            InterpolatorGroup.replicate(
                method = "linear",
                names = {
                    "indexer": "reactor_snf_anue.correction_indexer",
                    "interpolator": "reactor_snf_anue.correction_interpolated",
                    },
                replicate = index["reactor"],
            )
            outputs["reactor_snf_anue.correction_input.enu"] >> inputs["reactor_snf_anue.correction_interpolated.xcoarse"]
            outputs("reactor_snf_anue.correction_input.snf_correction") >> inputs("reactor_snf_anue.correction_interpolated.ycoarse")
            kinematic_integrator_enu >> inputs["reactor_snf_anue.correction_interpolated.xfine"]

            #
            # Offequilibrium part
            #
            Product.replicate(
                    outputs("reactor_anue.spec_nominal"),
                    outputs("reactor_offequilibrium_anue.correction_interpolated"),
                    name = "reactor_anue.spec_part_offeq_nominal",
                    allow_skip_inputs = True, # U238
                    replicate=index["isotope_offeq"],
                    )
            Product.replicate(
                    parameters("all.reactor.offequilibrium_scale"),
                    outputs("reactor_anue.spec_part_offeq_nominal"),
                    name = "reactor_anue.spec_part_offeq_scaled",
                    allow_skip_inputs = True, # U238
                    replicate=combinations["reactor.isotope_offeq"],
                    )

            #
            # SNF part
            #
            Product.replicate(
                    outputs("reactor_snf_anue.correction_interpolated"),
                    name = "reactor_anue.spec_part_snf_nominal",
                    replicate=index["reactor"],
                    )
            Product.replicate(
                    parameters("all.reactor.snf_scale"),
                    outputs("reactor_anue.spec_part_snf_nominal"),
                    name = "reactor_anue.spec_part_snf_scaled",
                    replicate=index["reactor"],
                    )

            #
            # Free antineutrino spectrum correction: spectrum model
            #
            Array.make_stored("reactor_anue.spec_model_edges", antineutrino_model_edges)

            from dagflow.lib import Exp
            from dagflow.lib.Concatenation import Concatenation

            if spectrum_correction_is_exponential:
                Concatenation.replicate(
                        parameters("all.reactor_anue_spectrum"),
                        name = "reactor_anue.spec_free_correction_input"
                        )
                Exp.replicate(
                        outputs["reactor_anue.spec_free_correction_input"],
                        name = "reactor_anue.spec_free_correction"
                        )
                outputs["reactor_anue.spec_free_correction_input"].dd.axes_meshes = (outputs["reactor_anue.spec_model_edges"],)
            else:
                Concatenation.replicate(
                        parameters("all.reactor_anue_spectrum"),
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
                    outputs("reactor_anue.spec_nominal"),
                    outputs["reactor_anue.spec_free_correction_interpolated"],
                    name = "reactor_anue.spec_part_main",
                    replicate=index["isotope"],
                    )

            Sum.replicate(
                    outputs("reactor_anue.spec_part_main"),
                    outputs("reactor_anue.spec_part_offeq_scaled"),
                    name = "reactor_anue.spec_part_core",
                    replicate=combinations["reactor.isotope"],
                    )

            #
            # Livetime
            #
            from dagflow.bundles.load_record import load_record_data
            load_record_data(
                name = "daily_data.detector_all",
                filenames = path_arrays/f"livetimes_Dubna_AdSimpleNL_all.{self._source_type}",
                replicate = index["detector"],
                objects = lambda idx, _: f"EH{idx[-2]}AD{idx[-1]}",
                columns = ("day", "ndet", "livetime", "eff", "efflivetime")
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
                replicate = ("core_data",),
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
                    name = "reactor.thermal_power_nominal_MeVs"
                    )

            # Time dependent, fit dependent (non-nominal) for IBD
            Product.replicate(
                    parameters("all.reactor.fission_fraction_scale"),
                    outputs("daily_data.reactor.fission_fraction"),
                    name = "daily_data.reactor.fission_fraction_ibd_scaled",
                    replicate=combinations["reactor.isotope.period"],
                    )

            Product.replicate(
                    parameters("all.reactor.energy_per_fission"),
                    outputs("daily_data.reactor.fission_fraction_ibd_scaled"),
                    name = "reactor.energy_per_fission_ibd_weighted_MeV",
                    replicate=combinations["reactor.isotope.period"],
                    )

            Sum.replicate(
                    outputs( "reactor.energy_per_fission_ibd_weighted_MeV"),
                    name = "reactor.energy_per_fission_ibd_average_MeV",
                    replicate=combinations["reactor.period"],
                    )

            Product.replicate(
                    outputs("daily_data.reactor.power"),
                    outputs("daily_data.reactor.fission_fraction_ibd_scaled"),
                    outputs["reactor.thermal_power_nominal_MeVs"],
                    name = "reactor.thermal_power_ibd_weighted_MeV",
                    replicate=combinations["reactor.isotope.period"],
                    )

            Division.replicate(
                    outputs("reactor.thermal_power_ibd_weighted_MeV"),
                    outputs("reactor.energy_per_fission_ibd_average_MeV"),
                    name = "reactor.fissions_per_second_ibd",
                    replicate=combinations["reactor.isotope.period"],
                    )

            # Time dependent, nominal (fit independent) for SNF
            Product.replicate(
                    parameters("all.reactor.energy_per_fission"),
                    parameters("all.reactor.fission_fraction_snf"),
                    name = "reactor.energy_per_fission_snf_weighted_MeV",
                    replicate=index["isotope"],
                    )

            Sum.replicate(
                    outputs("reactor.energy_per_fission_snf_weighted_MeV"),
                    name = "reactor.energy_per_fission_snf_average_MeV",
                    )

            Product.replicate(
                    outputs("daily_data.reactor.power"),
                    parameters("all.reactor.fission_fraction_snf"),
                    outputs["reactor.thermal_power_nominal_MeVs"],
                    name = "reactor.thermal_power_snf_weighted_MeV",
                    replicate=combinations["reactor.isotope.period"],
                    )

            Division.replicate(
                    outputs("reactor.thermal_power_snf_weighted_MeV"),
                    outputs["reactor.energy_per_fission_snf_average_MeV"],
                    name = "reactor.fissions_per_second_snf",
                    replicate=combinations["reactor.isotope.period"],
                    )
            
            # Number of fissions
            Product.replicate(
                    outputs("reactor.fissions_per_second_ibd"),
                    outputs("daily_data.detector.efflivetime"),
                    name = "reactor_detector.number_of_fissions_ibd_daily",
                    replicate=combinations["reactor.isotope.detector.period"],
                    )

            Product.replicate(
                    outputs("reactor.fissions_per_second_snf"),
                    outputs("daily_data.detector.efflivetime"),
                    name = "reactor_detector.number_of_fissions_snf_daily",
                    replicate=combinations["reactor.isotope.detector.period"],
                    )

            from dagflow.lib import ArraySum
            ArraySum.replicate(
                    outputs("reactor_detector.number_of_fissions_ibd_daily"),
                    name = "reactor_detector.number_of_fissions_ibd",
                    )

            ArraySum.replicate(
                    outputs("reactor_detector.number_of_fissions_snf_daily"),
                    name = "reactor_detector.number_of_fissions_snf",
                    )

            # Number of fissions × protons
            Product.replicate(
                    parameters["all.detector.nprotons_nominal_ad"],
                    parameters("all.detector.nprotons_correction"),
                    name = "detector.nprotons",
                    replicate = index["detector"]
            )

            Product.replicate(
                    outputs("reactor_detector.number_of_fissions_ibd"),
                    outputs("detector.nprotons"),
                    name = "reactor_detector.number_of_fissions_nprotons_ibd",
                    replicate=combinations["reactor.isotope.detector.period"],
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
            
            #
            # Integrand: flux × oscillation probability × cross section
            # [ν·cm²/fission/proton]
            #
            Product.replicate(
                    outputs["ibd.crosssection"],
                    outputs["ibd.jacobian"],
                    name="ibd.crosssection_jacobian",
            )

            Product.replicate(
                    outputs("oscprob"),
                    outputs("reactor_anue.spec_nominal"),
                    name="reactor_anue.spec_oscillated",
                    replicate=combinations["reactor.isotope.detector"]
            )

            Product.replicate(
                    outputs("reactor_anue.spec_oscillated"),
                    outputs["ibd.crosssection_jacobian"],
                    name="kinematics_integrand",
                    replicate=combinations["reactor.isotope.detector"]
            )
            outputs("kinematics_integrand") >> inputs("kinematics_integral")

            from dgf_reactoranueosc.InverseSquareLaw import InverseSquareLaw
            InverseSquareLaw.replicate(name="baseline_factor", replicate=combinations["reactor.detector"])
            parameters("constant.baseline") >> inputs("baseline_factor")

            InverseSquareLaw.replicate(name="baseline_factor_snf", replicate=combinations["reactor.detector"])
            parameters("constant.baseline_snf") >> inputs("baseline_factor_snf")

            Product.replicate(
                outputs("kinematics_integral"),
                outputs("baseline_factor"),
                name="countrate_reac",
                replicate=combinations["reactor.isotope.detector"]
            )

            Sum.replicate(
                outputs("countrate_reac"),
                replicate = index["detector"],
                name="countrate.raw"
            )

            #
            # Detector effects
            #
            load_array(
                name = "detector.iav",
                filenames = path_arrays/f"detector_IAV_matrix_P14A_LS.{self._source_type}",
                replicate = ("matrix_raw",),
                objects = {"matrix_raw": "iav_matrix"},
            )

            from dagflow.lib.NormalizeMatrix import NormalizeMatrix
            NormalizeMatrix.replicate(name="detector.iav.matrix")
            outputs["detector.iav.matrix_raw"] >> nodes["detector.iav.matrix"]

            from dagflow.lib.VectorMatrixProduct import VectorMatrixProduct
            VectorMatrixProduct.replicate(name="countrate.iav", replicate=index["detector"])
            outputs["detector.iav.matrix"] >> inputs("countrate.iav.matrix")
            outputs("countrate.raw") >> inputs("countrate.iav.vector")

            load_graph_data(
                name = "detector.lsnl.curves",
                x = "edep",
                y = "evis_parts",
                merge_x = True,
                filenames = path_arrays/f"detector_LSNL_curves_Jan2022_newE_v1.{self._source_type}",
                replicate = index["lsnl"],
            )

            from dgf_detector.bundles.refine_lsnl_data import refine_lsnl_data
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
            # TODO:
            # - LSNL weights
            # - escale per AD
            # - Monotonize
            Sum.replicate(outputs("detector.lsnl.curves.evis_parts"), name="detector.lsnl.curves.evis")
            InterpolatorGroup.replicate(
                method = "linear",
                names = {
                    "indexer": "detector.lsnl.indexer_fwd",
                    "interpolator": "detector.lsnl.interpolated_fwd",
                    },
                replicate = index["detector"]
            )
            outputs["detector.lsnl.curves.edep"] >> inputs["detector.lsnl.interpolated_fwd.xcoarse"]
            outputs["detector.lsnl.curves.evis"] >> inputs("detector.lsnl.interpolated_fwd.ycoarse")
            edges_energy_edep >> inputs["detector.lsnl.interpolated_fwd.xfine"]
            InterpolatorGroup.replicate(
                method = "linear",
                names = {
                    "indexer": "detector.lsnl.indexer_bwd",
                    "interpolator": "detector.lsnl.interpolated_bwd",
                    },
                replicate = index["detector"]
            )
            outputs["detector.lsnl.curves.evis"] >> inputs["detector.lsnl.interpolated_bwd.xcoarse"]
            outputs["detector.lsnl.curves.edep"]  >> inputs("detector.lsnl.interpolated_bwd.ycoarse")
            edges_energy_evis >> inputs["detector.lsnl.interpolated_bwd.xfine"]

            from dgf_detector.AxisDistortionMatrix import AxisDistortionMatrix
            AxisDistortionMatrix.replicate(name="detector.lsnl.matrix", replicate=index["detector"])
            edges_energy_edep.outputs[0] >> inputs("detector.lsnl.matrix.EdgesOriginal")
            outputs("detector.lsnl.interpolated_fwd") >> inputs("detector.lsnl.matrix.EdgesModified")
            outputs("detector.lsnl.interpolated_bwd") >> inputs("detector.lsnl.matrix.EdgesModifiedBackwards")

            VectorMatrixProduct.replicate(name="countrate.lsnl", replicate=index["detector"])
            outputs("detector.lsnl.matrix") >> inputs("countrate.lsnl.matrix")
            outputs("countrate.iav") >> inputs("countrate.lsnl.vector")

            from dgf_detector.EnergyResolution import EnergyResolution
            EnergyResolution.replicate(path="detector.eres")
            nodes["detector.eres.sigma_rel"] << parameters("constrained.detector.eres")
            outputs["edges.energy_evis"] >> inputs["detector.eres.matrix"]
            outputs["edges.energy_evis"] >> inputs["detector.eres.e_edges"]

            VectorMatrixProduct.replicate(name="countrate.erec", replicate=index["detector"])
            outputs["detector.eres.matrix"] >> inputs("countrate.erec.matrix")
            outputs("countrate.lsnl") >> inputs("countrate.erec.vector")

            from dgf_detector.Rebin import Rebin
            Rebin.replicate(names={"matrix": "detector.rebin_matrix", "product": "countrate.final"}, replicate=index["detector"])
            edges_energy_erec >> inputs["detector.rebin_matrix.edges_old"]
            edges_energy_final >> inputs["detector.rebin_matrix.edges_new"]
            outputs("countrate.erec") >> inputs("countrate.final")

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
                replicate = combinations["bkg.detector"],
                skip = inactive_detectors,
                key_order = (1, 0, 2),
                objects = lambda _, idx: f"DYB_{bkg_names[idx[0]]}_expected_spectrum_EH{idx[-1][-2]}_AD{idx[-1][-1]}"
            )

            ads_at_sites = {
                    "EH1": ("AD11", "AD12"),
                    "EH2": ("AD21", "AD22"),
                    "EH3": ("AD31", "AD32", "AD33", "AD34"),
                    }
            remap_items(
                parameters("all.bkg.rate.fastn"),
                outputs.child("bkg.rate.fastn"),
                indices = ads_at_sites,
                skip_indices = inactive_detectors,
                fcn = lambda par: par.output
            )
            remap_items(
                parameters("all.bkg.rate.lihe"),
                outputs.child("bkg.rate.lihe"),
                indices = ads_at_sites,
                skip_indices = inactive_detectors,
                fcn = lambda par: par.output
            )

            Product.replicate(
                    parameters("all.bkg.rate.acc"),
                    outputs("bkg.spectrum_shape.acc"),
                    name = "bkg.spectrum.acc",
                    replicate=combinations["period.detector"],
                    )

            Product.replicate(
                    outputs("bkg.rate.lihe"),
                    outputs("bkg.spectrum_shape.lihe"),
                    name = "bkg.spectrum.lihe",
                    replicate=combinations["period.detector"],
                    )

            Product.replicate(
                    outputs("bkg.rate.fastn"),
                    outputs("bkg.spectrum_shape.fastn"),
                    name = "bkg.spectrum.fastn",
                    replicate=combinations["period.detector"],
                    )

            Product.replicate(
                    parameters("all.bkg.rate.alphan"),
                    outputs("bkg.spectrum_shape.alphan"),
                    name = "bkg.spectrum.alphan",
                    replicate=combinations["period.detector"],
                    )

            Product.replicate(
                    parameters("all.bkg.rate.amc"),
                    outputs("bkg.spectrum_shape.amc"),
                    name = "bkg.spectrum.amc",
                    replicate=combinations["period.detector"],
                    )

            Sum.replicate(
                    outputs("bkg.spectrum"),
                    replicate=combinations["period.detector"],
                    name = "bkg.spectrum_total",
                    )

            #
            # Statistic
            #
            from dgf_statistics.MonteCarlo import MonteCarlo
            MonteCarlo.replicate(
                name="pseudo.data",
                mode="asimov",
                replicate=index["detector"],
                replicate_inputs=index["detector"]
            )
            outputs("countrate.final") >> inputs("pseudo.data.input")

            from dgf_statistics.Chi2 import Chi2
            Chi2.replicate(replicate_inputs=index["detector"], name="statistic.stat.chi2p")
            outputs("pseudo.data") >> inputs("statistic.stat.chi2p.data")
            outputs("countrate.final") >> inputs("statistic.stat.chi2p.theory")
            outputs("pseudo.data") >> inputs("statistic.stat.chi2p.errors")

            from dgf_statistics.CNPStat import CNPStat
            CNPStat.replicate(replicate_inputs=index["detector"], replicate=index["detector"], name="statistic.staterr.cnp")
            outputs("pseudo.data") >> inputs("statistic.staterr.cnp.data")
            outputs("countrate.final") >> inputs("statistic.staterr.cnp.theory")

            Chi2.replicate(replicate_inputs=index["detector"], name="statistic.stat.chi2cnp")
            outputs("pseudo.data") >> inputs("statistic.stat.chi2cnp.data")
            outputs("countrate.final") >> inputs("statistic.stat.chi2cnp.theory")
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
                    f"The following label groups were not used: {tuple(labels.keys())}"
                )
