from itertools import product
from pathlib import Path
from typing import Literal, Optional

from dagflow.bundles.load_array import load_array
from dagflow.bundles.load_graph import load_graph
from dagflow.bundles.load_parameters import load_parameters
from dagflow.graph import Graph
from dagflow.lib.arithmetic import Sum
from dagflow.logger import DEBUG, SUBINFO, SUBSUBINFO, set_level
from dagflow.storage import NodeStorage
from dagflow.tools.schema import LoadYaml
from multikeydict.nestedmkdict import NestedMKDict


class model_dayabay_v0:
    __slots__ = ("storage", "graph", "_datasource", "_sourcetype", "_strict", "_close")

    storage: NodeStorage
    graph: Optional[Graph]
    _datasource: Path
    _sourcetype: Literal["tsv", "hdf", "root"]
    _strict: bool
    _close: bool

    def __init__(
        self,
        *,
        sourcetype: Literal["tsv", "hdf", "root"] = "tsv",
        strict: bool = True,
        close: bool = True,
    ):
        self._strict = strict
        self._close = close

        set_level(SUBINFO)

        self.graph = None
        self.storage = NodeStorage()
        self._datasource = Path("data/dayabay-v0")
        self._sourcetype = sourcetype

        self.build()

    def build(self):
        storage = self.storage
        datasource = self._datasource

        index, combinations = {}, {}
        index["isotope"] = ("U235", "U238", "Pu239", "Pu241")
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
        index["period"] = ("6AD", "8AD", "7AD")
        index["background"] = ("acc", "lihe", "fastn", "alphan", "amc")
        index_all = (
            index["isotope"] + index["detector"] + index["reactor"] + index["period"]
        )
        set_all = set(index_all)
        if len(index_all) != len(set_all):
            raise RuntimeError("Repeated indices")
        combinations["reactor.detector"] = tuple(
            product(index["reactor"], index["detector"])
        )
        combinations["reactor.isotope"] = tuple(
            product(index["reactor"], index["isotope"])
        )
        combinations["reactor.isotopes.detector"] = tuple(
            product(index["reactor"], index["isotope"], index["detector"])
        )
        combinations["background.detector"] = tuple(
            product(index["background"], index["detector"])
        )
        inactive_detectors = [("6AD", "AD22"), ("6AD", "AD34"), ("7AD", "AD11")]
        combinations["period.detector"] = tuple(
            pair
            for pair in product(index["period"], index["detector"])
            if not pair in inactive_detectors
        )

        with Graph(close=self._close) as graph, storage:
            self.graph = graph
            #
            # Load parameters
            #
            load_parameters(path="oscprob", load=datasource / "parameters/oscprob.yaml")
            load_parameters(
                path="oscprob",
                load=datasource / "parameters/oscprob_solar.yaml",
                joint_nuisance=True,
            )
            load_parameters(
                path="oscprob", load=datasource / "parameters/oscprob_constants.yaml"
            )

            load_parameters(path="ibd", load=datasource / "parameters/pdg2012.yaml")
            load_parameters(
                path="ibd.csc", load=datasource / "parameters/ibd_constants.yaml"
            )
            load_parameters(
                path="conversion",
                load=datasource / "parameters/conversion_thermal_power.yaml",
            )
            load_parameters(
                path="conversion",
                load=datasource / "parameters/conversion_oscprob_argument.yaml",
            )

            load_parameters(load=datasource / "parameters/baselines.yaml")

            load_parameters(
                path="detector",
                load=datasource / "parameters/detector_nprotons_correction.yaml",
            )
            load_parameters(
                path="detector", load=datasource / "parameters/detector_eres.yaml"
            )

            load_parameters(
                path="reactor",
                load=datasource / "parameters/reactor_e_per_fission.yaml",
            )
            load_parameters(
                path="reactor",
                load=datasource / "parameters/reactor_thermal_power_nominal.yaml",
                replicate=index["reactor"],
            )
            load_parameters(
                path="reactor",
                load=datasource / "parameters/reactor_snf.yaml",
                replicate=index["reactor"],
            )
            load_parameters(
                path="reactor",
                load=datasource / "parameters/reactor_offequilibrium_correction.yaml",
                replicate=combinations["reactor.isotope"],
            )
            load_parameters(
                path="reactor",
                load=datasource / "parameters/reactor_fission_fraction_scale.yaml",
                replicate=index["reactor"],
                replica_key_offset=1,
            )

            load_parameters(
                path="bkg.rate", load=datasource / "parameters/bkg_rates.yaml"
            )

            # Create Nuisance parameters
            nuisanceall = Sum("nuisance total")
            storage["stat.nuisance.all"] = nuisanceall

            (
                output for output in storage("stat.nuisance_parts").walkvalues()
            ) >> nuisanceall

            #
            # Create nodes
            #
            labels = LoadYaml(datasource / "labels.yaml")
            parameters = storage("parameter")
            nodes = storage.child("nodes")
            inputs = storage.child("inputs")
            outputs = storage.child("outputs")

            from numpy import linspace

            from dagflow.lib.Array import Array
            from dagflow.lib.View import View

            edges_costheta, _ = Array.make_stored("edges.costheta", [-1, 1])
            edges_energy_common, _ = Array.make_stored(
                "edges.energy_common", linspace(0, 12, 241)
            )
            View.make_stored("edges.energy_enu", edges_energy_common)
            edges_energy_edep, _ = View.make_stored(
                "edges.energy_edep", edges_energy_common
            )
            View.make_stored("edges.energy_evis", edges_energy_common)
            View.make_stored("edges.energy_erec", edges_energy_common)

            integration_orders_edep, _ = Array.from_value(
                "kinematics_sampler.ordersx", 5, edges=edges_energy_edep
            )
            integration_orders_costheta, _ = Array.from_value(
                "kinematics_sampler.ordersy", 4, edges=edges_costheta
            )

            from dagflow.lib.IntegratorGroup import IntegratorGroup

            integrator, _ = IntegratorGroup.replicate(
                "2d",
                "kinematics_sampler",
                "kinematics_integral",
                name_x="mesh_edep",
                name_y="mesh_costheta",
                replicate=combinations["reactor.isotopes.detector"],
            )
            integration_orders_edep >> integrator.inputs["ordersX"]
            integration_orders_costheta >> integrator.inputs["ordersY"]

            from reactornueosc.IBDXsecVBO1Group import IBDXsecVBO1Group

            ibd, _ = IBDXsecVBO1Group.make_stored(use_edep=True)
            ibd << storage("parameter.constant.ibd")
            ibd << storage("parameter.constant.ibd.csc")
            outputs["kinematics_sampler.mesh_edep"] >> ibd.inputs["edep"]
            outputs["kinematics_sampler.mesh_costheta"] >> ibd.inputs["costheta"]

            from reactornueosc.NueSurvivalProbability import NueSurvivalProbability

            NueSurvivalProbability.replicate(
                "oscprob", distance_unit="m", replicate=combinations["reactor.detector"]
            )
            ibd.outputs["enu"] >> inputs("oscprob.enu")
            parameters("constant.baseline") >> inputs("oscprob.L")
            nodes("oscprob") << parameters("free.oscprob")
            nodes("oscprob") << parameters("constrained.oscprob")
            nodes("oscprob") << parameters("constant.oscprob")

            load_graph(
                name="reactor_anue.input_spectrum",
                x="enu",
                y="spec",
                merge_x=True,
                load=datasource / self._sourcetype / "reactor_anue_spectra_50kev.yaml",
                replicate=index["isotope"],
            )

            from dagflow.lib.InterpolatorGroup import InterpolatorGroup

            interpolator, _ = InterpolatorGroup.replicate(
                "exp",
                "reactor_anue.indexer",
                "reactor_anue.interpolator",
                replicate=index["isotope"],
            )
            (
                outputs["reactor_anue.input_spectrum.enu"]
                >> inputs["reactor_anue.interpolator.xcoarse"]
            )
            outputs("reactor_anue.input_spectrum.spec") >> inputs(
                "reactor_anue.interpolator.ycoarse"
            )
            ibd.outputs["enu"] >> inputs["reactor_anue.interpolator.xfine"]

            from dagflow.lib.arithmetic import Product

            Product.replicate(
                "kinematics_integrand",
                replicate=combinations["reactor.isotopes.detector"],
            )
            outputs("oscprob") >> nodes("kinematics_integrand")
            outputs["ibd.crosssection"] >> nodes("kinematics_integrand")
            outputs["ibd.jacobian"] >> nodes("kinematics_integrand")
            outputs("reactor_anue.interpolator") >> nodes("kinematics_integrand")
            outputs("kinematics_integrand") >> inputs("kinematics_integral")

            from reactornueosc.InverseSquareLaw import InverseSquareLaw

            InverseSquareLaw.replicate(
                "baseline_factor", replicate=combinations["reactor.detector"]
            )
            parameters("constant.baseline") >> inputs("baseline_factor")

            Product.replicate(
                "countrate_reac", replicate=combinations["reactor.isotopes.detector"]
            )
            outputs("kinematics_integral") >> nodes("countrate_reac")
            outputs("baseline_factor") >> nodes("countrate_reac")

            Sum.replicate(
                "countrate.raw", outputs("countrate_reac"), replicate=index["detector"]
            )

            load_array(
                name="detector.iav.matrix_raw",
                filenames=datasource / self._sourcetype / "detector_IAV_matrix_P14A_LS.tsv",
                replicate="iav_matrix"
            )

            from dagflow.lib.NormalizeMatrix import NormalizeMatrix

            NormalizeMatrix.replicate("detector.iav.matrix")
            outputs["detector.iav.matrix_raw"] >> nodes["detector.iav.matrix"]

            from dagflow.lib.VectorMatrixProduct import VectorMatrixProduct

            VectorMatrixProduct.replicate("countrate.iav", replicate=index["detector"])
            outputs["detector.iav.matrix"] >> inputs("countrate.iav.matrix")
            outputs("countrate.raw") >> inputs("countrate.iav.vector")

            # VectorMatrixProduct.replicate("countrate.lsnl", replicate=index["detector"])
            # # outputs["detector.lsnl.matrix"] >> inputs("countrate.lsnl.matrix")
            # outputs("countrate.iav") >> inputs("countrate.lsnl.vector")
            #
            # VectorMatrixProduct.replicate("countrate.erec", replicate=index["detector"])
            # # outputs["detector.erec.matrix"] >> inputs("countrate.erec.matrix")
            # outputs("countrate.lsnl") >> inputs("countrate.erec.vector")

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
