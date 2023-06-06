from dagflow.bundles.load_parameters import load_parameters
from dagflow.bundles.load_arrays import load_arrays
from pathlib import Path

from dagflow.graph import Graph
from dagflow.lib.arithmetic import Sum
from dagflow.tools.schema import LoadYaml
from dagflow.plot import plot_auto
from gindex import GNIndex
from dagflow.storage import NodeStorage

from itertools import product

def model_dayabay_v0():
    storage = NodeStorage()
    datasource = Path("data/dayabay-v0")

    index = GNIndex.from_dict({
                ("s", "site"): ("EH1", "EH2", "EH3"),
                ("d", "detector"): ("AD11", "AD12", "AD21", "AD22", "AD31", "AD32", "AD33", "AD34"),
                ("p", "period"): ("6AD", "8AD", "7AD"),
                ("r", "reactor"): ("DB1", "DB2", "LA1", "LA2", "LA3", "LA4"),
                ("i", "isotope"): ("U235", "U238", "Pu239", "Pu241"),
                ("b", "background"): ("acc", "lihe", "fastn", "amc", "alphan"),
                })
    idx_d = index.sub("d")
    idx_r = index.sub("r")
    idx_rd = index.sub(("r", "d"))
    idx_ri = index.sub(("r", "i"))

    list_isotopes = ("U235", "U238", "Pu239", "Pu241")
    list_detectors = ("AD11", "AD12", "AD21", "AD22", "AD31", "AD32", "AD33", "AD34")
    list_reactors = ("DB1", "DB2", "LA1", "LA2", "LA3", "LA4")
    combinations_reactors_detectors = tuple(product(list_reactors, list_detectors))
    combinations_reactors_isotopes = tuple(product(list_reactors, list_isotopes))

    close = True
    with Graph(close=close) as graph, storage:
        #
        # Load parameters
        #
        load_parameters(path="oscprob",    load=datasource/"parameters/oscprob.yaml")
        load_parameters(path="oscprob",    load=datasource/"parameters/oscprob_solar.yaml", joint_nuisance=True)
        load_parameters(path="oscprob",    load=datasource/"parameters/oscprob_constants.yaml")

        load_parameters(path="ibd",        load=datasource/"parameters/pdg2012.yaml")
        load_parameters(path="ibd.csc",    load=datasource/"parameters/ibd_constants.yaml")
        load_parameters(path="conversion", load=datasource/"parameters/conversion_thermal_power.yaml")
        load_parameters(path="conversion", load=datasource/"parameters/conversion_oscprob_argument.yaml")

        load_parameters(                   load=datasource/"parameters/baselines.yaml")

        load_parameters(path="detector",   load=datasource/"parameters/detector_nprotons_correction.yaml")
        load_parameters(path="detector",   load=datasource/"parameters/detector_eres.yaml")

        load_parameters(path="reactor",    load=datasource/"parameters/reactor_e_per_fission.yaml")
        load_parameters(path="reactor",    load=datasource/"parameters/reactor_thermal_power_nominal.yaml",     replicate=list_reactors)
        load_parameters(path="reactor",    load=datasource/"parameters/reactor_snf.yaml",                       replicate=list_reactors)
        load_parameters(path="reactor",    load=datasource/"parameters/reactor_offequilibrium_correction.yaml", replicate=combinations_reactors_isotopes)
        load_parameters(path="reactor",    load=datasource/"parameters/reactor_fission_fraction_scale.yaml",    replicate=list_reactors, replica_key_offset=1)

        # Create Nuisance parameters
        nuisanceall = Sum("nuisance total")
        storage["stat.nuisance.all"] = nuisanceall

        (output for output in storage("stat.nuisance_parts").walkvalues()) >> nuisanceall

        #
        # Create nodes
        #
        labels = LoadYaml(datasource/"labels.yaml")
        parameters = storage("parameter")
        nodes = storage.child("nodes")
        inputs = storage.child("inputs")
        outputs = storage.child("outputs")

        from dagflow.lib.Array import Array
        from dagflow.lib.View import View
        from numpy import linspace
        edges_costheta, _ = Array.make_stored("edges.costheta", [-1, 1], label_from=labels)
        edges_energy_common, _ = Array.make_stored("edges.energy_common", linspace(0, 12, 241), label_from=labels)
        View.make_stored("edges.energy_enu", edges_energy_common, label_from=labels)
        edges_energy_edep, _ = View.make_stored("edges.energy_edep", edges_energy_common, label_from=labels)
        View.make_stored("edges.energy_evis", edges_energy_common, label_from=labels)
        View.make_stored("edges.energy_erec", edges_energy_common, label_from=labels)

        integration_orders_edep, _ = Array.from_value("integration.ordersx", 4, edges=edges_energy_edep, label_from=labels)
        integration_orders_costheta, _ = Array.from_value("integration.ordersy", 4, edges=edges_costheta, label_from=labels)
        from dagflow.lib.IntegratorGroup import IntegratorGroup
        integrator, _ = IntegratorGroup.replicate("2d", "kinematics_sampler", "kinematics_integral", replicate=combinations_reactors_detectors)
        integration_orders_edep >> integrator.inputs["ordersX"]
        integration_orders_costheta >> integrator.inputs["ordersY"]
        outputs["integration.mesh_edep"] = (int_mesh_edep:=integrator.outputs["x"])
        outputs["integration.mesh_costheta"] = (int_mesh_costheta:=integrator.outputs["y"])

        from reactornueosc.IBDXsecO1Group import IBDXsecO1Group
        ibd, _ = IBDXsecO1Group.make_stored(use_edep=True)
        ibd << storage("parameter.constant.ibd")
        ibd << storage("parameter.constant.ibd.csc")
        int_mesh_edep >> ibd.inputs["edep"]
        int_mesh_costheta >> ibd.inputs["costheta"]

        from reactornueosc.NueSurvivalProbability import NueSurvivalProbability
        NueSurvivalProbability.replicate("oscprob", distance_unit="m", replicate=combinations_reactors_detectors)
        ibd.outputs["enu"] >> inputs("oscprob.enu")
        parameters("constant.baseline") >> inputs("oscprob.L")
        nodes("oscprob") << parameters("free.oscprob")
        nodes("oscprob") << parameters("constrained.oscprob")
        nodes("oscprob") << parameters("constant.oscprob")

        load_arrays(
                name = "reactor_anue_spectrum",
                filenames = [
                    datasource/"tsv/reactor_anue_spectra_50kev/Huber_anue_spectrum_extrap_U235_13.0_0.05_MeV.tsv",
                    datasource/"tsv/reactor_anue_spectra_50kev/Huber_anue_spectrum_extrap_Pu239_13.0_0.05_MeV.tsv",
                    datasource/"tsv/reactor_anue_spectra_50kev/Huber_anue_spectrum_extrap_Pu241_13.0_0.05_MeV.tsv",
                    datasource/"tsv/reactor_anue_spectra_50kev/Mueller_anue_spectrum_extrap_U238_13.0_0.05_MeV.tsv"
                    ],
                replicate = list_isotopes,
                x = 'enu',
                y = 'spec',
                merge_x = True
                )

        # load_arrays(
        #         name = "reactor_anue_spectrum",
        #         filenames = datasource/"hdf/anue_spectra_extrap_13.0_0.05_MeV.hdf5",
        #         replicate = list_isotopes
        #         )
        #
        # load_arrays(
        #         name = "reactor_anue_spectrum",
        #         filenames = datasource/"root/anue_spectra_extrap_13.0_0.05_MeV.root",
        #         replicate = list_isotopes
        #         )
        from dagflow.lib.InterpolatorGroup import InterpolatorGroup
        interpolator, _ = InterpolatorGroup.replicate("exp", "reactor_anue.indexer", "reactor_anue.interpolator", replicate=list_isotopes)
        outputs["reactor_anue_spectrum.enu"] >> inputs["reactor_anue.interpolator.xcoarse"]
        outputs("reactor_anue_spectrum.spec") >> inputs("reactor_anue.interpolator.ycoarse")
        ibd.outputs["enu"] >> inputs["reactor_anue.interpolator.xfine"]
        # outputs

        from dagflow.lib.arithmetic import Product
        # Product.replicate("kinematics_integrand", outputs("oscprob"), outputs["ibd"], replicate=combinations_reactors_detectors)
        Product.replicate("kinematics_integrand", replicate=combinations_reactors_detectors)
        outputs("oscprob") >> nodes("kinematics_integrand")
        outputs["ibd.crosssection"] >> nodes("kinematics_integrand")
        outputs["ibd.jacobian"] >> nodes("kinematics_integrand")
        outputs("kinematics_integrand") >> inputs("kinematics_integral")

        from reactornueosc.InverseSquareLaw import InverseSquareLaw
        InverseSquareLaw.replicate("baseline_factor", replicate=combinations_reactors_detectors)
        parameters("constant.baseline") >> inputs("baseline_factor")

        Product.replicate("countrate_reac", replicate=combinations_reactors_detectors)
        outputs("kinematics_integral")>>nodes("countrate_reac")
        outputs("baseline_factor")>>nodes("countrate_reac")

        Sum.replicate("count_rate", outputs("countrate_reac"), replicate=list_detectors)

    storage("outputs").read_labels(labels, strict=True)
    storage("inputs").remove_connected_inputs()
    storage.read_paths()
    # storage.process_indices(idx_unique)

    if not close:
        print(storage.to_table(truncate=True))
        return

    # storage("outputs").plot(folder='output/dayabay_v0_auto')
    # storage("outputs.oscprob").plot(show_all=True)

    storage["parameter.normalized.detector.eres.b_stat"].value = 1
    storage["parameter.normalized.detector.eres.a_nonuniform"].value = 2

    # p1 = storage["parameter.normalized.detector.eres.b_stat"]
    # p2 = storage["parameter.constrained.detector.eres.b_stat"]

    constrained = storage("parameter.constrained")
    normalized = storage("parameter.normalized")

    print("Everything")
    print(storage.to_table(truncate=True))

    # print("Constants")
    # print(storage("parameter.constant").to_table(truncate=True))
    #
    # print("Constrained")
    # print(constrained.to_table(truncate=True))
    #
    # print("Normalized")
    # print(normalized.to_table(truncate=True))
    #
    # print("Stat")
    # print(storage("stat").to_table(truncate=True))

    # print("Parameters (latex)")
    # print(storage["parameter"].to_latex())
    #
    # print("Constants (latex)")
    # tex = storage["parameter.constant"].to_latex(columns=["path", "value", "label"])
    # print(tex)

    storage.to_datax("output/dayabay_v0_data.tex")

    from dagflow.graphviz import GraphDot
    GraphDot.from_graph(graph, show="all").savegraph("output/dayabay_v0.dot")
    GraphDot.from_node(storage["parameter_node.constrained.reactor.fission_fraction_scale.DB1"].constraint._norm_node, show="all", minsize=2).savegraph("output/dayabay_v0_large.dot")
    GraphDot.from_node(storage["stat.nuisance.all"], show="all", mindepth=-1, no_forward=True).savegraph("output/dayabay_v0_nuisance.dot")
    GraphDot.from_output(storage["outputs.edges.energy_evis"], show="all", mindepth=-3, no_forward=True).savegraph("output/dayabay_v0_top.dot")

