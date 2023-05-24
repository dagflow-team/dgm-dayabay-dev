from dagflow.bundles.load_parameters import load_parameters
from pathlib import Path

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.arithmetic import Sum
from dagflow.tools.schema import LoadYaml
from gindex import GNIndex
from dagflow.storage import NodeStorage

def model_dayabay_v0():
    storage = NodeStorage({}, sep='.')
    datasource = Path('data/dayabay-v0')

    index = GNIndex.from_dict({
                ('s', 'site'): ('EH1', 'EH2', 'EH3'),
                ('d', 'detector'): ('AD11', 'AD12', 'AD21', 'AD22', 'AD31', 'AD32', 'AD33', 'AD34'),
                ('p', 'period'): ('6AD', '8AD', '7AD'),
                ('r', 'reactor'): ('DB1', 'DB2', 'LA1', 'LA2', 'LA3', 'LA4'),
                ('i', 'isotope'): ('U235', 'U238', 'Pu239', 'Pu241'),
                ('b', 'background'): ('acc', 'lihe', 'fastn', 'amc', 'alphan'),
                })
    idx_r= index.sub('r')
    idx_rd= index.sub(('r', 'd'))
    idx_ri= index.sub(('r', 'i'))
    list_reactors = idx_r.values
    list_dr = idx_rd.values
    list_reactors_isotopes = idx_ri.values

    with Graph(close=True) as graph, storage:
        #
        # Load parameters
        #
        load_parameters({'path': 'ibd'        , 'load': datasource/'parameters/pdg2012.yaml'})
        load_parameters({'path': 'ibd.csc'    , 'load': datasource/'parameters/ibd_constants.yaml'})
        load_parameters({'path': 'conversion' , 'load': datasource/'parameters/conversion_thermal_power.yaml'})
        load_parameters({'path': 'conversion' , 'load': datasource/'parameters/conversion_oscprob_argument.yaml'})

        load_parameters({                       'load': datasource/'parameters/baselines.yaml'})

        load_parameters({'path': 'detector'   , 'load': datasource/'parameters/detector_nprotons_correction.yaml'})
        load_parameters({'path': 'detector'   , 'load': datasource/'parameters/detector_eres.yaml'})

        load_parameters({'path': 'reactor'    , 'load': datasource/'parameters/reactor_e_per_fission.yaml'})
        load_parameters({'path': 'reactor'    , 'load': datasource/'parameters/reactor_thermal_power_nominal.yaml'     , 'replicate': list_reactors })
        load_parameters({'path': 'reactor'    , 'load': datasource/'parameters/reactor_snf.yaml'                       , 'replicate': list_reactors })
        load_parameters({'path': 'reactor'    , 'load': datasource/'parameters/reactor_offequilibrium_correction.yaml' , 'replicate': list_reactors_isotopes })
        load_parameters({'path': 'reactor'    , 'load': datasource/'parameters/reactor_fission_fraction_scale.yaml'    , 'replicate': list_reactors , 'replica_key_offset': 1 })

        # Create Nuisance parameters
        nuisanceall = Sum('nuisance total')
        storage['stat.nuisance.all'] = nuisanceall

        (output for output in storage('stat.nuisance_parts').walkvalues()) >> nuisanceall

        #
        # Create nodes
        #
        from dagflow.lib.Array import Array
        from dagflow.lib.View import View
        from numpy import linspace
        labels = LoadYaml(datasource/'labels.yaml')
        energy_common=Array.store("edges.energy_common", linspace(0, 12, 241), label_from=labels)
        energy_evis=View.store("edges.energy_evis", label_from=labels)
        energy_common >> energy_evis

    storage.read_paths()
    storage('outputs').plot(close=False, show=True)

    storage['parameter.normalized.detector.eres.b_stat'].value = 1
    storage['parameter.normalized.detector.eres.a_nonuniform'].value = 2

    # p1 = storage['parameter.normalized.detector.eres.b_stat']
    # p2 = storage['parameter.constrained.detector.eres.b_stat']

    constrained = storage('parameter.constrained')
    normalized = storage('parameter.normalized')

    print('Everything')
    print(storage.to_table(truncate=True))

    print('Constants')
    print(storage('parameter.constant').to_table(truncate=True))

    print('Constrained')
    print(constrained.to_table(truncate=True))

    print('Normalized')
    print(normalized.to_table(truncate=True))

    print('Stat')
    print(storage('stat').to_table(truncate=True))

    # print('Parameters (latex)')
    # print(storage['parameter'].to_latex())
    #
    # print('Constants (latex)')
    # tex = storage['parameter.constant'].to_latex(columns=['path', 'value', 'label'])
    # print(tex)

    storage.to_datax('output/dayabay_v0_data.tex')

    from dagflow.graphviz import GraphDot
    GraphDot.from_graph(graph, show='all').savegraph("output/dayabay_v0.dot")
    GraphDot.from_node(storage['parameter_node.constrained.reactor.fission_fraction_scale.DB1'].constraint._norm_node, show='all', minsize=2).savegraph("output/dayabay_v0_large.dot")
    GraphDot.from_node(storage['stat.nuisance.all'], show='all', mindepth=-1, no_forward=True).savegraph("output/dayabay_v0_nuisance.dot")
    GraphDot.from_output(storage['outputs.edges.energy_evis'], show='all', mindepth=-3, no_forward=True).savegraph("output/dayabay_v0_top.dot")

