from dagflow.bundles.load_parameters import load_parameters
from pathlib import Path

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Sum import Sum

from gindex import GNIndex

from model_tools.parameters_storage import ParametersStorage

def model_dayabay_v0():
    storage = ParametersStorage({}, sep='.')
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

    with Graph(close=True) as g:
        storage ^= load_parameters({'path': 'ibd'        , 'load': datasource/'parameters/pdg2012.yaml'})
        storage ^= load_parameters({'path': 'ibd.csc'    , 'load': datasource/'parameters/ibd_constants.yaml'})
        storage ^= load_parameters({'path': 'conversion' , 'load': datasource/'parameters/conversion_thermal_power.yaml'})
        storage ^= load_parameters({'path': 'conversion' , 'load': datasource/'parameters/conversion_oscprob_argument.yaml'})

        storage ^= load_parameters({                       'load': datasource/'parameters/baselines.yaml'})

        storage ^= load_parameters({'path': 'detector'   , 'load': datasource/'parameters/detector_nprotons_correction.yaml'})
        storage ^= load_parameters({'path': 'detector'   , 'load': datasource/'parameters/detector_eres.yaml'})

        storage ^= load_parameters({'path': 'reactor'    , 'load': datasource/'parameters/reactor_thermal_power_nominal.yaml'     , 'replicate': list_reactors })
        storage ^= load_parameters({'path': 'reactor'    , 'load': datasource/'parameters/reactor_snf.yaml'                       , 'replicate': list_reactors })
        storage ^= load_parameters({'path': 'reactor'    , 'load': datasource/'parameters/reactor_offequilibrium_correction.yaml' , 'replicate': list_reactors_isotopes })
        storage ^= load_parameters({'path': 'reactor'    , 'load': datasource/'parameters/reactor_fission_fraction_scale.yaml'    , 'replicate': list_reactors , 'replica_key_offset': 1 })

        nuisanceall = Sum('nuisance total')
        storage['stat.nuisance.all'] = nuisanceall

        (output for output in storage['stat.nuisance_parts'].walkvalues()) >> nuisanceall

    storage['parameter.normalized.detector.eres.b_stat'].value = 1
    storage['parameter.normalized.detector.eres.a_nonuniform'].value = 2

    # p1 = storage['parameter.normalized.detector.eres.b_stat']
    # p2 = storage['parameter.constrained.detector.eres.b_stat']
    # import IPython; IPython.embed(colors='neutral')

    print('Everything')
    print(storage.to_table(truncate=True))

    print('Constants')
    print(storage['parameter.constant'].to_table(truncate=True))

    print('Constrained')
    print(storage['parameter.constrained'].to_table(truncate=True))

    print('Normalized')
    print(storage['parameter.normalized'].to_table(truncate=True))

    print('Stat')
    print(storage['stat'].to_table(truncate=True))

    # print('Parameters (latex)')
    # print(storage['parameter'].to_latex())
    #
    # print('Constants (latex)')
    # tex = storage['parameter.constant'].to_latex(columns=['path', 'value', 'label'])
    # print(tex)

    savegraph(g, "output/dayabay_v0.dot", show='all')
