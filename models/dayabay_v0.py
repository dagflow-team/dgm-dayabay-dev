from dagflow.bundles.load_parameters import load_parameters
from multikeydict.nestedmkdict import NestedMKDict
from multikeydict.visitor import NestedMKDictVisitor
from pathlib import Path
from tabulate import tabulate

from typing import Union, Tuple, List, Optional
import pandas as pd
from pandas import DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 100)

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Sum import Sum

from gindex import GNIndex

class ParametersVisitor(NestedMKDictVisitor):
    __slots__ = ('_kwargs', '_data', '_localdata', '_path')
    _kwargs: dict
    _data: List[dict]
    _localdata: List[dict]
    _paths: List[Tuple[str, ...]]
    _path: Tuple[str, ...]
    # _npars: List[int]

    def __init__(self, kwargs: dict):
        self._kwargs = kwargs
        # self._npars = []

    @property
    def data(self):
        return self._data

    def start(self, dct):
        self._data = []
        self._path = ()
        self._paths = []
        self._localdata = []

    def enterdict(self, k, v):
        if self._localdata:
            self.exitdict(self._path, None)
        self._path = k
        self._paths.append(self._path)
        self._localdata = []

    def visit(self, key, value):
        try:
            dct = value.to_dict(**self._kwargs)
        except AttributeError:
            return

        subkey = key[len(self._path):]
        subkeystr = '.'.join(subkey)

        if self._path:
            dct['path'] = f'.. {subkeystr}'
        else:
            dct['path'] = subkeystr

        self._localdata.append(dct)

    def exitdict(self, k, v):
        if self._localdata:
            self._data.append({
                'path': f"group: {'.'.join(self._path)} [{len(self._localdata)}]"
                })
            self._data.extend(self._localdata)
            self._localdata = []
        if self._paths:
            del self._paths[-1]

            self._path = self._paths[-1] if self._paths else ()

    def stop(self, dct):
        pass

class ParametersWrapper(NestedMKDict):
    def to_dict(self, **kwargs) -> list:
        return self.visit(ParametersVisitor(kwargs)).data

    def to_df(self, *, columns: Optional[List[str]]=None, **kwargs) -> DataFrame:
        dct = self.to_dict(**kwargs)
        if columns is None:
            columns = ['path', 'value', 'central', 'sigma', 'label']
        df = DataFrame(dct, columns=columns)
        for key in ('central', 'sigma'):
            if df[key].isna().all():
                del df[key]
            else:
                df[key].fillna('-', inplace=True)

        df['value'].fillna('-', inplace=True)
        df['label'].fillna('', inplace=True)
        return df

    def to_string(self, **kwargs) -> str:
        df = self.to_df()
        return df.to_string(**kwargs)

    def to_table(self, *, df_kwargs: dict={}, **kwargs) -> str:
        df = self.to_df(**df_kwargs)
        kwargs.setdefault('headers', df.columns)
        ret = tabulate(df, **kwargs)

        return ret

    def to_latex(self, *, return_df: bool=False, **kwargs) -> Union[str, Tuple[str, DataFrame]]:
        df = self.to_df(label_from='latex', **kwargs)
        tex = df.to_latex(escape=False)

        if return_df:
            return tex, df

        return tex

def model_dayabay_v0():
    storage = ParametersWrapper({}, sep='.')
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
        storage ^= load_parameters({'path': 'conversion' , 'load': datasource/'parameters/conversion_thermal_power.py'})

        storage ^= load_parameters({                       'load': datasource/'parameters/baselines.yaml'})

        storage ^= load_parameters({'path': 'detector'   , 'load': datasource/'parameters/detector_nprotons_correction.yaml'})
        storage ^= load_parameters({'path': 'detector'   , 'load': datasource/'parameters/detector_eres.yaml'})

        storage ^= load_parameters({'path': 'reactor'    , 'load': datasource/'parameters/reactor_thermal_power_nominal.yaml' , 'replicate': list_reactors })
        storage ^= load_parameters({'path': 'reactor'    , 'load': datasource/'parameters/offequilibrium_correction.yaml'     , 'replicate': list_reactors_isotopes })

        nuisanceall = Sum('nuisance total')
        storage['stat.nuisance.all'] = nuisanceall

        (output for output in storage['stat.nuisance_parts'].walkvalues()) >> nuisanceall

    storage['parameter.normalized.detector.eres.b_stat'].value = 1
    storage['parameter.normalized.detector.eres.a_nonuniform'].value = 2

    print('Everything')
    print(storage.to_table())

    print('Constants')
    print(storage['parameter.constant'].to_table())

    print('Constrained')
    print(storage['parameter.constrained'].to_table())

    print('Normalized')
    print(storage['parameter.normalized'].to_table())

    print('Stat')
    print(storage['stat'].to_table())

    # print('Parameters (latex)')
    # print(storage['parameter'].to_latex())
    #
    # print('Constants (latex)')
    # tex = storage['parameter.constant'].to_latex(columns=['path', 'value', 'label'])
    # print(tex)

    savegraph(g, "output/dayabay_v0.dot", show='all')
