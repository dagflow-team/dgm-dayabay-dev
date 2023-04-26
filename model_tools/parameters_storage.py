from multikeydict.nestedmkdict import NestedMKDict
from multikeydict.visitor import NestedMKDictVisitor

from typing import Union, Tuple, List, Optional

from tabulate import tabulate
from pandas import DataFrame

from numpy import nan
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 100)

class ParametersStorage(NestedMKDict):
    def to_dict(self, **kwargs) -> list:
        return self.visit(ParametersVisitor(kwargs)).data

    def to_df(self, *, columns: Optional[List[str]]=None, **kwargs) -> DataFrame:
        dct = self.to_dict(**kwargs)
        if columns is None:
            columns = ['path', 'value', 'central', 'sigma', 'label']
        df = DataFrame(dct, columns=columns)

        df.insert(4, 'sigma_rel_perc', df['sigma'])
        df['sigma_rel_perc'] = df['sigma']/df['central']*100.
        df['sigma_rel_perc'].mask(df['central']==0, nan, inplace=True)

        for key in ('central', 'sigma', 'sigma_rel_perc'):
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
