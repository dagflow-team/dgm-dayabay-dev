from glob import glob

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pytest import mark


def ipy_notebooks():
    return glob("./extras/notebooks/**/*.ipynb", recursive=True)


@mark.parametrize("path", ipy_notebooks())
def test_notebook(path: str):
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb)
