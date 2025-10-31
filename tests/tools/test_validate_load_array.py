from pathlib import Path

import numpy as np
from pytest import raises

from dgm_dayabay_dev.tools.validate_load_array import validate_load_array


def test_validate_load_array():
    array_path = Path("data/dayabay-v1a/parameters-common/reactor_antineutrino_spectrum_edges.tsv")

    assert validate_load_array(array_path) == array_path
    assert validate_load_array(str(array_path)) == array_path

    edges_np = np.linspace(0, 12, 13)
    assert np.allclose(validate_load_array(edges_np), edges_np)

    assert np.allclose(validate_load_array(edges_np.tolist()), edges_np)

    with raises(FileNotFoundError):
        validate_load_array("not/found/file")

    assert validate_load_array(None) is None
