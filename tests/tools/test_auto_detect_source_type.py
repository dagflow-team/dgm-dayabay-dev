from os import remove
from pathlib import Path

from pytest import mark, raises

from dgm_dayabay_dev.tools import auto_detect_source_type


@mark.parametrize("extension", ["hdf5", "root", "npz", "tsv"])
def test_data_exists(extension):
    basepath = Path("data/dayabay-v1a/")

    assert extension == auto_detect_source_type(basepath / Path(extension))


def test_data_not_exists():
    data_path = Path("data/not/exists")

    with raises(RuntimeError) as excinfo:
        auto_detect_source_type(data_path)

        assert str(data_path) in str(excinfo.value)


def test_data_has_many_source_types():
    data_path = Path("data/dayabay-v1a/")

    fake_file = data_path / Path("hdf5/fake_file.npz")
    fake_file.touch()

    with raises(RuntimeError) as excinfo:
        auto_detect_source_type(data_path)

        assert str(data_path) in str(excinfo.value)

    remove(fake_file)
