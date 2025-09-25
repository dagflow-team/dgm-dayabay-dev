from dgm_dayabay_dev.bundles import auto_detect_source_type
from pathlib import Path
from pytest import mark, raises


@mark.parametrize("extension", ["hdf5", "root", "npz", "tsv"])
def test_data_exists(extension):
    basepath = Path("data/dayabay-v1a/")

    assert extension == auto_detect_source_type(basepath / Path(extension))


def test_data_not_exists():
    data_path = Path("data/not/exists")

    with raises(FileNotFoundError) as excinfo:
        auto_detect_source_type(data_path)

        assert str(data_path) in str(excinfo.value)

