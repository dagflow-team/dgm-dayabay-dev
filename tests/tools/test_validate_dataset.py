from os import remove
from pathlib import Path

from pytest import mark, raises
from yaml import safe_dump

from dgm_dayabay_dev.tools.validate_dataset import (
    auto_detect_source_type,
    validate_dataset_get_source_type,
)


@mark.parametrize("extension", ["hdf5", "root", "npz", "tsv"])
def test_autodetect_data_exists(extension):
    basepath = Path("data/dayabay-v1a/")

    assert extension == auto_detect_source_type(basepath / Path(extension))


def test_autodetect_data_not_exists():
    data_path = Path("data/not/exists")

    with raises(RuntimeError) as excinfo:
        auto_detect_source_type(data_path)

        assert str(data_path) in str(excinfo.value)


def test_autodetect_data_has_many_source_types():
    data_path = Path("data/dayabay-v1a/")

    fake_file = data_path / Path("hdf5/fake_file.npz")
    fake_file.touch()

    with raises(RuntimeError) as excinfo:
        auto_detect_source_type(data_path)

        assert str(data_path) in str(excinfo.value)

    remove(fake_file)


def test_validate_dataset(tmp_path):
    temp_dir = tmp_path / "test_validate_dataset"
    temp_dir.mkdir()

    meta_name = "data_info.yaml"
    temp_file = temp_dir / meta_name
    temp_file.write_text(
        """\
version: "0.1.0"
metadata:
  format: "hdf5"
  description: "Official data of the Daya Bay reactor electron antineutrino experiment"
            """
    )

    source_type = validate_dataset_get_source_type(
        temp_dir, meta_name, version_min="0.1.0", version_max="0.2.0"
    )
    assert source_type == "hdf5"

    with raises(RuntimeError):
        validate_dataset_get_source_type(
            temp_dir, meta_name, version_min="0.1.1", version_max="0.2.0"
        )

    with raises(RuntimeError):
        validate_dataset_get_source_type(
            temp_dir, meta_name, version_min="0.0.1", version_max="0.1.0"
        )


@mark.parametrize(
    "config",
    [
        {
            "version": "invalid",
            "metadata": {
                "format": "hdf5",
                "description": "Official data of the Daya Bay reactor electron antineutrino experiment",
            },
        },
        {
            "version": "0.1.0",
            "metadata_bad": {
                "format": "hdf5",
                "description": "Official data of the Daya Bay reactor electron antineutrino experiment",
            },
        },
        {
            "version": "0.1.0",
            "metadata": {
                "invalid_format": "hdf5",
                "description": "Official data of the Daya Bay reactor electron antineutrino experiment",
            },
        },
        {
            "version": "0.1.0",
            "metadata": {
                "format": "hdf5_invalid",
                "description": "Official data of the Daya Bay reactor electron antineutrino experiment",
            },
        },
    ],
)
def test_validate_dataset_bad_config(tmp_path, config):
    temp_dir = tmp_path / "test_validate_dataset_bad"
    temp_dir.mkdir()

    meta_name = "data_info.yaml"
    temp_file = temp_dir / meta_name
    temp_file.write_text(safe_dump(config))
    with raises(RuntimeError):
        validate_dataset_get_source_type(
            temp_dir, meta_name, version_min="0.1.0", version_max="0.2.0"
        )
