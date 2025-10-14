from pathlib import Path
from typing import Literal

from colorama import Fore
from dag_modelling.tools.logger import logger
from dag_modelling.tools.schema import LoadYaml
from semver import Version


def validate_dataset_get_source_type(
    path_data: Path, meta_name: str, *, version_min: str, version_max: str
) -> Literal["tsv", "hdf5", "root", "npz"]:
    """Validate the dataset version based on `meta_name` yaml file (`dataset_information.yaml`).

    Read or autodetect the source type (format).

    when the dataset's version is lower than minimal, it means that it was suitable for previous
    model, but not the current one and it is suggested to update the model. The current model is
    newer and thus the dataset should be updated. When the dataset's version is higher than maximal
    it means that the dataset is newer than the current model, thus the model should be updated.
    """
    source_type, version = load_manifest(path_data, meta_name)
    if source_type is None:
        source_type = auto_detect_source_type(path_data)
        logger.info(f"Source type automatically detected: {source_type}")
        return source_type
    assert version

    logger.info(f"Source type from {meta_name}: {source_type}")
    logger.info(f"Dataset version: {version!s}")

    version_min_v = Version.parse(version_min)
    version_max_v = Version.parse(version_max)

    if version < version_min_v:
        message = f"Dataset version {version!s} should be ≥{version_min} and <{version_max}"
        logger.critical(message)
        logger.info("Consider updating the dataset")
        raise RuntimeError(message)
    if version >= version_max_v:
        message = f"Dataset version {version!s} should be ≥{version_min} and <{version_max}"
        logger.critical(message)
        logger.info("Consider updating the model")
        raise RuntimeError(message)

    return source_type


def auto_detect_source_type(path_data: Path) -> Literal["tsv", "hdf5", "root", "npz"]:
    """Automatic detection of source type of data.

    It determines source type by path of data. Data must contain one of the next
    types: `tsv`, `hdf5`, `root`, or `npz`. It is not possible to mix data of
    different types. Parameters directory doesn't used in source type determination.

    Parameters
    ----------
    path_data : Path
        Path to data

    Returns
    -------
    Literal["tsv", "hdf5", "root", "npz"]
        Type of source data
    """
    extensions = {
        path.suffix[1:]
        for path in filter(
            lambda path: path.is_file() and "parameters" not in path.parts, path_data.rglob("*.*")
        )
    }
    extensions -= {"py", "yaml"}
    if len(extensions) == 1:
        source_type = extensions.pop()
        if source_type not in {"tsv", "hdf5", "root", "npz", "bz2"}:
            message = f"Unexpected data extension: {source_type}"
            logger.critical(message)
            raise RuntimeError(message)

        if source_type == "bz2":
            source_type = "tsv"

        return source_type  # pyright: ignore [reportReturnType]

    elif len(extensions) > 1:
        message = f"Find to many possibly loaded extensions: {', '.join(extensions)}"
        logger.critical(message)
        raise RuntimeError(message)

    message = f"Data directory `{path_data}` may not exists"
    logger.critical(message)
    raise RuntimeError(message)


def load_manifest(
    path_data: Path, meta_name: str
) -> tuple[Literal["tsv", "hdf5", "root", "npz"] | None, Version | None]:
    manifest_name = path_data / meta_name
    if not manifest_name.is_file():
        logger.warning(
            f"{Fore.RED}"
            f"{meta_name} not found. Version checking disabled. Trying to deduce the source type..."
            f"{Fore.RESET}"
        )
        return None, None

    manifest = LoadYaml(manifest_name)
    try:
        version_str = manifest["version"]
    except KeyError as e:
        message = f"Can not obtain 'version' from {meta_name}"
        logger.critical(message)
        raise RuntimeError(message) from e

    try:
        version = Version.parse(version_str)
    except ValueError as e:
        message = "Version format is not valid: {version_str}"
        logger.critical(message)
        raise RuntimeError(message) from e

    try:
        source_type = manifest["metadata"]["format"]
    except (KeyError, TypeError) as e:
        message = f"Can not obtain ['metadata']['format'] from the {meta_name}"
        logger.critical(message)
        raise RuntimeError(message) from e

    if source_type not in {"tsv", "hdf5", "root", "npz"}:
        message = f"Source type {source_type}, reported by {meta_name} is not supported"
        logger.critical(message)
        raise RuntimeError(message)

    return source_type, version
