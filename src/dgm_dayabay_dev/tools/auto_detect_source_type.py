from pathlib import Path
from typing import Literal

from dag_modelling.tools.logger import logger
from dag_modelling.tools.schema import LoadYaml


def auto_detect_source_type(path_data: Path) -> Literal["tsv", "hdf5", "root", "npz"]:
    """Automatic detection of source type of data.

    Detection is done in two steps:
        - try to read `Manifest.yaml` file and get the source type from `metadata.format` field.
        - check the actual files and determine the common extension.

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
    if (source_type := read_source_type_from_manifest(path_data)) is not None:
        logger.info(f"Source type from Manifest.yaml: {source_type}")
        return source_type

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

        logger.info(f"Source type automatically detected: {source_type}")

        return source_type  # pyright: ignore [reportReturnType]

    elif len(extensions) > 1:
        message = f"Find to many possibly loaded extensions: {', '.join(extensions)}"
        logger.critical(message)
        raise RuntimeError(message)

    message = f"Data directory `{path_data}` may not exists"
    logger.critical(message)
    raise RuntimeError(message)


def read_source_type_from_manifest(path_data: Path) -> Literal["tsv", "hdf5", "root", "npz"] | None:
    manifest_name = path_data / "Manifest.yaml"
    if not manifest_name.is_file():
        logger.warning(f"Manifest.yaml is not found. Try to deduce the source type...")
        return

    manifest = LoadYaml(manifest_name)
    try:
        source_type = manifest["version"]
    except KeyError:
        message = f"Can not obtain 'version' from Manifest.yaml"
        logger.critical(message)
        raise RuntimeError(message)

    try:
        source_type = manifest["metadata"]["format"]
    except (KeyError, TypeError):
        message = f"Can not obtain ['metadata']['format'] from the Manifest.yaml"
        logger.critical(message)
        raise RuntimeError(message)

    if source_type not in {"tsv", "hdf5", "root", "npz"}:
        message = f"Source type {source_type}, reported by Manifest.yaml is not supported"
        logger.critical(message)
        raise RuntimeError(message)

    return source_type
