from pathlib import Path
from typing import Literal


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
    extensions = {path.suffix[1:] for path in filter(
        lambda path: path.is_file(), path_data.rglob("[!p]*.*")
    )}
    extensions -= {"py", "yaml"}
    if len(extensions) == 1:
        source_type = extensions.pop()
        return source_type if source_type != "bz2" else "tsv"
    elif len(extensions) > 1:
        raise RuntimeError(f"Find to many possibly loaded extensions: {', '.join(extensions)}")
    raise RuntimeError(f"Data directory `{path_data}` may not exists")
