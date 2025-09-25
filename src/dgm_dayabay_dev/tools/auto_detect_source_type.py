from pathlib import Path
from typing import Literal


def auto_detect_source_type(path_data: Path) -> Literal["tsv", "hdf5", "root", "npz"]:
    extensions = set()
    for filepath in filter(lambda path: path.is_file(), path_data.rglob("*.*")):
        extensions.add(filepath.suffix[1:])
    extensions -= {"py", "yaml"}
    if len(extensions) == 1:
        source_type = extensions.pop()
        return source_type if source_type != "bz2" else "tsv"
    raise RuntimeError(f"Find to many possibly loaded extensions: {', '.join(extensions)}")
