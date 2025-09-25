import os
from pathlib import Path
from typing import Literal


def auto_detect_source_type(path_data: Path) -> Literal["tsv", "hdf5", "root", "npz"]:
    iav_relative_path = "detector_iav_matrix*"
    for filepath in path_data.rglob(iav_relative_path):
        *filename, ext = os.path.splitext(filepath)
        if ext == ".bz2":
            ext = ".tsv"
        return ext[1:]
    raise FileNotFoundError(f"File {iav_relative_path} was not found in {data_path}/")
