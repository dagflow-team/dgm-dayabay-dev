import os
from pathlib import Path


def auto_detect_source_type(data_path: Path):
    iav_relative_path = "detector_iav_matrix*"
    for filepath in data_path.rglob(iav_relative_path):
        *filename, ext = os.path.splitext(filepath)
        if ext == ".bz2":
            ext = ".tsv"
        return ext[1:]
    raise FileNotFoundError(f"File {iav_relative_path} was not found in {data_path}/")
    
