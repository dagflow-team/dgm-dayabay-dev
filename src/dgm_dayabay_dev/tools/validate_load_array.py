from pathlib import Path
from numpy import ndarray, ascontiguousarray
from collections.abc import Sequence
from numpy.typing import NDArray


def validate_load_array(array: str | Path | Sequence[int | float] | NDArray | None) -> Path | NDArray | None:
    result = None
    match array:
        case str() | Path():
            result = Path(array)
            if not result.is_file():
                raise FileNotFoundError(f"File {result} is not found")
        case Sequence() | ndarray():
            result = ascontiguousarray(array, dtype="d")
        case None:
            result = None
        case _:
            raise RuntimeError(
                f"Invalid array type: {type(array).__name__}"
            )
    return result
