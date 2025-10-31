from pathlib import Path
from numpy import ndarray, ascontiguousarray
from collections.abc import Sequence
from numpy.typing import NDArray


def validate_load_array(array: str | Path | Sequence[int | float] | NDArray | None) -> Path | NDArray | None:
    """Validate and load array.

    Parameters
    ----------
    array : str | Path | Sequence[int | float] | NDArray | None
        Path or array like object that will be validated. In case of array like
        object it will be converted to NDArray.

    Returns
    -------
    Path | NDArray | None
        NDArray object or validated path with description of array like object.
    """
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
