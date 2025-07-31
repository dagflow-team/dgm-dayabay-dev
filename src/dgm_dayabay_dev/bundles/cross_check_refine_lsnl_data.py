from numpy import arange
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from nested_mapping.nested_mapping import NestedMapping


def cross_check_refine_lsnl_data(
    storage: NestedMapping, *, xname: str, nominalname: str, **kwargs
) -> None:
    xcoarse = storage[xname]

    refiner = RefineGraph(xcoarse, **kwargs)

    for key, ycoarse in storage.walkitems():
        if ycoarse is xcoarse:
            continue

        nominal = storage[nominalname]
        storage[key] = refiner.process(ycoarse, nominal)

    storage[xname] = refiner.xfine


class RefineGraph:
    __slots__ = (
        "xcoarse",
        "xfine",
        "newmin",
        "newmax",
    )
    xcoarse: NDArray
    xfine: NDArray
    newmin: float
    newmax: float

    def __init__(
        self,
        xcoarse: NDArray,
        *,
        newmin: float,
        newmax: float,
    ):
        self.xcoarse = xcoarse
        self.newmin = newmin
        self.newmax = newmax

        self._process_x()

    def make_x(self) -> None:
        self.xfine = arange(0.0, 12.0000001, 0.05)

    def _process_x(self) -> None:
        self.make_x()

    def process(self, y: NDArray, nominal: NDArray) -> NDArray:
        skip_diff = y is nominal
        yfine = self._method_interpolate(y)
        yabs = self._method_reltoabs(yfine)

        return yabs if skip_diff else self._method_diff(nominal, yabs)

    def _method_reltoabs(self, yrel: NDArray) -> NDArray:
        return yrel * self.xfine

    def _method_interpolate(self, ycoarse) -> NDArray:
        fcn = interp1d(
            self.xcoarse,
            ycoarse,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        return fcn(self.xfine)

    def _method_diff(self, nominal: NDArray, y: NDArray) -> NDArray:
        return nominal if nominal is y else y - nominal
