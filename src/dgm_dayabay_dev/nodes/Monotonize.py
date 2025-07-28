from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit

from dagflow.core.exception import InitializationError
from dagflow.core.node import Node
from dagflow.core.type_functions import (
    check_dimension_of_inputs,
    check_inputs_have_same_shape,
    copy_from_inputs_to_outputs,
)

if TYPE_CHECKING:
    from numpy import double
    from numpy.typing import NDArray

    from dagflow.core.input import Input
    from dagflow.core.output import Output


@njit(cache=True)
def _monotonize_with_x(
    x: NDArray[double],
    y: NDArray[double],
    result: NDArray[double],
    gradient: float,
    index: int,
) -> None:
    # forward loop
    i = index
    result[i] = y[i]
    direction = 1 if y[i + 1] > y[i] else -1
    while i < len(y) - 1:
        direction_current = 1 if y[i + 1] > result[i] else -1
        if direction == direction_current:
            result[i + 1] = y[i + 1]
        else:
            result[i + 1] = result[i] + direction * gradient * (x[i + 1] - x[i])  # fmt:skip
        i += 1

    # backward loop
    if index == 0:
        return
    i = index + 1
    while i > 0:
        direction_current = 1 if result[i] > y[i - 1] else -1
        if direction == direction_current:
            result[i - 1] = y[i - 1]
        else:
            result[i - 1] = result[i] - direction * gradient * (x[i] - x[i - 1])  # fmt:skip
        i -= 1


@njit(cache=True)
def _monotonize_without_x(
    y: NDArray[double],
    result: NDArray[double],
    gradient: float,
    index: int,
) -> None:
    # forward loop
    i = index
    result[i] = y[i]
    direction = 1 if y[i + 1] > y[i] else -1
    while i < len(y) - 1:
        direction_current = 1 if y[i + 1] > result[i] else -1
        if direction == direction_current:
            result[i + 1] = y[i + 1]
        else:
            result[i + 1] = result[i] + direction * gradient
        i += 1

    # backward loop
    if index == 0:
        return
    i = index + 1
    while i > 0:
        direction_current = 1 if result[i] > y[i - 1] else -1
        if direction == direction_current:
            result[i - 1] = y[i - 1]
        else:
            result[i - 1] = result[i] - direction * gradient
        i -= 1


class Monotonize(Node):
    r"""
    Monotonizes a function.

    inputs:
        `y`: f(x) array
        `x` (**optional**): arguments array

    outputs:
        `0` or `result`: the resulting array

    constructor arguments:
        `index_fraction`: fraction of array to monotonize (must be >=0 and <1)
        `gradient`: set gradient to monotonize (takes absolute value)
    """

    __slots__ = ("_y", "_x", "_result", "_index_fraction", "_gradient", "_index")

    _y: Input
    _x: Input
    _result: Output
    _index_fraction: float
    _gradient: float
    _index: int

    def __init__(
        self,
        name,
        *args,
        with_x: bool = False,
        index_fraction: float = 0,
        gradient: float = 0,
        **kwargs,
    ) -> None:
        super().__init__(name, *args, **kwargs, allowed_kw_inputs=("y", "x"))
        if gradient > 0.0:
            self._labels.setdefault("mark", "↗")
        elif gradient < 0.0:
            self._labels.setdefault("mark", "↘")
        else:
            self._labels.setdefault("mark", "→")

        if index_fraction < 0 or index_fraction >= 1:
            raise InitializationError(
                f"`index_fraction` must be 0 <= x < 1, but given {index_fraction}",
                node=self,
            )
        self._index_fraction = index_fraction
        self._gradient = abs(gradient)
        if with_x:
            self._x = self._add_input("x", positional=False)  # input: "x"
        self._y = self._add_input("y", positional=True)  # input: "y"
        self._result = self._add_output("result")  # output: 0
        self._functions_dict.update({"with_x": self._function_with_x, "without_x": self._function_without_x})

    @property
    def gradient(self) -> float:
        return self._gradient

    @property
    def index_fraction(self) -> float:
        return self._index_fraction

    @property
    def index(self) -> int:
        return self._index

    def _function_with_x(self) -> None:
        _monotonize_with_x(self._x.data, self._y.data, self._result._data, self.gradient, self.index)

    def _function_without_x(self) -> None:
        _monotonize_without_x(self._y.data, self._result._data, self.gradient, self.index)

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        self._x = self.inputs.get("x")
        isGivenX = self._x is not None
        inputsToCheck = ("x", "y") if isGivenX else "y"

        check_dimension_of_inputs(self, inputsToCheck, 1)
        check_inputs_have_same_shape(self, inputsToCheck)
        copy_from_inputs_to_outputs(self, "y", "result")

        self._index = int((self.inputs["y"].dd.shape[0] - 1) * self.index_fraction)
        self.function = self._functions_dict["with_x" if isGivenX else "without_x"]
