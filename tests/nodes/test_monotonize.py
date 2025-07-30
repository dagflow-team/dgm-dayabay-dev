#!/usr/bin/env python

"""Check the Monotonize transformation."""
from typing import Literal

from matplotlib import pyplot as plt
from numpy import allclose, fabs, finfo, linspace, log
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.plot.graphviz import savegraph
from dgm_dayabay_dev.nodes.Monotonize import Monotonize


@mark.parametrize("direction", [+1, -1])
@mark.parametrize("gradient", [0, 0.5])
@mark.parametrize("start", [0, 0.5])
@mark.parametrize("positive", [True, False])
def test_monotonize(
    direction: Literal[-1, +1],
    gradient: float | int,
    start: float | int,
    positive: bool,
    debug_graph,
    test_name: str,
    output_path: str
):
    x = linspace(0.0, 10, 101)[1:]
    y = log(x)
    if not positive:
        y -= y.max()
    mask = x < 1.0
    y[mask] = (x[mask] - 1.0) ** 2

    if start == 0:
        mask = x > 1.0
    elif start > 0.4:
        mask = x < 1.0
    else:
        assert False

    if direction < 0:
        y = -y
    frac = (len(x) - len(x[mask]) - 1) / len(x)

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        X = Array("x", x, mode="fill")
        Y = Array("y", y, mode="fill")
        m = Monotonize(name="monotonize", index_fraction=frac, gradient=gradient)
        X >> m("x")
        Y >> m("y")
        # Using monotonize node again does not change data
        m2 = Monotonize(name="monotonize", index_fraction=frac, gradient=gradient)
        X >> m2("x")
        m.outputs["result"] >> m2("y")

    ym = m.outputs["result"].data
    xmod = x[mask]
    ymod = ym[mask]
    ykept = ym[~mask]
    ymod_grad = (ymod[1:] - ymod[:-1]) / (xmod[1:] - xmod[:-1])

    assert (ykept == y[~mask]).all(), "Modified the region, that should not be modified"
    assert allclose(
        fabs(ymod_grad), gradient, atol=finfo("f").resolution, rtol=0
    ), "Gradient of the modified region is not correct"
    if gradient != 0:
        diff = ym[1:] - ym[:-1]
        assert ((diff > 0) == (diff[0] > 0)).all(), "The result is not monotonous"
    # Using monotonize node again does not change data
    ym2 = m2.outputs["result"].data
    assert (ym == ym2).all()

    fig = plt.figure()
    ax = plt.subplot(
        111, xlabel="x", ylabel="y", title=f"Grad {gradient}, start {start}"
    )
    ax.grid()
    ax.plot(x, y, "+", label="input")
    ax.plot(x, ym, "x", label="monotonize 1")
    ax.plot(x, ym2, ".", label="monotonize 2")
    ax.legend()
    fig.savefig(f"{output_path}/{test_name}-plot.png")
    plt.close()

    fname = f"{output_path}/{test_name}.png"
    print(f"Save graph: {fname}")
    savegraph(graph, fname)
