#!/usr/bin/env python

from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING

from h5py import File
from matplotlib import pyplot as plt
from numpy import arange, diagonal, isnan
from numpy.typing import NDArray

from dagflow.plot import add_colorbar

if TYPE_CHECKING:
    from typing import Literal, Mapping, Sequence


def main(opts: Namespace) -> None:
    ifile = File(opts.input, "r")
    group = ifile[opts.mode]

    try:
        elements0: NDArray = group["elements"][:]
        if isinstance(elements0[0], (str, bytes)):
            elements = tuple(l.decode() for l in elements0)
        else:
            elements = tuple(
                f"{l[0].decode()[-2:]}:{l[1].decode()[0]}" for l in elements0
            )
    except KeyError:
        elements = None

    model = group["model"][:]
    # edges = group["edges"][:]
    # widths = edges[1:] - edges[:-1]

    if opts.mode == "detector":
        figsize_1d = (12, 6)
        figsize_2d = (6, 6)
    else:
        figsize_1d = (18, 6)
        figsize_2d = (12, 12)
    plt.figure(figsize=figsize_1d)
    ax = plt.subplot(111, xlabel="", ylabel="entries", title="Model")
    ax.grid(axis="y")
    stairs_with_blocks(model, blocks=elements)

    for name, matrix_cov in group["covmat_syst"].items():
        matrix_cov = matrix_cov[:]
        array_sigma = diagonal(matrix_cov) ** 0.5
        array_sigma_rel = 100 * (array_sigma / model)
        matrix_cor = matrix_cov / array_sigma[None, :] / array_sigma[:, None]
        matrix_cor[isnan(matrix_cor)] = 0.0
        matrix_cov_rel = 100 * 100 * (matrix_cov / model[None, :] / model[:, None])

        plt.figure(figsize=figsize_1d)
        ax = plt.subplot(111, xlabel="", ylabel=r"$\sigma$", title="Uncertainty")
        ax.grid(axis="y")
        stairs_with_blocks(array_sigma, blocks=elements)

        plt.figure(figsize=figsize_1d)
        ax = plt.subplot(
            111, xlabel="", ylabel=r"$\sigma$, %", title="Relative uncertainty"
        )
        ax.grid(axis="y")
        stairs_with_blocks(array_sigma_rel, blocks=elements)

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111, xlabel="", ylabel="bin", title=f"Covariance matrix {name}"
        )
        pcolorfast_with_blocks(matrix_cov, blocks=elements)

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="bin",
            title=rf"Relative covariance matrix {name}, %²",
        )
        pcolorfast_with_blocks(matrix_cov_rel, blocks=elements)

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111, xlabel="", ylabel="bin", title=f"Correlation matrix {name}"
        )
        pcolorfast_with_blocks(matrix_cor, blocks=elements)

        plt.show()

    if opts.show:
        plt.show()


def stairs_with_blocks(
    a0: NDArray, /, *args, blocks: Sequence[str], sep_kwargs: Mapping = {}, **kwargs
):
    ax = plt.gca()
    ax.stairs(a0, *args, **kwargs)

    xs = _get_blocks_data(a0.shape[0], blocks)
    _plot_separators("x", xs, blocks, **sep_kwargs)


def pcolorfast_with_blocks(
    data: NDArray,
    /,
    *args,
    blocks: Sequence[str],
    sep_kwargs: Mapping = {},
    colorbar: bool = True,
    **kwargs,
):
    from numpy import fabs
    from numpy.ma import array

    data = array(data, mask=(fabs(data) < 1.0e-9))
    ax = plt.gca()
    ax.set_aspect("equal")
    hm = ax.pcolorfast(data, *args, **kwargs)
    if colorbar:
        add_colorbar(hm)
    ax.set_ylim(*reversed(ax.get_ylim()))

    sep_kwargs = dict({"color": "white"}, **kwargs)
    positions = _get_blocks_data(data.shape[0], blocks)
    _plot_separators("x", positions, blocks, **sep_kwargs)
    _plot_separators("y", positions, blocks, **sep_kwargs)


def _get_blocks_data(size: int, blocks: Sequence[str]) -> NDArray:
    n_blocks = len(blocks)
    bins_in_block = size // n_blocks
    xs = arange(0, n_blocks + 1) * bins_in_block

    return xs


def _plot_separators(
    axis: Literal["x", "y"],
    positions: NDArray,
    blocks: Sequence[str],
    xpos: float = 1.12,
    ypos: float = -0.1,
    **kwargs,
):
    ax = plt.gca()
    if axis == "x":
        linefcn = ax.axvline

        def textfcn(pos: float, text: str):
            ax.text(pos, ypos, text, transform=ax.get_xaxis_transform(), ha="center")

    elif axis == "y":
        linefcn = ax.axhline

        def textfcn(pos: float, text: str):
            ax.text(xpos, pos, text, transform=ax.get_yaxis_transform(), va="center")

    else:
        raise ValueError(axis)

    sep_kwargs = dict(
        {"linestyle": "--", "color": "black", "linewidth": 1, "alpha": 0.5}, **kwargs
    )
    prev = 0
    for i, pos in enumerate(positions):

        linefcn(pos, 0.0, 1.0, **sep_kwargs)

        if i == 0:
            continue

        midpos = 0.5 * (pos + prev)
        text = blocks[i - 1]
        textfcn(midpos, text)

        prev = pos


plt.style.use(
    {
        "axes.titlepad": 20,
        "axes.formatter.limits": (-3, 3),
        "axes.formatter.use_mathtext": True,
        "axes.grid": False,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.top": True,
        "ytick.right": True,
        "errorbar.capsize": 2,
        "lines.markerfacecolor": "none",
        "savefig.dpi": 300,
    }
)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("input", help="input h5py file")
    parser.add_argument(
        "-m",
        "--mode",
        default="detector",
        choices=("detector", "detector_period"),
        help="mode",
    )
    parser.add_argument("-o", "--output", help="output folder file")
    parser.add_argument("-s", "--show", action="store_true")

    main(parser.parse_args())
