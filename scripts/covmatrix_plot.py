#!/usr/bin/env python

from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING

from h5py import File
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy import arange, diagonal, isnan
from numpy.typing import NDArray

from dagflow.logger import logger
from dagflow.plot import add_colorbar

if TYPE_CHECKING:
    from typing import Literal, Mapping, Sequence


def main(opts: Namespace) -> None:
    cmap = "RdBu_r"

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
    edges = group["edges"][:]
    blocksize = edges.size - 1
    # widths = edges[1:] - edges[:-1]

    if opts.output:
        pdf = PdfPages(opts.output)
        pdf.__enter__()
    else:
        pdf = None

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
    if pdf:
        pdf.savefig()

    for name, matrix_cov in group["covmat_syst"].items():
        matrix_cov = matrix_cov[:]
        (
            array_sigma,
            array_sigma_rel,
            matrix_cov_rel,
            matrix_cor,
            bmatrix_cov,
            barray_sigma,
            bmatrix_cor,
        ) = covariance_get_matrices(matrix_cov, model, blocksize)

        plt.figure(figsize=figsize_1d)
        ax = plt.subplot(
            111, xlabel="", ylabel=r"$\sigma$", title=f"Uncertainty {name} (diagonal)"
        )
        ax.grid(axis="y")
        stairs_with_blocks(array_sigma, blocks=elements)
        if pdf:
            pdf.savefig()

        plt.figure(figsize=figsize_1d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel=r"$\sigma$, %",
            title=f"Relative uncertainty {name} (diagonal)",
        )
        ax.grid(axis="y")
        stairs_with_blocks(array_sigma_rel, blocks=elements)
        if pdf:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111, xlabel="", ylabel="bin", title=f"Covariance matrix {name}"
        )
        pcolor_with_blocks(matrix_cov, blocks=elements, cmap=cmap)
        if pdf:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111, xlabel="", ylabel="bin", title=f"Covariance matrix {name} (blocks)"
        )
        pcolor_with_blocks(bmatrix_cov, blocks=elements, cmap=cmap)
        if pdf:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="bin",
            title=f"Covariance matrix {name} ({elements[0]})",
        )
        pcolor_with_blocks(
            matrix_cov[:blocksize, :blocksize], blocks=elements[:1], cmap=cmap
        )
        if pdf:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="bin",
            title=rf"Relative covariance matrix {name}, %²",
        )
        pcolor_with_blocks(matrix_cov_rel, blocks=elements, cmap=cmap)
        if pdf:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="bin",
            title=f"Relative covariance matrix {name} ({elements[0]})",
        )
        pcolor_with_blocks(
            matrix_cov_rel[:blocksize, :blocksize], blocks=elements[:1], cmap=cmap
        )
        if pdf:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111, xlabel="", ylabel="bin", title=f"Correlation matrix {name}"
        )
        pcolor_with_blocks(matrix_cor, blocks=elements, cmap=cmap)
        if pdf:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="bin",
            title=f"Correlation matrix {name} ({elements[0]})",
        )
        hm = pcolor_with_blocks(
            matrix_cor[:blocksize, :blocksize],
            blocks=elements[:1],
            pcolormesh=True,
            cmap=cmap,
        )
        # heatmap_show_values(hm, lower_triangle=True)
        if pdf:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111, xlabel="", ylabel="bin", title=f"Correlation matrix {name} (blocks)"
        )
        hm = pcolor_with_blocks(
            bmatrix_cor, blocks=elements, pcolormesh=True, cmap=cmap
        )
        heatmap_show_values(hm, lower_triangle=True)

        logger.info(f"Plot {name}")

        if pdf:
            pdf.savefig()

        if opts.show:
            plt.show()

        if not opts.show:
            plt.close("all")

    if pdf:
        pdf.__exit__(None, None, None)
        logger.info(f"Write output file: {opts.output}")

    if opts.show:
        plt.show()


def covariance_get_matrices(
    matrix_cov: NDArray, model: NDArray, blocksize: int
) -> tuple[
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
]:
    array_sigma = diagonal(matrix_cov) ** 0.5
    array_sigma_rel = 100 * (array_sigma / model)
    matrix_cor = matrix_cov / array_sigma[None, :] / array_sigma[:, None]
    matrix_cor[isnan(matrix_cor)] = 0.0
    matrix_cov_rel = 100 * 100 * (matrix_cov / model[None, :] / model[:, None])

    bmatrix_cov = matrix_sum_blocks(matrix_cov, blocksize=blocksize)
    barray_sigma = diagonal(bmatrix_cov) ** 0.5
    bmatrix_cor = bmatrix_cov / barray_sigma[None, :] / barray_sigma[:, None]
    bmatrix_cor[isnan(bmatrix_cor)] = 0.0

    return (
        array_sigma,
        array_sigma_rel,
        matrix_cov_rel,
        matrix_cor,
        bmatrix_cov,
        barray_sigma,
        bmatrix_cor,
    )


def heatmap_show_values(
    pc: "QuadMesh", fmt: str = "%.2f", lower_triangle: bool = False, **kwargs
):
    from numpy import mean, unravel_index

    pc.update_scalarmappable()
    data = pc.get_array()
    ax = plt.gca()
    for i, (path, color, value) in enumerate(
        zip(pc.get_paths(), pc.get_facecolors(), data.flatten())
    ):
        x, y = path.vertices[:-1].mean(0)
        row, col = unravel_index(i, data.shape)

        if lower_triangle and col > row:
            continue

        x -= 0.1 * (path.vertices[2, 0] - path.vertices[1, 0])
        if mean(color[:3]) > 0.5:
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kwargs)


from numba import njit
from numpy import empty


@njit
def matrix_sum_blocks(matrix: NDArray, blocksize: int) -> NDArray:
    nr, nc = matrix.shape
    assert nr == nc
    assert (nr % blocksize) == 0

    nblocks = nr // blocksize
    ret = empty((nblocks, nblocks), dtype=matrix.dtype)
    for row in range(nblocks):
        for col in range(nblocks):
            row1 = row * blocksize
            row2 = row1 + blocksize
            col1 = col * blocksize
            col2 = col1 + blocksize

            ret[row, col] = matrix[row1:row2, col1:col2].sum()

    return ret


def stairs_with_blocks(
    a0: NDArray, /, *args, blocks: Sequence[str], sep_kwargs: Mapping = {}, **kwargs
):
    ax = plt.gca()
    ax.stairs(a0, *args, **kwargs)

    xs = _get_blocks_data(a0.shape[0], blocks)
    _plot_separators("x", xs, blocks, **sep_kwargs)


def pcolor_with_blocks(
    data: NDArray,
    /,
    *args,
    blocks: Sequence[str],
    colorbar: bool = True,
    pcolormesh: bool = False,
    sep_kwargs: Mapping = {},
    **kwargs,
):
    from numpy import fabs
    from numpy.ma import array

    dmin = data.min()
    dmax = data.max()

    bound = max(fabs(dmin), dmax)
    vmin, vmax = -bound, bound

    # fdata = fabs(data)
    # data = array(data, mask=(fdata < 1.0e-9))
    ax = plt.gca()
    ax.set_aspect("equal")
    if pcolormesh:
        hm = ax.pcolormesh(data, *args, vmin=vmin, vmax=vmax, **kwargs)
    else:
        hm = ax.pcolorfast(data, *args, vmin=vmin, vmax=vmax, **kwargs)
    hm.set_rasterized(True)
    if colorbar:
        add_colorbar(hm, rasterized=True)
    ax.set_ylim(*reversed(ax.get_ylim()))

    nblocks = len(blocks)
    if nblocks < 2:
        return hm

    sep_kwargs = dict({"color": "green"}, **sep_kwargs)
    positions = _get_blocks_data(data.shape[0], blocks)
    _plot_separators("x", positions, blocks, **sep_kwargs)
    _plot_separators("y", positions, blocks, **sep_kwargs)

    return hm


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
    textopts = {"fontsize": "small"}
    ax = plt.gca()
    if axis == "x":
        linefcn = ax.axvline

        def textfcn(pos: float, text: str):
            ax.text(
                pos,
                ypos,
                text,
                transform=ax.get_xaxis_transform(),
                ha="center",
                **textopts,
            )

    elif axis == "y":
        linefcn = ax.axhline

        def textfcn(pos: float, text: str):
            ax.text(
                xpos,
                pos,
                text,
                transform=ax.get_yaxis_transform(),
                va="center",
                **textopts,
            )

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
    parser.add_argument("-o", "--output", help="output pdf file")
    parser.add_argument("-s", "--show", action="store_true")

    main(parser.parse_args())
