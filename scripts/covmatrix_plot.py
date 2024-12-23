#!/usr/bin/env python

from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING

from h5py import File, Group
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import SymLogNorm
from numpy import arange, diagonal, isnan, quantile
from numpy.typing import NDArray

from dagflow.plot.plot import add_colorbar
from dagflow.tools.logger import logger

if TYPE_CHECKING:
    from typing import Literal, Mapping, Sequence

    from matplotlib.collections import QuadMesh


def main(opts: Namespace) -> None:
    cmap = "RdBu_r"

    ifile = File(opts.input, "r")
    group = ifile[opts.mode]

    model = group["model"][:]
    edges = group["edges"][:]

    sfile, sgroup, smodel = None, None, None
    if opts.subtract:
        sfile = File(opts.subtract, "r")
        sgroup = sfile[opts.mode]
        smodel = sgroup["model"][:]
        plotmodel = model - smodel
    else:
        plotmodel = model

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

    blocksize = edges.size - 1
    block_selected = 1
    block_selection = slice(
        blocksize * block_selected, blocksize * (block_selected + 1)
    )
    block_name = elements[block_selected]

    if opts.output is not None:
        ofile = opts.output and opts.output[0] or opts.input.replace(".hdf5", ".pdf")
        assert ofile != opts.input and ofile.endswith(".pdf")
        pdf = PdfPages(ofile)
        pdf.__enter__()
    else:
        ofile = None
        pdf = None

    title_suffix = f" {opts.title_suffix}" if opts.title_suffix else ""
    if opts.subtract:
        title_suffix += " [diff]"
    if opts.mode == "detector":
        figsize_1d = (12, 6)
        figsize_2d = (6, 6)
    else:
        figsize_1d = (18, 6)
        figsize_2d = (12, 12)
    plt.figure(figsize=figsize_1d)
    ax = plt.subplot(111, xlabel="", ylabel="entries", title=f"Model{title_suffix}")
    ax.grid(axis="y")
    stairs_with_blocks(plotmodel, blocks=elements)
    if pdf:
        pdf.savefig()

    items_to_process = dict(group["covmat_syst"].items())
    for name, obj in tuple(items_to_process.items()):
        if not isinstance(obj, Group):
            continue

        del items_to_process[name]
        for subname, subobj in obj.items():
            items_to_process[f"{name}.{subname}"] = subobj

    for name, matrix_cov in items_to_process.items():
        matrix_cov = matrix_cov[:]
        (
            array_sigma,
            array_sigma_rel,
            matrix_cov_rel,
            matrix_cor,
            bmatrix_cov,
            _,  # barray_sigma,
            bmatrix_cor,
        ) = covariance_get_matrices(matrix_cov, model, blocksize)

        if sgroup is not None:
            matrix_cov_subtract = sgroup["covmat_syst"][name][:]
            (
                array_sigma_subtract,
                array_sigma_rel_subtract,
                matrix_cov_rel_subtract,
                matrix_cor_subtract,
                bmatrix_cov_subtract,
                _,  # barray_sigma_subtract,
                bmatrix_cor_subtract,
            ) = covariance_get_matrices(matrix_cov_subtract, smodel, blocksize)
            matrix_cov = matrix_cov - matrix_cov_subtract
            array_sigma -= array_sigma_subtract
            array_sigma_rel -= array_sigma_rel_subtract
            matrix_cov_rel -= matrix_cov_rel_subtract
            matrix_cor -= matrix_cor_subtract
            bmatrix_cov -= bmatrix_cov_subtract
            bmatrix_cor -= bmatrix_cor_subtract

        as_min = max(array_sigma.min(), 0)
        as_range = array_sigma.max() - as_min
        as_coef = as_range / array_sigma.mean()
        vmax, vmax_rel = None, None
        if sgroup is None and as_coef > 10:
            vmax = float(quantile(array_sigma, 0.95))
            vmax_rel = float(quantile(array_sigma_rel, 0.95))

        plt.figure(figsize=figsize_1d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel=r"$\sigma$",
            title=f"Uncertainty {name} (diagonal){title_suffix}",
        )
        ax.grid(axis="y")
        stairs_with_blocks(array_sigma, blocks=elements)
        if vmax is not None:
            ax.axhline(vmax, linestyle="--", color="black", alpha=0.5)
        if pdf:
            pdf.savefig()

        plt.figure(figsize=figsize_1d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel=r"$\sigma$, %",
            title=f"Relative uncertainty {name} (diagonal){title_suffix}",
        )
        ax.grid(axis="y")
        stairs_with_blocks(array_sigma_rel, blocks=elements)
        if vmax_rel is not None:
            ax.axhline(vmax_rel, linestyle="--", color="black", alpha=0.5)
        if pdf:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="bin",
            title=f"Covariance matrix {name}{title_suffix}",
        )
        hm = pcolor_with_blocks(matrix_cov, blocks=elements, cmap=cmap)
        if pdf and hm is not None:
            pdf.savefig()

        if vmax is not None:
            plt.figure(figsize=figsize_2d)
            ax = plt.subplot(
                111,
                xlabel="",
                ylabel="bin",
                title=f"Covariance matrix {name} (truncated){title_suffix}",
            )
            hm = pcolor_with_blocks(matrix_cov, blocks=elements, cmap=cmap, vmax=vmax)
            if pdf and hm is not None:
                pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="bin",
            title=f"Covariance matrix {name} (symlog){title_suffix}",
        )
        hm = pcolor_with_blocks(matrix_cov, blocks=elements, cmap=cmap, symlog=True)
        if pdf and hm is not None:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="bin",
            title=f"Covariance matrix {name} (blocks){title_suffix}",
        )
        hm = pcolor_with_blocks(bmatrix_cov, blocks=elements, cmap=cmap)
        if pdf and hm is not None:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="bin",
            title=f"Covariance matrix {name} (blocks, symlog){title_suffix}",
        )
        hm = pcolor_with_blocks(bmatrix_cov, blocks=elements, cmap=cmap, symlog=True)
        if pdf and hm is not None:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="bin",
            title=f"Covariance matrix {name} (block {block_name}){title_suffix}",
        )
        hm = pcolor_with_blocks(
            matrix_cov[block_selection, block_selection], blocks=elements[:1], cmap=cmap
        )
        if pdf and hm is not None:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="bin",
            title=f"Covariance matrix {name} (block {block_name}, symlog){title_suffix}",
        )
        hm = pcolor_with_blocks(
            matrix_cov[block_selection, block_selection],
            blocks=elements[:1],
            cmap=cmap,
            symlog=True,
        )
        if pdf and hm is not None:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="bin",
            title=rf"Relative covariance matrix {name}, %²{title_suffix}",
        )
        hm = pcolor_with_blocks(matrix_cov_rel, blocks=elements, cmap=cmap)
        if pdf and hm is not None:
            pdf.savefig()

        if vmax_rel is not None:
            plt.figure(figsize=figsize_2d)
            ax = plt.subplot(
                111,
                xlabel="",
                ylabel="bin",
                title=rf"Relative covariance matrix {name} (truncated), %²{title_suffix}",
            )
            hm = pcolor_with_blocks(
                matrix_cov_rel, blocks=elements, cmap=cmap, vmax=vmax_rel
            )
            if pdf and hm is not None:
                pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="bin",
            title=rf"Relative covariance matrix {name} (symlog), %²{title_suffix}",
        )
        hm = pcolor_with_blocks(matrix_cov_rel, blocks=elements, cmap=cmap, symlog=True)
        if pdf and hm is not None:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="bin",
            title=f"Relative covariance matrix {name} (block {block_name}){title_suffix}",
        )
        hm = pcolor_with_blocks(
            matrix_cov_rel[block_selection, block_selection],
            blocks=elements[:1],
            cmap=cmap,
        )
        if pdf and hm is not None:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="bin",
            title=f"Relative covariance matrix {name} (block {block_name}, symlog){title_suffix}",
        )
        hm = pcolor_with_blocks(
            matrix_cov_rel[block_selection, block_selection],
            blocks=elements[:1],
            cmap=cmap,
            symlog=True,
        )
        if pdf and hm is not None:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="bin",
            title=f"Correlation matrix {name}{title_suffix}",
        )
        hm = pcolor_with_blocks(matrix_cor, blocks=elements, cmap=cmap)
        if pdf and hm is not None:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="bin",
            title=f"Correlation matrix {name} (block {block_name}){title_suffix}",
        )
        hm = pcolor_with_blocks(
            matrix_cor[block_selection, block_selection],
            blocks=elements[:1],
            pcolormesh=True,
            cmap=cmap,
        )
        # heatmap_show_values(hm, lower_triangle=True)
        if pdf and hm is not None:
            pdf.savefig()

        plt.figure(figsize=figsize_2d)
        ax = plt.subplot(
            111,
            xlabel="",
            ylabel="bin",
            title=f"Correlation matrix {name} (blocks){title_suffix}",
        )
        hm = pcolor_with_blocks(
            bmatrix_cor, blocks=elements, pcolormesh=True, cmap=cmap
        )
        heatmap_show_values(hm, lower_triangle=True)

        logger.info(f"Plot {name}")

        if pdf and hm is not None:
            pdf.savefig()

        if opts.show:
            plt.show()

        if not opts.show:
            plt.close("all")

    if pdf and hm is not None:
        pdf.__exit__(None, None, None)
        logger.info(f"Write output file: {ofile}")

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
    pc: QuadMesh, fmt: str = "%.2f", lower_triangle: bool = False, **kwargs
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
        row1 = row * blocksize
        row2 = row1 + blocksize
        for col in range(nblocks):
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
    vmax: float | None = None,
    symlog: bool = False,
    **kwargs,
):
    from numpy import fabs

    dmin = data.min()
    dmax = data.max()

    if dmin == dmax == 0:
        return None

    bound = max(fabs(dmin), dmax)
    if vmax is not None:
        vmin, vmax = -vmax, vmax
    else:
        vmin, vmax = -bound, bound

    if symlog:
        pcolor_kwargs = dict(
            norm=SymLogNorm(
                linthresh=bound * 0.01, linscale=0.05, vmin=vmin, vmax=vmax
            ),
            **kwargs,
        )
    else:
        pcolor_kwargs = dict(vmin=vmin, vmax=vmax, **kwargs)

    # fdata = fabs(data)
    # data = array(data, mask=(fdata < 1.0e-9))
    ax = plt.gca()
    ax.set_aspect("equal")
    if pcolormesh:
        hm = ax.pcolormesh(data, *args, **pcolor_kwargs)
    else:
        hm = ax.pcolorfast(data, *args, **pcolor_kwargs)
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
    return arange(0, n_blocks + 1) * bins_in_block


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
    parser.add_argument("--subtract", help="input h5py file to subtract")
    parser.add_argument(
        "-m",
        "--mode",
        default="detector_period",
        choices=("detector", "detector_period"),
        help="mode",
    )
    parser.add_argument("-o", "--output", nargs="*", help="output pdf file")
    parser.add_argument("-s", "--show", action="store_true")
    parser.add_argument("--title-suffix", help="figure title suffix")

    main(parser.parse_args())
