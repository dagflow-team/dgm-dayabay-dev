#!/usr/bin/env python
from argparse import Namespace

import numpy as np
from dag_modelling.tools.logger import INFO1, set_level
from matplotlib import pyplot as plt

from dgm_dayabay_dev.models.dayabay_v0 import model_dayabay_v0

set_level(INFO1)


def plot_hist_mean(fits, key, title, xlabel, args) -> None:
    fig, ax = plt.subplots(1, 1)
    print(title)
    for i, (fit_key, fit_group) in enumerate(fits.items()):
        data = np.array([fit[key] for fit in fit_group]).reshape(-1)
        ax.hist(data, label=fit_key + f" {data.mean():1.5e}", alpha=0.5, color=f"C{i}")
        if key == "x":
            ax.plot(
                [data.mean(), data.mean()], [0, 500], linestyle="--", color=f"C{i+3}", label=fit_key
            )
            print(fit_key + f" {data.mean():1.5e}, {1 - data.mean():+1.5e}")
        else:
            print(fit_key + f" {data.mean():1.5e}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Entries")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"gn/{args.concatenation_mode}-{args.data_mc_mode}-{key}.png")
    plt.savefig(f"gn/{args.concatenation_mode}-{args.data_mc_mode}-{key}.pdf")


def main(args: Namespace) -> None:

    model = model_dayabay_v0(
        source_type=args.source_type,
        spectrum_correction_mode=args.spec,
        monte_carlo_mode=args.data_mc_mode,
        concatenation_mode=args.concatenation_mode,
        seed=args.seed,
    )

    parameters = model.storage("parameters.all")
    statistic = model.storage("outputs.statistic")

    model._touch()

    from dgm_fit.iminuit_minimizer import IMinuitMinimizer

    minimization_pars = [parameters[par_name] for par_name in args.min_par]
    minimizers = dict(
        [
            (key, IMinuitMinimizer(statistic[key], parameters=minimization_pars))
            for key in args.statistic
        ]
    )
    if not minimizers:
        exit(0)

    par_value_inits = []
    for par_name, par_value in args.par:
        par_value_inits.append(parameters[par_name].value)
        model.set_parameters({par_name: par_value})
    for i, (par_name, _) in enumerate(args.par):
        model.set_parameters({par_name: par_value_inits[i]})

    observations = []
    fits = dict(zip(args.statistic, [[] for _ in range(len(args.statistic))]))
    for _ in range(args.repeat):
        model.next_sample()
        model._touch()
        observations.append(
            model.storage.get_value("outputs.eventscount.final.concatenated").data.copy()
        )
        for key, minimizer in minimizers.items():
            minimizer.push_initial_values()
            fit = minimizer.fit()
            minimizer.push_initial_values()
            fits[key].append(fit)

    from matplotlib import pyplot as plt

    plot_hist_mean(
        fits,
        "x",
        f"{args.concatenation}, {args.data_mc_mode}" + ", global normalization, value",
        r"$N^{\mathrm{global}}$",
        args,
    )
    plot_hist_mean(
        fits,
        "errors",
        f"{args.concatenation}, {args.data_mc_mode}" + ", global normalization, error",
        r"$\sigma_{N}$",
        args,
    )
    plot_hist_mean(
        fits,
        "fun",
        f"{args.concatenation}, {args.data_mc_mode}" + r", $\chi^2$ distribution",
        r"$\chi^2$",
        args,
    )

    plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", default=0, action="count", help="verbosity level")

    input = parser.add_argument_group("input", "input related options")
    input.add_argument(
        "--input",
        nargs=2,
        action="append",
        metavar=("STAT_TYPE", "FILENAME"),
        default=[],
        help="input file with fit to compare to",
    )

    model = parser.add_argument_group("model", "model related options")
    model.add_argument(
        "-s",
        "--source-type",
        "--source",
        choices=("tsv", "hdf5", "root", "npz"),
        default="npz",
        help="Data source type",
    )
    model.add_argument(
        "--spec",
        choices=("linear", "exponential"),
        default="exponential",
        help="antineutrino spectrum correction mode",
    )
    model.add_argument("--seed", default=0, type=int, help="seed of randomization")
    model.add_argument(
        "--concatenation-mode",
        default="detector",
        choices=["detector", "detector-period"],
        help="Choose type of concatenation",
    )
    model.add_argument(
        "--data-mc-mode",
        default="asimov",
        choices=["asimov", "normal-stats", "poisson"],
        help="type of data to be analyzed",
    )

    pars = parser.add_argument_group("pars", "setup pars")
    pars.add_argument(
        "--par",
        nargs=2,
        action="append",
        default=[],
        help="set parameter value",
    )
    pars.add_argument(
        "--min-par",
        nargs="*",
        default=[],
        help="choose minimization parameters",
    )

    stats = parser.add_argument_group("statistic", "statistic parameters")
    stats.add_argument(
        "--statistic",
        nargs="*",
        default=[],
        help="choose statistic of minimization",
    )
    stats.add_argument(
        "--repeat",
        default=100,
        type=int,
        help="Number of MC samples to be fitted",
    )

    outputs = parser.add_argument_group("outputs", "set outputs")
    outputs.add_argument(
        "--fit-output",
        help="path to save fit, without extension",
    )

    args = parser.parse_args()

    main(args)
