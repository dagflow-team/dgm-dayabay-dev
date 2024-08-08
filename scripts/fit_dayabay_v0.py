#!/usr/bin/env python
from yaml import safe_load

from dagflow.logger import INFO1
from dagflow.logger import INFO2
from dagflow.logger import INFO3
from dagflow.logger import DEBUG as INFO4
from dagflow.logger import set_level
from dagflow.storage import NodeStorage

from models.dayabay_v0 import model_dayabay_v0


set_level(INFO1)


dagflow_parameters_to_gna = {
    "detector.global_normalization": "dayabay.global_norm",
    "oscprob.DeltaMSq32": "dayabay.pmns.DeltaMSq23",
    "oscprob.SinSq2Theta13": "dayabay.pmns.SinSqDouble13",
}

gna_parameters_to_dagflow = {
    "dayabay.global_norm": "detector.global_normalization",
    "dayabay.pmns.DeltaMSq23": "oscprob.DeltaMSq32",
    "dayabay.pmns.SinSqDouble13": "oscprob.SinSq2Theta13",
}


def _compare_parameters(gna_pars: list, dagflow_pars: list) -> bool:
    dagflow_transformed = sorted([dagflow_parameters_to_gna[name] for name in dagflow_pars])
    gna_transformed = sorted(gna_pars)
    return gna_transformed == dagflow_transformed


def next_sample(storage: NodeStorage, parameter, val: tuple[float, float]) -> None:
    parameter.value = val[0]
    print(parameter)
    for node in storage("nodes.pseudo.data").walkvalues():
        node.next_sample()
    parameter.value = val[1]


def compare_gna(dagflow_fit: dict, gna_fit_filename: str) -> None:
    from matplotlib import pyplot as plt
    with open(gna_fit_filename, "r") as f:
        gna_fits = safe_load(f)["fitresult"]
    for gna_fit in gna_fits.values():
        if _compare_parameters(gna_fit["names"], dagflow_fit["names"]):
            print(f"ChiSquared:     {gna_fit['fun']:+1.3e}  {dagflow_fit['fun']:+1.3e}      {(gna_fit['fun'] - dagflow_fit['fun']) / gna_fit['fun']:+1.3e}")
            for gna_par_name in gna_fit["names"]:
                dagflow_par_name = gna_parameters_to_dagflow[gna_par_name]
                gna_par_value = gna_fit["xdict"][gna_par_name]
                gna_par_error = gna_fit["errorsdict"][gna_par_name]
                dagflow_par_value = dagflow_fit["xdict"][dagflow_par_name]
                dagflow_par_error = dagflow_fit["errorsdict"][dagflow_par_name]
                print(gna_par_name)
                print(f"Central:        {gna_par_value:+1.3e}  {dagflow_par_value:+1.3e}      {(gna_par_value - dagflow_par_value) / gna_par_value:+1.3e}")
                print(f"Error:          {gna_par_error:+1.3e}  {dagflow_par_error:+1.3e}      {(gna_par_error - dagflow_par_error) / gna_par_error:+1.3e}")
                fig, axs = plt.subplots(1, 1)
                axs.errorbar(x=[gna_par_value], y=[0], xerr=[gna_par_error], fmt="o", markerfacecolor="none", label="GNA")
                axs.errorbar(x=[dagflow_par_value], y=[1], xerr=[dagflow_par_error], fmt="X", markerfacecolor="none", label="dag-flow")
                axs.set_xlabel("value")
                axs.set_title(dagflow_par_name)
                axs.legend()
                axs.set_ylim(-1, 2)
                plt.tight_layout()
                plt.savefig(f"output/comparison/{dagflow_par_name}.png")


def main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", default=0, action="count", help="verbosity level"
    )

    input = parser.add_argument_group("input", "input related options")
    input.add_argument(
        "--input", nargs=2, action="append", metavar=("STAT_TYPE", "FILENAME"),
        default=[], help="input file to compare to",
    )

    model = parser.add_argument_group("model", "model related options")
    model.add_argument(
        "-s", "--source-type", "--source",
        choices=("tsv", "hdf5", "root", "npz"), default="npz",
        help="Data source type",
    )
    model.add_argument(
        "--spec", choices=("linear", "exponential"), default="exponential",
        help="antineutrino spectrum correction mode",
    )
    model.add_argument(
        "--fission-fraction-normalized", action="store_true",
        help="fission fraction correction",
    )

    pars = parser.add_argument_group("pars", "setup pars")
    pars.add_argument(
        "--par", nargs=2, action="append", default=[], help="set parameter value",
    )
    pars.add_argument(
        "--min-par", nargs="*", default=[], help="choose minimization parameters",
    )

    args = parser.parse_args()

    model = model_dayabay_v0(
        source_type=args.source_type,
        spectrum_correction_mode=args.spec,
        fission_fraction_normalized=args.fission_fraction_normalized,
    )

    graph = model.graph
    storage = model.storage
    parameters = model.storage("parameter")
    statistic = model.storage("outputs.statistic")

    from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer
    chi2p_stat = statistic["stat.chi2p"]
    chi2p_syst = statistic["full.chi2p"]
    mc_parameters = {}
    for (par_name, par_value) in args.par:
        mc_parameters[parameters[par_name]] = (par_value, parameters[par_name].value.copy())

    minimizer_stat = IMinuitMinimizer(
        statistic=chi2p_stat, parameters=[parameters[par_name] for par_name in args.min_par]
    )
    minimizer_syst = IMinuitMinimizer(
        statistic=chi2p_syst, parameters=[parameters[par_name] for par_name in args.min_par]
    )

    for stat_type, filename in args.input:
        minimizer = minimizer_stat if stat_type == "stat" else minimizer_syst
        for parameter, values in mc_parameters.items():
            next_sample(storage, parameter, values)
            dagflow_fit = minimizer.fit()
            print(f"Fit {stat_type}:{'GNA':>17}{'dag-flow':>12}  relative-error")
            compare_gna(dagflow_fit, filename)
            minimizer.push_initial_values()


if __name__ == "__main__":
    main()
