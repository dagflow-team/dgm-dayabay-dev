#!/usr/bin/env python
from yaml import safe_load
from argparse import Namespace
from numpy import ndarray

from dagflow.logger import INFO1, set_level

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


def _simplify_fit_dict(data: dict[str, bool | float | ndarray]) -> None:
    for key, value in data.items():
        if isinstance(value, ndarray):
            data[key] = value.tolist()
    data.pop("summary")


def _compare_parameters(gna_pars: list, dagflow_pars: list) -> bool:
    dagflow_transformed = sorted([dagflow_parameters_to_gna[name] for name in dagflow_pars])
    gna_transformed = sorted(gna_pars)
    return gna_transformed == dagflow_transformed


def compare_gna(dagflow_fit: dict, gna_fit_filename: str) -> None:
    from matplotlib import pyplot as plt

    with open(gna_fit_filename, "r") as f:
        gna_fits = safe_load(f)["fitresult"]
    for gna_fit in gna_fits.values():
        if _compare_parameters(gna_fit["names"], dagflow_fit["names"]):
            print(
                f"ChiSquared:     {gna_fit['fun']:+1.3e}  {dagflow_fit['fun']:+1.3e}      {(gna_fit['fun'] - dagflow_fit['fun']) / gna_fit['fun']:+1.3e}"
            )
            for gna_par_name in gna_fit["names"]:
                dagflow_par_name = gna_parameters_to_dagflow[gna_par_name]
                gna_par_value = gna_fit["xdict"][gna_par_name]
                gna_par_error = gna_fit["errorsdict"][gna_par_name]
                dagflow_par_value = dagflow_fit["xdict"][dagflow_par_name]
                dagflow_par_error = dagflow_fit["errorsdict"][dagflow_par_name]
                print(gna_par_name)
                print(
                    f"Central:        {gna_par_value:+1.3e}  {dagflow_par_value:+1.3e}      {(gna_par_value - dagflow_par_value) / gna_par_value:+1.3e}"
                )
                print(
                    f"Error:          {gna_par_error:+1.3e}  {dagflow_par_error:+1.3e}      {(gna_par_error - dagflow_par_error) / gna_par_error:+1.3e}"
                )
                fig, axs = plt.subplots(1, 1)
                axs.errorbar(
                    x=[gna_par_value],
                    y=[0],
                    xerr=[gna_par_error],
                    fmt="o",
                    markerfacecolor="none",
                    label="GNA",
                )
                axs.errorbar(
                    x=[dagflow_par_value],
                    y=[1],
                    xerr=[dagflow_par_error],
                    fmt="X",
                    markerfacecolor="none",
                    label="dag-flow",
                )
                axs.set_xlabel("value")
                axs.set_title(dagflow_par_name)
                axs.legend()
                axs.set_ylim(-1, 2)
                plt.tight_layout()
                plt.savefig(f"output/comparison/{dagflow_par_name}.png")


def main(args: Namespace) -> None:

    model = model_dayabay_v0(
        source_type=args.source_type,
        spectrum_correction_mode=args.spec,
        fission_fraction_normalized=args.fission_fraction_normalized,
        monte_carlo_mode=args.data_mc_mode,
        seed=args.seed,
    )

    storage = model.storage
    parameters = model.storage("parameters.all")
    statistic = model.storage("outputs.statistic")

    model.touch()

    from yaml import safe_dump
    from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer

    chi2p_stat = statistic["stat.chi2p"]
    chi2p_syst = statistic["full.chi2p"]

    if args.full_fit:
        minimization_pars = list(parameters("oscprob").walkvalues())[:-1]
        minimization_pars.extend(list(parameters("detector.eres").walkvalues()))
        minimization_pars.extend(list(parameters("detector.lsnl_scale_a").walkvalues()))
        minimization_pars.extend(list(parameters("detector.iav_offdiag_scale_factor").walkvalues()))
        minimization_pars.extend(list(parameters("detector.detector_relative").walkvalues()))
        minimization_pars.extend(list(parameters("reactor.energy_per_fission").walkvalues()))
        minimization_pars.extend(list(parameters("reactor.nominal_thermal_power").walkvalues()))
        minimization_pars.extend(list(parameters("reactor.snf_scale").walkvalues()))
        minimization_pars.extend(list(parameters("reactor.offequilibrium_scale").walkvalues()))
        minimization_pars.extend(list(parameters("reactor.fission_fraction_scale").walkvalues()))
        minimization_pars.extend(list(parameters("bkg").walkvalues()))
        minimization_pars.extend([parameters.get_value("detector.global_normalization")])
        model.set_parameters({"detector.global_normalization": 1.0})
        model.next_sample()
        model.set_parameters({"detector.global_normalization": 1.0})
        chi2 = chi2p_stat if args.full_fit == "stat" else chi2p_syst
        minimizer = IMinuitMinimizer(chi2, parameters=minimization_pars)
        fit = minimizer.fit()
        print(fit)
        if args.full_fit_output:
            _simplify_fit_dict(dagflow_fit)
            dagflow_fit = dict(**dagflow_fit)
            with open(f"{args.full_fit_output}-{stat_type}.yaml", "w") as f:
                safe_dump(dagflow_fit, f)

    minimization_pars = [parameters[par_name] for par_name in args.min_par]
    minimizer_stat = IMinuitMinimizer(chi2p_stat, parameters=minimization_pars)
    minimizer_syst = IMinuitMinimizer(chi2p_syst, parameters=minimization_pars)
    for stat_type, filename in args.input:
        minimizer = minimizer_stat if stat_type == "stat" else minimizer_syst
        for par_name, par_value in args.par:
            par_value_init = parameters[par_name].value
            model.set_parameters({par_name: par_value})
            model.next_sample()
            model.set_parameters({par_name: par_value_init})
            dagflow_fit = minimizer.fit()
            print(f"Fit {stat_type}:{'GNA':>17}{'dag-flow':>12}  relative-error")
            compare_gna(dagflow_fit, filename)
            minimizer.push_initial_values()
            if args.fit_output:
                _simplify_fit_dict(dagflow_fit)
                dagflow_fit = dict(**dagflow_fit)
                with open(f"{args.fit_output}-{stat_type}.yaml", "w") as f:
                    safe_dump(dagflow_fit, f)

    if args.profile_par:
        profile_parameter_str, l_edge, r_edge, n_points = args.profile_par
        profile_parameter = parameters[profile_parameter_str]

        assert profile_parameter not in minimization_pars, "You can not minimize profiling parameter"

        from numpy import linspace, savetxt, vstack
        from matplotlib import pyplot as plt

        profile_values = linspace(float(l_edge), float(r_edge), int(n_points))
        chi2_stat_values = []
        chi2_syst_values = []
        model.set_parameters({profile_parameter_str: profile_parameter.value})
        model.next_sample()
        model.set_parameters({profile_parameter_str: profile_parameter.value})
        for value in profile_values:
            for chi2_values, minimizer in zip(
                [chi2_stat_values, chi2_syst_values], [minimizer_stat, minimizer_syst]
            ):
                profile_parameter.value = value
                fit = minimizer.fit()
                chi2_values.append(fit["fun"])
                minimizer.push_initial_values()

        plt.plot(profile_values, chi2_stat_values, label="stat")
        plt.plot(profile_values, chi2_syst_values, label="syst", linestyle="--")
        plt.xlabel(profile_parameter.to_dict()["label"])
        plt.ylabel(r"$\chi^2$")
        plt.ylim(0, 20)
        plt.legend()
        plt.tight_layout()
        if args.profile_output:
            plt.savefig(f"{args.profile_output}.png")
            savetxt(
                f"{args.profile_output}.txt",
                vstack((profile_values, chi2_stat_values, chi2_syst_values)).T,
                header="parameter\tchi2stat\tchi2syst",
            )


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
    input.add_argument(
        "--input-profile",
        action="append",
        default=[],
        help="input file with profile to compare to",
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
    model.add_argument(
        "--fission-fraction-normalized",
        action="store_true",
        help="fission fraction correction",
    )
    model.add_argument("--seed", default=0, type=int, help="seed of randomization")
    model.add_argument(
        "--data-mc-mode",
        default="asimov",
        choices=["asimov", "normalstats", "poisson"],
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
    pars.add_argument(
        "--profile-par",
        nargs=4,
        metavar=("PARAMETER", "LEFT_EDGE", "RIGHT_EDGE", "N_POINTS"),
        help="choose profiling parameters",
    )
    pars.add_argument(
        "--full-fit",
        default=None,
        choices=["stat", "syst"],
        help="Fit model with all parameters",
    )

    outputs = parser.add_argument_group("outputs", "set outputs")
    outputs.add_argument(
        "--fit-output",
        help="path to save fit, without extension",
    )
    outputs.add_argument(
        "--full-fit-output",
        help="path to save full fit, without extension",
    )
    outputs.add_argument(
        "--profile-output",
        help="path to save plot and data, without extension",
    )

    args = parser.parse_args()

    main(args)
