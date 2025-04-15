PERIODS = ["6AD", "8AD", "7AD"]
REACTORS = ["DB1", "DB2", "LA1", "LA2", "LA3", "LA4"]
ISOTOPES = ["U235", "U238", "Pu239", "Pu241"]
DETECTORS = ["AD11", "AD12", "AD21", "AD22", "AD31", "AD32", "AD33", "AD34"]

LATEX_SYMBOLS = {
    r"DeltaMSq32": r"$\Delta m^2_{32}$",
    r"DeltaMSq31": r"$\Delta m^2_{31}$",
    r"DeltaMSq21": r"$\Delta m^2_{21}$",
    r"SinSq2Theta23": r"$\sin^2 2\theta_{23}$",
    r"SinSq2Theta13": r"$\sin^2 2\theta_{13}$",
    r"SinSq2Theta12": r"$\sin^2 2\theta_{12}$",
    r"nominal_thermal_power.DB1": r"$r^{\mathrm{th}}_{\mathrm{DB1}}$",
    r"nominal_thermal_power.DB2": r"$r^{\mathrm{th}}_{\mathrm{DB2}}$",
    r"nominal_thermal_power.LA1": r"$r^{\mathrm{th}}_{\mathrm{LA1}}$",
    r"nominal_thermal_power.LA2": r"$r^{\mathrm{th}}_{\mathrm{LA2}}$",
    r"nominal_thermal_power.LA3": r"$r^{\mathrm{th}}_{\mathrm{LA3}}$",
    r"nominal_thermal_power.LA4": r"$r^{\mathrm{th}}_{\mathrm{LA4}}$",
    r"nominal_thermal_power.LA5": r"$r^{\mathrm{th}}_{\mathrm{LA5}}$",
    r"nominal_thermal_power.LA6": r"$r^{\mathrm{th}}_{\mathrm{LA6}}$",
    r"lsnl_scale_a.pull0": r"$\omega_0$",
    r"lsnl_scale_a.pull1": r"$\omega_1$",
    r"lsnl_scale_a.pull2": r"$\omega_2$",
    r"lsnl_scale_a.pull3": r"$\omega_3$",
    r"global_normalization": r"$N^{\mathrm{global}}$",
    r"eres.a_nonuniform": r"$\sigma_a$",
    r"eres.b_stat": r"$\sigma_b$",
    r"eres.c_noise": r"$\sigma_c$",
}

for isotope in ISOTOPES:
    key = f"energy_per_fission.{isotope}"
    value = f"e_{{{isotope}}}"
    LATEX_SYMBOLS[r"{}".format(key)] = r"${}$".format(value)

    for reactor in REACTORS:
        for key, value in [
            (
                f"fission_fraction_scale.{reactor}.{isotope}",
                f"f_{{{reactor},{isotope}}}",
            ),
            (
                f"nonequilibrium_scale.{reactor}.{isotope}",
                f"OffEq_{{{reactor},{isotope}}}",
            ),
        ]:
            LATEX_SYMBOLS[r"{}".format(key)] = r"${}$".format(value)

for reactor in REACTORS:
    key = f"snf_scale.{reactor}"
    value = f"SNF_{{{reactor}}}"
    LATEX_SYMBOLS[r"{}".format(key)] = r"${}$".format(value)


for detector in DETECTORS:
    for key, value in [
        (f"iav_offdiag_scale_factor.{detector}", f"IAV_{{{detector}}}"),
        (
            f"detector_relative.{detector}.efficiency_factor",
            f"\\varepsilon_{{{detector}}}",
        ),
        (
            f"detector_relative.{detector}.energy_scale_factor",
            f"\\epsilon_{{{detector}}}",
        ),
    ]:
        LATEX_SYMBOLS[r"{}".format(key)] = r"${}$".format(value)

    for period in PERIODS:
        for key, value in [
            (
                f"rate.alphan.{period}.{detector}",
                f"r^{{\\alpha-n}}_{{{detector}, {period}}}",
            ),
            (f"rate.acc.{period}.{detector}", f"r^{{acc}}_{{{detector}, {period}}}"),
            (f"rate.amc.{period}.{detector}", f"r^{{AmC}}_{{{detector}, {period}}}"),
            (
                f"rate.fastn.{period}.{detector}",
                f"r^{{\\mathrm{{fast}}\\ n}}_{{{detector}, {period}}}",
            ),
            (
                f"rate.lihe.{period}.{detector}",
                f"r^{{\\mathrm{{LiHe}}}}_{{{detector}, {period}}}",
            ),
        ]:
            LATEX_SYMBOLS[r"{}".format(key)] = r"${}$".format(value)


for i in range(24):
    key = f"spec_scale_{i:02d}"
    value = f"\\zeta_{{{i}}}"
    LATEX_SYMBOLS[r"{}".format(key)] = r"${}$".format(value)
