PERIODS = ["6AD", "8AD", "7AD"]
REACTORS = ["DB1", "DB2", "LA1", "LA2", "LA3", "LA4"]
ISOTOPES = ["U235", "U238", "Pu239", "Pu241"]
DETECTORS = ["AD11", "AD12", "AD21", "AD22", "AD31", "AD32", "AD33", "AD34"]
EXPERIMENTAL_HALLS = ["EH1", "EH2", "EH3"]

LATEX_SYMBOLS = {
    r"oscprob.DeltaMSq32": r"$\Delta m^2_{32}$",
    r"oscprob.DeltaMSq31": r"$\Delta m^2_{31}$",
    r"oscprob.DeltaMSq21": r"$\Delta m^2_{21}$",
    r"oscprob.SinSq2Theta23": r"$\sin^2 2\theta_{23}$",
    r"oscprob.SinSq2Theta13": r"$\sin^2 2\theta_{13}$",
    r"oscprob.SinSq2Theta12": r"$\sin^2 2\theta_{12}$",
    r"reactor.nominal_thermal_power.DB1": r"$r^{\mathrm{th}}_{\mathrm{DB1}}$",
    r"reactor.nominal_thermal_power.DB2": r"$r^{\mathrm{th}}_{\mathrm{DB2}}$",
    r"reactor.nominal_thermal_power.LA1": r"$r^{\mathrm{th}}_{\mathrm{LA1}}$",
    r"reactor.nominal_thermal_power.LA2": r"$r^{\mathrm{th}}_{\mathrm{LA2}}$",
    r"reactor.nominal_thermal_power.LA3": r"$r^{\mathrm{th}}_{\mathrm{LA3}}$",
    r"reactor.nominal_thermal_power.LA4": r"$r^{\mathrm{th}}_{\mathrm{LA4}}$",
    r"reactor.nominal_thermal_power.LA5": r"$r^{\mathrm{th}}_{\mathrm{LA5}}$",
    r"reactor.nominal_thermal_power.LA6": r"$r^{\mathrm{th}}_{\mathrm{LA6}}$",
    r"detector.lsnl_scale_a.pull0": r"$\omega_0$",
    r"detector.lsnl_scale_a.pull1": r"$\omega_1$",
    r"detector.lsnl_scale_a.pull2": r"$\omega_2$",
    r"detector.lsnl_scale_a.pull3": r"$\omega_3$",
    r"detector.global_normalization": r"$N^{\mathrm{global}}$",
    r"detector.eres.a_nonuniform": r"$\sigma_a$",
    r"detector.eres.b_stat": r"$\sigma_b$",
    r"detector.eres.c_noise": r"$\sigma_c$",
}

for isotope in ISOTOPES:
    key = f"reactor.energy_per_fission.{isotope}"
    value = f"e_{{{isotope}}}"
    LATEX_SYMBOLS[r"{}".format(key)] = r"${}$".format(value)

    for reactor in REACTORS:
        for key, value in [
            (
                f"reactor.fission_fraction_scale.{reactor}.{isotope}",
                f"f_{{{reactor},{isotope}}}",
            ),
            (
                f"reactor.nonequilibrium_scale.{reactor}.{isotope}",
                f"OffEq_{{{reactor},{isotope}}}",
            ),
        ]:
            LATEX_SYMBOLS[r"{}".format(key)] = r"${}$".format(value)

for reactor in REACTORS:
    key = f"reactor.snf_scale.{reactor}"
    value = f"SNF_{{{reactor}}}"
    LATEX_SYMBOLS[r"{}".format(key)] = r"${}$".format(value)

for detector in DETECTORS:
    for key, value in [
        (f"detector.iav_offdiag_scale_factor.{detector}", f"IAV_{{{detector}}}"),
        (
            f"detector.detector_relative.{detector}.efficiency_factor",
            f"\\varepsilon_{{{detector}}}",
        ),
        (
            f"detector.detector_relative.{detector}.energy_scale_factor",
            f"\\epsilon_{{{detector}}}",
        ),
    ]:
        LATEX_SYMBOLS[r"{}".format(key)] = r"${}$".format(value)

    for period in PERIODS:
        for key, value in [
            (
                f"bkg.rate.alphan.{period}.{detector}",
                f"r^{{\\alpha-n}}_{{{detector}, {period}}}",
            ),
            (f"bkg.rate_scale.acc.{period}.{detector}", f"r^{{acc}}_{{{detector}, {period}}}"),
            (f"bkg.rate.amc.{period}.{detector}", f"r^{{AmC}}_{{{detector}, {period}}}"),
            (
                f"bkg.rate.fastn.{period}.{detector}",
                f"r^{{\\mathrm{{fast}}\\ n}}_{{{detector}, {period}}}",
            ),
            (
                f"bkg.rate.lihe.{period}.{detector}",
                f"r^{{\\mathrm{{LiHe}}}}_{{{detector}, {period}}}",
            ),
        ]:
            LATEX_SYMBOLS[r"{}".format(key)] = r"${}$".format(value)

for eh in EXPERIMENTAL_HALLS:
    for period in PERIODS:
        symbol = f"r^{{AmC}}_{{{eh}, {period}}}"
        LATEX_SYMBOLS[f"bkg.uncertainty_scale_by_site.lihe.{eh}.{period}"] = r"${}$".format(symbol)
        symbol = f"r^{{\\mathrm{{fast}}\\ n}}_{{{eh}, {period}}}"
        LATEX_SYMBOLS[f"bkg.uncertainty_scale_by_site.fastn.{eh}.{period}"] = r"${}$".format(symbol)

LATEX_SYMBOLS["bkg.uncertainty_scale.amc"] = "r^{AmC}"

for i in range(31):
    key = f"neutrino_per_fission_factor.spec_scale_{i:02d}"
    value = f"n_{{{i}}}"
    LATEX_SYMBOLS[r"{}".format(key)] = r"${}$".format(value)
