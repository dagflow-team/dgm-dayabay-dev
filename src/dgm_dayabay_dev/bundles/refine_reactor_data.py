from collections.abc import Sequence

from numba import njit
from numpy import empty, isnan
from numpy.typing import NDArray

from nested_mapping import NestedMapping


@njit
def weekly_to_daily(array: NDArray) -> NDArray:
    ret = empty(array.shape[0] * 7)

    for i in range(7):
        ret[i::7] = array

    return ret


@njit
def inperiod_to_daily(array: NDArray, ndays: int) -> NDArray:
    ret = empty(array.shape[0] * ndays)

    for i in range(ndays):
        ret[i::ndays] = array

    return ret


@njit
def periods_to_days(array: NDArray, ndays: int) -> NDArray:
    ret = empty(array.shape[0] * ndays)

    for i in range(ndays):
        ret[i::ndays] = array
        ret[i::ndays] += i

    return ret


@njit
def weeks_to_days(array: NDArray) -> NDArray:
    ret = empty(array.shape[0] * 7)

    for i in range(7):
        ret[i::7] = array
        ret[i::7] += i

    return ret


def refine_reactor_data(
    source: NestedMapping,
    target: NestedMapping,
    *,
    reactors: Sequence[str],
    isotopes: Sequence[str],
    periods: Sequence[int] = (6, 8, 7),
    clean_source: bool = True,
) -> None:
    """Refine reactor reactor data.

    Does the following items:
        - checks input consistency
        - split arrays based on isotope number and period name
        - provides similar data on a daily basis
        - splits into multiple arrays based on the reactor index

    Source should contain the following fields:
        - period : int
              period number
        - day : int
              number of the first day of the period, relative to some common day.
        - ndays : int
              length of the period in days. This function can deal only with periods of
              equal length and with no gaps.
        - core : int
              number of the current reactor (1 to 6).
        - ndet : int
              number of active detectors during the period (6, 8 or 7).
        - power : dict[str, NDArray]
              thermal power as a fraction to nominal value (0 to 1) for each reactor
              core.
        - u235 : dict[str, NDArray]
              fission fraction for the ²³⁵U ("U235" should be contained in isotopes) for
              each reactor core.
        - <isotope> : dict[str, NDArray]
              fission fraction for the <isotope> according to isotopes for each reactor
              core.

    The target dictionary will be populated as follows:
    ```python
    target = {
        "days": array(),
        "power": {
            "period_name": {
                "reactor_name": array(),
                ...
            },
            ...
        },
        "fission_fraction": {
            "isotope_name": {
                "period_name": {
                    "reactor_name": array(),
                    ...
                },
                ...
            },
        }
    }
    ```
    with all the arrays of same lengths.

    Parameters
    ----------
    source : NestedMapping
        storage/record/mapping with source data.
    target : NestedMapping
        storage to write target arrays to.
    reactors : Sequence[str]
        reactor names to use when writing data.
    isotopes : Sequence[str]
        names of the isotopes to read and write fission fractions.
    periods : Sequence[int], default=(6, 8, 7)
        periods (number of active detectors) to select.
    clean_source : bool, default=True
        if True, remove used data from source.
    """
    for corename in reactors:
        period = source["period", corename]
        day = source["day", corename]
        try:
            ndays = source["n_days", corename]
        except KeyError:
            ndays = source["ndays", corename]
        try:
            ndet = source["n_det", corename]
        except KeyError:
            ndet = source["ndet", corename]

        ndays0 = ndays[0]
        if not (ndays[:-1] == ndays0).all():
            raise ValueError(
                "refine_reactor_data expects information with constant periodicity"
            )

        power = source["power"][corename]
        fission_fractions = {key: source[key.lower(), corename] for key in isotopes}

        step = period[1:] - period[:-1]
        assert (
            step == 1
        ).all(), "Expect reactor data for with distinct period, no gaps"

        target["days"] = (days_storage := {})
        for period in periods:
            mask_period = ndet == period
            periodname = f"{period}AD"

            mask = mask_period
            key = (
                periodname,
                corename,
            )
            target[("power",) + key] = inperiod_to_daily(power[mask], ndays0)
            for isotope in isotopes:
                target[("fission_fraction",) + key + (isotope,)] = inperiod_to_daily(
                    fission_fractions[isotope][mask], ndays0
                )

            days = periods_to_days(day[mask], ndays0)
            days_stored = days_storage.setdefault(periodname, days)
            if days is not days_stored:
                assert all(days == days_stored)

    for key, array in target.walkjoineditems():
        if isnan(array).any():
            raise ValueError(f"Invalid refined reactor data for {key}")

    if clean_source:
        for key in tuple(source.walkkeys()):
            source.delete_with_parents(key)


def split_refine_reactor_data(
    source: NestedMapping,
    target: NestedMapping,
    *,
    reactors: Sequence[str],
    isotopes: Sequence[str],
    periods: Sequence[int] = (6, 8, 7),
    reactor_number_start: int = 1,
    clean_source: bool = True,
) -> None:
    """Filter/split and refine reactor reactor data.

    Does the following items:
        - checks input consistency
        - split arrays based on reactor number, isotope number and period name
        - provides similar data on a daily basis
        - splits into multiple arrays based on the reactor index

    Source should contain the following fields:
        - week : int
              week/period number
        - day : int
              number of the first day of the period, relative to some common day.
        - ndays : int
              length of the period in days. This function can deal only with periods of
              equal length and with no gaps.
        - core : int
              number of the current reactor (1 to 6).
        - ndet : int
              number of active detectors during the period (6, 8 or 7).
        - power : float
              thermal power as a fraction to nominal value (0 to 1).
        - u235 : float
              fission fraction for the ²³⁵U ("U235" should be contained in isotopes).
        - <isotope> : float
              fission fraction for the <isotope> according to isotopes.

    The target dictionary will be populated as follows:
    ```python
    target = {
        "days": array(),
        "power": {
            "period_name": {
                "reactor_name": array(),
                ...
            },
            ...
        },
        "fission_fraction": {
            "isotope_name": {
                "period_name": {
                    "reactor_name": array(),
                    ...
                },
                ...
            },
        }
    }
    ```
    with all the arrays of same lengths.

    Parameters
    ----------
    source : NestedMapping
        storage/record/mapping with source data.
    target : NestedMapping
        storage to write target arrays to.
    reactors : Sequence[str]
        reactor names to use when writing data.
    isotopes : Sequence[str]
        names of the isotopes to read and write fission fractions.
    periods : Sequence[int], default=(6, 8, 7)
        periods (number of active detectors) to select.
    reactor_number_start : int, default=1
        number of the first reactor in the input data.
    clean_source : bool, default=True
        if True, remove used data from source.
    """
    week = source["week"]
    day = source["day"]
    ndays = source["ndays"]
    core = source["core"]
    try:
        ndet = source["n_det"]
    except KeyError:
        ndet = source["ndet"]

    if not (ndays == 7).all():
        raise ValueError("refine_reactor_data expects weekly information")

    power = source["power"]
    fission_fractions = {key: source[key.lower()] for key in isotopes}

    ncores = 6
    for i in range(ncores):
        rweek = week[i::ncores]
        step = rweek[1:] - rweek[:-1]
        assert (step == 1).all(), "Expect reactor data for each week, no gaps"

    target["days"] = (days_storage := {})
    for period in periods:
        mask_period = ndet == period
        periodname = f"{period}AD"

        for icore, corename in enumerate(reactors, reactor_number_start):
            mask = mask_period * (core == icore)
            key = (
                periodname,
                corename,
            )
            target[("power",) + key] = weekly_to_daily(power[mask])
            for isotope in isotopes:
                target[("fission_fraction",) + key + (isotope,)] = weekly_to_daily(
                    fission_fractions[isotope][mask]
                )

            days = weeks_to_days(day[mask])
            days_stored = days_storage.setdefault(periodname, days)
            if days is not days_stored:
                assert all(days == days_stored)

    for key, array in target.walkjoineditems():
        if isnan(array).any():
            raise ValueError(f"Invalid refined reactor data for {key}")

    if clean_source:
        for key in tuple(source.walkkeys()):
            source.delete_with_parents(key)
