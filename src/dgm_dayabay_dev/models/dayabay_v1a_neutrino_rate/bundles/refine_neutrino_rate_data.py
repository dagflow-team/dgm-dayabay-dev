from collections.abc import Sequence

from nested_mapping import NestedMapping
from numba import njit
from numpy import empty, isnan, logical_or
from numpy.typing import NDArray


@njit
def weekly_to_daily(array: NDArray) -> NDArray:
    ret = empty(array.shape[0] * 7)

    for i in range(7):
        ret[i::7] = array

    return ret


@njit
def inperiod_to_daily(array: NDArray, ndays: NDArray) -> NDArray:
    ndays_total = ndays.sum()
    ret = empty(ndays_total)

    i = 0
    for ndays_i, arr in zip(ndays, array):
        ret[i : i + ndays_i] = arr
        i += ndays_i

    return ret


@njit
def periods_to_days(array: NDArray, ndays: NDArray) -> NDArray:
    ndays_total = ndays.sum()
    ret = empty(ndays_total)

    i = 0
    for arr, ndays_p in zip(array, ndays):
        for day in range(ndays_p):
            ret[i] = float(arr) + float(day)
            i += 1

    return ret


@njit
def weeks_to_days(array: NDArray) -> NDArray:
    ret = empty(array.shape[0] * 7)

    for i in range(7):
        ret[i::7] = array
        ret[i::7] += i

    return ret


def refine_neutrino_rate_data(
    source: NestedMapping,
    target: NestedMapping,
    *,
    reactors: Sequence[str],
    isotopes: Sequence[str],
    periods: Sequence[int] = (6, 8, 7),
    clean_source: bool = True,
) -> None:
    """Refine neutrino rate data.

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
        - u235 : dict[str, NDArray]
              number of neutrinos per second from the ²³⁵U ("U235" should be contained in isotopes) for
              each reactor core.
        - <isotope> : dict[str, NDArray]
              number of neutrinos per second for the <isotope> according to isotopes for each reactor
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
        ndays = source["n_days", corename]
        ndet = source["n_det", corename]

        neutrino_rate = {key: source[key.lower(), corename] for key in isotopes}

        step = period[1:] - period[:-1]
        assert (step == 1).all(), "Expect reactor data for with distinct period, no gaps"

        target["days"] = (days_storage := {})
        for period in periods:
            mask_period = logical_or((ndet == period), (ndet > 8))
            periodname = f"{period}AD"

            mask = mask_period
            key = (
                periodname,
                corename,
            )
            ndays_p = ndays[mask]
            for isotope in isotopes:
                target[("fission_fraction",) + key + (isotope,)] = inperiod_to_daily(
                    neutrino_rate[isotope][mask], ndays_p
                )

            days = periods_to_days(day[mask], ndays_p)
            if isnan(days).any():
                raise RuntimeError()
            days_stored = days_storage.setdefault(periodname, days)
            if days is not days_stored:
                assert all(days == days_stored)

    for key, array in target.walkjoineditems():
        if isnan(array).any():
            raise ValueError(f"Invalid refined reactor data for {key}")

    if clean_source:
        for key in tuple(source.walkkeys()):
            source.delete_with_parents(key)

