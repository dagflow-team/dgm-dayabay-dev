from collections.abc import Sequence

from numba import njit
from numpy import empty
from numpy.typing import NDArray

from multikeydict.nestedmkdict import NestedMKDict


@njit
def weekly_to_daily(array: NDArray) -> NDArray:
    ret = empty(array.shape[0] * 7)

    for i in range(7):
        ret[i::7] = array

    return ret


def refine_reactor_data(
    source: NestedMKDict,
    target: NestedMKDict,
    *,
    reactors: Sequence[str],
    isotopes: Sequence[str],
    periods: Sequence[int] = (6, 8, 7),
    reactor_number_start: int = 1,
    clean_source: bool = True
) -> None:
    week = source["week"]
    core = source["core"]
    # ndays = source["ndays"]
    ndet = source["ndet"]
    start_utc = source["start_utc"]

    earliest_utc = start_utc[0]

    power = source["power"]
    fission_fractions = {key: source[key.lower()] for key in isotopes}

    ncores = 6
    for i in range(ncores):
        rweek = week[i::ncores]
        step = rweek[1:] - rweek[:-1]
        assert (step==1).all()

    for period in periods:
        mask_period = ndet == period
        period = (f"{period}AD",)

        for icore, corename in enumerate(reactors, reactor_number_start):
            mask = mask_period * (core == icore)
            key = period + (corename,)
            target[("power",) + key] = weekly_to_daily(power[mask])
            for isotope in isotopes:
                target[("fission_fraction",) + key + (isotope,)] = weekly_to_daily(fission_fractions[isotope][mask])

    if clean_source:
        for key in tuple(source.walkkeys()):
            source.delete_with_parents(key)
