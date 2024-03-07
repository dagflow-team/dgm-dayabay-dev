from collections.abc import Sequence

from numba import njit
from numpy import empty
from numpy.typing import NDArray

from multikeydict.nestedmkdict import NestedMKDict


def refine_detector_data(
    source: NestedMKDict,
    target: NestedMKDict,
    *,
    detectors: Sequence[str],
    periods: Sequence[int] = (6, 8, 7),
    clean_source: bool = True,
) -> None:
    fields = ("livetime", "eff", "efflivetime")
    target["days"] = (days_storage := {})
    for det in detectors:
        day = source["day", det]
        step = day[1:] - day[:-1]
        assert (step==1).all(), "Expect detector data for each day"

        ndet = source["ndet", det]
        for period in periods:
            mask_period = ndet == period
            periodname = f"{period}AD"

            for field in fields:
                data = source[field, det]

                key = (field, periodname, det)
                target[key] = data[mask_period]

            days = source["day", det][mask_period]
            days_stored = days_storage.setdefault(periodname, days)
            if days is not days_stored:
                assert all(days==days_stored)

    if clean_source:
        for key in tuple(source.walkkeys()):
            source.delete_with_parents(key)
