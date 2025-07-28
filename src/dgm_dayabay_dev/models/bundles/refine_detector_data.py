from collections.abc import Sequence

from nested_mapping import NestedMapping


def refine_detector_data(
    source: NestedMapping,
    target: NestedMapping,
    *,
    detectors: Sequence[str],
    periods: Sequence[str] = ("6AD", "8AD", "7AD"),
    clean_source: bool = True,
    columns=("livetime", "eff", "efflivetime"),
    skip: Sequence[set[str]] | None = None,
) -> None:
    """Read arrays with detector data for the whole data taking period, write detector
    data for each period in separate array.
    """
    target["days"] = (days_storage := {})
    for det in detectors:
        day = source["day", det]
        step = day[1:] - day[:-1]
        assert (step == 1).all(), "Expect detector data for each day"

        ndet = source["ndet", det]
        for periodname in periods:
            period_ndet = int(periodname[0])
            mask_period = ndet == period_ndet

            for field in columns:
                data = source[field, det]

                key = (field, periodname, det)
                if skip is not None and any(skipkey.issubset(key) for skipkey in skip):
                    continue

                target[key] = data[mask_period]

            days = source["day", det][mask_period]
            days_stored = days_storage.setdefault(periodname, days)
            if days is not days_stored:
                assert all(days == days_stored)

    if clean_source:
        for key in tuple(source.walkkeys()):
            source.delete_with_parents(key)
