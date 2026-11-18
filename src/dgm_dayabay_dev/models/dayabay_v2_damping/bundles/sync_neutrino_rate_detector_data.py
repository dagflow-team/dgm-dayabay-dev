from nested_mapping import NestedMapping


def sync_neutrino_rate_detector_data(
    neutrino_rate_data: NestedMapping,
    detector_data: NestedMapping,
) -> None:
    neutrino_rate_day = neutrino_rate_data("days")
    detector_day = detector_data("days")

    for period, detector_day_p in detector_day.items():
        neutrino_rate_day_p = neutrino_rate_day[period]

        offset1 = int(detector_day_p[0] - neutrino_rate_day_p[0])
        offset2 = int(detector_day_p[-1] - neutrino_rate_day_p[-1])
        if offset1 < 0 or offset2 > 0:
            raise RuntimeError(
                "Neutrino rate data is expected to start not later and end later then detector data. "
                f"{period} got start/end for neutrino rate ({neutrino_rate_day_p[0]:.0f}, {neutrino_rate_day_p[-1]:.0f})"
                f" and detector ({detector_day_p[0]:.0f}, {detector_day_p[-1]:.0f})"
            )
        if offset2 == 0:
            offset2 = None

        slc = slice(offset1, offset2)
        neutrino_rate_day_p_new = neutrino_rate_day_p[slc]
        assert (
            neutrino_rate_day_p_new == detector_day_p
        ).all(), f"Unable to synchronize neutrino rate and detector data for {period}"

        neutrino_rate_day[period] = neutrino_rate_day_p_new

        for key, data in neutrino_rate_data.walkitems():
            if key[0] == "days":
                continue
            if not period in key:
                continue

            neutrino_rate_data[key] = data[slc]
