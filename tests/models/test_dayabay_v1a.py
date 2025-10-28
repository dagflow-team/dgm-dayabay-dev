import numpy as np
from pytest import raises

from dgm_dayabay_dev.models import model_dayabay_v1a


def test_dayabay_v1a_custom_cov_pull():
    nbins = 12
    model = model_dayabay_v1a(
        covariance_groups=["eres"],
        pull_groups=["eres"],
        strict=False,
    )

    assert len(model.storage["outputs.covariance.jacobians"]) == 1
    assert len(model.storage["nodes.statistic.nuisance.pull_extra"].inputs[0].parent_node.inputs) == 3


def test_dayabay_v1a_edges_loading():
    nbins = 12
    model = model_dayabay_v1a(final_erec_bin_edges=np.linspace(0, 12, nbins + 1))

    assert (
        model.storage["outputs.eventscount.final.concatenated.detector_period"].data.shape[0]
        == 21 * nbins
    )

    nweights = 8
    model = model_dayabay_v1a(antineutrino_spectrum_segment_edges=np.linspace(1, 8, nweights))

    assert (
        len(list(model.storage["parameters.free.neutrino_per_fission_factor"].walkvalues()))
        == nweights
    )

    with raises(RuntimeError):
        model_dayabay_v1a(
            final_erec_bin_edges=np.linspace(0, 12, nbins + 1),
            override_cfg_files={"final_erec_bin_edges": "non/existed/edges.tsv"},
        )

    with raises(RuntimeError):
        model_dayabay_v1a(
            antineutrino_spectrum_segment_edges=np.linspace(0, 12, nbins + 1),
            override_cfg_files={"antineutrino_spectrum_segment_edges": "non/existed/edges.tsv"},
        )
