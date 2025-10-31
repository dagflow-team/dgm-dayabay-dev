#!/usr/bin/env python

from dag_modelling.plot.graphviz import GraphDot
from dag_modelling.tools.logger import set_verbosity

from dgm_dayabay_dev.models import load_model

set_verbosity(1)

graph_accept_index = {
    "reactor": [0],
    "detector": [0, 1],
    "isotope": [0],
    "period": [1, 2],
}


def main(args):
    model = load_model("v1a_distorted")
    model.switch_data("asimov")
    model.update_frozen_nodes()

    outpath = "output/test_distortion"

    distortion_path = "nodes.survival_probability_fake.spectrum_distortion"
    # chi2_path = "outputs.statistic.stat.chi2p"
    chi2_path = "outputs.statistic.stat.chi2cnp"
    chi2 = model.storage[chi2_path]
    nuisance_path = "outputs.statistic.nuisance.pull_extra"
    nuisance = model.storage[nuisance_path]

    fakebaseline = model.storage["parameters.all.survival_probability_fake.baseline"]
    fakepar = model.storage["parameters.all.survival_probability_fake.target.DeltaMSq32"]

    graphopts = dict(
        min_depth=-2,
        max_depth=2,
        keep_direction=True,
        show="all",
        filter=graph_accept_index,
    )

    def savegraphs(name: str):
        GraphDot.from_object(model.storage[distortion_path], **graphopts).savegraph(
            f"{outpath}/{name}_d.dot"
        )
        GraphDot.from_object(model.storage[chi2_path], **graphopts).savegraph(
            f"{outpath}/{name}_c.dot"
        )

    def prnt():
        print(f"{fakepar!s} {fakebaseline!s} {chi2.data=} {nuisance.data=}")


    print("Initial: consistent")
    prnt()
    # savegraphs("it0")

    print("change splitting]")
    fakepar.push(0.002)
    prnt()
    # savegraphs("it1")

    print("disable baseline: consistent")
    fakebaseline.push(0)
    prnt()
    # savegraphs("it2")

    print("disable baseline: revert")
    fakepar.pop()
    fakebaseline.pop()
    prnt()

    fakepar.push(0.002)
    model.update_frozen_nodes()
    print("fake splitting as asimov: consistent")
    prnt()

    print("disable baseline: inconsistent")
    fakebaseline.push(0)
    prnt()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    main(parser.parse_args())
