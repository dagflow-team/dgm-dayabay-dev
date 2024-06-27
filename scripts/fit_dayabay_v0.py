#!/usr/bin/env python
from argparse import Namespace

from dagflow.graph import Graph
from dagflow.logger import INFO1
from dagflow.logger import INFO2
from dagflow.logger import INFO3
from dagflow.logger import DEBUG as INFO4
from dagflow.logger import set_level
from dagflow.storage import NodeStorage

from models.dayabay_v0 import model_dayabay_v0


set_level(INFO1)


def main() -> None:

    model = model_dayabay_v0(
        close=True,
        strict=True,
        source_type="hdf5",
        override_indices=[],
        spectrum_correction_mode="exponential",
        fission_fraction_normalized=False,
    )

    graph = model.graph
    storage = model.storage
    parameters = model.storage("parameter")
    statistic = model.storage("outputs.statistic")

    from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer
    chi2p_stat = statistic["stat.chi2p"]
    minimizer = IMinuitMinimizer(statistic=chi2p_stat, parameters=[parameters["all.oscprob.SinSq2Theta13"], parameters["all.oscprob.DeltaMSq32"]])
    minimizer_sin2t13 = IMinuitMinimizer(statistic=chi2p_stat, parameters=[parameters["all.oscprob.SinSq2Theta13"]])
    minimizer_dm32 = IMinuitMinimizer(statistic=chi2p_stat, parameters=[parameters["all.oscprob.DeltaMSq32"]])
    sin2t13 = parameters["all.oscprob.SinSq2Theta13"]
    dm32 = parameters["all.oscprob.DeltaMSq32"]

    from pprint import pprint

    pprint([sin2t13.to_dict(), dm32.to_dict()])
    pprint(minimizer.fit())
    pprint([sin2t13.to_dict(), dm32.to_dict()])
    pprint(minimizer_dm32.fit())
    pprint([sin2t13.to_dict(), dm32.to_dict()])
    pprint(minimizer_sin2t13.fit())
    pprint([sin2t13.to_dict(), dm32.to_dict()])



if __name__ == "__main__":
    main()
