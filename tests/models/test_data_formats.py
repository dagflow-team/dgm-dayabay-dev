from pytest import mark

from os import environ
from dag_modelling.core import Graph, NodeStorage
from dag_modelling.plot.graphviz import GraphDot
from models import model_dayabay_v0c as reference_model
from numpy import allclose

source_type_reference = "hdf5"
source_types_other = ["tsv", "npz"]
if "ROOTSYS" in environ:
    print("ROOT is enabled")
    source_types_other.append("root")
else:
    print("ROOT is disabled")

precision_requirement = {
    "tsv": 8.e-11,
    "root": 2.e-11,
    "npz": 0
    }

@mark.parametrize("source_type", source_types_other)
def test_dayabay_v0(source_type: str):
    model_ref = reference_model(source_type=source_type_reference)
    model = reference_model(source_type=source_type)

    outname = "outputs.eventscount.final.concatenated.detector_period"
    output_ref = model_ref.storage[outname]
    output = model.storage[outname]

    data_ref = output_ref.data
    data = output.data

    atol = precision_requirement[source_type]
    assert allclose(data, data_ref, rtol=0, atol=atol), f"{source_type} requires {atol=}"





