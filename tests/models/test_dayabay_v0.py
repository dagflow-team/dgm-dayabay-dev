from pytest import mark

from dagflow.core import Graph, NodeStorage
from dagflow.plot.graphviz import GraphDot
from models import available_models, load_model


@mark.parametrize("model_version", available_models())
def test_dayabay_v0(model_version: str):
    model = load_model(model_version, close=True, strict=True)

    graph = model.graph
    storage = model.storage

    if not graph.closed:
        print("Nodes")
        print(storage("nodes").to_table(truncate=True))
        print("Outputs")
        print(storage("outputs").to_table(truncate=True))
        print("Not connected inputs")
        print(storage("inputs").to_table(truncate=True))

        plot_graph(graph, storage)
        return

    print(storage.to_table(truncate=True))
    if len(storage("inputs")) > 0:
        print("Not connected inputs")
        print(storage("inputs").to_table(truncate=True))

    storage.to_datax("output/dayabay_v0_data.tex")
    plot_graph(graph, storage)


@mark.parametrize("model_version", available_models())
def test_dayabay_v0_proxy_switch(model_version: str):
    # TODO: remove when the model is done
    if model_version in ("v0e", "latest"):
        return
    model = load_model(model_version, close=True, strict=True, monte_carlo_mode="poisson")

    storage = model.storage

    proxy_node = storage["nodes.data.proxy"]
    obs = storage.get_value("outputs.eventscount.final.concatenated.selected")
    chi2 = storage["outputs.statistic.stat.chi2p"]
    assert chi2.data != 0.0

    proxy_node.open()
    obs >> proxy_node
    proxy_node.close()
    proxy_node.switch_input(-1)
    assert chi2.data == 0.0


def plot_graph(graph: Graph, storage: NodeStorage) -> None:
    GraphDot.from_graph(graph, show="all").savegraph("output/dayabay_v0.dot")
    GraphDot.from_graph(
        graph,
        show="all",
        filter={
            "reactor": [0],
            "detector": [0, 1],
            "isotope": [0],
            "period": [0],
            "background": [0],
        },
    ).savegraph("output/dayabay_v0_reduced.dot")
    GraphDot.from_node(
        storage["nodes.statistic.nuisance.all"],
        show="all",
        mindepth=-1,
        keep_direction=True,
    ).savegraph("output/dayabay_v0_nuisance.dot")
    GraphDot.from_output(
        storage["outputs.edges.energy_evis"],
        show="all",
        mindepth=-3,
        keep_direction=True,
    ).savegraph("output/dayabay_v0_top.dot")
