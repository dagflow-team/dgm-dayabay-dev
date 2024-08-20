from dagflow.graph import Graph
from dagflow.plot import plot_auto
from dagflow.storage import NodeStorage
from models.dayabay_v0 import model_dayabay_v0


def test_dayabay_v0():
    model = model_dayabay_v0(close=True, strict=False)

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
    if len(storage("inputs"))>0:
        print("Not connected inputs")
        print(storage("inputs").to_table(truncate=True))

    storage.to_datax("output/dayabay_v0_data.tex")
    plot_graph(graph, storage)


def plot_graph(graph: Graph, storage: NodeStorage) -> None:
    from dagflow.graphviz import GraphDot

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
        storage["nodes.statistic.nuisance.all"], show="all", mindepth=-1, keep_direction=True
    ).savegraph("output/dayabay_v0_nuisance.dot")
    GraphDot.from_output(
        storage["outputs.edges.energy_evis"], show="all", mindepth=-3, keep_direction=True
    ).savegraph("output/dayabay_v0_top.dot")
