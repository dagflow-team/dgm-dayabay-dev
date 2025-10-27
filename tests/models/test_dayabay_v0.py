from dag_modelling.core import Graph, NodeStorage
from dag_modelling.plot.graphviz import GraphDot
from parameterized import parameterized_class
from pytest import mark

from dgm_dayabay_dev.models import available_models, load_model


@parameterized_class(
    [{"model_version": version} for version in available_models() if version != "latest"]
)
@mark.usefixtures("output_path")
class TestModel:
    model_version = None
    model = None

    @classmethod
    def setup_class(cls):
        cls.model = load_model(cls.model_version, close=True, strict=False)

    def test_dayabay_v0(self, output_path: str):
        graph = self.model.graph
        storage = self.model.storage

        if not graph.closed:
            print("Nodes")
            print(storage("nodes").to_table(truncate=True))
            print("Outputs")
            print(storage("outputs").to_table(truncate=True))
            print("Not connected inputs")
            print(storage("inputs").to_table(truncate=True))

            plot_graph(graph, storage, output_path=output_path)
            return

        print(storage.to_table(truncate=True))
        if len(storage("inputs")) > 0:
            print("Not connected inputs")
            print(storage("inputs").to_table(truncate=True))

        storage.to_datax(f"{output_path}/dayabay_v0_data.tex")
        plot_graph(graph, storage, output_path=output_path)

    def test_dayabay_v0_proxy_switch(self):
        model = self.model

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


def plot_graph(graph: Graph, storage: NodeStorage, output_path: str) -> None:
    GraphDot.from_graph(graph, show="all").savegraph(f"{output_path}/dayabay_v0.dot")
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
    ).savegraph(f"{output_path}/dayabay_v0_reduced.dot")
    GraphDot.from_node(
        storage["nodes.statistic.nuisance.all"],
        show="all",
        min_depth=-1,
        keep_direction=True,
    ).savegraph(f"{output_path}/dayabay_v0_nuisance.dot")
    GraphDot.from_output(
        storage["outputs.edges.energy_evis"],
        show="all",
        min_depth=-3,
        keep_direction=True,
    ).savegraph(f"{output_path}/dayabay_v0_top.dot")
