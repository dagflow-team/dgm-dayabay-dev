from models.dayabay_v0 import model_dayabay_v0

from dagflow.plot import plot_auto

def test_dayabay_v0():
	model = model_dayabay_v0()

	graph = model.graph
	storage = model.storage

	if not graph.closed:
		return

	print(storage.to_table(truncate=True))

	# storage("outputs").plot(folder='output/dayabay_v0_auto')
	# storage("outputs.oscprob").plot(folder='output/dayabay_v0_auto')
	# storage("outputs.countrate").plot(show_all=True)
	# storage("outputs").plot(
	#     folder = 'output/dayabay_v0_auto',
	#     overlay_priority = (index["isotope"], index["reactor"], index['background'], index["detector"])
	# )

	storage["parameter.normalized.detector.eres.b_stat"].value = 1
	storage["parameter.normalized.detector.eres.a_nonuniform"].value = 2

	# p1 = storage["parameter.normalized.detector.eres.b_stat"]
	# p2 = storage["parameter.constrained.detector.eres.b_stat"]

	constrained = storage("parameter.constrained")
	normalized = storage("parameter.normalized")

	print("Everything")
	print(storage.to_table(truncate=True))

	# print("Constants")
	# print(storage("parameter.constant").to_table(truncate=True))
	#
	# print("Constrained")
	# print(constrained.to_table(truncate=True))
	#
	# print("Normalized")
	# print(normalized.to_table(truncate=True))
	#
	# print("Stat")
	# print(storage("stat").to_table(truncate=True))

	# print("Parameters (latex)")
	# print(storage["parameter"].to_latex())
	#
	# print("Constants (latex)")
	# tex = storage["parameter.constant"].to_latex(columns=["path", "value", "label"])
	# print(tex)

	storage.to_datax("output/dayabay_v0_data.tex")

	from dagflow.graphviz import GraphDot
	GraphDot.from_graph(
		graph,
		show="all"
	).savegraph("output/dayabay_v0.dot")
	GraphDot.from_graph(
		graph,
		show="all",
		filter = {
			'reactor': [0],
			'detector': [0],
			'isotope': [0],
			'period': [0],
			'background': [0],
		}
	).savegraph("output/dayabay_v0_reduced.dot")
	GraphDot.from_node(
		storage["stat.nuisance.all"],
		show="all",
		mindepth=-1,
		no_forward=True
	).savegraph("output/dayabay_v0_nuisance.dot")
	GraphDot.from_output(
		storage["outputs.edges.energy_evis"],
		show="all",
		mindepth=-3,
		no_forward=True
	).savegraph("output/dayabay_v0_top.dot")
