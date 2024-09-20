from models.dayabay_v0 import model_dayabay_v0
from dagflow.logger import INFO1, set_level

set_level(INFO1)


#def test_dayabay_v0_invalidation():
#    model = model_dayabay_v0(close=True, strict=False, include_covariance=False)
#
#    graph = model.graph
#    if not graph.closed:
#        raise RuntimeError("The model graph is not closed!")
#
#    storage = model.storage
#    nodes = graph._nodes
#
#    get_n_tainted = lambda: len([flag for node in nodes if (flag := node.fd.tainted)])
#    logger = model._logger
#
#    logger.log(INFO1, f"Number of tainted nodes before evaluation: {get_n_tainted()}.")
#    for force in (True, False):
#        logger.log(INFO1, f"Evaluating full model: force_computation={force}.")
#        model.eval(force_computation=force)
#        logger.log(INFO1, f"Number of tainted nodes after evaluation: {get_n_tainted()}.")
#
#    parameters = storage("parameters.all")
#    with open("output/tainted.txt", "w") as f:
#        f.write("# parname\tntainted\tevaltime\n")
#        for keys, par in parameters.walkitems():
#            for node in nodes:
#                node._n_calls = 0
#            parname = ".".join(keys)
#            logger.log(INFO1, f"Tainting parameter {parname}...")
#
#            par._value_output._node.taint_children()
#            ntainted = get_n_tainted()
#            logger.log(INFO1, f"Number of tainted nodes after taint: {ntainted}.")
#
#            logger.log(INFO1, "Reevaluating model...")
#            _, evaltime = model.eval(force_computation=False)
#            f.write(f"{parname}\t{ntainted}\t{evaltime}\n")
#
#            ntainted = get_n_tainted()
#            assert ntainted == 0

def test_dayabay_v0_invalidation_histo():
    from pandas import read_csv

    df = read_csv("output/tainted.txt", delimiter="\t", skiprows=1, names=["name", "ntainted", "evaltime"])

    print(f"All parameters: npars={len(df.index)}")
    for key in ("ntainted", "evaltime"):
        print(f"mean({key})={df[key].mean():0.3f}, max({key})={df[key].max()}") 

    from matplotlib import pyplot as plt
    _ = df.plot.hist(column=["ntainted"], bins=100)
    plt.yscale("log")
    plt.ylim(1, 1e3)
    plt.xlabel("tainted nodes number")
    plt.tight_layout()
    plt.savefig("output/tainted_nodes_histo.png")

    df = df.drop_duplicates(subset=["ntainted"], keep="first")
    print(f"Unique parameters: npars={len(df.index)}")
    for key in ("ntainted", "evaltime"):
        print(f"mean({key})={df[key].mean():0.3f}, max({key})={df[key].max()}")
