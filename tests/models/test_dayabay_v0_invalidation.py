from time import time

from matplotlib import pyplot as plt

from dagflow.logger import INFO1, set_level
from models.dayabay_v0 import model_dayabay_v0

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
#        f.write("# parname\tntainted\tevaltime\tevaltime+tainttime\n")
#        for keys, par in parameters.walkitems():
#            for node in nodes:
#                node._n_calls = 0
#            parname = ".".join(keys)
#            logger.log(INFO1, f"Tainting parameter {parname}...")
#
#            tainttime = time()
#            par._value_output._node.taint_children()
#            tainttime = time() - tainttime
#
#            ntainted = get_n_tainted()
#            logger.log(INFO1, f"Number of tainted nodes after taint: {ntainted}.")
#
#            logger.log(INFO1, "Reevaluating model...")
#            _, evaltime = model.eval(force_computation=False)
#            f.write(f"{parname}\t{ntainted}\t{evaltime}\t{evaltime+tainttime}\n")
#
#            ntainted = get_n_tainted()
#            assert ntainted == 0


def test_dayabay_v0_invalidation_histo():
    from pandas import read_csv

    df = read_csv(
        "output/tainted.txt",
        delimiter="\t",
        skiprows=1,
        names=["name", "ntainted", "evaltime", "evaltime+tainttime"],
    )
    keys_to_proccess = ("ntainted", "evaltime", "evaltime+tainttime")

    npars = len(df.index)
    print(f"All parameters: npars={npars}")
    meandict, maxdict = {}, {}
    for key in keys_to_proccess:
        meanv, maxv = df[key].mean(), df[key].max()
        meandict[key] = meanv
        maxdict[key] = maxv
        print(f"mean({key})={meanv:0.3f}, max({key})={maxv}")

    _ = df.plot.hist(column=["ntainted"], bins=100, legend=False)
    plt.yscale("log")
    plt.xlim(0, 3500)
    plt.ylim(1, 1e3)
    plt.xlabel("tainted nodes number")
    plt.title("tainted node distribution")
    plt.text(0.75*3500, 0.5e3, f"mean={meandict['ntainted']:0.3f}")
    plt.text(0.75*3500, 0.35e3, f"max={maxdict['ntainted']}")
    plt.tight_layout()
    plt.savefig("output/tainted_nodes_histo.png", dpi=200)

    for arg in ("evaltime", "evaltime+tainttime"):
        draw_time_hist(df, arg, meandict[arg], maxdict[arg])

    df = df.drop_duplicates(subset=["ntainted"], keep="first")
    print(f"Unique parameters: npars={len(df.index)}")
    for key in keys_to_proccess:
        print(f"mean({key})={df[key].mean():0.3f}, max({key})={df[key].max()}")


def draw_time_hist(df, arg, meanv, maxv):
    result = df.copy()
    result[arg] *= 1e3
    _ = result.plot.hist(column=[arg], bins=100, legend=False)
    xmax, ymax = 50, 200
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.title(f"{arg} distribution")
    plt.xlabel("time [ms]")
    plt.text(0.75*xmax, 0.9*ymax, f"mean={meanv*1e3:0.3f}ms")
    plt.text(0.75*xmax, 0.85*ymax, f"max={maxv*1e3:0.3f}ms")
    plt.tight_layout()
    plt.savefig(f"output/{arg}_histo.png", dpi=200)
