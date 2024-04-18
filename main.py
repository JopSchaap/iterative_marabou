import itertools
from pathlib import Path
import pickle
import sys
import time
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np

from maraboupy import MarabouCore as mb_core
from maraboupy import Marabou
from tqdm import tqdm, trange

from iterator import Iterator

# SETTINGS ------------------------------------------------------
# NNET_FILE = Path("networks/ACASXU_run2a_1_1_tiny.nnet") # Super small DNN handy for testing!
NNET_FILE = Path("networks/ACASXU_experimental_v2a_1_1.nnet")
OUT_DIR = Path("result")
EXPERIMENT_FILE = OUT_DIR / "experiment.pk"
FIG_NAME = "figure"
OLD_RESULTS_FILE = Path("prev_result/experiment.pk")
# ---------------------------------------------------------------


def eprint(*args, **vargs):
    __builtins__.print(*args, file=sys.stderr, **vargs)

def saveFig(figure_name: str = FIG_NAME):
    path = OUT_DIR / figure_name
    pngPath = path.with_suffix(".png")
    pdfPath = path.with_suffix(".pdf")
    for i in itertools.count(1):
        if (not pngPath.exists()) and (not pdfPath.exists()):
             break 
        path = OUT_DIR / (figure_name + f"-{i}")
        pngPath = path.with_suffix(".png")
        pdfPath = path.with_suffix(".pdf")
    plt.savefig(pngPath)
    plt.savefig(pdfPath)

def plot_box(index1:int,index2:int,box:np.ndarray):
    x1 = box[index1]
    x2 = box[index2]
    width = x1[1] - x1[0]
    height = x2[1] - x2[0]
    rect = patches.Rectangle((x1[0],x2[0]), width, height, edgecolor="red", facecolor="none")
    # rect = patches.Rectangle((x1[0],x2[0]), width, height, color="red")
    # ax = plt.axes()
    plt.gca().add_patch(rect)
    

def plot_time_per_iteration(time_spent):
    plt.figure()
    plt.plot(time_spent)
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("time spent solving")
    saveFig()

def plot_response(net: Marabou.MarabouNetwork, axis = [0,1], out_axis=None, min_value=-1, max_value=1, nSamples=(100, 100), base = None, box=None):
    nOutputVars = len(net.outputVars[0][0])
    out_axis = out_axis or np.arange(nOutputVars)
    result = np.zeros((len(out_axis),*nSamples))
    extra = np.zeros(nSamples)
    base = base or np.zeros(len(net.inputVars[0][0]))
    for i, x in enumerate(tqdm(np.linspace(min_value, max_value,nSamples[0]))):
        for j, y in enumerate(np.linspace(min_value,max_value,nSamples[1])):
            base[axis] = x,y
            res = np.array(net.evaluate(base, False))[0]
            extra[i,j] = 1 if res[2] < 0.8 else 0
            result[:,i,j] = res[out_axis]

    for i in trange(result.shape[0]):
        plt.figure()
        plt.imshow(result[i, :, :].T, alpha=extra.T, extent=(min_value, max_value, min_value, max_value))
        plt.colorbar(label=f"$y_{out_axis[i]}$")
        plt.xlabel(f"$x_{axis[0]}$")
        plt.ylabel(f"$x_{axis[1]}$")
        if not box is None:
            plot_box(axis[0],axis[1], box)
        saveFig(f"input-{axis[0]}-{axis[1]}_output-{out_axis[i]}")

def plot_box_sizes(optimistic_sizes: np.ndarray, pessimistic_sizes:np.ndarray):
    plt.figure()
    plt.plot(optimistic_sizes, label="optimistic")
    plt.plot(pessimistic_sizes, label="pessimistic")
    plt.legend()
    plt.ylim(bottom=0, top=max(pessimistic_sizes)*2)
    saveFig()

def plot_uncertainty(optimistic_sizes: np.ndarray, pessimistic_sizes:np.ndarray):
    uncertainty = np.array(optimistic_sizes) - np.array(pessimistic_sizes)
    plt.plot(uncertainty)
    # plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Size of the box(product of the widths in all directions)")
    saveFig()

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    net: Marabou.MarabouNetworkNNet = Marabou.read_nnet(NNET_FILE)
    eprint("number of variables:", net.numVars)
    eprint("number of layers:", net.numLayers)
    output_vars: np.ndarray = net.outputVars[0]
    param = output_vars[0][0]
    # net.setUpperBound(output_vars[0][2],1.0)
    eprint("ouptut: ", net.evaluate(np.zeros(5), False))
    net.setLowerBound(output_vars[0][2],0.8)

    if "--no-run" in sys.argv:
        with OLD_RESULTS_FILE.open("rb") as f:
            iterator = pickle.load(f)
            print(iterator.box_size_pessimistic_list)
    else:
        try:
            iterator = Iterator(net)
            # iterator.n_in_vars = 2
            # iterator.upper_bounds = iterator.upper_bounds[:2]
            # iterator.lower_bounds = iterator.lower_bounds[:2]
            iterator.tighten()
        except KeyboardInterrupt:
            pass
        with EXPERIMENT_FILE.open("wb") as f:
            pickle.dump(iterator, f)


    plot_box_sizes(iterator.box_size_optimistic_list, iterator.box_size_pessimistic_list)
    plot_uncertainty(iterator.box_size_optimistic_list, iterator.box_size_pessimistic_list)

    # net.setLowerBound(param, .5)
    # net.setLowerBound(net.outputVars[0][0][0], -.5)
    
    # plot_response(net,out_axis=[2], box=iterator.lower_bounds)
    # plot_response(net,axis = [2,3], out_axis=[2], box=iterator.lower_bounds)
    # plot_response(net,axis = [0,4], out_axis=[2], box=iterator.lower_bounds)
    eprint(iterator.lower_bounds)
    for axis in itertools.combinations(range(iterator.n_in_vars),2):
        plot_response(net,list(axis), out_axis=[2], box=iterator.lower_bounds)

    # exit_code, vals, stats = net.solve("out.txt", True, Marabou.createOptions())
    # stats: mb_core.Statistics = stats

    # print(f"exit_code: {exit_code}")
    # print(f"total time: {getStatistics(stats)}")
    # print(f"num splits: {stats.getNumSplist()}")