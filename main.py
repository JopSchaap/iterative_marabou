from pathlib import Path
import numpy as np

from maraboupy import Marabou

from marabou import maraboupy
import maraboupy.MarabouCore as mb_core

# SETTINGS ------------------------------------------------------
NNET_FILE = Path("networks/ACASXU_run2a_1_1_tiny.nnet")
# ---------------------------------------------------------------


if __name__ == "__main__":
    net: Marabou.MarabouNetworkNNet = Marabou.read_nnet(NNET_FILE)
    output_vars: np.ndarray = net.outputVars[0]
    param = output_vars[0][0]
    print(net.outputVars[0][0].shape)
    net.setLowerBound(param, .5)


    exit_code, vals, stats = net.solve()
    stats: mb_core.Statistics = stats

    print(f"exit_code: {exit_code}")
    print(f"total time: {stats.getTotalTime()}")
    # print(f"num splits: {stats.getNumSplist()}")