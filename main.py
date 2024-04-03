from pathlib import Path
import numpy as np

from maraboupy import MarabouCore as mb_core
from maraboupy import Marabou

# SETTINGS ------------------------------------------------------
NNET_FILE = Path("networks/ACASXU_run2a_1_1_tiny.nnet")
OUT_DIR = Path("result")
# ---------------------------------------------------------------


def getStatistics(stats: mb_core.Statistics) -> dict:
    return {
        "num splits": stats.getUnsignedAttribute(mb_core.NUM_SPLITS),
        "current decission level": stats.getUnsignedAttribute(mb_core.CURRENT_DECISION_LEVEL),
        "max decision level": stats.getUnsignedAttribute(mb_core.MAX_DECISION_LEVEL),
        "num accepted phase pattern update": stats.getLongAttribute(mb_core.NUM_ACCEPTED_PHASE_PATTERN_UPDATE),
        "num added rows": stats.getLongAttribute(mb_core.NUM_ADDED_ROWS),
        "num main loop iterations": stats.getLongAttribute(mb_core.NUM_MAIN_LOOP_ITERATIONS),
        "total time": stats.getTotalTimeInMicro(),
        "total time applying stored tightenings micro": stats.getLongAttribute(mb_core.TOTAL_TIME_APPLYING_STORED_TIGHTENINGS_MICRO),
        "total time constraint matrix bound tightening micro": stats.getLongAttribute(mb_core.TOTAL_TIME_CONSTRAINT_MATRIX_BOUND_TIGHTENING_MICRO),
        "total time degredation checking": stats.getLongAttribute(mb_core.TOTAL_TIME_DEGRADATION_CHECKING),
        "total time explicit basis bound tightening micro": stats.getLongAttribute(mb_core.TOTAL_TIME_EXPLICIT_BASIS_BOUND_TIGHTENING_MICRO),
        "total time performing valid case splits micro": stats.getLongAttribute(mb_core.TOTAL_TIME_PERFORMING_VALID_CASE_SPLITS_MICRO),
        "total time applying stored tightenings micro": stats.getLongAttribute(mb_core.TOTAL_TIME_APPLYING_STORED_TIGHTENINGS_MICRO),
        "total time handling statistics micro": stats.getLongAttribute(mb_core.TOTAL_TIME_HANDLING_STATISTICS_MICRO),
        "total time precision restoration": stats.getLongAttribute(mb_core.TOTAL_TIME_PRECISION_RESTORATION),
        "total time constraint matrix bound tightening micro": stats.getLongAttribute(mb_core.TOTAL_TIME_CONSTRAINT_MATRIX_BOUND_TIGHTENING_MICRO),
        "total time obtain current assignment micro": stats.getLongAttribute(mb_core.TOTAL_TIME_OBTAIN_CURRENT_ASSIGNMENT_MICRO),
        "total time smt core micro": stats.getLongAttribute(mb_core.TOTAL_TIME_SMT_CORE_MICRO),
        "total time degradation checking": stats.getLongAttribute(mb_core.TOTAL_TIME_DEGRADATION_CHECKING),
        "total time performing symbolic bound tightening": stats.getLongAttribute(mb_core.TOTAL_TIME_PERFORMING_SYMBOLIC_BOUND_TIGHTENING),
        "total time updating soi phase pattern micro": stats.getLongAttribute(mb_core.TOTAL_TIME_UPDATING_SOI_PHASE_PATTERN_MICRO),
    }


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    net: Marabou.MarabouNetworkNNet = Marabou.read_nnet(NNET_FILE)
    output_vars: np.ndarray = net.outputVars[0]
    param = output_vars[0][0]
    print(net.inputVars[0])
    print(output_vars)
    # net.setLowerBound(param, .5)
    net.setLowerBound(net.outputVars[0][0][0], -.5)


    exit_code, vals, stats = net.solve("out.txt", True, Marabou.createOptions())
    stats: mb_core.Statistics = stats

    print(f"exit_code: {exit_code}")
    print(f"total time: {getStatistics(stats)}")
    # print(f"num splits: {stats.getNumSplist()}")