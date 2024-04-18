import itertools
from pathlib import Path
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np

from maraboupy import MarabouCore as mb_core
from maraboupy import Marabou
from tqdm import tqdm, trange

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


class Iterator(object):
    # Iterator settings
    max_iterations = 200
    marabou_options = Marabou.createOptions(timeoutInSeconds=4*60)
    marabou_verbose = False
    verbose = True
    point_select = 0.5
    epsilon = 0.001

    # Holds test data


    def __init__(self, net: Marabou.MarabouNetworkNNet, rng = np.random.default_rng(), starting_upper_bound = 1) -> None:
        self.net = net
        self.n_in_vars = len(net.inputVars[0][0])
        self.n_out_vars = len(net.outputVars[0][0])
        self.lower_bounds = np.zeros((self.n_in_vars, 2),dtype=float) # represents a point which is verified to be correct
        self.upper_bounds = np.full((self.n_out_vars, 2), starting_upper_bound, dtype=float) # represents extreme points where the true bound may still lie
        self.upper_bounds = self.upper_bounds * [-1, 1]
        self.rng = rng
        for input_var in self.net.inputVars[0][0]:
            # start with lower bound all bounds set to [0.0, 0.0]
            self.net.setLowerBound(input_var, 0.0)
            self.net.setUpperBound(input_var, 0.0)
        self.chosen_dim_list = []
        self.stat_list = []
        self.time_list = []
        self.box_size_optimistic_list = []
        self.box_size_pessimistic_list = []
        self.solve()

    def marabou_out_file(self):
        return "" if self.marabou_verbose else "/dev/null"

    def solve(self) -> Tuple[bool, None|np.ndarray]:
        exit_code, vals, stats = self.net.solve(self.marabou_out_file(), self.marabou_verbose, self.marabou_options)
        self.stat_list.append(getStatistics(stats))
        return exit_code != "unsat", vals

    def step_bound(self, dim: int, neg_or_pos: int):
        lb = self.lower_bounds[dim, neg_or_pos]
        ub = self.upper_bounds[dim, neg_or_pos]
        mid = lb + (ub - lb) * self.point_select
        param = self.net.inputVars[0][0][dim]
        if neg_or_pos == 0:
            self.net.setLowerBound(param, mid)
        else:
            self.net.setUpperBound(param,mid)
        isSat, _ = self.solve()
        if not isSat:
            new_lb = mid
            self.lower_bounds[dim,neg_or_pos] = new_lb
        else:
            # TODO improve by observing counter example
            self.upper_bounds[dim,neg_or_pos] = mid
        if neg_or_pos == 0:
            self.net.setLowerBound(param, lb)
        else:
            self.net.setUpperBound(param,lb)

    def calc_box_size(self, optimistic=False):
        bounds = self.upper_bounds if optimistic else self.lower_bounds
        sizes = bounds[:, 1] - bounds[:, 0]
        return np.prod(sizes)

    def record_stats(self):
        self.box_size_optimistic_list.append(self.calc_box_size(True))
        self.box_size_pessimistic_list.append(self.calc_box_size(False))
        

    def step(self):
        dim = self.rng.integers(0,self.n_in_vars)
        self.step_bound(dim, 0)
        self.step_bound(dim, 1)
        


    def tighten(self) -> np.ndarray:
        for i in trange(self.max_iterations, disable=not self.verbose):
            self.step()
            self.record_stats()
        return self.lower_bounds

    