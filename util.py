from colorama import Fore
import typing as T
import time


def ERROR(*texts, verbose: bool = True):
    if verbose:
        print(Fore.RED + " ".join([str(text) for text in texts]))


def WARN(*texts, verbose: bool = True):
    if verbose:
        print(Fore.YELLOW + " ".join([str(text) for text in texts]))


def INFO(*texts, verbose: bool = True):
    if verbose:
        print(Fore.BLUE + " ".join([str(text) for text in texts]))


def YAY(*texts, verbose: bool = True):
    if verbose:
        print(Fore.GREEN + " ".join([str(text) for text in texts]))


def all_possible_combinations_of_items(item_set: T.List[str], num_items: int):
    """
    Recursively generate a set of all possible ordered strings of items of length num_items.
    """
    if num_items == 0:
        return [""]
    result = []
    possible_n_1 = all_possible_combinations_of_items(item_set, num_items - 1)
    for item in item_set:
        result += [item + x for x in possible_n_1]
    return result


class timeit:
    def __init__(self):
        self.times = []
        self.times.append(time.time())
        self.totals = 0
        self.a_start = None

    def dt(self, descriptor=None, print_stuff=True):
        self.times.append(time.time())
        if print_stuff:
            if descriptor is None:
                INFO("%.3fs since last time-check" % (self.times[-1] - self.times[-2]))
            else:
                descriptor = str(descriptor)
                INFO(descriptor + " took %.3fs" % (self.times[-1] - self.times[-2]))
        return self.times[-1] - self.times[-2]

    def T(self, descriptor=None):
        self.times.append(time.time())
        if descriptor is None:
            INFO("%.3fs since the start" % (self.times[-1] - self.times[0]))
        else:
            INFO(
                descriptor
                + " took %.3fs since the start" % (self.times[-1] - self.times[0])
            )

    def start(self):
        self.a_start = time.time()

    def end(self):
        self.totals += time.time() - self.a_start
        self.a_start = None

    def total(self, descriptor=None):
        INFO("All " + descriptor + " took %.3fs" % (self.totals))


from pydrake.solvers import MathematicalProgram
from pydrake.solvers import Solve
import numpy as np
import matplotlib.pyplot as plt

from pydrake.geometry.optimization import HPolyhedron


def ChebyshevCenter(poly: HPolyhedron):
    # Ax <= b
    m = poly.A().shape[0]
    n = poly.A().shape[1]

    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(n, "x")
    r = prog.NewContinuousVariables(1, "r")
    prog.AddLinearCost(np.array([-1]), 0, r)

    big_num = 100000

    prog.AddBoundingBoxConstraint(0, big_num, r)

    a = np.zeros((1, n + 1))
    for i in range(m):
        a[0, 0] = np.linalg.norm(poly.A()[i, :])
        a[0, 1:] = poly.A()[i, :]
        prog.AddLinearConstraint(
            a, -np.array([big_num]), np.array([poly.b()[i]]), np.append(r, x)
        )

    result = Solve(prog)
    if not result.is_success():
        return False, None, None
    else:
        return True, result.GetSolution(x), result.GetSolution(r)[0]
