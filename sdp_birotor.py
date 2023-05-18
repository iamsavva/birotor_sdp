import typing as T

import numpy as np
import numpy.typing as npt
# import scipy as sp

from matplotlib import pyplot as plt
import matplotlib.patches as patches

from pydrake.solvers import (
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
    SolverOptions,
    CommonSolverOption, 
    IpoptSolver,
    GurobiSolver
)
from pydrake.symbolic import Polynomial, Variable, Variables, Evaluate
from sdp import create_sdp_relaxation, _get_sol_from_svd

from pydrake.math import eq, le, ge

from util import timeit, YAY, WARN, INFO, ERROR

import math

import matplotlib.pyplot as plt
import mpld3
import numpy as np
from IPython.display import HTML, display
from pydrake.all import  DiagramBuilder 
from pydrake.solvers import MathematicalProgram, Solve

# from underactuated import ConfigureParser, running_as_notebook
from underactuated import running_as_notebook
from underactuated.quadrotor2d import Quadrotor2D, Quadrotor2DVisualizer

from nonlinear_birotor import make_a_nonconvex_birotor_program, solve_nonconvex_birotor

def get_solution_from_X(N, X, verbose = False):
    builder = DiagramBuilder()
    plant = builder.AddSystem(Quadrotor2D())
    x_vec = np.real(_get_sol_from_svd(X))[1:]
    res = dict()
    N1 = N
    res["x"] = x_vec[:N1]
    res["y"] = x_vec[N1:2*N1]
    res["th"] = x_vec[2*N1:3*N1]

    res["dx"] = x_vec[3*N1:4*N1]
    res["dy"] = x_vec[4*N1:5*N1]
    res["dth"] = x_vec[5*N1:6*N1]

    res["c"] = x_vec[6*N1:7*N1]
    res["s"] = x_vec[7*N1:8*N1]

    res["dc"] = x_vec[8*N1:9*N1]
    res["ds"] = x_vec[9*N1:10*N1]

    res["v"] = x_vec[10*N1:10*N1+N]
    res["w"] = x_vec[10*N1+N:10*N1+2*N]
    if verbose:
        for name in res.keys():
            r = 2
            if name in ("v","w"):
                r = 3
            YAY( name, np.round(res[name],r) )
    return res

def solve_sdp_birotor(N:int, desired_pos:npt.NDArray = np.array([2,0]), dt:float = 0.2, multiply_equality_constraints = False):
    prog = make_a_nonconvex_birotor_program(N, desired_pos, dt, True)

    timer = timeit()
    relaxed_prog, X, basis = create_sdp_relaxation(prog, multiply_equality_constraints=multiply_equality_constraints, sample_random_equality_constraints=False, sample_percentage=0.2)

    timer.dt("SDP generation")
    relaxed_solution = Solve(relaxed_prog)
    timer.dt("SDP solving")
    print("---")
    print(relaxed_solution.is_success())
    print(relaxed_solution.get_optimal_cost())
    print(relaxed_solution.get_solution_result())
    X_val = relaxed_solution.GetSolution(X)
    eigenvals, _ = np.linalg.eig(X_val)    
    print("matrix shape", X_val.shape)
    print("Matrix rank", np.sum(eigenvals>1e-4))
    res = get_solution_from_X(N, X_val, verbose=True)

    print("---")

    # warmstart = make_a_warmstart_from_initial_condition_and_control_sequence(res["v"], res["w"], horizon)

    # solve_nonconvex_birotor(horizon, warmstart=res["full-state"])
    solve_nonconvex_birotor(N, warmstart=res)

    # return relaxed_prog, relaxed_solution, res

    
if __name__ == "__main__":
    solve_sdp_birotor(12, multiply_equality_constraints=False)