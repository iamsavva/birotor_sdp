import typing as T

import numpy as np
import numpy.typing as npt

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

from util import timeit, YAY, WARN, INFO, ERROR, diditwork


import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import  DiagramBuilder 
from pydrake.solvers import MathematicalProgram, Solve

# from underactuated import ConfigureParser, running_as_notebook
from underactuated.quadrotor2d import Quadrotor2D, Quadrotor2DVisualizer

from nonlinear_birotor import make_a_nonconvex_birotor_program, solve_nonconvex_birotor, evalaute_square_feasibility_violation, make_interpolation_init

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

def solve_sdp_birotor(N:int, desired_pos:npt.NDArray = np.array([2,0]), dt:float = 0.2, multiply_equality_constraints = False, evaluation = False):
    prog = make_a_nonconvex_birotor_program(N, desired_pos, dt, True)

    timer = timeit()
    relaxed_prog, X, basis = create_sdp_relaxation(prog, multiply_equality_constraints=multiply_equality_constraints, sample_random_equality_constraints=False, sample_percentage=0.50)

    timer.dt("SDP generation")


    relaxed_solution = Solve(relaxed_prog)
    sdp_solve_time = timer.dt("SDP solving")
    if not evaluation:
        diditwork(relaxed_solution)
    X_val = relaxed_solution.GetSolution(X)
    eigenvals, _ = np.linalg.eig(X_val)
    INFO("Matrix shape", X_val.shape)
    INFO("Matrix rank", np.sum(eigenvals>1e-3))
    res = get_solution_from_X(N, X_val, verbose=not evaluation)
    
    INFO("--------", verbose = not evaluation)

    solve_nonconvex_birotor(N, desired_pos = desired_pos, warmstart=res, evaluation=evaluation)
    YAY("-----------")

    if evaluation:
        violation = evalaute_square_feasibility_violation(res, N, dt)
        INFO("SDP solved:", relaxed_solution.is_success())
        INFO("SDP time:", sdp_solve_time)
        INFO("SDP cost:  ", relaxed_solution.get_optimal_cost())
        INFO("SDP error: ", violation)
        print("x", np.round(res["x"],3) )
        print("y", np.round(res["y"],3))
        INFO("--------")
    return X_val, res

    
if __name__ == "__main__":
    # desired_pos = np.array([2,0, 2*np.pi])
    # desired_pos = np.array([0,-2, 2*np.pi])
    desired_pos = np.array([3,3,0])
    N = 14
    solve_nonconvex_birotor(N, desired_pos = desired_pos, warmstart = make_interpolation_init(N), evaluation=True)
    YAY("-----------")
    solve_sdp_birotor(N, desired_pos = desired_pos, multiply_equality_constraints=True, evaluation=True)