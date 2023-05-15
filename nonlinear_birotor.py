import typing as T

import numpy as np
import numpy.typing as npt
import scipy as sp

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

if running_as_notebook:
    mpld3.enable_notebook()



def make_a_nonconvex_birotor_program(N:int, desired_pos:npt.NDArray = np.array([2,0]), dt:float = 0.2, for_sdp_solver:bool = False):
    builder = DiagramBuilder()
    plant = builder.AddSystem(Quadrotor2D())

    thrust_2_mass_ratio = 3 # 3:1 thrust : mass ratio
    r = plant.length
    m = plant.mass
    I = plant.inertia
    g = plant.gravity

    Q = np.diag([10, 10, 10,  1, 1, 20,   1, 1,1,1 ])
    Qf = Q
    R = np.array([[0.1, 0.05], [0.05, 0.1]])

    prog = MathematicalProgram()
    x = np.hstack( ([0], prog.NewContinuousVariables(N, "x")))
    y = np.hstack( ([0], prog.NewContinuousVariables(N, "y")))
    th = np.hstack( ([0], prog.NewContinuousVariables(N, "th")))
    dx = np.hstack( ([0], prog.NewContinuousVariables(N, "dx")))
    dy = np.hstack( ([0], prog.NewContinuousVariables(N, "dy")))
    dth = np.hstack( ([0], prog.NewContinuousVariables(N, "dth")))
    c = np.hstack( ([1], prog.NewContinuousVariables(N, "c")))
    s = np.hstack( ([0], prog.NewContinuousVariables(N, "s")))
    dc = np.hstack( ([0], prog.NewContinuousVariables(N, "dc")))
    ds = np.hstack( ([0], prog.NewContinuousVariables(N, "ds")))
    # N timesteps for control inputs
    v = prog.NewContinuousVariables(N, "v")
    w = prog.NewContinuousVariables(N, "w")
    # full state and control vectors
    z = np.vstack( (x,y,th,dx,dy,dth,c,s,dc,ds)).T
    u = np.vstack( (v,w) ).T

    if for_sdp_solver:
        for n in range(1, N+1):
            prog.AddConstraint( s[n]**2 + c[n]**2 >= 0.99 )
            prog.AddConstraint( s[n]**2 + c[n]**2 <= 1.05 )
    else:
        for n in range(1, N+1):
            prog.AddConstraint( s[n]**2 + c[n]**2 >= 0.85 )
            prog.AddConstraint( s[n]**2 + c[n]**2 <= 1.1 )
        # if solving with nonlinear solver -- relax cos^2 + sin^2 = 1, add bounding boxes
        prog.AddBoundingBoxConstraint(-5*np.ones(N),  5*np.ones(N),  x[1:])
        prog.AddBoundingBoxConstraint(-5*np.ones(N),  5*np.ones(N),  y[1:])
        prog.AddBoundingBoxConstraint( -np.pi*np.ones(N),  np.pi*np.ones(N),  th[1:])
        prog.AddBoundingBoxConstraint( -10*np.ones(N),  10*np.ones(N),  dx[1:])
        prog.AddBoundingBoxConstraint( -10*np.ones(N),  10*np.ones(N),  dy[1:])
        prog.AddBoundingBoxConstraint( -np.pi*np.ones(N),  np.pi*np.ones(N),  dth[1:])

    # lower and upper bounds on control inputs
    prog.AddBoundingBoxConstraint(0, m*g/2 * thrust_2_mass_ratio, v)
    prog.AddBoundingBoxConstraint(0, m*g/2 * thrust_2_mass_ratio, w)

    # dynamics
    for n in range(N):
        prog.AddLinearEqualityConstraint( x[n+1] == x[n] + dt * dx[n]  )
        prog.AddLinearEqualityConstraint( y[n+1] == y[n] + dt * dy[n] )
        prog.AddLinearEqualityConstraint( th[n+1] == th[n] + dt * dth[n] )
        prog.AddLinearEqualityConstraint( dth[n+1] == dth[n] + dt * (v[n] - w[n]) * r / I )
        # quadratic constraints
        prog.AddConstraint( dx[n+1] == dx[n] + dt * (-(v[n] + w[n]) * s[n] / m) )
        prog.AddConstraint( dy[n+1] == dy[n] + dt * ( (v[n] + w[n]) * c[n] / m - g) )
        prog.AddConstraint( c[n+1] == c[n] + dt * dc[n] )
        prog.AddConstraint( s[n+1] == s[n] + dt * ds[n] )
        prog.AddConstraint( dc[n+1] == dc[n] - dt * ( r/I*(v[n]-w[n])*s[n] + dth[n]*ds[n] ) )
        prog.AddConstraint( ds[n+1] == ds[n] + dt * ( r/I*(v[n]-w[n])*c[n] + dth[n]*dc[n] ) )


    # desired point
    z_star = np.hstack( (desired_pos, np.array([0, 0,0,0, 1,0, 0,0]) ))
    u_star = m * g / 2.0 * np.array([1, 1])

    # build the cost
    cost = 0
    for i in range(N):
        cost = cost + (z[i]-z_star).dot(Q).dot(z[i]-z_star) + (u[i]-u_star).dot(R).dot(u[i]-u_star)
    cost += (z[N]-z_star).dot(Qf).dot(z[N]-z_star)
    prog.AddCost(cost)

    return prog


def solve_nonconvex_birotor(N:int, desired_pos:npt.NDArray = np.array([2,0]), dt:float = 0.2):
    prog = make_a_nonconvex_birotor_program(N, desired_pos, dt, False)
    INFO("Program built.")

    timer = timeit()
    solution = Solve(prog)
    timer.dt("Program solved.")
    
    if not solution.is_success():
        ERROR("solve failed")
    else:
        YAY("solved!")
    print(solution.get_solver_id())
    print(solution.get_optimal_cost())
    print(solution.get_solution_result())

    INFO( "x", np.round(solution.GetSolution(x[1:]),2) )
    INFO( "y",  np.round(solution.GetSolution(y[1:]),2) )
    INFO( "th", np.round(solution.GetSolution(th[1:]),2) )
    print("---")
    INFO( "dx", np.round(solution.GetSolution(dx[1:]),2) )
    INFO( "dy",  np.round(solution.GetSolution(dy[1:]),2) )
    INFO( "dth", np.round(solution.GetSolution(dth[1:]),2) )
    print("---")
    INFO( "c", np.round(solution.GetSolution(c[1:]),2) )
    INFO( "s",  np.round(solution.GetSolution(s[1:]),2) )
    print("---")
    INFO( "dc", np.round(solution.GetSolution(dc[1:]),2) )
    INFO( "ds",  np.round(solution.GetSolution(ds[1:]),2) )
    cc = np.round(solution.GetSolution(c[1:]),2) ** 2
    ss = np.round(solution.GetSolution(s[1:]),2) ** 2
    INFO( "c2+s2", cc+ss)
    print("---")
    INFO( "v", np.round(solution.GetSolution(v),2) )
    INFO( "w",  np.round(solution.GetSolution(w),2) )
    return prog, solution