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
    SnoptSolver,
    GurobiSolver,
    SolverId
)
from pydrake.symbolic import Polynomial, Variable, Variables, Evaluate
from sdp import create_sdp_relaxation, _get_sol_from_svd

from pydrake.math import eq, le, ge

from util import timeit, YAY, WARN, INFO, ERROR, diditwork

import numpy as np
from pydrake.all import  DiagramBuilder 
from pydrake.solvers import MathematicalProgram, Solve

from underactuated.quadrotor2d import Quadrotor2D, Quadrotor2DVisualizer


def make_a_nonconvex_birotor_program(N:int, desired_pos:npt.NDArray = np.array([2,0]), dt:float = 0.2, for_sdp_solver:bool = False, get_vars = False):
    builder = DiagramBuilder()
    plant = builder.AddSystem(Quadrotor2D())
    # define constants
    thrust_2_mass_ratio = 3 # 3:1 thrust : mass ratio
    r = plant.length
    m = plant.mass
    I = plant.inertia
    g = plant.gravity

    Q = np.diag([10, 10, 10,  1, 1, 20,   1, 1,1,1 ])
    Qf = Q
    R = np.array([[0.1, 0.05], [0.05, 0.1]])

    prog = MathematicalProgram()
    # generate state optimization variables
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

    # "dynamics" of cos^2 + sin^2 = 1
    for n in range(1, N+1):
        prog.AddConstraint( s[n]*ds[n] + c[n]*dc[n] == 0)

    # add bounding constraints -- nonconvex solver needs them else it swears
    if not for_sdp_solver:
        prog.AddBoundingBoxConstraint(-5*np.ones(N),  5*np.ones(N),  x[1:])
        prog.AddBoundingBoxConstraint(-5*np.ones(N),  5*np.ones(N),  y[1:])
        prog.AddBoundingBoxConstraint( -np.pi*np.ones(N),  np.pi*np.ones(N),  th[1:])
        prog.AddBoundingBoxConstraint( -10*np.ones(N),  10*np.ones(N),  dx[1:])
        prog.AddBoundingBoxConstraint( -10*np.ones(N),  10*np.ones(N),  dy[1:])
        prog.AddBoundingBoxConstraint( -np.pi*np.ones(N),  np.pi*np.ones(N),  dth[1:])

        prog.AddBoundingBoxConstraint(-1*np.ones(N),  1.03*np.ones(N),  c[1:])
        prog.AddBoundingBoxConstraint(-1*np.ones(N),  1.03*np.ones(N),  s[1:])
        prog.AddBoundingBoxConstraint(-1*np.ones(N),  1.03*np.ones(N),  dc[1:])
        prog.AddBoundingBoxConstraint(-1*np.ones(N),  1.03*np.ones(N),  ds[1:])
        # lower and upper bounds on control inputs
        prog.AddBoundingBoxConstraint(0, m*g/2 * thrust_2_mass_ratio, v)
        prog.AddBoundingBoxConstraint(0, m*g/2 * thrust_2_mass_ratio, w)

    # dynamics
    for n in range(N):
        prog.AddLinearEqualityConstraint( x[n+1] == x[n] + dt * dx[n]  )
        prog.AddLinearEqualityConstraint( y[n+1] == y[n] + dt * dy[n] )
        prog.AddLinearEqualityConstraint( th[n+1] == th[n] + dt * dth[n] )
        prog.AddConstraint( c[n+1] == c[n] + dt * dc[n] )
        prog.AddConstraint( s[n+1] == s[n] + dt * ds[n] )
        # use drag dynamics
        drag = 0.15 / dt
        ddth_n = (v[n] - w[n]) * r / I - drag * dth[n]
        prog.AddLinearEqualityConstraint( 0 == -dth[n+1] + dth[n]  + dt * ddth_n ) # - dth[n]*dt*drag  )
        # prog.AddLinearEqualityConstraint( dth[n+1] == dth[n]*(1-drag)  + dt * (v[n] - w[n]) * r / I) # - dth[n]*dt*drag  )
        # prog.AddLinearEqualityConstraint( dth[n+1] == dth[n] + dt * (v[n] - w[n]) * r / I )
        # quadratic constraints
        prog.AddConstraint( dx[n+1] == dx[n] + dt * (-(v[n] + w[n]) * s[n] / m) )
        prog.AddConstraint( dy[n+1] == dy[n] + dt * ( (v[n] + w[n]) * c[n] / m - g) )

        prog.AddConstraint( dc[n+1] == dc[n] - dt * ( ddth_n*s[n] + dth[n]*ds[n] ) )
        prog.AddConstraint( ds[n+1] == ds[n] + dt * ( ddth_n*c[n] + dth[n]*dc[n] ) )
        
        # prog.AddConstraint( dc[n+1] == dc[n] - dt * ( r/I*(v[n]-w[n])*s[n] + dth[n]*ds[n] ) )
        # prog.AddConstraint( ds[n+1] == ds[n] + dt * ( r/I*(v[n]-w[n])*c[n] + dth[n]*dc[n] ) )


    # desired point
    z_star = np.hstack( (desired_pos, np.array([0, 0,0,0, 1,0, 0,0]) ))
    u_star = m * g / 2.0 * np.array([1, 1])

    # build quadratic cost
    cost = 0
    for i in range(N):
        cost = cost + (z[i]-z_star).dot(Q).dot(z[i]-z_star) + (u[i]-u_star).dot(R).dot(u[i]-u_star)
    # add final cost
    cost += (z[N]-z_star).dot(Qf).dot(z[N]-z_star)
    prog.AddCost(cost)

    if not get_vars:
        return prog
    else: 
        return prog, (x,y,th,dx,dy,dth,c,s,dc,ds,v,w)
    

def make_a_warmstart_from_initial_condition_and_control_sequence( v,w, N:int, x0:npt.NDArray = np.array([0,0,0, 0,0,0, 1,0, 0,0]), dt:float=0.2 ):
    assert 1 <= len(v) <= N
    assert len(v) == len(w)
    while len(v) < N:
        v = np.hstack((v, v[-1]))
        w = np.hstack((w, w[-1]))
    
    builder = DiagramBuilder()
    plant = builder.AddSystem(Quadrotor2D())
    # define constants
    r = plant.length
    m = plant.mass
    I = plant.inertia
    g = plant.gravity

    x = np.array([x0[0]] + [0] * (N))
    y = np.array([x0[1]] + [0] * (N) )
    th = np.array([x0[2]] + [0] * (N) )
    dx = np.array([x0[3]] + [0] * (N) )
    dy = np.array([x0[4]] + [0] * (N) )
    dth = np.array([x0[5]] + [0] * (N) )
    c = np.array([x0[6]] + [0] * (N) )
    s = np.array([x0[7]] + [0] * (N) )
    dc = np.array([x0[8]] + [0] * (N) )
    ds = np.array([x0[9]] + [0] * (N) )
    for n in range(N):
        print(v[n], w[n])
        x[n+1] = x[n] + dt * dx[n]
        y[n+1] = y[n] + dt * dy[n]
        th[n+1] = th[n] + dt * dth[n]
        # use drag dynamics
        drag = 0.15
        dth[n+1] = dth[n]*(1-drag) + dt * (v[n] - w[n]) * r / I
        # prog.AddLinearEqualityConstraint( dth[n+1] == dth[n] + dt * (v[n] - w[n]) * r / I )
        # quadratic constraints
        dx[n+1] = dx[n] + dt * (-(v[n] + w[n]) * s[n] / m)
        dy[n+1] = dy[n] + dt * ( (v[n] + w[n]) * c[n] / m - g)
        c[n+1] = c[n] + dt * dc[n]
        s[n+1] = s[n] + dt * ds[n]
        dc[n+1] = dc[n] - dt * ( r/I*(v[n]-w[n])*s[n] + dth[n]*ds[n] )
        ds[n+1] = ds[n] + dt * ( r/I*(v[n]-w[n])*c[n] + dth[n]*dc[n] )

    print("x", x)
    print("y", y)
    print("th", th)
    print("dx", dx)
    print("dy", dy)
    print("dth", dth)
    print("c", c)
    print("s", s)
    print("dc", dc)
    print("ds", ds)
    print("v", v)
    print("w", w)
    
    return np.hstack((x[1:],y[1:],th[1:],dx[1:],dy[1:],dth[1:],c[1:],s[1:],dc[1:],ds[1:],v,w))

def make_nonlinear_warmstart_from_small_solution(big_N, small_N, desired_pos:npt.NDArray = np.array([2,0]), dt:float = 0.2):
    prog, (x,y,th,dx,dy,dth,c,s,dc,ds,v,w) = make_a_nonconvex_birotor_program(small_N, desired_pos, dt, False, get_vars=True)
    solution = Solve(prog)
    assert solution.is_success(), ERROR("small problem didn't solve, use smaller N")
    INFO("Small program got solved")
    v_sol = solution.GetSolution(v).reshape(small_N)
    w_sol = solution.GetSolution(w).reshape(small_N)

    x0 = np.array([0,0,0, 0,0,0, 1,0, 0,0])
    warmstart = make_a_warmstart_from_initial_condition_and_control_sequence(v_sol, w_sol, x0, big_N, dt)
    return warmstart

def make_interpolation_init(N, desired_pos:npt.NDArray = np.array([2,0])):
    res = dict()
    res["x"] = np.linspace(0,desired_pos[0],N)
    res["y"] = np.linspace(0,desired_pos[1],N)
    res["th"] = np.zeros(N)
    res["dx"] = np.zeros(N)
    res["dy"] = np.zeros(N)
    res["dth"] = np.zeros(N)
    res["c"] = np.ones(N)
    res["s"] = np.zeros(N)
    res["dc"] = np.zeros(N)
    res["ds"] = np.zeros(N)
    res["v"] = np.zeros(N)
    res["w"] = np.zeros(N)
    return res

def evalaute_square_feasibility_violation(res, N):
    for n in range(N):
        


def solve_nonconvex_birotor(N:int, desired_pos:npt.NDArray = np.array([2,0]), dt:float = 0.2, warmstart = None):
    prog, (x,y,th,dx,dy,dth,c,s,dc,ds,v,w) = make_a_nonconvex_birotor_program(N, desired_pos, dt, False, get_vars=True)
    INFO("Program built.")
    
    if warmstart is not None:
        res = warmstart
        prog.SetInitialGuess(x[1:], res["x"])
        prog.SetInitialGuess(y[1:], res["y"])
        prog.SetInitialGuess(th[1:], res["th"])
        prog.SetInitialGuess(dx[1:], res["dx"])
        prog.SetInitialGuess(dy[1:], res["dy"])
        prog.SetInitialGuess(dth[1:], res["dth"])
        prog.SetInitialGuess(c[1:], res["c"])
        prog.SetInitialGuess(s[1:], res["s"])
        prog.SetInitialGuess(dc[1:], res["dc"])
        prog.SetInitialGuess(ds[1:], res["ds"])
        prog.SetInitialGuess(v, res["v"])
        prog.SetInitialGuess(w, res["w"])

    INFO("Solving ", N)
    timer = timeit()
    solver = SnoptSolver()
    # solver = IpoptSolver()
    solution = solver.Solve(prog)
    timer.dt()
    diditwork(solution)
    
    if solution.is_success():
        print(solution.get_optimal_cost())

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
        INFO( "v", np.round(solution.GetSolution(v),3) )
        INFO( "w",  np.round(solution.GetSolution(w),3) )
    return prog, solution

if __name__ == "__main__":    
    solve_nonconvex_birotor(12)
    print("----")
    solve_nonconvex_birotor(12, warmstart = make_interpolation_init(12))
    # solve_nonconvex_birotor(16)