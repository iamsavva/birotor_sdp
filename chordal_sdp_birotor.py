import typing as T

import numpy as np
import numpy.typing as npt

import pydrake.symbolic as sym

from pydrake.solvers import (
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
    SolverOptions,
    CommonSolverOption, 
    IpoptSolver,
    GurobiSolver
)

from pydrake.math import eq, le, ge

from util import timeit, YAY, WARN, INFO, ERROR, diditwork

import numpy as np
from pydrake.all import  DiagramBuilder 
from pydrake.solvers import MathematicalProgram, Solve

# from underactuated import ConfigureParser, running_as_notebook
from underactuated.quadrotor2d import Quadrotor2D, Quadrotor2DVisualizer

from utilities import unit_vector
from nonlinear_birotor import make_a_nonconvex_birotor_program
from sdp_birotor import get_solution_from_X, solve_sdp_birotor
from sdp import (
    _collect_bounding_box_constraints, _quadratic_cost_binding_to_homogenuous_form, _quadratic_cost_binding_to_homogenuous_form, _linear_bindings_to_homogenuous_form,
    _generic_constraint_bindings_to_polynomials, _assert_max_degree, _construct_symmetric_matrix_from_triang, _generic_constraint_binding_to_polynomials,
    _get_monomial_coeffs, _get_sol_from_svd, _linear_binding_to_expressions, _quadratic_polynomial_to_homoenuous_form, 
    _generic_constraint_bindings_to_polynomials, add_constraints_to_psd_mat_from_prog
)
from nonlinear_birotor import make_a_warmstart_from_initial_condition_and_control_sequence, solve_nonconvex_birotor, DRAG_COEF, evalaute_square_feasibility_violation, make_interpolation_init
from nonlinear_birotor import Q, Qf, R, thrust_2_mass_ratio

def make_chordal_sdp_relaxation(N:int, state_dim:int, control_dim:int, prog: MathematicalProgram, boundary_conditions:T.Tuple[npt.NDArray, npt.NDArray, npt.NDArray], multiply_equality_constraints:bool = True):
    # ----------------------------------------------------------------------
    # define the variables
    Qf, z_star, z0 = boundary_conditions
    z_star = z_star.reshape(state_dim, 1)

    psd_mat_dim = 1 + state_dim + control_dim + state_dim

    # generating PSD matrices
    #  --------------------------------
    # | 1       z_n.T     u_n.T   z_n+1.T | 
    # | z_n      X          X        X    |
    # | u_n      X          X        X    |--------- 
    # |z_n+1     X          ZU       Z     
    # |--------------------------|          
    #                            |  
    #                            |
    
    relaxed_prog = MathematicalProgram()

    # one vector
    one = relaxed_prog.NewContinuousVariables(1, "1").reshape(1,1)
    # First variable is 1
    relaxed_prog.AddLinearConstraint(one[0,0] == 1)  

    # generate all the vectors and matrices
    zn = []
    un = []
    zn_zn = []
    zn_un = []
    zn_zn1 = []
    un_un = []
    un_zn1 = []

    # initial state zo
    z0 = z0.reshape(state_dim, 1)
    z0_z0 = z0 @ z0.T
    zn.append(z0)
    zn_zn.append(z0_z0)

    # add state and control inputs
    for n in range(N):
        # z_n
        zn_name = "z"+str(n+1)
        zn.append( relaxed_prog.NewContinuousVariables(state_dim, zn_name).reshape(state_dim, 1))
        # u_n
        un_name = "u"+str(n)
        un.append( relaxed_prog.NewContinuousVariables(control_dim, un_name).reshape(control_dim, 1))
        # z_n x z_n
        zn_zn_name = "z"+str(n+1)+"z"+str(n+1)
        zn_zn.append( relaxed_prog.NewSymmetricContinuousVariables(state_dim, zn_zn_name) )
        # u_n x u_n
        un_un_name = "u"+str(n)+"u"+str(n)
        un_un.append( relaxed_prog.NewSymmetricContinuousVariables(control_dim, un_un_name ) )
    
    # add product matrices
    for n in range(N):
        if n == 0:
            # make initial conditions constraint explicit
            z0 = zn[0]
            u0 = un[0]
            z1 = zn[1]
            zn_un.append( z0 @ u0.T )
            zn_zn1.append( z0 @ z1.T )
            un_zn1.append( relaxed_prog.NewContinuousVariables(control_dim, state_dim, "u0z1") )
        else:
            zn_un_name = "z" + str(n) + "u" + str(n)
            zn_zn1_name = "z" + str(n) + "z" + str(n+1)
            un_zn1_name = "u" + str(n) + "z" + str(n+1)
            zn_un.append( relaxed_prog.NewContinuousVariables(state_dim, control_dim, zn_un_name ) )
            zn_zn1.append( relaxed_prog.NewContinuousVariables(state_dim, state_dim, zn_zn1_name ) )
            un_zn1.append( relaxed_prog.NewContinuousVariables(control_dim, state_dim, un_zn1_name ) )

    # having formed the parts, form the psd matrices
    psd_mats = []
    for n in range(N):
        # construct the matrix
        row1 = np.hstack( (one, zn[n].T, un[n].T, zn[n+1].T) )
        row2 = np.hstack( (zn[n], zn_zn[n], zn_un[n], zn_zn1[n]) )
        row3 = np.hstack( (un[n], zn_un[n].T, un_un[n], un_zn1[n]) )
        row4 = np.hstack( (zn[n+1], zn_zn1[n].T, un_zn1[n].T, zn_zn[n+1]) )
        mat = np.vstack((row1, row2, row3, row4) )
        # make it PSD YEAAAAAH CHORDALITY BABY
        relaxed_prog.AddPositiveSemidefiniteConstraint(mat)
        psd_mats.append(mat)

    # ----------------------------------------------------------------------
    # great, let's add all the constraints
    # notice how all of the above would work for any 

    # constraints get added to each an every matrix
    for n in range(N):
        add_constraints_to_psd_mat_from_prog(prog, relaxed_prog, psd_mats[n], multiply_equality_constraints)

    # add final cost
    row1 = np.hstack( (one, zn[N].T) )
    row2 = np.hstack( (zn[N], zn_zn[N]))
    final_mat = np.vstack( (row1, row2) )

    cost_row1 = np.hstack( ( (z_star.T @ Qf @ z_star).reshape(1,1), -z_star.T @ Qf ) )
    cost_row2 = np.hstack( ( -Qf @ z_star, Qf ) )
    final_cost = np.vstack((cost_row1, cost_row2))
                          
    relaxed_prog.AddLinearCost( np.sum( final_mat * final_cost ) )

    return relaxed_prog, zn, un, psd_mats

def get_chordal_solution(solution, zn, un, horizon, verbose = True):
    x = []
    y = []
    th = []
    dx = []
    dy = []
    dth = []
    c = []
    s = []
    dc = []
    ds = []
    v = []
    w = []


    for n in range(horizon+1):
        if n != 0:
            x.append(solution.GetSolution(zn[n][0]))
            y.append(solution.GetSolution(zn[n][1]))
            th.append(solution.GetSolution(zn[n][2]))
            dx.append(solution.GetSolution(zn[n][3]))
            dy.append(solution.GetSolution(zn[n][4]))
            dth.append(solution.GetSolution(zn[n][5]))
            c.append(solution.GetSolution(zn[n][6]))
            s.append(solution.GetSolution(zn[n][7]))
            dc.append(solution.GetSolution(zn[n][8]))
            ds.append(solution.GetSolution(zn[n][9]))
        if n != horizon:
            v.append(solution.GetSolution(un[n][0]))
            w.append(solution.GetSolution(un[n][1]))
    res = dict()
    res["x"] = np.array(x).reshape(horizon)
    res["y"] = np.array(y).reshape(horizon)
    res["th"] = np.array(th).reshape(horizon)
    res["dx"] = np.array(dx).reshape(horizon)
    res["dy"] = np.array(dy).reshape(horizon)
    res["dth"] = np.array(dth).reshape(horizon)
    res["c"] = np.array(c).reshape(horizon)
    res["s"] = np.array(s).reshape(horizon)
    res["dc"] = np.array(dc).reshape(horizon)
    res["ds"] = np.array(ds).reshape(horizon)
    res["v"] = np.array(v).reshape(horizon)
    res["w"] = np.array(w).reshape(horizon)
    res["full-state"] = np.hstack((res["x"], res["y"], res["th"], res["dx"],res["dy"],res["dth"],res["c"],res["s"],res["dc"],res["ds"],res["v"], res["w"]))
    if verbose:
        for name in res.keys():
            r = 2
            if name in ("v","w"):
                r = 3
            YAY( name, np.round(res[name],r) )
    return res

def make_a_1_step_nonconvex_problem(N, desired_pos = np.array([2,0,0]), dt = 0.2):
    builder = DiagramBuilder()
    plant = builder.AddSystem(Quadrotor2D())
    # define constants
    r = plant.length
    m = plant.mass
    I = plant.inertia
    g = plant.gravity

    # Q = np.diag([10, 10, 10,  1, 1, 1,   1, 1,1,1 ])
    # Qf = Q
    # R = np.array([[0.1, 0.05], [0.05, 0.1]])

    # order in which we define the variables matters -- hence the odd repetition.
    # the logic behind N = 1 is to make chordal matrix gen easier
    N = 1
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, "x_n")
    y = prog.NewContinuousVariables(1, "y_n")
    th = prog.NewContinuousVariables(1, "th_n")
    dx = prog.NewContinuousVariables(1, "dx_n")
    dy = prog.NewContinuousVariables(1, "dy_n")
    dth = prog.NewContinuousVariables(1, "dth_n")
    c = prog.NewContinuousVariables(1, "c_n")
    s = prog.NewContinuousVariables(1, "s_n")
    dc = prog.NewContinuousVariables(1, "dc_n")
    ds = prog.NewContinuousVariables(1, "ds_n")
    # N timesteps for control inputs
    v = prog.NewContinuousVariables(1, "v_n")
    w = prog.NewContinuousVariables(1, "w_n")
    # N+1 staet
    x = np.hstack( (x, prog.NewContinuousVariables(1, "x_n1")))
    y = np.hstack( (y, prog.NewContinuousVariables(1, "y_n1") ))
    th = np.hstack( (th, prog.NewContinuousVariables(1, "th_n1") ))
    dx = np.hstack( (dx, prog.NewContinuousVariables(1, "dx_n1") ))
    dy =np.hstack( (dy,  prog.NewContinuousVariables(1, "dy_n1") ))
    dth = np.hstack( (dth, prog.NewContinuousVariables(1, "dth_n1") ))
    c = np.hstack( (c, prog.NewContinuousVariables(1, "c_n1") ))
    s = np.hstack( (s, prog.NewContinuousVariables(1, "s_n1") ))
    dc = np.hstack( (dc, prog.NewContinuousVariables(1, "dc_n1") ))
    ds =np.hstack( (ds,  prog.NewContinuousVariables(1, "ds_n1") ))
    # full state and control vectors
    z = np.vstack( (x,y,th,dx,dy,dth,c,s,dc,ds)).T
    u = np.vstack( (v,w) ).T
    # dimensions
    state_dim = 10
    control_dim = 2

    # dynamics of s^2 + c^2 = 1a
    prog.AddConstraint( s[1]*ds[1] + c[1]*dc[1] == 0)
    # prog.AddConstraint( v[0] <= )
    prog.AddBoundingBoxConstraint(0, m*g/2 * thrust_2_mass_ratio, v[0])
    prog.AddBoundingBoxConstraint(0, m*g/2 * thrust_2_mass_ratio, w[0])
    # prog.AddConstraint( c[1] <= 1.03)
    # linear dynamics
    prog.AddLinearEqualityConstraint( x[1] == x[0] + dt * dx[0]  ) 
    prog.AddLinearEqualityConstraint( y[1] == y[0] + dt * dy[0] ) 
    prog.AddLinearEqualityConstraint( th[1] == th[0] + dt * dth[0] ) 
    drag = DRAG_COEF / dt
    ddth_n = (v[0] - w[0]) * r / I - drag * dth[0]
    prog.AddLinearEqualityConstraint( dth[1] == dth[0] + dt * ddth_n ) # - dth[0]*dt*drag  )

    # prog.AddLinearEqualityConstraint( dth[1] == dth[0]*(1-drag) + dt * (v[0] - w[0]) * r / I) # - dth[0]*dt*drag  )
    # cons.append( prog.AddLinearEqualityConstraint( dth[1] == dth[0] + dt * (v[0] - w[0]) * r / I ) )
    # quadratic dynamics
    prog.AddConstraint( dx[1] == dx[0] + dt * (-(v[0] + w[0]) * s[0] / m) ) 
    prog.AddConstraint( dy[1] == dy[0] + dt * ( (v[0] + w[0]) * c[0] / m - g) ) 
    prog.AddLinearEqualityConstraint( c[1] == c[0] + dt * dc[0] ) 
    prog.AddLinearEqualityConstraint( s[1] == s[0] + dt * ds[0] ) 
    prog.AddConstraint( dc[1] == dc[0] - dt * ( ddth_n*s[0] + dth[0]*ds[0] ) ) 
    prog.AddConstraint( ds[1] == ds[0] + dt * ( ddth_n*c[0] + dth[0]*dc[0] ) ) 
    # boundary conditions
    th_star = desired_pos[2]
    z_star = np.hstack( (desired_pos, np.array([0,0,0, np.cos(th_star),np.sin(th_star), 0,0]) ))
    u_star = m * g / 2.0 * np.array([1, 1])
    z_0 = np.array( [0,0,0, 0,0,0, 1,0, 0,0] )
    boundary_conditions = Qf, z_star, z_0 # need for final cost / initial conditions
    
    # build the cost
    cost = 0
    for i in range(N):
        cost = cost + (z[i]-z_star).dot(Q).dot(z[i]-z_star) + (u[i]-u_star).dot(R).dot(u[i]-u_star)
    prog.AddCost(cost)
    return state_dim, control_dim, prog, boundary_conditions

def make_chordal_sdp_program(N, desired_pos = np.array([2,0,0]), dt = 0.2, evaluation= False):
    state_dim, control_dim, prog, boundary_conditions = make_a_1_step_nonconvex_problem(N,desired_pos,dt)

    timer = timeit()
    relaxed_prog, zn, un, psd_mats = make_chordal_sdp_relaxation( N, state_dim, control_dim, prog, boundary_conditions)

    timer.dt("making the chordal sdp", verbose = not evaluation)
    solution = Solve(relaxed_prog)
    CHORDAL_solve_time = timer.dt("solving the chordal sdp", verbose = not evaluation)

    if not evaluation:
        diditwork(solution)

    res = get_chordal_solution(solution, zn, un, N, verbose= not evaluation)
    INFO("--------", verbose = not evaluation)

    solve_nonconvex_birotor(N, dt = dt, desired_pos=desired_pos, warmstart=res, evaluation=evaluation)

    if evaluation:
        violation = evalaute_square_feasibility_violation(res, N, dt)
        # INFO("CHORDAL solved:", solution.is_success())
        # INFO("CHORDAL time:", CHORDAL_solve_time)
        # INFO("CHORDAL cost:  ", solution.get_optimal_cost())
        # INFO("CHORDAL error: ", violation)
        # print("th", res["th"])
        # print("v", res["v"])
        # print("w", res["w"])
        # INFO("--------")

    
if __name__ == "__main__":
    desired_pos = np.array([0,0, 2*np.pi])
    N = 17
    dt = 0.1
    solve_nonconvex_birotor(N, dt=dt,desired_pos = desired_pos, warmstart = make_interpolation_init(N), evaluation=True)
    make_chordal_sdp_program(N, dt=dt,desired_pos = desired_pos, evaluation=True)
    # solve_sdp_birotor(N, dt=dt, desired_pos = desired_pos, multiply_equality_constraints=False, evaluation=True)