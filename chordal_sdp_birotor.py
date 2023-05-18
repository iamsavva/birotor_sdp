import typing as T

import numpy as np
import numpy.typing as npt
import scipy as sp

from matplotlib import pyplot as plt
import matplotlib.patches as patches

import pydrake.symbolic as sym
from utilities import unit_vector

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

from nonlinear_birotor import make_a_nonconvex_birotor_program

from sdp_birotor import get_solution_from_X, solve_sdp_birotor

from sdp import (
    _collect_bounding_box_constraints, _quadratic_cost_binding_to_homogenuous_form, _quadratic_cost_binding_to_homogenuous_form, _linear_bindings_to_homogenuous_form,
    _generic_constraint_bindings_to_polynomials, _assert_max_degree, _construct_symmetric_matrix_from_triang, _generic_constraint_binding_to_polynomials,
    _get_monomial_coeffs, _get_sol_from_svd, _linear_binding_to_expressions, _quadratic_polynomial_to_homoenuous_form, 
    _generic_constraint_bindings_to_polynomials,
)

def add_constraints_to_psd_mat_from_prog(prog:MathematicalProgram, relaxed_prog:MathematicalProgram, X:npt.NDArray, multiply_equality_constraints:bool):
    DEGREE_QUADRATIC = 2  # We are only relaxing (non-convex) quadratic programs
    
    decision_vars = np.array( sorted(prog.decision_variables(), key=lambda x: x.get_id()) )
    num_vars = ( len(decision_vars) + 1 )
    assert X.shape == (num_vars, num_vars) # else something is off

    basis = np.flip(sym.MonomialBasis(decision_vars, DEGREE_QUADRATIC))
    
    bounding_box_eqs, bounding_box_ineqs = _collect_bounding_box_constraints(
        prog.bounding_box_constraints()
    )

    has_linear_costs = len(prog.linear_costs()) > 0
    if has_linear_costs:
        raise NotImplementedError("Linear costs not yet implemented!")

    has_quadratic_costs = len(prog.quadratic_costs()) > 0
    if has_quadratic_costs:
        quadratic_costs = prog.quadratic_costs()
        Q_cost = [
            _quadratic_cost_binding_to_homogenuous_form(c, basis, num_vars)
            for c in quadratic_costs
        ]
        for Q in Q_cost:
            c = np.trace(Q.dot(X))
            relaxed_prog.AddCost(c)

    has_linear_eq_constraints = (
        len(prog.linear_equality_constraints()) > 0 or len(bounding_box_eqs) > 0
    )
    A_eq = None
    if has_linear_eq_constraints:
        A_eq = _linear_bindings_to_homogenuous_form(
            prog.linear_equality_constraints(), bounding_box_eqs, decision_vars
        )
        m,_ = A_eq.shape
        I = np.eye(num_vars)
        j = 0
        if multiply_equality_constraints:
            num_cons = num_vars
        else:
            num_cons = 1

        np.random.seed(1)
        for a in A_eq:
            for i in range(num_cons):
                j += 1
                A = np.outer(a, I[i])
                relaxed_prog.AddLinearConstraint( np.sum( X * ( A + A.T ) ) == 0 )

    has_linear_ineq_constraints = (
        len(prog.linear_constraints()) > 0 or len(bounding_box_ineqs) > 0
    )
    A_ineq = None
    if has_linear_ineq_constraints:
        A_ineq = _linear_bindings_to_homogenuous_form(
            prog.linear_constraints(), bounding_box_ineqs, decision_vars
        )
        m,_ = A_ineq.shape

        multiplied_constraints = ge(A_ineq.dot(X).dot(A_ineq.T), 0)
        for c in multiplied_constraints.flatten():
            relaxed_prog.AddLinearConstraint(c)

        e_1 = unit_vector(0, X.shape[0])
        linear_constraints = ge(A_ineq.dot(X).dot(e_1), 0)
        for c in linear_constraints:
            relaxed_prog.AddLinearConstraint(c)

    has_generic_constaints = len(prog.generic_constraints()) > 0
    # TODO: I can use Hongkai's PR once that is merged
    if has_generic_constaints:
        (
            generic_eq_constraints_as_polynomials,
            generic_ineq_constraints_as_polynomials,
        ) = _generic_constraint_bindings_to_polynomials(prog.generic_constraints())

        # check degree of all generic constraints
        generic_constraints_as_polynomials = np.concatenate(
            (
                generic_eq_constraints_as_polynomials.flatten(),
                generic_ineq_constraints_as_polynomials.flatten(),
            )
        )
        _assert_max_degree(generic_constraints_as_polynomials, DEGREE_QUADRATIC)

        Q_eqs = [
            _quadratic_polynomial_to_homoenuous_form(p, basis, num_vars)
            for p in generic_eq_constraints_as_polynomials
        ]
        for Q in Q_eqs:
            constraints = eq( np.sum(X * Q), 0).flatten()
            for c in constraints:  # Drake requires us to add one constraint at the time
                relaxed_prog.AddLinearConstraint(c)
        Q_ineqs = [
            _quadratic_polynomial_to_homoenuous_form(p, basis, num_vars)
            for p in generic_ineq_constraints_as_polynomials
        ]
        for Q in Q_ineqs:
            constraints = ge(np.sum(X * Q), 0).flatten()
            for c in constraints:  # Drake requires us to add one constraint at the time
                relaxed_prog.AddLinearConstraint(c)
    # return relaxed_prog, X, basis


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
    # relaxed_prog.AddLinearConstraint(zn[N][0][0] == 2.0)
    # relaxed_prog.AddLinearConstraint(zn[N][1][0] == 0.0)
    # relaxed_prog.AddLinearConstraint(zn[N][2][0] == 0.0)

    # relaxed_prog.AddLinearConstraint(zn[N][3][0] == 0.0)
    # relaxed_prog.AddLinearConstraint(zn[N][4][0] == 0.0)
    # relaxed_prog.AddLinearConstraint(zn[N][5][0] == 0.0)

    return relaxed_prog, zn, un, psd_mats


def print_chordal_solution(solution, zn, un, horizon):
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
    YAY("x", np.round(np.array(x).reshape(horizon),2) )
    YAY("y", np.round(np.array(y).reshape(horizon),2) )
    YAY("th", np.round(np.array(th).reshape(horizon),2) )
    YAY("dx", np.round(np.array(dx).reshape(horizon),2) )
    YAY("dy", np.round(np.array(dy).reshape(horizon),2) )
    YAY("dth", np.round(np.array(dth).reshape(horizon),2) )
    YAY("c", np.round(np.array(c).reshape(horizon),2) )
    YAY("s", np.round(np.array(s).reshape(horizon),2) )
    YAY("dc", np.round(np.array(dc).reshape(horizon),2) )
    YAY("ds", np.round(np.array(ds).reshape(horizon),2) )
    INFO("v", np.round(np.array(v).reshape(horizon),3) )
    INFO("w", np.round(np.array(w).reshape(horizon),3) )

def make_chordal_sdp_program(N, desired_pos = np.array([2,0]), dt = 0.2):
    horizon = N
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
    # R = np.array([[1, 0.5], [0.5, 1]])

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

    state_dim = 10
    control_dim = 2

    # let's stack all binding per step at respective parts
    
    for n in range(N):
        # quadratic inequality constraints: s^2 + c^2 = 1
        prog.AddConstraint( s[n+1]*ds[n+1] + c[n+1]*dc[n+1] == 0)
        prog.AddConstraint( c[n+1] <= 1)
        # prog.AddConstraint( c[n+1] <= 1)

        # lower and upper bounds on control inputs
        # prog.AddBoundingBoxConstraint(0, m*g/2 * thrust_2_mass_ratio, v[n]) 
        # prog.AddBoundingBoxConstraint(0, m*g/2 * thrust_2_mass_ratio, w[n]) 

        # linear dynamics
        prog.AddLinearEqualityConstraint( x[n+1] == x[n] + dt * dx[n]  ) 
        prog.AddLinearEqualityConstraint( y[n+1] == y[n] + dt * dy[n] ) 
        prog.AddLinearEqualityConstraint( th[n+1] == th[n] + dt * dth[n] ) 
        drag = 0.2
        prog.AddLinearEqualityConstraint( dth[n+1] == dth[n]*(1-drag) + dt * (v[n] - w[n]) * r / I) # - dth[n]*dt*drag  )
        # cons.append( prog.AddLinearEqualityConstraint( dth[n+1] == dth[n] + dt * (v[n] - w[n]) * r / I ) )
        # quadratic dynamics
        prog.AddConstraint( dx[n+1] == dx[n] + dt * (-(v[n] + w[n]) * s[n] / m) ) 
        prog.AddConstraint( dy[n+1] == dy[n] + dt * ( (v[n] + w[n]) * c[n] / m - g) ) 
        prog.AddLinearEqualityConstraint( c[n+1] == c[n] + dt * dc[n] ) 
        prog.AddLinearEqualityConstraint( s[n+1] == s[n] + dt * ds[n] ) 
        prog.AddConstraint( dc[n+1] == dc[n] - dt * ( r/I*(v[n]-w[n])*s[n] + dth[n]*ds[n] ) ) 
        prog.AddConstraint( ds[n+1] == ds[n] + dt * ( r/I*(v[n]-w[n])*c[n] + dth[n]*dc[n] ) ) 
        # append all the constraints

    # add the cost -- this is fine, cost we can put over the whole thing
    z_star = np.hstack( (desired_pos, np.array([0, 0,0,0, 1,0, 0,0]) ))
    u_star = m * g / 2.0 * np.array([1, 1])
    # build the cost
    cost = 0
    for i in range(N):
        cost = cost + (z[i]-z_star).dot(Q).dot(z[i]-z_star) + (u[i]-u_star).dot(R).dot(u[i]-u_star)

    prog.AddCost(cost)

    z_0 = np.array( [0,0,0, 0,0,0, 1,0, 0,0] )
    boundary_conditions = Qf, z_star, z_0

    timer = timeit()
    relaxed_prog, zn, un, psd_mats = make_chordal_sdp_relaxation( horizon, state_dim, control_dim, prog, boundary_conditions)

    timer.dt("making the chordal sdp")
    solution = Solve(relaxed_prog)
    
    timer.dt("solving the chordal sdp")
    print( solution.is_success() )
    print( solution.get_optimal_cost() )
    print( solution.get_solution_result() )

    print_chordal_solution(solution, zn, un, horizon)

    
if __name__ == "__main__":
    make_chordal_sdp_program(15)