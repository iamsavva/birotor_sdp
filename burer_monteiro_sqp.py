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
    SnoptSolver,
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
from nonlinear_birotor import make_a_warmstart_from_initial_condition_and_control_sequence, solve_nonconvex_birotor, make_a_nonconvex_birotor_program


def make_dumb_init(N, desired_pos, p):
    one = np.array([1])
    x = np.linspace(0,desired_pos[0],N)
    y = np.linspace(0,desired_pos[1],N)
    th = np.zeros(N)
    dx = np.zeros(N)
    dy = np.zeros(N)
    dth = np.zeros(N)
    c = np.ones(N)
    s = np.zeros(N)
    dc = np.zeros(N)
    ds = np.zeros(N)
    v = np.zeros(N)
    w = np.zeros(N)
    full_state = np.hstack((one, x,y,th,dx,dy,dth,c,s,dc,ds,v,w))
    n = len(full_state)
    full_state = full_state.reshape(n,1)
    return full_state
    # return np.hstack((full_state, np.zeros((n,p-1))))
    



def make_burer_monteiro_program(N, desired_pos = np.array([2,0]), dt = 0.2):
    # construct a nonlinear program
    prog, (x,y,th,dx,dy,dth,c,s,dc,ds,v,w) = make_a_nonconvex_birotor_program(N, desired_pos, dt, for_sdp_solver=True, get_vars=True)
    decision_vars = np.array( sorted(prog.decision_variables(), key=lambda x: x.get_id()) )
    num_vars = ( len(decision_vars))
    
    bm_prog = MathematicalProgram()

    p = 20 # Y is (n x p)
    # Y = bm_prog.NewContinuousVariables(num_vars, p, "Y")
    Y_opt = bm_prog.NewContinuousVariables(num_vars, p, "Y")
    Y = np.vstack((np.ones((1,p)),Y_opt))
    X = Y @ Y.T / p

    for c in le(Y_opt, 50):
        bm_prog.AddConstraint(c)
    for c in ge(Y_opt, -50):
        bm_prog.AddConstraint(c)

    X_val, res = solve_sdp_birotor(N, desired_pos, dt)

    warmstart = X_val[0][1:].reshape(num_vars,1)

    # passing derivatives fucks it up
    # passing controls fucks it up too
    state_only = N*3

    # i can warmstart initial stuff and get not complete rubbish
    # dumb warmstart gives dumb 000 solution
    for i in range(p):
        if i == 0:
            bm_prog.SetInitialGuess( Y_opt[:state_only, i], warmstart[:state_only] )
        else:
            bm_prog.SetInitialGuess( Y_opt[:state_only, i], np.ones(state_only) )
    # bm_prog.SetInitialGuess( Y_opt[:state_only, 1], warmstart[:state_only] ) 
    # bm_prog.SetInitialGuess( Y_opt[:state_only, 2], warmstart[:state_only] )

    # INFO("Warmstart: ", np.round(warmstart,2).reshape(num_vars) )
    # bm_prog.SetInitialGuess( Y, np.hstack((X_val[0].reshape(n,1), X_val[0].reshape(n,1)))/2 )
    # bm_prog.SetInitialGuess( Y, np.hstack((X_val[0].reshape(n,1), np.zeros((n, p-1)))))
    # bm_prog.SetInitialGuess( Y, np.random.random((num_vars, p)))
    # there is something that's clearly wrong about my costs
    # am i picking highest value?

    print("----------")
    print("----------")
    print("----------")
    print("----------")
    timer = timeit()
    add_constraints_to_psd_mat_from_prog(prog, bm_prog, X, True, verbose=True)
    timer.dt("Constructing Burer Monteiro program")

    INFO("Solving Burer Monteiro ", N)
    timer = timeit()
    solver = SnoptSolver()
    # solver = IpoptSolver()
    solution = solver.Solve(bm_prog) # type: MathematicalProgramResult
    timer.dt("Solving Burer Monteiro program")
    diditwork(solution)

    solution.GetInfeasibleConstraints(bm_prog)

    def evaluate_expression(x):
        return x.Evaluate()

    ev = np.vectorize(evaluate_expression)
    X_val = ev(solution.GetSolution(X))
    eigenvals, _ = np.linalg.eig(X_val)

    YAY("Solution for Y", np.round(solution.GetSolution(Y_opt[:,0]).reshape(num_vars), 3) )    

    print("Matrix shape", X_val.shape)
    print("Matrix rank", np.sum(eigenvals>1e-4))
    print("Non-negative eigenvalues are", np.round(np.real(eigenvals[eigenvals>1e-4]),3) )
    res = get_solution_from_X(N, X_val, verbose=True)


if __name__ == "__main__":    
    make_burer_monteiro_program(7)