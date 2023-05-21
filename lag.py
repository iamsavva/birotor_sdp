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
    _generic_constraint_bindings_to_polynomials, add_constraints_to_psd_mat_from_prog, extract_constraints_from_prog
)
from nonlinear_birotor import make_a_warmstart_from_initial_condition_and_control_sequence, solve_nonconvex_birotor, make_a_nonconvex_birotor_program


def make_linear_interpolation_BM_init(N, desired_pos):
    one = np.array([1])
    x = np.linspace(0,desired_pos[0],N)
    y = np.linspace(0,desired_pos[1],N)
    th = np.linspace(0,desired_pos[2],N)
    dx = np.zeros(N)
    dy = np.zeros(N)
    dth = np.zeros(N)
    c = np.cos(th)
    s = np.sin(th)
    dc = np.zeros(N)
    ds = np.zeros(N)
    v = np.zeros(N)
    w = np.zeros(N)
    full_state = np.hstack((one, x,y,th,dx,dy,dth,c,s,dc,ds,v,w))
    n = len(full_state)
    full_state = full_state.reshape(n,1)
    return full_state

def langrangian_method(prog:MathematicalProgram, nonc_prog:MathematicalProgram, X:npt.NDArray, Y:npt.NDArray, Y_opt):
    cost_expression, cost_matrix, constraint_expressions, constraint_matrices = extract_constraints_from_prog(nonc_prog, X, Y,False)
    assert len(constraint_expressions) == len(constraint_matrices)
    constraint_expressions_squared = np.sum(constraint_expressions ** 2)
    num_constraints = len(constraint_matrices)

    # initial lambdas -- all ones
    lambda_i = np.ones(num_constraints)
    mu_i = 10
    omega_i = 1/mu_i
    Y_i = np.ones(Y.shape)
    Y_opt_i = np.ones(Y_opt.shape)

    gamma = 2
    eta = 0.75
    v_k = float("inf")

    for i in range(5):
        # form multiplier program

        # for augmented lagrangian
        timer = timeit()
        cost_function = cost_expression - lambda_i.dot(constraint_expressions) + mu_i/2 * constraint_expressions_squared
        timer.dt("creating cost function")
        cost_binding = prog.AddCost(cost_function)
        # solve the program
        timer.dt("adding cost binding")
        solver = SnoptSolver()
        INFO("\t", omega_i)
        prog.SetSolverOption(SnoptSolver.id(), "Major iterations limit", 10000)
        prog.SetSolverOption(solver.solver_id(), "Feasibility tolerance", omega_i)
        prog.SetSolverOption(solver.solver_id(), "Major feasibility tolerance", omega_i)
        prog.SetSolverOption(solver.solver_id(), "Major optimality tolerance", omega_i)

        # prog.SetSolverOption(solver.solver_id(), "Minor feasibility tolerance", omega_i)
        # prog.SetSolverOption(solver.solver_id(), "Minor optimality tolerance", omega_i)
        timer.dt("Adding solver otions")
        # solver = IpoptSolver()
        solution = solver.Solve(prog) # type: MathematicalProgramResult # should add initial guess
        timer.dt("Solving Burer Monteiro program") 
        diditwork(solution)
        if not solution.is_success():
            XX = Y_i @ Y_i.T
            print(XX[0]/XX[0,0]) 
        assert solution.is_success()

        Y_i = solution.GetSolution(Y)

        timer = timeit()
        prog.SetInitialGuess( Y, Y_i )
        prog.RemoveCost(cost_binding)
        timer.dt("init_guess, bindings")

        # extract solution

        violation = 0
        X_i = Y_i @ Y_i.T
        for (j,a) in constraint_matrices:
            if j == "all":
                violation += (np.sum(a*(X_i)))**2
            else:
                violation += (Y_i[:,j].dot(a))**2
        WARN("violation", violation)
        
        if violation <= 0.75 * v_k:
            for i in range(num_constraints):
                (j,a) = constraint_matrices[i]
                if j == "all":
                    con = np.sum(a*(X_i))
                else:
                    con = Y_i[:,j].dot(a)
                lambda_i[i] = lambda_i[i] - mu_i * con
            mu_i = mu_i
            v_k = violation
            omega_i = omega_i / mu_i
        else:
            mu_i = 2 * mu_i
            omega_i = 1 / mu_i
            # v_k = violation
            
    XX = Y_i @ Y_i.T
    print(XX[0]/XX[0,0]) 
        # if all_passed:
        #     for i in range(num_constraints):
        #         all_passed = True
        #         (j,a) = constraint_matrices[i]
        #         con = Y_i[:,j].dot(a)
        #         lambda_i[i] = lambda_i[i] - mu_i * con
        #     mu_i = mu_i
        #     eta_i  = eta_i / mu_i**0.9
        #     omega_i = omega_i / mu_i
        # else:
        #     mu_i = 5*mu_i
        #     eta_i  = 1 / mu_i
        #     omega_i = 1 / mu_i

        


        #prog.RemoveCost(cost_binding)


    


def make_burer_monteiro_program(N, desired_pos = np.array([2,0,0]), dt = 0.2):
    p = 5 # Y will be (num_vars x p)
    
    # construct a nonlinear program without any inequality constraints
    nonc_prog, (x,y,th,dx,dy,dth,c,s,dc,ds,v,w) = make_a_nonconvex_birotor_program(N, desired_pos, dt, for_sdp_solver=True, get_vars=True)
    # define prog
    decision_vars = np.array( sorted(nonc_prog.decision_variables(), key=lambda x: x.get_id()) )
    num_vars = len(decision_vars)
    bm_prog = MathematicalProgram()
    # define optimization variables
    Y_opt = bm_prog.NewContinuousVariables(num_vars, p, "Y")
    # Y = np.vstack((np.ones((1,p)),Y_opt))
    Y = np.vstack((bm_prog.NewContinuousVariables(1, p, "Y0"),Y_opt))
    X = Y @ Y.T
    # finite values just in case -- may want to remove that
    for c in le(Y, 100):
        bm_prog.AddConstraint(c)
    for c in ge(Y, -100):
        bm_prog.AddConstraint(c)
    
    # warmstarting -- skip that for now
    if True:
        # X_val, res = solve_sdp_birotor(N, desired_pos, dt)
        # warmstart = X_val[0][1:].reshape(num_vars,1)
        state_only = N*3
        warmstart = make_linear_interpolation_BM_init(N, desired_pos)
        for i in range(p):
            # bm_prog.SetInitialGuess( Y_opt[:, i], warmstart )
            bm_prog.SetInitialGuess( Y[:, i], warmstart )
            

            # if i == 0:
            #     bm_prog.SetInitialGuess( Y[0,i], 1 )
            #     bm_prog.SetInitialGuess( Y_opt[:state_only, i], warmstart[:state_only] )

            # else:
            #     bm_prog.SetInitialGuess( Y[0,i], 1 )
            #     bm_prog.SetInitialGuess( Y_opt[:state_only, i], np.ones(state_only) )

    
    
    # cost_expressions, constraint_expressions = extract_constraints_from_prog(nonc_prog, X, Y, False)
    # INFO(len(cost_expressions), len(constraint_expressions))

    langrangian_method(bm_prog, nonc_prog, X, Y, Y_opt)




    # INFO("Warmstart: ", np.round(warmstart,2).reshape(num_vars) )
    # bm_prog.SetInitialGuess( Y, np.hstack((X_val[0].reshape(n,1), X_val[0].reshape(n,1)))/2 )
    # bm_prog.SetInitialGuess( Y, np.hstack((X_val[0].reshape(n,1), np.zeros((n, p-1)))))
    # bm_prog.SetInitialGuess( Y, np.random.random((num_vars, p)))
    # there is something that's clearly wrong about my costs
    # am i picking highest value?

    # print("----------")
    # print("----------")
    # print("----------")
    # print("----------")
    # timer = timeit()
    # add_constraints_to_psd_mat_from_prog(prog, bm_prog, X, True, verbose=True)
    # timer.dt("Constructing Burer Monteiro program")

    # INFO("Solving Burer Monteiro ", N)
    # timer = timeit()
    # solver = SnoptSolver()
    # # solver = IpoptSolver()
    # solution = solver.Solve(bm_prog) # type: MathematicalProgramResult
    # timer.dt("Solving Burer Monteiro program")
    # diditwork(solution)

    # solution.GetInfeasibleConstraints(bm_prog)

    # def evaluate_expression(x):
    #     return x.Evaluate()

    # ev = np.vectorize(evaluate_expression)
    # X_val = ev(solution.GetSolution(X))
    # eigenvals, _ = np.linalg.eig(X_val)

    # YAY("Solution for Y", np.round(solution.GetSolution(Y_opt[:,0]).reshape(num_vars), 3) )    

    # print("Matrix shape", X_val.shape)
    # print("Matrix rank", np.sum(eigenvals>1e-4))
    # print("Non-negative eigenvalues are", np.round(np.real(eigenvals[eigenvals>1e-4]),3) )
    # res = get_solution_from_X(N, X_val, verbose=True)


if __name__ == "__main__":    
    make_burer_monteiro_program(6)