"""
$(TYPEDSIGNATURES)

Build OCP functional solution from DOCP vector solution (given as a GenericExecutionStats)
"""
function OCPSolutionFromDOCP(docp, docp_solution_ipopt)

    # could pass some status info too (get_status ?)
    return OCPSolutionFromDOCP_raw(docp, docp_solution_ipopt.solution, objective=docp_solution_ipopt.objective, constraints_violation=docp_solution_ipopt.primal_feas, iterations=docp_solution_ipopt.iter,multipliers_constraints=docp_solution_ipopt.multipliers, multipliers_LB=docp_solution_ipopt.multipliers_L, multipliers_UB=docp_solution_ipopt.multipliers_U, message=docp_solution_ipopt.solver_specific[:internal_msg])
end


"""
$(TYPEDSIGNATURES)

Build OCP functional solution from DOCP vector solution (given as raw variables and multipliers plus some optional infos)
"""
function OCPSolutionFromDOCP_raw(docp, solution; objective=nothing, constraints_violation=nothing, iterations=0, multipliers_constraints=nothing, multipliers_LB=nothing, multipliers_UB=nothing, message=nothing)
   
    # NB. still missing: stopping and success info...

    # set objective if needed
    if objective==nothing
        objective = DOCP_objective(solution, docp)
    end
    # adjust objective sign for maximization problems
    if !is_min(docp.ocp)
        objective = - objective
    end

    # recompute value of constraints at solution
    # NB. the constraint formulation is LB <= C <= UB
    constraints = zeros(docp.dim_NLP_constraints)
    DOCP_constraints!(constraints, solution, docp)
    # set constraint violation if needed
    if constraints_violation==nothing
        constraints_check = zeros(docp.dim_NLP_constraints)
        DOCP_constraints_check!(constraints_check, constraints, docp)
        variables_check = zeros(docp.dim_NLP_variables)
        DOCP_variables_check!(variables_check, solution, docp)
        constraints_violation = norm(append!(variables_check, constraints_check), Inf)
    end
    
    # parse NLP variables, constraints and multipliers 
    X, U, v, P, sol_control_constraints, sol_state_constraints, sol_mixed_constraints, sol_variable_constraints, mult_control_constraints, mult_state_constraints, mult_mixed_constraints, mult_variable_constraints, mult_state_box_lower, mult_state_box_upper, mult_control_box_lower, mult_control_box_upper, mult_variable_box_lower, mult_variable_box_upper = parse_DOCP_solution(docp, solution, multipliers_constraints, multipliers_LB, multipliers_UB, constraints)

    # time grid
    N = docp.dim_NLP_steps
    t0 = get_initial_time(solution, docp)
    tf = max(get_final_time(solution, docp), t0 + 1e-9)
    T = collect(LinRange(t0, tf, N+1))

    # variables: remove additional state for lagrange cost
    x = ctinterpolate(T, matrix2vec(X[:,1:docp.ocp.state_dimension], 1))
    p = ctinterpolate(T[1:end-1], matrix2vec(P[:,1:docp.ocp.state_dimension], 1))
    u = ctinterpolate(T, matrix2vec(U, 1))

    # generate ocp solution
    sol = OptimalControlSolution() # +++ constructor with ocp as argument ?
    copy!(sol, docp.ocp)
    sol.times      = T
    # use scalar output for x,u,v,p if dim=1
    sol.state      = (sol.state_dimension==1)    ? deepcopy(t -> x(t)[1]) : deepcopy(t -> x(t)) 
    sol.costate    = (sol.state_dimension==1)    ? deepcopy(t -> p(t)[1]) : deepcopy(t -> p(t))
    sol.control    = (sol.control_dimension==1)  ? deepcopy(t -> u(t)[1]) : deepcopy(t -> u(t))
    sol.variable   = (sol.variable_dimension==1) ? v[1] : v
    sol.objective  = objective
    sol.iterations = iterations
    sol.stopping   = nothing #+++
    sol.message    = String(message)
    sol.success    = nothing #+++

    # nonlinear constraints and multipliers
    if docp.has_state_constraints
        cx = ctinterpolate(T, matrix2vec(sol_state_constraints, 1))
        mcx = ctinterpolate(T, matrix2vec(mult_state_constraints, 1))
        sol.infos[:dim_state_constraints] = docp.dim_state_constraints    
        sol.infos[:state_constraints] = t -> cx(t)
        sol.infos[:mult_state_constraints] = t -> mcx(t)
    end
    if docp.has_control_constraints
        cu = ctinterpolate(T, matrix2vec(sol_control_constraints, 1))
        mcu = ctinterpolate(T, matrix2vec(mult_control_constraints, 1))
        sol.infos[:dim_control_constraints] = docp.dim_control_constraints  
        sol.infos[:control_constraints] = t -> cu(t)
        sol.infos[:mult_control_constraints] = t -> mcu(t)
    end
    if docp.has_mixed_constraints
        cxu = ctinterpolate(T, matrix2vec(sol_mixed_constraints, 1))
        mcxu = ctinterpolate(T, matrix2vec(mult_mixed_constraints, 1))
        sol.infos[:dim_mixed_constraints] = docp.dim_mixed_constraints    
        sol.infos[:mixed_constraints] = t -> cxu(t)
        sol.infos[:mult_mixed_constraints] = t -> mcxu(t)
    end
    if docp.has_variable_constraints
        sol.infos[:dim_variable_constraints] = docp.dim_variable_constraints
        sol.infos[:variable_constraints] = sol_variable_constraints
        sol.infos[:mult_variable_constraints] = mult_variable_constraints
    end

    # box constraints multipliers
    if docp.has_state_box
        # remove additional state for lagrange cost
        mbox_x_l = ctinterpolate(T, matrix2vec(mult_state_box_lower[:,1:docp.ocp.state_dimension], 1))
        mbox_x_u = ctinterpolate(T, matrix2vec(mult_state_box_upper[:,1:docp.ocp.state_dimension], 1))
        sol.infos[:mult_state_box_lower] = t -> mbox_x_l(t)
        sol.infos[:mult_state_box_upper] = t -> mbox_x_u(t)    
    end
    if docp.has_control_box
        mbox_u_l = ctinterpolate(T, matrix2vec(mult_control_box_lower, 1))
        mbox_u_u = ctinterpolate(T, matrix2vec(mult_control_box_upper, 1))
        sol.infos[:mult_control_box_lower] = t -> mbox_u_l(t)
        sol.infos[:mult_control_box_upper] = t -> mbox_u_u(t)
    end
    if docp.has_variable_box
        sol.infos[:mult_variable_box_lower] = mult_variable_box_lower
        sol.infos[:mult_variable_box_upper] = mult_variable_box_upper 
    end

    return sol

end


"""
$(TYPEDSIGNATURES)

Parse DOCP solution into OCP variables, constraints and multipliers
"""
function parse_DOCP_solution(docp, solution, multipliers_constraints, multipliers_LB, multipliers_UB, constraints)
    
    # states and controls variables, with box multipliers
    N = docp.dim_NLP_steps
  
    X = zeros(N+1,docp.dim_NLP_state)
    U = zeros(N+1,docp.ocp.control_dimension)
    v = get_variable(solution, docp)
    # if box multipliers are empty, use dummy vectors for size consistency
    if length(multipliers_LB) > 0
        mult_L = multipliers_LB
    else
        mult_L = zeros(docp.dim_NLP_variables)
    end
    if length(multipliers_UB) > 0
        mult_U = multipliers_UB
    else
        mult_U = zeros(docp.dim_NLP_variables)
    end  
    mult_state_box_lower = zeros(N+1,docp.dim_NLP_state)
    mult_state_box_upper = zeros(N+1,docp.dim_NLP_state)
    mult_control_box_lower = zeros(N+1,docp.ocp.control_dimension)
    mult_control_box_upper = zeros(N+1,docp.ocp.control_dimension)
    mult_variable_box_lower = zeros(N+1,docp.variable_dimension)
    mult_variable_box_upper = zeros(N+1,docp.variable_dimension)

    for i in 1:N+1
        # state and control variables
        X[i,:] = vget_state_at_time_step(solution, docp, i-1)
        U[i,:] = vget_control_at_time_step(solution, docp, i-1)
        # box multipliers (same layout as variables !)
        # NB. will return 0 if box constraints are not present
        mult_state_box_lower[i,:] = vget_state_at_time_step(mult_L, docp, i-1)
        mult_control_box_lower[i,:] = vget_control_at_time_step(mult_L, docp, i-1)
        mult_state_box_upper[i,:] = vget_state_at_time_step(mult_U, docp, i-1)
        mult_control_box_upper[i,:] = vget_control_at_time_step(mult_U, docp, i-1)
    end
    if docp.has_variable_box
        mult_variable_box_lower = get_variable(mult_L, docp)
        mult_variable_box_upper = get_variable(mult_U, docp)
    end

    # constraints, costate and constraints multipliers
    P = zeros(N, docp.dim_NLP_state)
    lambda = multipliers_constraints
    sol_control_constraints = zeros(N+1,docp.dim_control_constraints)
    sol_state_constraints = zeros(N+1,docp.dim_state_constraints)
    sol_mixed_constraints = zeros(N+1,docp.dim_mixed_constraints) 
    sol_variable_constraints = zeros(docp.dim_variable_constraints)
    mult_control_constraints = zeros(N+1,docp.dim_control_constraints)
    mult_state_constraints = zeros(N+1,docp.dim_state_constraints)
    mult_mixed_constraints = zeros(N+1,docp.dim_mixed_constraints)
    mult_variable_constraints = zeros(docp.dim_variable_constraints)

    index = 1
    for i in 1:N
        # state equation
        P[i,:] = lambda[index:index+docp.dim_NLP_state-1]
        index = index + docp.dim_NLP_state
        # path constraints
        # +++ use aux function for the 3 blocks, see eval c also
        if docp.has_control_constraints
            sol_control_constraints[i,:] = constraints[index:index+docp.dim_control_constraints-1]
            mult_control_constraints[i,:] = lambda[index:index+docp.dim_control_constraints-1]
            index = index + docp.dim_control_constraints
        end
        if docp.has_state_constraints
            sol_state_constraints[i,:] = constraints[index:index+docp.dim_state_constraints-1]
            mult_state_constraints[i,:] = lambda[index:index+docp.dim_state_constraints-1]
            index = index + docp.dim_state_constraints
        end
        if docp.has_mixed_constraints
            sol_mixed_constraints[i,:] = constraints[index:index+docp.dim_mixed_constraints-1]
            mult_mixed_constraints[i,:] = lambda[index:index+docp.dim_mixed_constraints-1]
            index = index + docp.dim_mixed_constraints
        end
    end
    # path constraints at final time
    # +++ use aux function for the 3 blocks, see eval c also
    if docp.has_control_constraints
        sol_control_constraints[N+1,:] = constraints[index:index+docp.dim_control_constraints-1]
        mult_control_constraints[N+1,:] = lambda[index:index+docp.dim_control_constraints-1]
        index = index + docp.dim_control_constraints
    end
    if docp.has_state_constraints
        sol_state_constraints[N+1,:] = constraints[index:index+docp.dim_state_constraints-1] 
        mult_state_constraints[N+1,:] = lambda[index:index+docp.dim_state_constraints-1]
        index = index + docp.dim_state_constraints
    end
    if docp.has_mixed_constraints
        sol_mixed_constraints[N+1,:] = constraints[index:index+docp.dim_mixed_constraints-1]        
        mult_mixed_constraints[N+1,:] =  lambda[index:index+docp.dim_mixed_constraints-1]
        index = index + docp.dim_mixed_constraints
    end

    # boundary conditions and multipliers
    if docp.has_boundary_conditions
        sol_boundary_conditions = constraints[index:index+docp.dim_boundary_conditions-1]
        mult_boundary_conditions = lambda[index:index+docp.dim_boundary_conditions-1]
        index = index + docp.dim_boundary_conditions
    end

    # variable constraints and multipliers
    if docp.has_variable_constraints
        sol_variable_constraints = constraints[index:index+docp.dim_variable_constraints-1]
        mult_variable_constraints = lambda[index:index+docp.dim_variable_constraints-1]
        index = index + docp.dim_variable_constraints
    end

    return X, U, v, P, sol_control_constraints, sol_state_constraints, sol_mixed_constraints, sol_variable_constraints, mult_control_constraints, mult_state_constraints, mult_mixed_constraints, mult_variable_constraints, mult_state_box_lower, mult_state_box_upper, mult_control_box_lower, mult_control_box_upper, mult_variable_box_lower, mult_variable_box_upper
end
