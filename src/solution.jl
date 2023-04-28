# build generic OCP solution from direct method (NLP) solution from ipopt
function _OptimalControlSolution(ocp, ipopt_solution, ctd)

    # save general solution data
    ctd.NLP_stats = ipopt_solution
    if ismin(ocp)
        ctd.NLP_objective = ipopt_solution.objective
    else
        ctd.NLP_objective = - ipopt_solution.objective
    end
    ctd.NLP_constraints_violation = ipopt_solution.primal_feas
    ctd.NLP_iterations = ipopt_solution.iter
    ctd.NLP_solution = ipopt_solution.solution
    ctd.NLP_sol_constraints = ipopt_constraint(ipopt_solution.solution, ctd)

    # parse NLP variables, constraints and multipliers
    X, U, P, sol_control_constraints, sol_state_constraints, sol_mixed_constraints, mult_control_constraints, mult_state_constraints, mult_mixed_constraints, mult_state_box_lower, mult_state_box_upper, mult_control_box_lower, mult_control_box_upper, U_step = parse_ipopt_sol(ctd)

    # variables and misc infos
    N = ctd.dim_NLP_steps
    t0 = ctd.initial_time
    tf = get_final_time(ctd.NLP_solution, ctd.final_time, ctd.has_free_final_time)
    T = collect(LinRange(t0, tf, N+1))
    x = ctinterpolate(T, matrix2vec(X, 1))
    # +++ NB. interpolation on control stages may fail for RK schemes with non-distinct c_i !
    # also, the interpolation requires the time stages grid instead of the time steps one
    # for now, we interpolate the 'average' control that is constant on each time step 
    u = ctinterpolate(T, matrix2vec(U_step, 1))
    p = ctinterpolate(T[1:end-1], matrix2vec(P, 1))
    sol = OptimalControlSolution()
    sol.state_dimension = ctd.state_dimension
    sol.control_dimension = ctd.control_dimension
    sol.times = T
    sol.time_name = ocp.time_name
    sol.state = t -> x(t)
    sol.state_names = ocp.state_names
    sol.adjoint = t -> p(t)
    sol.control = t -> u(t)
    sol.control_names = ocp.control_names
    sol.objective = ctd.NLP_objective
    sol.iterations = ctd.NLP_iterations
    sol.stopping = :dummy 
    sol.message = "no message" 
    sol.success = false #todo cf ipopt return codes

    # constraints and multipliers
    if ctd.has_state_constraints
        cx = ctinterpolate(T, matrix2vec(sol_state_constraints, 1))
        mcx = ctinterpolate(T, matrix2vec(mult_state_constraints, 1))
        sol.infos[:dim_state_constraints] = ctd.dim_state_constraints    
        sol.infos[:state_constraints] = t -> cx(t)
        sol.infos[:mult_state_constraints] = t -> mcx(t)
    end
    if ctd.has_control_constraints
        cu = ctinterpolate(T, matrix2vec(sol_control_constraints, 1))
        mcu = ctinterpolate(T, matrix2vec(mult_control_constraints, 1))
        sol.infos[:dim_control_constraints] = ctd.dim_control_constraints  
        sol.infos[:control_constraints] = t -> cu(t)
        sol.infos[:mult_control_constraints] = t -> mcu(t)
    end
    if ctd.has_mixed_constraints
        cxu = ctinterpolate(T, matrix2vec(sol_mixed_constraints, 1))
        mcxu = ctinterpolate(T, matrix2vec(mult_mixed_constraints, 1))
        sol.infos[:dim_mixed_constraints] = ctd.dim_mixed_constraints    
        sol.infos[:mixed_constraints] = t -> cxu(t)
        sol.infos[:mult_mixed_constraints] = t -> mcxu(t)
    end
    if ctd.has_state_box
        mbox_x_l = ctinterpolate(T, matrix2vec(mult_state_box_lower, 1))
        mbox_x_u = ctinterpolate(T, matrix2vec(mult_state_box_upper, 1))
        sol.infos[:mult_state_box_lower] = t -> mbox_x_l(t)
        sol.infos[:mult_state_box_upper] = t -> mbox_x_u(t)    
    end
    if ctd.has_control_box
        mbox_u_l = ctinterpolate(T, matrix2vec(mult_control_box_lower, 1))
        mbox_u_u = ctinterpolate(T, matrix2vec(mult_control_box_upper, 1))
        sol.infos[:mult_control_box_lower] = t -> mbox_u_l(t)
        sol.infos[:mult_control_box_upper] = t -> mbox_u_u(t)
    end
    
    return sol

end


# parse NLP solution from ipopt into OCP variables, constraints and multipliers
function parse_ipopt_sol(ctd)
    
    N = ctd.dim_NLP_steps
    s = ctd.rk.stage
    nx = ctd.dim_NLP_state
    m = ctd.control_dimension

    # states and controls variables, with box multipliers
    nlp_x = ctd.NLP_solution
    mult_L = ctd.NLP_stats.multipliers_L
    mult_U = ctd.NLP_stats.multipliers_U
    X = zeros(N+1,nx)
    mult_state_box_lower = zeros(N+1,nx)
    mult_state_box_upper = zeros(N+1,nx)
    U = zeros(N*s,ctd.control_dimension)    
    mult_control_box_lower = zeros(N*s,m)
    mult_control_box_upper = zeros(N*s,m)
    U_step = zeros(N+1,m)

    # parse state variables and box multipliers
    for i in 1:N+1
        X[i,:] = get_state_at_time_step(nlp_x, i-1, nx, N)
        if length(mult_L) > 0
            mult_state_box_lower[i,:] = get_state_at_time_step(mult_L, i-1, nx, N)        
        end
        if length(mult_U) > 0
            mult_state_box_upper[i,:] = get_state_at_time_step(mult_U, i-1, nx, N)
        end
    end

    # parse control variables and box multipliers
    for i in 1:N
        for j in 1:s
            U[(i-1)*s + j,:] = get_control_at_time_stage(nlp_x, i-1, j, nx, N, m, s)
            if length(mult_L) > 0
                mult_control_box_lower[(i-1)*s + j,:] = get_control_at_time_stage(mult_L, i-1, j, nx, N, m, s)
            end
            if length(mult_U) > 0
                mult_control_box_upper[(i-1)*s + j,:] = get_control_at_time_stage(mult_U, i-1, j, nx, N, m, s)
            end
        end
    end

    # compute the 'average' control (constant on each step, duplicate last value for tf)
    for i in 1:N+1
        U_step[i,:] = get_control_at_time_step(nlp_, i-1, nx, N, m, ctd.rk)
    end

    # +++ recover kstage variables (NB. they have no bounds) ?

    # parse constraints, costate and constraints multipliers
    P = zeros(N, nx)
    lambda = ctd.NLP_stats.multipliers
    c = ctd.NLP_sol_constraints
    sol_control_constraints = zeros(N+1,ctd.dim_control_constraints)
    sol_state_constraints = zeros(N+1,ctd.dim_state_constraints)
    sol_mixed_constraints = zeros(N+1,ctd.dim_mixed_constraints)    
    mult_control_constraints = zeros(N+1,ctd.dim_control_constraints)
    mult_state_constraints = zeros(N+1,ctd.dim_state_constraints)
    mult_mixed_constraints = zeros(N+1,ctd.dim_mixed_constraints)
    index = 1
    for i in 1:N
        # skip kstage equations (+++ recover those as well ?)
        index = index + s*nx

        # state equation
        P[i,:] = lambda[index:index+nx-1]
        index = index + nx
        
        # path constraints
        if ctd.has_control_constraints
            sol_control_constraints[i,:] = c[index:index+ctd.dim_control_constraints-1]
            mult_control_constraints[i,:] = lambda[index:index+ctd.dim_control_constraints-1]
            index = index + ctd.dim_control_constraints
        end
        if ctd.has_state_constraints
            sol_state_constraints[i,:] = c[index:index+ctd.dim_state_constraints-1]
            mult_state_constraints[i,:] = lambda[index:index+ctd.dim_state_constraints-1]
            index = index + ctd.dim_state_constraints
        end
        if ctd.has_mixed_constraints
            sol_mixed_constraints[i,:] = c[index:index+ctd.dim_mixed_constraints-1]
            mult_mixed_constraints[i,:] = lambda[index:index+ctd.dim_mixed_constraints-1]
            index = index + ctd.dim_mixed_constraints
        end
    end
    # path constraints at final time
    if ctd.has_control_constraints
        sol_control_constraints[N+1,:] = c[index:index+ctd.dim_control_constraints-1]
        mult_control_constraints[N+1,:] = lambda[index:index+ctd.dim_control_constraints-1]
        index = index + ctd.dim_control_constraints
    end
    if ctd.has_state_constraints
        sol_state_constraints[N+1,:] = c[index:index+ctd.dim_state_constraints-1] 
        mult_state_constraints[N+1,:] = lambda[index:index+ctd.dim_state_constraints-1]
        index = index + ctd.dim_state_constraints
    end
    if ctd.has_mixed_constraints
        sol_mixed_constraints[N+1,:] = c[index:index+ctd.dim_mixed_constraints-1]        
        mult_mixed_constraints[N+1,:] =  lambda[index:index+ctd.dim_mixed_constraints-1]
        index = index + ctd.dim_mixed_constraints
    end

    return X, U, P, sol_control_constraints, sol_state_constraints, sol_mixed_constraints, mult_control_constraints, mult_state_constraints, mult_mixed_constraints, mult_state_box_lower, mult_state_box_upper, mult_control_box_lower, mult_control_box_upper, U_step
end
