function get_state_at_time_step(nlp_x, i, nx, N)
    """
        return
        x(t_i)
        i must be in 0:N
    """
    @assert i <= N "trying to get x(t_i) for i > N"
    return nlp_x[i*nx + 1 : (i+1)*nx]
end

function get_control_at_time_stage(nlp_x, i, j, nx, N, m, s)
    """
        return the control 
        u(t_{ij})
        i must be in 0:N-1
    """
    @assert i <= N-1 "trying to get u(t_{i,j}) for i >= N"
    @assert 1 <= j <= s "trying to get u(t_{i,j}) for j > s"
    start = (N+1)*nx 
    return nlp_x[start + i*m*s + (j-1)*m + 1 : start + i*m*s + (j-1)*m + m]
end

function get_control_at_time_step(nlp_x, i, nx, N, m, rk)
    """
        return 'average control'
        u(t_i) = sum_j=1..s b_j u_i^s
    """
    @assert i <= N "trying to get u(t_i) for i > N"
    # final time case: use penultimate control ie u_N := u_{N-1}
    if i == N
        i = N-1
    end
    ui = zeros(m)
    s = rk.stage
    for j in 1:s
        ui = ui + rk.butcher_b[j] * get_control_at_time_stage(nlp_x, i, j, nx, N, m, s)
    end
    return ui 
end

function get_k_at_time_stage(nlp_x, i, j, nx, N, m, s)
    @assert i <= N-1 "trying to get k_i^j for i >= N"
    @assert 1 <= j <= s "trying to get k_i^j for j > s"
    start = (N+1)*nx + N*s*m
    return nlp_x[start + i*nx*s + (j-1)*nx + 1 : start + i*nx*s + (j-1)*nx + nx]
end

function get_state_at_time_stage(nlp_x, i, j, nx, N, m, rk, h)
    s = rk.stage    
    @assert i <= N-1 "trying to get x_i^j for i >= N"
    @assert 1 <= j <= s "trying to get x_i^j for j > s"
    xij = get_state_at_time_step(nlp_x, i, nx, N)
    for l in 1:s
        xij = xij + h * rk.butcher_a[j,l] * get_k_at_time_stage(nlp_x, i, l, nx, N, m, s)
    end
    return xij
end

function get_time_stages(time_steps, rk)
    N = length(time_steps) - 1
    h = time_steps[2] - time_steps[1]
    s = rk.stage
    time_stages = zeros(N*s)
    for i in 1:N
        ti = time_steps[i]
        for j in 1:s
            time_stages[(i-1)*s + j] = ti + h * rk.butcher_c[j]
        end 
    end
    return time_stages
end

function get_final_time(nlp_x, fixed_final_time, has_free_final_time)
    if has_free_final_time 
        return nlp_x[end] 
    else
        return fixed_final_time
    end
end


## Initialization for the NLP problem
function set_state_at_time_step!(nlp_x, x, i, nx, N)
    @assert i <= N "trying to set init for x(t_i) with i > N"
    nlp_x[1+i*nx:(i+1)*nx] = x[1:nx]
end
    
function set_stage_controls_at_time_step!(nlp_x, u, i, nx, N, m, s)
    @assert i <= N-1 "trying to set init for u(t_i) with i >= N"
    start = (N+1)*nx 
    for j in 1:s
        nlp_x[start+i*m*s+(j-1)*m+1:start+i*m*s+(j-1)*m+m] = u[1:m]
    end
end

function initial_guess(ctd)

    N = ctd.dim_NLP_steps
    init = ctd.NLP_init
    nlp_x0 = 1.1*ones(ctd.dim_NLP_variables)

    if init !== nothing
        if length(init) != (ctd.state_dimension + ctd.control_dimension)
            error("vector for initialization should be of size n+m",ctd.state_dimension+ctd.control_dimension)
        end
        # split state / control init values
        x_init = zeros(ctd.dim_NLP_state)
        x_init[1:ctd.state_dimension] = init[1:ctd.state_dimension]
        u_init = zeros(ctd.control_dimension)
        u_init[1:ctd.control_dimension] = init[ctd.state_dimension+1:ctd.state_dimension+ctd.control_dimension]
        
        # mayer -> lagrange additional state
        if ctd.has_lagrange_cost
            x_init[ctd.dim_NLP_state] = 0.1
        end

        # set constant initialization for state / control variables
        for i in 0:N
            set_state_at_time_step!(nlp_x0, x_init, i, ctd.dim_NLP_state, N)
        end
        for i in 0:N-1
            set_stage_controls_at_time_step!(nlp_x0, u_init, i, ctd.dim_NLP_state, N, ctd.control_dimension, ctd.rk.stage)
        end
    end

    # free final time case, put back 0.1 here ?
    # +++todo: add a component in init vector for tf and put this part in main if/then above
    if ctd.has_free_final_time
        nlp_x0[end] = 1.0
    end

    return nlp_x0
end