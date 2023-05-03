#=
1) NLP variables layout: +++ try some reshape instead of getters ?
X = [ STATE VARIABLES, CONTROL VARIABLES, KSTAGE VARIABLES, PARAMETERS]

STATE VARIABLES: 
[x_0, x_1 ... x_N] 
with x_i the state at time step i (dim n)

CONTROL VARIABLES: 
[u_0^1 ,.., u_0^s,  ...  , u_{N-1}^1 ,.., u_{N-1}^s]
with u_i^j the control at time step i and stage j (dim m)

KSTAGE VARIABLES:
[k_0^1 ,.., k_0^s,  ...  , k_{N-1}^1 ,.., k_{N-1}^s]
with k_i^j the kstage at time step i and stage j (dim n)

PARAMETERS: optional vector of scalar paremeters to be optimized
Currently there is only the case of the free final time

2) NLP constraints layout:
C(X) = [ (KSTAGE EQUATIONS i, DYNAMIC EQUATION i, PATH CONSTRAINTS i)_{i=0,..N-1}  PATH CONSTRAINTS(tf) BOUNDARY CONDITIONS ]

KSTAGE EQUATIONS: on each stage
(k_i^j - f(t_i + c_j h, x_i + h sum_{l=1,..,s} a_jl k_i^l) )_{j=1,..,s}         (dim n)

DYNAMIC EQUATION:
x_{i+1} - (x_i + h sum_{j=1,..,s} b_j k_i^j)        (dim n)
with 
h = t_{i+1} - t_i the time step, 
b_j the butcher coefficients 
k_i^j the kstage variables at time step i and stage j

PATH CONSTRAINTS:
(state contraints, control constraints, mixed constraints)
NB path constraints are evaluated at time steps (including tf) and not at time stages !

BOUNDARY CONDITIONS

=#

# struct for ocop/nlp info
mutable struct rk_method_data
    stage::Integer
    butcher_a::Matrix{<:MyNumber}
    butcher_b::Vector{<:MyNumber}
    butcher_c::Vector{<:MyNumber}
    properties::Dict{Symbol, Any}

    function rk_method_data(name::Symbol)
    rk = new()
    if name == :midpoint
        rk.stage  = 1
        rk.butcher_a = reshape([0.5],1,1)
        rk.butcher_b = [1]
        rk.butcher_c = [0.5]
        rk.properties = Dict(:name => "implicit_midpoint", :order => 2, :implicit => true)
    elseif name == :trapeze
        rk.stage  = 2
        rk.butcher_a = [0 0; 0.5]
        rk.butcher_b = [0.5, 0.5]
        rk.butcher_c = [0, 1]
        rk.properties = Dict(:name => "trapeze", :order => 2, :implicit => true)
    elseif name == :gauss2
        rk_stage = 2
        rk.butcher_a = [0.25 0.25 - sqrt(3)/6; 0.25 + sqrt(3)/6 0.25]
        rk.butcher_b = [0.5, 0.5]
        rk.butcher_c = [0.5 - sqrt(3)/6, 0.5 + sqrt(3)/6]
        rk.properties = Dict(:name => "Gauss II (Hammer Hollingworth", :order => 4; :implicit => true, :Astable => true, :Bstable => true, :symplectic => true, :ref => "Geometric Numerical Integration Table 1.1 p34")
    else
        error(name, " method not yet implemented")
    end
    return rk
    end 
end

mutable struct CTDirect_data

    ## OCP
    # OCP variables and functions
    initial_time
    final_time
    state_dimension
    control_dimension
    dynamics
    mayer
    lagrange
    criterion_min_max
    has_free_final_time
    has_lagrange_cost
    has_mayer_cost

    # OCP constraints
    # indicators
    has_control_constraints
    has_state_constraints
    has_mixed_constraints
    has_boundary_conditions
    has_control_box
    has_state_box

    # dimensions
    dim_control_constraints
    dim_state_constraints
    dim_mixed_constraints
    dim_path_constraints
    dim_boundary_conditions
    dim_control_box
    dim_state_box

    # functions
    control_constraints
    state_constraints
    mixed_constraints
    boundary_conditions
    control_box
    state_box

    ## NLP
    # NLP problem
    dim_NLP_state
    dim_NLP_constraints
    dim_NLP_variables
    dim_NLP_steps       # grid_size
    rk::rk_method_data  
    dynamics_lagrange_to_mayer
    NLP_init

    # NLP solution
    NLP_solution
    NLP_objective
    NLP_sol_constraints
    NLP_constraints_violation
    NLP_iterations
    NLP_stats       # remove later ? type is https://juliasmoothoptimizers.github.io/SolverCore.jl/stable/reference/#SolverCore.GenericExecutionStats

    function CTDirect_data(ocp::OptimalControlModel, N::Integer, rk_method::Symbol, init=nothing)

        ctd = new()

        ## Optimal Control Problem OCP
        # time
        ctd.initial_time = ocp.initial_time
        ctd.final_time = ocp.final_time
        ctd.has_free_final_time = isnothing(ctd.final_time)

        # dimensions and functions
        ctd.state_dimension = ocp.state_dimension
        ctd.control_dimension = ocp.control_dimension
        ctd.dynamics = ocp.dynamics
        ctd.has_lagrange_cost = !isnothing(ocp.lagrange)
        ctd.lagrange = ocp.lagrange
        ctd.has_mayer_cost = !isnothing(ocp.mayer)
        ctd.mayer = ocp.mayer
        
        # constraints
        ctd.control_constraints, ctd.state_constraints, ctd.mixed_constraints, ctd.boundary_conditions, ctd.control_box, ctd.state_box = nlp_constraints(ocp)
        ctd.dim_control_constraints = length(ctd.control_constraints[1])
        ctd.dim_state_constraints = length(ctd.state_constraints[1])
        ctd.dim_mixed_constraints = length(ctd.mixed_constraints[1])
        ctd.dim_path_constraints = ctd.dim_control_constraints + ctd.dim_state_constraints + ctd.dim_mixed_constraints
        ctd.dim_boundary_conditions = length(ctd.boundary_conditions[1])
        ctd.dim_control_box = length(ctd.control_box[1])
        ctd.dim_state_box = length(ctd.state_box[1])
        ctd.has_control_constraints = !isempty(ctd.control_constraints[1])
        ctd.has_state_constraints = !isempty(ctd.state_constraints[1])
        ctd.has_mixed_constraints = !isempty(ctd.mixed_constraints[1])
        ctd.has_boundary_conditions = !isempty(ctd.boundary_conditions[1])
        ctd.has_control_box = !isempty(ctd.control_box[1])
        ctd.has_state_box = !isempty(ctd.state_box[1])

        ## Non Linear Programming NLP
        ctd.dim_NLP_steps = N
        ctd.rk = rk_method_data(rk_method)
        ctd.NLP_init = init

        # Mayer to Lagrange reformulation: 
        # additional state with Lagrange cost as dynamics and null initial condition
        ctd.dim_NLP_state = ctd.state_dimension  
        ctd.dim_NLP_constraints = N * ((ctd.rk.stage+1)*ctd.dim_NLP_state + ctd.dim_path_constraints) + ctd.dim_path_constraints + ctd.dim_boundary_conditions
        if ctd.has_lagrange_cost
            ctd.dim_NLP_state = ctd.dim_NLP_state + 1  
            ctd.dim_NLP_constraints = ctd.dim_NLP_constraints + N * (ctd.rk.stage+1) + 1
        end
        # augmented dynamics (+++try to evaluate the condition only once cf below)
        #ctd.dynamics_lagrange_to_mayer(t, x, u) = ctd.has_lagrange_cost ? [ctd.dynamics(t, x[1:ctd.state_dimension], u); ctd.lagrange(t, x[1:ctd.state_dimension], u)] : ctd.dynamics(t, x, u) DOES NOT COMPILE
        function f(t, x, u)
            if ctd.has_lagrange_cost
                return [ctd.dynamics(t, x[1:ctd.state_dimension], u); ctd.lagrange(t, x[1:ctd.state_dimension], u)]
            else
                return ctd.dynamics(t, x, u)
            end
        end
        ctd.dynamics_lagrange_to_mayer = f

        # min or max problem (unused ?)
        ctd.criterion_min_max = ocp.criterion

        # additional variable for free final time
        ctd.dim_NLP_variables = (N + 1) * ctd.dim_NLP_state + N * ctd.rk.stage * (ctd.control_dimension + ctd.dim_NLP_state)
        if ctd.has_free_final_time
            ctd.dim_NLP_variables =  ctd.dim_NLP_variables + 1
        end
        return ctd

    end

end

function is_solvable(ocp)
    solvable = true

    # free initial time
    if isnothing(ocp.initial_time)
        solvable = false
    end
    
    return solvable
end


# bounds for the constraints
function  constraints_bounds(ctd)

    N = ctd.dim_NLP_steps
    s = ctd.rk.stage
    lb = zeros(ctd.dim_NLP_constraints)
    ub = zeros(ctd.dim_NLP_constraints)

    index = 1 # counter for the constraints
    for i in 0:N-1
        
        # skip (ie leave 0) bound for equality kstage constraints
        index = index + s * ctd.dim_NLP_state
        # skip (ie leave 0) bound for equality dynamics constraint
        index = index + ctd.dim_NLP_state

        # path constraints 
        if ctd.has_control_constraints
            lb[index:index+ctd.dim_control_constraints-1] = ctd.control_constraints[1]
            ub[index:index+ctd.dim_control_constraints-1] = ctd.control_constraints[3]
            index = index + ctd.dim_control_constraints
        end
        if ctd.has_state_constraints
            lb[index:index+ctd.dim_state_constraints-1] = ctd.state_constraints[1]
            ub[index:index+ctd.dim_state_constraints-1] = ctd.state_constraints[3]
            index = index + ctd.dim_state_constraints
        end
        if ctd.has_mixed_constraints
            lb[index:index+ctd.dim_mixed_constraints-1] = ctd.mixed_constraints[1]
            ub[index:index+ctd.dim_mixed_constraints-1] = ctd.mixed_constraints[3]
            index = index + ctd.dim_mixed_constraints
        end
    end
    # path constraints at final time
    if ctd.has_control_constraints
        lb[index:index+ctd.dim_control_constraints-1] = ctd.control_constraints[1]
        ub[index:index+ctd.dim_control_constraints-1] = ctd.control_constraints[3]
        index = index + ctd.dim_control_constraints
    end
    if ctd.has_state_constraints
        lb[index:index+ctd.dim_state_constraints-1] = ctd.state_constraints[1]
        ub[index:index+ctd.dim_state_constraints-1] = ctd.state_constraints[3]
        index = index + ctd.dim_state_constraints
    end
    if ctd.has_mixed_constraints
        lb[index:index+ctd.dim_mixed_constraints-1] = ctd.mixed_constraints[1]
        ub[index:index+ctd.dim_mixed_constraints-1] = ctd.mixed_constraints[3]
        index = index + ctd.dim_mixed_constraints
    end
    # boundary conditions
    lb[index:index+ctd.dim_boundary_conditions-1] = ctd.boundary_conditions[1]
    ub[index:index+ctd.dim_boundary_conditions-1] = ctd.boundary_conditions[3]
    index = index + ctd.dim_boundary_conditions
    if ctd.has_lagrange_cost
        lb[index] = 0.
        ub[index] = 0.
        index = index + 1
    end

    return lb, ub
end

# box constraints for variables
function variables_bounds(ctd)

    N = ctd.dim_NLP_steps
    s = ctd.rk.stage
    l_var = -Inf*ones(ctd.dim_NLP_variables)
    u_var = Inf*ones(ctd.dim_NLP_variables)

    # state box
    if ctd.has_state_box
        index = 0
        for i in 0:N
            for j in 1:ctd.dim_state_box
                indice = ctd.state_box[2][j]
                l_var[index+indice] = ctd.state_box[1][j]
                u_var[index+indice] = ctd.state_box[3][j]
            end
            index = index + ctd.dim_NLP_state
        end
    end

    # control box
    if ctd.has_control_box
        index = (N+1)*ctd.dim_NLP_state 
        for i in 0:N
            for l in 1:s
                for j in 1:ctd.dim_control_box
                    indice = ctd.control_box[2][j]
                    l_var[index+indice] = ctd.control_box[1][j]
                    u_var[index+indice] = ctd.control_box[3][j]
                end
            end
            index = index + s * ctd.control_dimension
        end
    end

    # free final time case
    if ctd.has_free_final_time
        l_var[end] = 1.e-3
    end

    return l_var, u_var
end


# IPOPT objective
function ipopt_objective(nlp_x, ctd)

    t0 = ctd.initial_time
    tf = get_final_time(nlp_x, ctd.final_time, ctd.has_free_final_time)
    N = ctd.dim_NLP_steps
    obj = 0
    
    if ctd.has_mayer_cost
        x0 = get_state_at_time_step(nlp_x, 0, ctd.dim_NLP_state, N)
        xf = get_state_at_time_step(nlp_x, N, ctd.dim_NLP_state, N)
        obj = obj + ctd.mayer(t0, x0[1:ctd.state_dimension], tf, xf[1:ctd.state_dimension])
    end
    
    if ctd.has_lagrange_cost
        obj = obj + nlp_x[(N+1)*ctd.dim_NLP_state]
    end

    if ctd.criterion_min_max == :min
        return obj
    else
        return -obj
    end
end


# IPOPT constraints +++ add bounds computation here at first call
function ipopt_constraint(nlp_x, ctd)
    """
    compute the constraints for the NLP : 
        - discretization of the dynamics via the trapeze method
        - boundary conditions
    inputs
    ocp :: ocp model
    return
    c :: 
    """
    t0 = ctd.initial_time
    tf = get_final_time(nlp_x, ctd.final_time, ctd.has_free_final_time)
    N = ctd.dim_NLP_steps
    nx = ctd.dim_NLP_state
    m = ctd.control_dimension
    rk = ctd.rk
    s = rk.stage
    h = (tf - t0) / N
    c = zeros(eltype(nlp_x), ctd.dim_NLP_constraints)

    index = 1 # counter for the constraints
    for i in 0:N-1
        ti = t0 + i*h
        
        # stage equation
        for j in 1:s
            tij = ti + rk.butcher_c[j]*h
            kij = get_k_at_time_stage(nlp_x, i, j, nx, N, m, s)
            xij = get_state_at_time_stage(nlp_x, i, j, nx, N, m, ctd.rk, h)
            uij = get_control_at_time_stage(nlp_x, i, j, nx, N, m, s)
            c[index:index+nx-1] = kij - ctd.dynamics_lagrange_to_mayer(tij, xij, uij)
            index = index + nx
        end

        # state equation
        xi = get_state_at_time_step(nlp_x, i, nx, N)
        xip1 = get_state_at_time_step(nlp_x, i+1, nx, N)
        sum_bk = zeros(nx)
        for j in 1:s
            sum_bk = sum_bk + rk.butcher_b[j] * get_k_at_time_stage(nlp_x, i, j, nx, N, m, s)
        end
        c[index:index+nx-1] = xip1 - (xi + h * sum_bk)
        index = index + nx

        # path constraints
        ui = get_control_at_time_step(nlp_x, i, nx, N, m, rk)
        if ctd.has_control_constraints
            c[index:index+ctd.dim_control_constraints-1] = ctd.control_constraints[2](ti, ui)
            index = index + ctd.dim_control_constraints
        end
        if ctd.has_state_constraints
            c[index:index+ctd.dim_state_constraints-1] = ctd.state_constraints[2](ti, xi[1:ctd.state_dimension])
            index = index + ctd.dim_state_constraints
        end
        if ctd.has_mixed_constraints
            c[index:index+ctd.dim_mixed_constraints-1] = ctd.mixed_constraints[2](ti, xi[1:ctd.state_dimension], ui)
            index = index + ctd.dim_mixed_constraints
        end
    end

    # path constraints at final time
    xf = get_state_at_time_step(nlp_x, N, nx, N)
    uf = get_control_at_time_step(nlp_x, N, nx, N, m, rk)
    if ctd.has_control_constraints
        c[index:index+ctd.dim_control_constraints-1] = ctd.control_constraints[2](tf, uf)      
        index = index + ctd.dim_control_constraints
    end  
    if ctd.has_state_constraints
        c[index:index+ctd.dim_state_constraints-1] = ctd.state_constraints[2](tf, xf[1:ctd.state_dimension])      
        index = index + ctd.dim_state_constraints
    end 
    if ctd.has_mixed_constraints
        c[index:index+ctd.dim_mixed_constraints-1] = ctd.mixed_constraints[2](tf, xf[1:ctd.state_dimension], uf)
        index = index + ctd.dim_mixed_constraints
    end

    # boundary conditions
    x0 = get_state_at_time_step(nlp_x, 0, nx, N)
    c[index:index+ctd.dim_boundary_conditions-1] = ctd.boundary_conditions[2](t0, x0[1:ctd.state_dimension], tf, xf[1:ctd.state_dimension])
    index = index + ctd.dim_boundary_conditions
    # null initial condition for augmented state (reformulated lagrangian cost)
    if ctd.has_lagrange_cost
        c[index] = nlp_x[nx]
        index = index + 1
    end

    return c
end
