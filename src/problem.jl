function ADNLProblem(ocp::OptimalControlModel, N::Integer, init=nothing)

    # direct_infos
    t0, tf_, n_x, m, f, ξ, ψ, ϕ, dim_ξ, dim_ψ, dim_ϕ, 
    has_ξ, has_ψ, has_ϕ, hasLagrangeCost, hasMayerCost, 
    dim_x, nc, dim_xu, g, f_Mayer, has_free_final_time, criterion = direct_infos(ocp, N)

    # IPOPT objective
    function ipopt_objective(xu)
        tf = get_final_time(xu, tf_, has_free_final_time)
        obj = 0
        if hasMayerCost
            x0 = get_state_at_time_step(xu, 0, dim_x, N)
            xf = get_state_at_time_step(xu, N, dim_x, N)
            obj = obj + g(t0, x0[1:n_x], tf, xf[1:n_x])
        end
        if hasLagrangeCost
            obj = obj + xu[(N+1)*dim_x]
        end
        return ismin(ocp) ? obj : -obj
    end

    # IPOPT constraints
    function ipopt_constraint(xu)
        """
        compute the constraints for the NLP : 
            - discretization of the dynamics via the trapeze method
            - boundary conditions
        inputs
        ocp :: ocp model
        xu :: 
            layout of the nlp unknown xu for trapeze discretization 
            additional state variable x_{n+1}(t) for the objective (Lagrange to Mayer formulation)
            [x_1(t_0), ... , x_{n+1}(t_0),
            ... , 
            x_{1}(t_N), ... , x_{n+1}(t_N),
            u_1(t_0), ... , u_m(t_0), 
            ... , 
            u_m(t_N), ..., u_m(t_N)]
        return
        c :: 
        """
        tf = get_final_time(xu, tf_, has_free_final_time)
        h = (tf-t0)/N
        c = zeros(eltype(xu),nc)
        #

        # state equation
        index = 1 # counter for the constraints
        for i in 0:N-1
            ti = t0 + i*h
            tip1 = t0 + i*h + h
            # state and control at the current state
            xi = get_state_at_time_step(xu, i, dim_x, N)
            xip1 = get_state_at_time_step(xu, i+1, dim_x, N)
            ui = get_control_at_time_step(xu, i, dim_x, N, m)
            uip1 = get_control_at_time_step(xu, i+1, dim_x, N, m)
            # state equation
            c[index:index+dim_x-1] = xip1 - (xi + 0.5*h*(f_Mayer(ti, xi, ui)+f_Mayer(tip1, xip1, uip1)))
            index = index + dim_x
            if has_ξ
                c[index:index+dim_ξ-1] = ξ[2](ti, ui)        # ui vector
                index = index + dim_ξ
            end
            if has_ψ
                c[index:index+dim_ψ-1] = ψ[2](ti, xi[1:n_x], ui)        # ui vector
                index = index + dim_ψ
            end
        end
        if has_ξ
            uf = get_control_at_time_step(xu, N, dim_x, N, m)
            c[index:index+dim_ξ-1] = ξ[2](tf, uf)      
            index = index + dim_ξ
          end  
        if has_ψ
            xf = get_state_at_time_step(xu, N, dim_x, N)
            uf = get_control_at_time_step(xu, N-1, dim_x, N, m)
            c[index:index+dim_ψ-1] = ψ[2](tf, xf, uf)        # ui is false because Euler
            index = index + dim_ψ
        end

        # boundary conditions
        # -------------------
        x0 = get_state_at_time_step(xu, 0, dim_x, N)
        xf = get_state_at_time_step(xu, N, dim_x, N)
        c[index:index+dim_ϕ-1] = ϕ[2](t0, x0[1:n_x], tf, xf[1:n_x])  # because Lagrange cost possible
        index = index + dim_ϕ
        if hasLagrangeCost
            c[index] = xu[dim_x]
            index = index + 1
        end

        return c
    end

    # bounds for the constraints
    function  constraints_bounds()
        lb = zeros(nc)
        ub = zeros(nc)
        index = 1 # counter for the constraints
        for i in 0:N-1
            index = index + dim_x          # leave 0 for the state equation
            if has_ξ
                lb[index:index+dim_ξ-1] = ξ[1]
                ub[index:index+dim_ξ-1] = ξ[3]
                index = index + dim_ξ
            end
            if has_ψ
                lb[index:index+dim_ψ-1] = ψ[1]
                ub[index:index+dim_ψ-1] = ψ[3]
                index = index + dim_ψ
            end
        end
        if has_ξ
            lb[index:index+dim_ξ-1] = ξ[1]
            ub[index:index+dim_ξ-1] = ξ[3]
            index = index + dim_ξ
        end 
        if has_ψ
            lb[index:index+dim_ψ-1] = ψ[1]
            ub[index:index+dim_ψ-1] = ψ[3]
            index = index + dim_ψ
        end
        # boundary conditions
        lb[index:index+dim_ϕ-1] = ϕ[1]
        ub[index:index+dim_ϕ-1] = ϕ[3]
        index = index + dim_ϕ
        if hasLagrangeCost
            lb[index] = 0.
            ub[index] = 0.
            index = index + 1
        end

        return lb, ub
    end

    # todo: retrieve optional bounds from ocp parsed constraints
    function variables_bounds()
        # unbounded case
        l_var = -Inf*ones(dim_xu)
        u_var = Inf*ones(dim_xu)
        return l_var, u_var
    end

    # generate initial guess
    function set_state_at_time_step!(x, i, dim_x, N, xu)
        if i > N
            error("trying to set x(t_i) for i > N")
        else
            xu[1+i*dim_x:(i+1)*dim_x] = x[1:dim_x]
        end
    end
    
    function set_control_at_time_step!(u, i, dim_x, N, m, xu)
        if i > N
            error("trying to set (t_i) for i > N")
        else
            xu[1+(N+1)*dim_x+i*m:m+(N+1)*dim_x+i*m] = u[1:m]
        end
    end

    function initial_guess()
        #println("Initialization: ", init)

        if init === nothing
            # default initialization
            xu0 = 1.1*ones(dim_xu)
        else
            if length(init) != (n_x + m)
                error("vector for initialization should be of size n+m",n_x+m)
            end
            # split state / control
            x_init = zeros(dim_x)
            x_init[1:n_x] = init[1:n_x]
            u_init = zeros(m)
            u_init[1:m] = init[n_x+1:n_x+m]
            
            # mayer -> lagrange additional state
            if hasLagrangeCost
                x_init[dim_x] = 0.1
            end

            # constant initialization
            xu0 = zeros(dim_xu)
            for i in 0:N
                set_state_at_time_step!(x_init, i, dim_x, N, xu0)
                set_control_at_time_step!(u_init, i, dim_x, N, m, xu0)
            end
        end
        return xu0
    end

    # variables bounds   
    l_var, u_var = variables_bounds()

    # initial guess
    xu0 = initial_guess()

    # free final time case
    if has_free_final_time
      xu0[end] = 1.0
      l_var[end] = 1.e-3
    end

    lb, ub = constraints_bounds()

    nlp = ADNLPModel(ipopt_objective, xu0, l_var, u_var, ipopt_constraint, lb, ub)    

    return nlp

end
