using CTDirect
using CTBase
using CTProblems


# simple integrator - time min
prob = Problem(:exponential, :time, :x_dim_1, :u_dim_1, :lagrange)
ocp = prob.model

@testset verbose = true showtiming = true "control_box_constraints" begin
    N = 2

    ctd = CTDirect.CTDirect_data(ocp, N, nothing)
    @test ctd.dim_control_constraints == 0
    @test ctd.has_control_box         == true
    @test ctd.has_control_constraints == false
    @test ctd.dim_control_box == 1
    @test ctd.dim_NLP_state == 1

    lb, ub = CTDirect.constraints_bounds(ctd)
    l_var, u_var = CTDirect.variables_bounds(ctd)
    true_lb = zeros(3*N) # test without boundary conditions because of the dictionary
    # true_lb[end-4:end] = [-1,0,0,0,0]
    true_l_var =  -Inf*ones((N+1)*4)
    true_l_var[3*(N+1)+1:4*(N+1)] .= -umax
    @test lb[1:end-5] == true_lb
    @test ub[1:end-5] == -true_lb
    @test l_var == true_l_var
    @test u_var == -l_var
 end

sol = solve(ocp, grid_size=100, print_level=5, tol=1e-12, rk_method=:trapeze)
p1 = plot(sol)
