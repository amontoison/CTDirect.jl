using CTDirect
using CTBase
using CTProblems


# simple integrator - time min
prob = Problem(:exponential, :time, :state_dim_1, :control_dim_1, :lagrange)
ocp = prob.model
sol = solve(ocp, grid_size=100, print_level=5, tol=1e-12, rk_method=:gauss2)
p1 = plot(sol)