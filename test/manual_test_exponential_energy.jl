using CTDirect
using CTBase
using CTProblems


# simple integrator - energy min
prob = Problem(:exponential, :energy, :state_dim_1, :control_dim_1, :lagrange)
ocp = prob.model
sol = solve(ocp, grid_size=100, print_level=5, tol=1e-12, mu_strategy="adaptive", init=nothing)
p1 = plot(sol)