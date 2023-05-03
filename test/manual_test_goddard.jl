using CTDirect
using CTBase
using CTProblems


# simple integrator - time min
prob = Problem(:goddard, :classical)
ocp = prob.model
sol = solve(ocp, grid_size=50, print_level=5, tol=1e-12, rk_method=:gauss2)
p1 = plot(sol)