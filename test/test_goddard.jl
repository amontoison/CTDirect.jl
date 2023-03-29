# goddard with state constraint - maximize altitude
println("Goddard test")
prob = Problem(:goddard, :state_constraint)
ocp = prob.model

# initial guess (constant state and control functions)
init = [1.01, 0.05, 0.8, 0.1]

# solve problem
sol = solve(ocp, grid_size=100, print_level=5, tol=1e-12, mu_strategy="adaptive", init=init)

# check solution
@test sol.objective ≈ prob.solution.objective atol=5e-3
