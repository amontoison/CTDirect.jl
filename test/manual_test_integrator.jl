using CTDirect
using CTBase
using CTProblems


# double integrator - energy min
prob = Problem(:integrator, :energy)
ocp = prob.model

init = [1., 0.5, 0.3]

# basic
sol = solve(ocp, grid_size=100, print_level=5, tol=1e-12, mu_strategy="adaptive", init=init)
p1 = plot(sol)

# control box
constraint!(ocp, :control, -4.01, 4.01, :control_con1)
sol = solve(ocp, grid_size=100, print_level=5, tol=1e-12, mu_strategy="adaptive", init=init)
p2 =  plot(sol)

plot(p1,p2)