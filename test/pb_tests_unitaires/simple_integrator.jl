using CTDirect
using CTBase

println("Test: simple integrator")

OCP = []
# min energy
ocp1 = Model()
state!(ocp1, 1)
control!(ocp1, 1)
time!(ocp1, [0, 1])
constraint!(ocp1, :initial, -1, :initial_constraint)
constraint!(ocp1, :final, 0, :final_constraint)
dynamics!(ocp1, (x, u) -> -x + u)
objective!(ocp1, :lagrange, (x, u) -> u^2)

push!(OCP,ocp1)