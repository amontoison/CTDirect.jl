"""
$(TYPEDSIGNATURES)

Solve the optimal control problem

Input : 
ocp : functional description of the optimal control problem (cf. ocp.jl)
grid_size   : number of time steps for the discretization
      Int
rk_method : rRunge-Kutta method

Output
sol : solution of the discretized problem
      ...
```@examples
julia> using CTDirect
julia> using CTProblems
julia> ocp =  Problem((:integrator, :dim2, :energy)
julia> solve(ocp)
```
"""

function solve(ocp::OptimalControlModel, 
  description...;
  grid_size::Integer=__grid_size_direct(),
  rk_method:: Symbol=__rk_method(),
  print_level::Integer=__print_level_ipopt(),
  mu_strategy::String=__mu_strategy_ipopt(),
  display::Bool=__display(),
  init=nothing,  #NB. for now, can be nothing or (n+m) vector
  kwargs...)

  # description... is unused here. See OptimalControl.jl/src/solve.jl for an example of use

  # no display
  print_level = display ?  print_level : 0

  # build internal structure for direct method
  ctd = CTDirect_data(ocp, grid_size, rk_method, init)

  # from OCP to NLP
  nlp = ADNLProblem(ocp, ctd)

  # solve by IPOPT: more info at 
  # https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl/blob/main/src/NLPModelsIpopt.jl#L119
  # options of ipopt: https://coin-or.github.io/Ipopt/OPTIONS.html
  # callback: https://github.com/jump-dev/Ipopt.jl#solver-specific-callback
  # sb="yes": remove ipopt header
  ipopt_solution = ipopt(nlp, print_level=print_level, mu_strategy=mu_strategy, sb="yes"; kwargs...)

  # Parse solution from NLP to OCP variables and constraints
  sol = DirectSolution(ocp, ctd, ipopt_solution)

  return sol

end
