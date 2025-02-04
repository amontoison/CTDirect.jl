module CTDirect

using CTBase
using DocStringExtensions
using Symbolics                 # for optimized AD
using ADNLPModels               # docp model with AD
using NLPModelsIpopt            # NLP solver
using LinearAlgebra             # norm

# Other declarations
const __grid_size_direct() = 100
const __print_level_ipopt = CTBase.__print_level_ipopt
const __mu_strategy_ipopt = CTBase.__mu_strategy_ipopt
const __display = CTBase.__display
const nlp_constraints = CTBase.nlp_constraints
const matrix2vec = CTBase.matrix2vec

# includes
#include("init.jl")
include("utils.jl")
include("problem.jl")
include("solution.jl")
include("solve.jl")

# export functions only for user
export available_methods
export is_solvable
export solve
#export OptimalControlInit
export directTranscription
export getNLP
export setDOCPInit
export OCPSolutionFromDOCP

# CTBase reexports
export @def
export Model
export Index
export state!
export control!
export variable!
export time!
export constraint!
export dynamics!
export objective!
export remove_constraint!
export OCPInit

end