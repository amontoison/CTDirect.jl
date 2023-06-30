using CTDirect
using Test
using CTBase
#using LinearAlgebra
include("test_utils.jl")

# check local test suite
#@testset verbose = true showtiming = true "All problems" begin
    # run all scripts in subfolder suite/
#    include.(filter(contains(r".jl$"), readdir("./suite"; join=true)))  
#end


include("pb_tests_unitaires/simple_integrator.jl")
include("pb_tests_unitaires/double_integrator.jl")
include("pb_tests_unitaires/goddard_all_constraints.jl")

include("test_getters.jl")

