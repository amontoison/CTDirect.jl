using CTDirect
using Test
using CTBase
using CTProblems
using LinearAlgebra

include("test_utils.jl")

include("test_getters.jl")

# test all problems in CTProblems (except consumption ones)
@testset verbose = true showtiming = true "Smooth problems" begin

    problems_list = @Problems !:consumption
    #problems_list = @Problems !:non_diff_wrt_u
    for prob in problems_list
        println("Test: ",prob.title)
        @testset "$(prob.title)" begin
            for rk_method in CTDirect.liste_of_rk_methods
                @testset "$(rk_method)" begin
                    grid_size = 50
                    if rk_method == :exp_euler || rk_method == :imp_euler
                        grid_size  = 300
                        println("gris_size = ", grid_size)
                    end
                    sol = solve(prob.model, grid_size=grid_size, rk_method=rk_method, print_level=0)
                    @test sol.objective ≈ prob.solution.objective rtol=1e-2
                end
            end
        end
    end        
end
