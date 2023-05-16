using CTDirect
using Test
using CTBase
using CTProblems
using LinearAlgebra

include("test_utils.jl")

# test all problems in CTProblems (except consumption ones)
@testset verbose = true showtiming = true "RK: midpoint // step control" begin
    problems_list = Problems(:(!:consumption & !:classical))
    for prob in problems_list
        println("Test: ",prob.description)
        @testset "$(prob.description)" begin
            sol = solve(prob.model, grid_size=50, print_level=0, max_iter=500, rk_method=:midpoint)
            @test sol.objective ≈ prob.solution.objective rtol=1e-2
        end
    end        
end
@testset verbose = true showtiming = true "RK: midpoint // stage control" begin
    problems_list = Problems(:(!:consumption & !:classical))
    for prob in problems_list
        println("Test: ",prob.description)
        @testset "$(prob.description)" begin
            sol = solve(prob.model, grid_size=50, print_level=0, max_iter=500, rk_method=:midpoint, control_disc_method=:stage)
            @test sol.objective ≈ prob.solution.objective rtol=1e-2
        end
    end        
end


@testset verbose = true showtiming = true "RK: gauss2 // step control" begin
    problems_list = Problems(:(!:consumption & !:classical))
    for prob in problems_list
        println("Test: ",prob.description)
        @testset "$(prob.description)" begin
            sol = solve(prob.model, grid_size=50, print_level=0, max_iter=500, rk_method=:gauss2)
            @test sol.objective ≈ prob.solution.objective rtol=1e-2
        end
    end        
end
@testset verbose = true showtiming = true "RK: gauss2 // stage control" begin
    problems_list = Problems(:(!:consumption & !:classical))
    for prob in problems_list
        println("Test: ",prob.description)
        @testset "$(prob.description)" begin
            sol = solve(prob.model, grid_size=50, print_level=0, max_iter=500, rk_method=:gauss2, control_disc_method=:stage)
            @test sol.objective ≈ prob.solution.objective rtol=1e-2
        end
    end        
end