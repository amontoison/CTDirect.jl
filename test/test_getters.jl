#using CTDirect
#using Test
#using CTBase
#using CTProblems

@testset verbose = true showtiming = true "Getters" begin
    for ocp in OCP
        #@testset "$(ocp.title)" begin
            #for rk_method in CTDirect.liste_of_rk_methods_for_tests
                #@testset "$(rk_method)" begin
                    #ocp = prob.model
                    init = CTDirect.OptimalControlInit()
                    ctd = CTDirect.CTDirect_data(ocp, 1, init) # rk_method, init)
                    xu = CTDirect.initial_guess(ctd)
                    N = ctd.dim_NLP_steps
                    nlp_x = 1:ctd.dim_NLP_variables
                    n  = ctd.state_dimension
                    nx = ctd.dim_NLP_state
                    m = ctd.control_dimension
                    #rk =  ctd.rk
                    #s = rk.stage
                    #s_u = rk.s_u
                    X = reshape(xu[1:(N+1)*nx],(nx,N+1))
                    for i in 0:N
                        @test vector_scalar(X[1:n,i+1]) == CTDirect.get_state_at_time_step(xu, ctd, i)
                    end
                    start =  (N+1)*nx
                    # à voir
                    U = reshape(xu[start+1:start+(N+1)*m],(m,N+1))
                 
                    for i in 0:N
                        @test vector_scalar(U[:,i+1] )== CTDirect.get_control_at_time_step(xu, ctd, i)
                    end
                    start = start + N*m
                    # variable

                #end
            #end
        #end
    end
end