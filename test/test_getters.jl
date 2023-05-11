#using CTDirect
#using Test
#using CTBase
#using CTProblems

@testset verbose = true showtiming = true "Getters" begin
    problems_list = Problems() #Problems(:(!:consumption))
    for prob in problems_list
        @testset "$(prob.title)" begin
            for rk_method in CTDirect.liste_of_rk_methods_for_tests
                @testset "$(rk_method)" begin
                    ocp = prob.model
                    ctd = CTDirect.CTDirect_data(ocp, 100, rk_method, nothing)
                    N = ctd.dim_NLP_steps
                    nlp_x = 1:ctd.dim_NLP_variables
                    nx = ctd.dim_NLP_state
                    m = ctd.control_dimension
                    N = ctd.dim_NLP_steps
                    rk =  ctd.rk
                    s = rk.stage
                    s_u = rk.s_u
                    X = reshape(nlp_x[1:(N+1)*nx],(nx,N+1))
                    for i in 1:N+1
                        @test X[:,i] == CTDirect.get_state_at_time_step(nlp_x, i-1, nx, N)
                    end
                    start =  (N+1)*nx
                    U = reshape(nlp_x[start+1:start+N*s_u*m],(m,s_u,N))
                    for i in 0:N-1
                        # ajouter j --> j'
                        for j in 1:s
                            if rk.lobatto
                                if rk.butcher_c[j] == 1 
                                    if i != N-1
                                        @test U[:,1,i+2] == CTDirect.   get_control_at_time_stage(nlp_x, i, j, nx, N, m, rk)
                                    end
                                else
                                    @test U[:,rk.u_stage[j],i+1] == CTDirect.get_control_at_time_stage(nlp_x, i, j, nx, N, m, rk)
                                end
                            else
                                @test U[:,rk.u_stage[j],i+1] == CTDirect.get_control_at_time_stage(nlp_x, i, j, nx, N, m, rk)
                            end
                        end
                    end
                    start = start + N*s_u*m
                    if ctd.rk.lobatto
                        u_tf = nlp_x[start+1:start+m]
                        @test u_tf == CTDirect.get_control_at_time_stage(nlp_x, N-1, s, nx, N, m, rk)
                     start = start + m
                    end

                    K = reshape(nlp_x[start+1:start+N*s*nx],(nx,s,N))
                    for i in 1:N
                        for j in 1:s
                            @test K[:,j,i] == CTDirect.get_k_at_time_stage(nlp_x, i-1, j, nx, N, m, rk)
                        end

                    end
                end
            end
        end
    end
end


