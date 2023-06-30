
using Plots

using Match

toto = 7

@match toto begin
    Int => println("Int")
    x::Float => println("Float")
    end


    function fct(T)
    x0 = -1
    xf = 0
    
    obj = []
    P0 = []
    for tf in T
    a = xf - x0*exp(-tf)
    b = sinh(tf)
    p0 = a/b
    x(t) = p0*sinh(t) + x0*exp(-t)
    p(t) = exp(t)*p0
    u(t) = p(t)
    objective = (exp(2*tf)-1)*p0^2/4
    obj = push!(obj, objective)
    P0  = push!(P0, p0)
    end
    return obj,P0
    end

    T = 0.5:0.1:5
    obj,P0 = fct(T)
    display([T P0 obj])
    p1 = plot(T,obj)
    p2 = plot(T,P0)
    plot(p1,p2)

    # sol tf = 2


