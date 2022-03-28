using JuMP, Pavito, Ipopt, GLPK, LinearAlgebra

#k = 4 # number arms
#m = 3 # number agents
#p = 3 # context dimension

#covs = randn(k, p, p)
#contexts = randn(m, p)



function KL_optimal_allocations(covs, contexts)

    k, p = size(covs)[1:2]
    m = size(contexts)[1]
    invcovs = zeros(k, p, p)
    for i in 1:k
        invcovs[i, :, :] = inv(covs[i, :, :])
    end
    solver = JuMP.optimizer_with_attributes(
            Pavito.Optimizer,
            "mip_solver" => optimizer_with_attributes(GLPK.Optimizer, "msg_lev" => GLPK.GLP_MSG_OFF),
            "cont_solver" => optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0),
            "mip_solver_drives" => false,
            "log_level" => 0
           )

    mod = Model(solver)

    set_silent(mod)

    @variable(mod, S[1:m, 1:k] >= 0, Bin)
    @constraint(mod, sum(S, dims=2) .<= 1)
    # @expression(mod, expr[i=1:k], invcovs[i,:,:]+ contexts' * diagm(S[:, i]) * contexts)


    function f(x...)

        X = reshape(collect(x), m, k)
        return sum(logdet(invcovs[i,:,:]+ contexts' * diagm(X[:, i]) * contexts) for i=1:k)

    end
    JuMP.register(mod, :f, m*k, f, autodiff=true)
    @NLobjective(mod, Max, f(S...))

    optimize!(mod)
    #print(value.(S))
    return value.(S)
end

#KL_optimal_allocations(invcovs, contexts)


function KL_optimal_allocations_with_restrictions(covs, contexts, restrictions)

    k, p = size(covs)[1:2]
    m = size(contexts)[1]
    invcovs = zeros(k, p, p)
    for i in 1:k
        invcovs[i, :, :] = inv(covs[i, :, :])
    end
    solver = JuMP.optimizer_with_attributes(
            Pavito.Optimizer,
            "mip_solver" => optimizer_with_attributes(GLPK.Optimizer, "msg_lev" => GLPK.GLP_MSG_OFF),
            "cont_solver" => optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0),
            "mip_solver_drives" => false,
            "log_level" => 0
           )

    mod = Model(solver)

    set_silent(mod)

    @variable(mod, S[1:m, 1:k] >= 0, Bin)
    @constraint(mod, sum(S, dims=2) .<= 1)
    @constraint(mod, S .<= restrictions)
    # @expression(mod, expr[i=1:k], invcovs[i,:,:]+ contexts' * diagm(S[:, i]) * contexts)


    function f(x...)

        X = reshape(collect(x), m, k)
        return sum(logdet(invcovs[i,:,:]+ contexts' * diagm(X[:, i]) * contexts) for i=1:k)

    end
    JuMP.register(mod, :f, m*k, f, autodiff=true)
    @NLobjective(mod, Max, f(S...))

    optimize!(mod)
    #print(value.(S))
    return value.(S)
end

