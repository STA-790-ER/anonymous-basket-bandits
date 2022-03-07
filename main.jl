
using LinearAlgebra, Statistics, Plots, Distributions, StatsBase, DelimitedFiles, CSV, Tables, Random, BenchmarkTools

#using BayesianOptimization, GaussianProcesses, Distributions
using Flux
using BSON: @load
using StatsBase
#BLAS.set_num_threads(1)

include("rollouts.jl")
include("policies.jl")
include("opt_alloc.jl")

#cnst mts = MersenneTwister.(1:Threads.nthreads())
# Parameters
const context_dim = 2
const context_mean = 0
const context_sd = 1
const obs_sd = 1

const bandit_count = 10
const bandit_prior_mean = 0
const bandit_prior_sd = 10

# Multi Action
const multi_count = 10

# SIMULATION HORIZON
const T = 100

# NUMBER OF GLOBAL SIMULATION EPISODES (PER INDEX JOB)
const n_episodes = 10

# DISCOUNT PARAMETER
const discount = 1.

# PARAMETER FOR EPSILON GREEDY POLICY
const epsilon = .02

# PARAMETERS FOR ALL ROLLOUT METHODS
const rollout_length = 5
const n_rollouts = 50

# PARAMETERS FOR SPSA OPTIMIZATION METHOD
const n_opt_rollouts = 100000
const n_spsa_iter = 300



## PARAMETERS FOR GRID OPTIMIZATION METHOD
const n_grid_iter = 7
const grid_ratio = 2
const grid_num = 6
const int_length = 2
const n_grid_rollouts = 50

### NON ADAPTIVE GRID (n_opt_rollouts used for opt rollouts)
#grid_margin_1 = [0, .5, 1, 2, 4]
#grid_margin_2 = [0, .5, 1, 2]
#grid_margin_3 = [.5, 1, 2]

grid_margin_1 = [0., 1.]
grid_margin_2 = [0., 1.]
grid_margin_3 = [1., 2.]

## SIMULATOR FUNCTION


const idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

neural_net_list = []
scale_list = []



## NEED TO DECIDE WHETHER TRUE IS USED
for i in 1:T
    @load "/hpc/home/jml165/rl/valneuralnets/valnn_$(i).bson" m
    scales = CSV.File("/hpc/home/jml165/rl/neuralnetscales/scales_$(i).csv", header = false) |> Tables.matrix
    push!(neural_net_list, m)
    push!(scale_list, scales)
end


upper_triangular_vec = function(M)

    d = size(M)[1]

    output = zeros(convert(Int64,(d+1) * d / 2))

    count = 1
    for i in 1:d
        for j in i:d
            output[count] = M[i, j]
            count += 1
        end
    end
        return output
end


function ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
        
    
        bandit_posterior_means = zeros(bandit_count, context_dim)
        bandit_posterior_covs = zeros(bandit_count, context_dim, context_dim)
    	bandit_param = copy(global_bandit_param)
        true_bandit_param = copy(global_bandit_param)
        EPREWARDS = zeros(T)
	    EPOPTREWARDS = zeros(T)

        for i in 1:bandit_count
            bandit_posterior_means[i, :] = repeat([bandit_prior_mean], context_dim)
            bandit_posterior_covs[i, :, :] = Diagonal(repeat([bandit_prior_sd^2], context_dim))
        end
        
        for t in 1:T
            context = randn(context_dim) * context_sd .+ context_mean
            true_expected_rewards = true_bandit_param * context
            #true_expected_rewards = bandit_posterior_means * context
            action = action_function(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
            true_expected_reward = true_expected_rewards[action]
            EPREWARDS[t] = true_expected_reward
            EPOPTREWARDS[t] = maximum(true_expected_rewards)
            #obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:],context)) + true_expected_reward
            obs = randn() * obs_sd + true_expected_reward
	        old_cov = bandit_posterior_covs[action, :, :]
	        CovCon = old_cov * context ./ obs_sd
	        #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)
	        bandit_posterior_covs[action, :, :] = old_cov - CovCon * CovCon' ./ (1 + dot(context, old_cov, context))
	        bandit_posterior_covs[action, :, :] = ((bandit_posterior_covs[action,:,:]) + bandit_posterior_covs[action,:,:]')/2
	        bandit_posterior_means[action, :] = (bandit_posterior_covs[action, :, :]) * (old_cov \ (bandit_posterior_means[action,:]) + context * obs / obs_sd^2)

	        println("Ep: ", ep, " - ", t, " of ", T, " for ", String(Symbol(action_function)))
            flush(stdout)


        end
	return EPREWARDS, EPOPTREWARDS
end

function contextual_bandit_simulator(action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

    REWARDS = zeros(T, n_episodes)
    OPTREWARDS = zeros(T, n_episodes)
    ep_count = 1
    

    ## USING BAYESIAN FORMULATION FOR FAIR COMPARISON
    #global_bandit_param = [1 0; 0 1; 2 -1]
    global_bandit_param = rand(Normal(bandit_prior_mean, bandit_prior_sd), bandit_count, context_dim)
    
    
    #threadreps = zeros(Threads.nthreads())


    for ep in 1:n_episodes
        EPREWARDS, EPOPTREWARDS = ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                       context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
        ep_count += 1
	REWARDS[:, ep] = EPREWARDS
	OPTREWARDS[:, ep] = EPOPTREWARDS
    end
    #print(threadreps)
    return REWARDS', OPTREWARDS'
end

function multi_ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
        
    
        bandit_posterior_means = zeros(bandit_count, context_dim)
        bandit_posterior_covs = zeros(bandit_count, context_dim, context_dim)
    	bandit_param = copy(global_bandit_param)
        true_bandit_param = copy(global_bandit_param)
        EPREWARDS = zeros(T)
	    EPOPTREWARDS = zeros(T)
        
        actions = zeros(multi_count)
        contexts = zeros(multi_count, context_dim)
        obss = zeros(multi_count)
        for i in 1:bandit_count
            bandit_posterior_means[i, :] = repeat([bandit_prior_mean], context_dim)
            bandit_posterior_covs[i, :, :] = Diagonal(repeat([bandit_prior_sd^2], context_dim))
        end
        
        for t in 1:T

            for m in 1:multi_count

                context = randn(context_dim) * context_sd .+ context_mean
                contexts[m, :] = context
                true_expected_rewards = true_bandit_param * context
                #true_expected_rewards = bandit_posterior_means * context
                action = action_function(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
                actions[m] = action
                true_expected_reward = true_expected_rewards[action]
                EPREWARDS[t] += true_expected_reward
                EPOPTREWARDS[t] += maximum(true_expected_rewards)
                #obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:],context)) + true_expected_reward
                obss[m] = randn() * obs_sd + true_expected_reward
	        
            end


            for act in 1:bandit_count

                action_obss = obss[actions .== act]
                action_contexts = contexts[actions .== act, :]
                old_cov = bandit_posterior_covs[act, :, :]
                #CovCon = old_cov * context ./ obs_sd
                #bandit_posterior_covs[act, :, :] = old_cov - CovCon * CovCon' ./ (1 + dot(context, old_cov, context))
                #bandit_posterior_covs[act, :, :] = ((bandit_posterior_covs[act,:,:]) + bandit_posterior_covs[act,:,:]')/2
                #bandit_posterior_means[act, :] = (bandit_posterior_covs[act, :, :]) * (old_cov \ (bandit_posterior_means[act,:]) + context * obs / obs_sd^2)
                #
                bandit_posterior_covs[act, :, :] = inv(inv(old_cov) + action_contexts' * action_contexts ./ obs_sd^2)
                bandit_posterior_covs[act, :, :] = ((bandit_posterior_covs[act,:,:]) + bandit_posterior_covs[act,:,:]')/2
                bandit_posterior_means[act, :, :] = bandit_posterior_covs[act, :, :] * (old_cov \ bandit_posterior_means[act, :] + action_contexts' * action_obss ./ obs_sd^2)

            end


	        #println("Ep: ", ep, " - ", t, " of ", T, " for ", String(Symbol(action_function)))
            #flush(stdout)


        end
	return EPREWARDS, EPOPTREWARDS
end



function single_expected_entropy_gain(cov, contexts)
    return log(det(Matrix(1.0I, context_dim, context_dim) + cov * contexts' * contexts ./ obs_sd.^2))
end


function get_adjacent(actions, remains, bandit_count)
    adj_mat = zeros(length(remains) * (bandit_count-1), length(actions))
    count = 0
    for i in 1:length(remains)
        
        for j in 1:bandit_count
            if actions[remains[i]] != j
                count += 1
                adj_mat[count, :] = actions
                adj_mat[count, remains[i]] = j
            end
        end
    end
    return adj_mat
end

function expected_entropy_gain(bandit_posterior_covs, contexts, actions)
    gain = 0
    for b in 1:bandit_count
       gain += single_expected_entropy_gain(@view(bandit_posterior_covs[b, :, :]), @view(contexts[actions .== b, :]))
    end
    return gain
end


function sss_step(bandit_posterior_covs, contexts, actions, remains, bandit_count, greedy_skew)
    adj_mat = get_adjacent(actions, remains, bandit_count)
    m = size(adj_mat)[1]
    scores = zeros(m)
    for i in 1:m
        scores[i] = expected_entropy_gain(bandit_posterior_covs, contexts, adj_mat[i, :])
    end

    scores ./= sum(scores)
    max_score = findmax(scores)
    return adj_mat[sample(1:m, Weights(scores)), :], adj_mat[max_score[2], :], max_score[1]
end
    
function sss(bandit_posterior_covs, contexts, actions, remains, niter, greedy_skew)
    

    max_actions = actions
    max_score = expected_entropy_gain(bandit_posterior_covs, contexts, actions)

    for i in 1:niter
        actions, local_max_actions, local_max_score = sss_step(bandit_posterior_covs, contexts, actions, remains, bandit_count, greedy_skew)

        if local_max_score > max_score
            max_actions = local_max_actions
            max_score = local_max_score
        end

    end

    return max_actions
end

# generate a first pass estimate of the max entropy action set by greedily sequentially constructing the action set
function sequential_max_entropy_step(bandit_posterior_covs, contexts, actions)

    max_score = 0
    max_actions = actions

    for i in 1:length(actions)
        if actions[i] == 0
            temp_actions = copy(actions)
            for j in 1:bandit_count
                temp_actions[i] = j
                eeg = expected_entropy_gain(bandit_posterior_covs, contexts[actions .!= 0, :], actions[actions .!= 0])
                if eeg > max_score
                    max_score = eeg
                    max_actions = temp_actions
                end
            end
        end
    end

    return max_actions
end

function sequential_max_entropy(bandit_posterior_covs, contexts, actions)
    seq_actions = copy(actions)

    while 0 in seq_actions
        seq_actions = sequential_max_entropy_step(bandit_posterior_covs, contexts, seq_actions)
    end

    return seq_actions


end

function coord_greedy_ep_contextual_bandit_simulator(ep, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
        
    
        bandit_posterior_means = zeros(bandit_count, context_dim)
        bandit_posterior_covs = zeros(bandit_count, context_dim, context_dim)
    	bandit_param = copy(global_bandit_param)
        true_bandit_param = copy(global_bandit_param)
        EPREWARDS = zeros(T)
	    EPOPTREWARDS = zeros(T)
        
        actions = convert(Vector{Int64}, zeros(multi_count))
        contexts = zeros(multi_count, context_dim)
        obss = zeros(multi_count)
        for i in 1:bandit_count
            bandit_posterior_means[i, :] = repeat([bandit_prior_mean], context_dim)
            bandit_posterior_covs[i, :, :] = Diagonal(repeat([bandit_prior_sd^2], context_dim))
        end
        
        agree_count = 0
        base_count = 0
        for t in 1:T
            remains = []
            for m in 1:multi_count

                context = randn(context_dim) * context_sd .+ context_mean
                contexts[m, :] = context
                #true_expected_rewards = true_bandit_param * context
                posterior_expected_rewards = bandit_posterior_means * context
                if rand() > epsilon
                    actions[m] = findmax(posterior_expected_rewards)[2]
                else
                    push!(remains, m)
                end
                #true_expected_reward = true_expected_rewards[action]
                #EPREWARDS[t] += true_expected_reward
                #EPOPTREWARDS[t] += maximum(true_expected_rewards)
                #obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:],context)) + true_expected_reward
                #obss[m] = randn() * obs_sd + true_expected_reward
	        
            end
            
            if length(remains) > 0

                post_covs_copy = copy(bandit_posterior_covs)
                
                remain_contexts = contexts[remains, :]

                for m in 1:multi_count
                    if ! (m in remains)
                        m_context = contexts[m, :]
                        post_covs_copy[actions[m], :, :] = inv(inv(post_covs_copy[actions[m], :, :]) + m_context * m_context' / obs_sd^2)
                    end
                end
                    
                if length(remains) == 1

                    actions[remains[1]] = findmax([logdet(I + post_covs_copy[i, :, :] * contexts[remains[1], :] * contexts[remains[1], :]' / obs_sd^2) for i = 1:bandit_count])[2]
                else
                    
                    print("\n")
                    print(post_covs_copy)
                    print("\n")
                    print(remain_contexts ./ obs_sd)
                    print("\n")
                    flush(stdout)
                    alloc = KL_optimal_allocations(post_covs_copy, remain_contexts ./ obs_sd)
                    remain_actions = [findmax(alloc[i, :])[2] for i = 1:length(remains)]
                    for i in 1:length(remains)
                        actions[remains[i]] = remain_actions[i]
                    end
                end
            end
   #         if length(remains) > 0
   #             max_val = 0
   #             free_actions = copy(actions)
   #             max_tup = zeros(length(remains))
   #             for tup in CartesianIndices(ntuple(d->1:bandit_count, length(remains)))
   #                 for i in 1:length(remains)
   #                     free_actions[remains[i]] = tup[i]
   #                 end
   #                 eeg = expected_entropy_gain(bandit_posterior_covs, contexts, free_actions)
   #                 if eeg > max_val
   #                     max_val = eeg
   #                     max_tup .= free_actions[remains]
   #                 end
   #             end


   #             sss_actions = sequential_max_entropy(bandit_posterior_covs, contexts, actions)
   #             sss_actions = sss(bandit_posterior_covs, contexts, sss_actions, remains, 1000, 0)
   #         
   #             actions[remains] .= max_tup

   #             agree_count += (actions == sss_actions)            
   #             base_count += 1
   #             actions = convert(Vector{Int64}, sss_actions)
   #         
   #         end
   #



            for m in 1:multi_count
                #print(actions)
                #print("\n")
                action = actions[m]
                context = contexts[m, :]
                true_expected_rewards = true_bandit_param * context
                EPREWARDS[t] += true_expected_rewards[action]
                EPOPTREWARDS[t] += maximum(true_expected_rewards)
                #obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:],context)) + true_expected_reward
                #obs = randn() * obs_sd + true_expected_rewards[action]
                obss[m] = randn() * obs_sd + true_expected_rewards[action]
	        
            end


            for act in 1:bandit_count

                action_obss = obss[actions .== act]
                action_contexts = contexts[actions .== act, :]
                old_cov = bandit_posterior_covs[act, :, :]
                #CovCon = old_cov * context ./ obs_sd
                #bandit_posterior_covs[act, :, :] = old_cov - CovCon * CovCon' ./ (1 + dot(context, old_cov, context))
                #bandit_posterior_covs[act, :, :] = ((bandit_posterior_covs[act,:,:]) + bandit_posterior_covs[act,:,:]')/2
                #bandit_posterior_means[act, :] = (bandit_posterior_covs[act, :, :]) * (old_cov \ (bandit_posterior_means[act,:]) + context * obs / obs_sd^2)
                #
                bandit_posterior_covs[act, :, :] = inv(inv(old_cov) + action_contexts' * action_contexts ./ obs_sd^2)
                bandit_posterior_covs[act, :, :] = ((bandit_posterior_covs[act,:,:]) + bandit_posterior_covs[act,:,:]')/2
                bandit_posterior_means[act, :, :] = bandit_posterior_covs[act, :, :] * (old_cov \ bandit_posterior_means[act, :] + action_contexts' * action_obss ./ obs_sd^2)

            end


	        #println("Ep: ", ep, " - ", t, " of ", T, " for coord eps greedy")
            #flush(stdout)


        end
	return EPREWARDS, EPOPTREWARDS, agree_count, base_count
end

function contextual_bandit_simulator(action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

    REWARDS = zeros(T, n_episodes)
    OPTREWARDS = zeros(T, n_episodes)
    ep_count = 1
    

    ## USING BAYESIAN FORMULATION FOR FAIR COMPARISON
    #global_bandit_param = [1 0; 0 1; 2 -1]
    global_bandit_param = rand(Normal(bandit_prior_mean, bandit_prior_sd), bandit_count, context_dim)
    
    
    #threadreps = zeros(Threads.nthreads())

    for ep in 1:n_episodes
        EPREWARDS, EPOPTREWARDS= ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                       context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
        
        
        ep_count += 1
	    REWARDS[:, ep] = EPREWARDS
	    OPTREWARDS[:, ep] = EPOPTREWARDS
    end
    #print(threadreps)
    return REWARDS', OPTREWARDS'
end

function multi_contextual_bandit_simulator(action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

    REWARDS = zeros(T, n_episodes)
    OPTREWARDS = zeros(T, n_episodes)
    ep_count = 1
    

    ## USING BAYESIAN FORMULATION FOR FAIR COMPARISON
    #global_bandit_param = [1 0; 0 1; 2 -1]
    global_bandit_param = rand(Normal(bandit_prior_mean, bandit_prior_sd), bandit_count, context_dim)
    
    
    #threadreps = zeros(Threads.nthreads())


    for ep in 1:n_episodes
    
        EPREWARDS, EPOPTREWARDS = multi_ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                       context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
        ep_count += 1
	    REWARDS[:, ep] = EPREWARDS
	    OPTREWARDS[:, ep] = EPOPTREWARDS
    
    end
    
    return REWARDS', OPTREWARDS'

end

function coord_greedy_contextual_bandit_simulator(T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

    REWARDS = zeros(T, n_episodes)
    OPTREWARDS = zeros(T, n_episodes)
    ep_count = 1
    

    ## USING BAYESIAN FORMULATION FOR FAIR COMPARISON
    #global_bandit_param = [1 0; 0 1; 2 -1]
    global_bandit_param = rand(Normal(bandit_prior_mean, bandit_prior_sd), bandit_count, context_dim)
    
    
    #threadreps = zeros(Threads.nthreads())

    tot_agree_count = 0
    tot_base_count = 0
    for ep in 1:n_episodes
    
        EPREWARDS, EPOPTREWARDS, agree_count, base_count = coord_greedy_ep_contextual_bandit_simulator(ep,T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                       context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
        ep_count += 1
	    REWARDS[:, ep] = EPREWARDS
	    OPTREWARDS[:, ep] = EPOPTREWARDS
        tot_agree_count += agree_count
        tot_base_count += base_count
    
	    println("Ep: ", ep, " of ", n_episodes, " for coord eps greedy")
        flush(stdout)
    end
    
    return REWARDS', OPTREWARDS', tot_agree_count, tot_base_count

end

## SIMULATIONS

# Sims


const discount_vector = discount .^ collect(0:(T-1))
#const run_policies = [greedy_policy, greedy_policy, thompson_policy, thompson_policy]
#const multi_ind = [true, false, true, false]

#const run_policies = [vftp2, vftp4, vftp8, vftp16, vftp32, vftp64, vegp2, vegp4, vegp8, vegp16, vegp32, vegp64]
#const multi_ind = [false, false, false, false, false, false, false, false, false, false, false, false]

#const run_policies = [greedy_policy, epsilon_greedy_policy] 
#const multi_ind = [true, true]


const run_policies = [greedy_policy, epsilon_greedy_policy]
const multi_ind = [true, true]
const coord_epsilon_greedy = true


# don't modify
regret_header = []

if coord_epsilon_greedy
    cumulative_discounted_regrets = zeros(T, 1+length(run_policies))
else
    cumulative_discounted_regrets = zeros(T, length(run_policies))
end

for pol in 1:length(run_policies)
    
    if multi_ind[pol]
        pol_rewards, pol_opt_rewards = multi_contextual_bandit_simulator(run_policies[pol], T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
        push!(regret_header, "$(run_policies[pol])_multi")
    
    else
        pol_rewards, pol_opt_rewards = contextual_bandit_simulator(run_policies[pol], T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
        push!(regret_header, "$(run_policies[pol])")
    end

    print("\n")

    pol_cumulative_discounted_regret = cumsum(discount_vector .* [mean(c) for c in eachcol(pol_opt_rewards - pol_rewards)])
    cumulative_discounted_regrets[:, pol] = pol_cumulative_discounted_regret

end

if coord_epsilon_greedy
    ceg_rewards, ceg_opt_rewards, agree_count, base_count = coord_greedy_contextual_bandit_simulator(T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
    cdr = cumsum(discount_vector .* [mean(c) for c in eachcol(ceg_opt_rewards - ceg_rewards)])
    cumulative_discounted_regrets[:, 1+length(run_policies)] = cdr
    push!(regret_header, "Coord_Epsilon_Greedy")
    CSV.write("/hpc/home/jml165/rl/arrayresults/agreeprop_$(idx).csv", Tables.table([agree_count base_count]))
end



#const combReg = [greedyCumDiscReg epsgreedyCumDiscReg thompsonCumDiscReg bayesucbCumDiscReg squarecbCumDiscReg lambdaCumDiscReg lambdameanCumDiscReg greedyrolloutCumDiscReg betterlambdaCumDiscReg optlambdaCumDiscReg bayesoptlambdaCumDiscReg]

#const combReg = [greedyCumDiscReg thompsonCumDiscReg greedyrolloutCumDiscReg valgreedyrolloutCumDiscReg valbettergridlambdaCumDiscReg valgreedythompsonCumDiscReg]
CSV.write("/hpc/home/jml165/rl/arrayresults/results_$(idx).csv", Tables.table(cumulative_discounted_regrets), header=regret_header)
#CSV.write("/hpc/home/jml165/rl/arrayresults/results_$(idx).csv",  Tables.table(combReg), header=["Greedy", "EpsGreedy", "Thompson", "BayesUCB", "SquareCB", "Lambda","LambdaMean","GreedyRollout", "BetterLambda","OptLambda", "BayesOptLambda"])
