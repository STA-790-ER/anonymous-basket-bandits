
using LinearAlgebra, Statistics, Plots, Distributions, StatsBase, DelimitedFiles, CSV, Tables, Random, BenchmarkTools
using XGBoost
#using BayesianOptimization, GaussianProcesses, Distributions
using Flux
using BSON: @load
using StatsBase
using Suppressor
#BLAS.set_num_threads(1)

mutable struct exp4
    policy_list::Vector{Function}
    policy_probs::Vector{Float64}
    #policy::Function
end
include("rollouts.jl")
include("policies.jl")
include("opt_alloc.jl")
include("coordinated_utilities.jl")
include("glm.jl")
include("mcmc.jl")
include("vb.jl")
include("gp.jl")
include("mab.jl")
#cnst mts = MersenneTwister.(1:Threads.nthreads())
# Parameters
const context_dim = 5
const context_mean = 0
const context_sd = .25 
# set to false for gp
const context_constant = true
const obs_sd = 4

const bandit_count = 3
const bandit_prior_mean = 0
const bandit_prior_sd = 1


# MCMC parameters
const prior_mean = repeat([bandit_prior_mean], context_dim)
const prior_cov = diagm(repeat([bandit_prior_sd^2], context_dim))
const proposal_sd = .1
const n_burn = 200
const n_dup = 10

# Multi Action
const multi_count = 10

# SIMULATION HORIZON
const T = 100

# NUMBER OF GLOBAL SIMULATION EPISODES (PER INDEX JOB)
const n_episodes = 1 

# DISCOUNT PARAMETER
const discount = .95

# PARAMETER FOR EPSILON GREEDY POLICY
const epsilon = .4
const decreasing = true
# PARAMETERS FOR ALL ROLLOUT METHODS
const rollout_length = 30# 20
const n_rollouts = 10000 # 100000

# PARAMETERS FOR SPSA OPTIMIZATION METHOD
const n_opt_rollouts = 15000# 100000
const n_spsa_iter = 20000



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

### NON STATIONARY PARAMETERS

const delta = 1.


### GP PARAMETERS

const kernel_scale = 10
const kernel_bandwidth = 10

### MAB PARAMETERS
const a_beta_prior = .5
const b_beta_prior = .5

### VB PARAMETER

const vb_rollout_mult = 5
const vb_rollout_tol = .00001
const vb_policy_tol = .01

### ADAPTIVE PARAM

const expected_regret_thresh = .0002
const action_expected_regret_thresh = .0002

## DP PARAM
const dp_rollout_count = 5000
const dp_within_rollout_count = 30
const num_round  = 3
const dp_length = 3
const param = ["max_depth" => 3,
         "eta" => .5,
         "objective" => "reg:squarederror",
         "verbosity" => 0]
const metrics = ["rmse"]

## SIMULATOR FUNCTION


const idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

neural_net_list = []
scale_list = []

multi_neural_net_list = []
multi_scale_list = []

bern_neural_net_list = []
bern_scale_list = []

use_nn = false

#GC.enable_logging(true)

## NEED TO DECIDE WHETHER TRUE IS USED
if use_nn
for i in 1:T
    @load "/hpc/home/jml165/rl/valneuralnets/valnn_$(i).bson" m
    scales = CSV.File("/hpc/home/jml165/rl/neuralnetscales/scales_$(i).csv", header = false) |> Tables.matrix
    push!(neural_net_list, m)
    push!(scale_list, scales)
end
for i in 1:T
    @load "/hpc/home/jml165/rl/multivalneuralnets/multivalnn_$(i).bson" m
    scales = CSV.File("/hpc/home/jml165/rl/multineuralnetscales/multiscales_$(i).csv", header = false) |> Tables.matrix
    push!(multi_neural_net_list, m)
    push!(multi_scale_list, scales)
end
for i in 1:T
    @load "/hpc/home/jml165/rl/bernvalneuralnets/bernvalnn_$(i).bson" m
    scales = CSV.File("/hpc/home/jml165/rl/bernneuralnetscales/bernscales_$(i).csv", header = false) |> Tables.matrix
    push!(bern_neural_net_list, m)
    push!(bern_scale_list, scales)
end
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

function generate_context(context_dim, context_mean, context_sd, context_constant)
    if context_constant
        return [1; context_mean .+ context_sd .* randn(context_dim - 1)]
    else
        return context_mean .+ context_sd .* randn(context_dim)
    end
end
const ground_truth_param = bandit_prior_mean .+ bandit_prior_sd .* randn(bandit_count, context_dim, n_episodes)
const ground_truth_context = [[generate_context(context_dim, context_mean, context_sd, context_constant) for _ in 1:T] for _ in 1:n_episodes]
const ground_truth_obs = obs_sd .* randn(T, n_episodes)

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
            #context = generate_context(context_dim, context_mean, context_sd, context_constant)
            context = ground_truth_context[ep][t]
            true_expected_rewards = true_bandit_param * context
            #true_expected_rewards = bandit_posterior_means * context
            action = action_function(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
            true_expected_reward = true_expected_rewards[action]
            EPREWARDS[t] = true_expected_reward
            EPOPTREWARDS[t] = maximum(true_expected_rewards)
            #obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:],context)) + true_expected_reward
            obs = ground_truth_obs[t, ep] + true_expected_reward
	        old_cov = bandit_posterior_covs[action, :, :]
	        CovCon = old_cov * context ./ obs_sd
	        #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)
	        bandit_posterior_covs[action, :, :] = old_cov - CovCon * CovCon' ./ (1 + dot(context, old_cov, context) / obs_sd^2)
	        bandit_posterior_covs[action, :, :] = ((bandit_posterior_covs[action,:,:]) + bandit_posterior_covs[action,:,:]')/2
	        bandit_posterior_means[action, :] = (bandit_posterior_covs[action, :, :]) * (old_cov \ (bandit_posterior_means[action,:]) + context * obs / obs_sd^2)

	        println("Ep: ", ep, " - ", t, " of ", T, " for ", String(Symbol(action_function)))
            flush(stdout)

            #GC.gc()
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
    
    
    #threadreps = zeros(Threads.nthreads())


    for ep in 1:n_episodes
    
        #global_bandit_param = rand(Normal(bandit_prior_mean, bandit_prior_sd), bandit_count, context_dim)
        global_bandit_param = ground_truth_param[:, :, ep]
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

                context = generate_context(context_dim, context_mean, context_sd, context_constant)
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
            if decreasing
                thresh = epsilon / t
            else
                thresh = epsilon
            end
            
            for m in 1:multi_count

                context = generate_context(context_dim, context_mean, context_sd, context_constant)
                contexts[m, :] = context
                #true_expected_rewards = true_bandit_param * context
                posterior_expected_rewards = bandit_posterior_means * context
                if rand() > thresh
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
function approx_coord_greedy_ep_contextual_bandit_simulator(ep, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
            if decreasing
                thresh = epsilon / t
            else
                thresh = epsilon
            end
            for m in 1:multi_count

                context = generate_context(context_dim, context_mean, context_sd, context_constant)
                contexts[m, :] = context
                #true_expected_rewards = true_bandit_param * context
                posterior_expected_rewards = bandit_posterior_means * context
                if rand() > thresh
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
            
            #if length(remains) > 0

            #    post_covs_copy = copy(bandit_posterior_covs)
            #    
            #    remain_contexts = contexts[remains, :]

            #    for m in 1:multi_count
            #        if ! (m in remains)
            #            m_context = contexts[m, :]
            #            post_covs_copy[actions[m], :, :] = inv(inv(post_covs_copy[actions[m], :, :]) + m_context * m_context' / obs_sd^2)
            #        end
            #    end
            #        
            #    if length(remains) == 1

            #        actions[remains[1]] = findmax([logdet(I + post_covs_copy[i, :, :] * contexts[remains[1], :] * contexts[remains[1], :]' / obs_sd^2) for i = 1:bandit_count])[2]
            #    else
            #        
            #        print("\n")
            #        print(post_covs_copy)
            #        print("\n")
            #        print(remain_contexts ./ obs_sd)
            #        print("\n")
            #        flush(stdout)
            #        alloc = KL_optimal_allocations(post_covs_copy, remain_contexts ./ obs_sd)
            #        remain_actions = [findmax(alloc[i, :])[2] for i = 1:length(remains)]
            #        for i in 1:length(remains)
            #            actions[remains[i]] = remain_actions[i]
            #        end
            #    end
            #end
            if length(remains) > 0
                #max_val = 0
                #free_actions = copy(actions)
                #max_tup = zeros(length(remains))
                #for tup in CartesianIndices(ntuple(d->1:bandit_count, length(remains)))
                #    for i in 1:length(remains)
                #        free_actions[remains[i]] = tup[i]
                #    end
                #    eeg = expected_entropy_gain(bandit_posterior_covs, contexts, free_actions)
                #    if eeg > max_val
                #        max_val = eeg
                #        max_tup .= free_actions[remains]
                #    end
                #end

                sa_actions = sequential_max_entropy(bandit_posterior_covs, contexts, actions)
                
                sa_actions = sa(bandit_posterior_covs, contexts, sa_actions, remains, 10000, 10, .99)
            
                #actions = se_actions

                #agree_count += (actions == sss_actions)            
                #base_count += 1
                actions = convert(Vector{Int64}, sa_actions)
            
            end
   



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

function coord_thompson_ep_contextual_bandit_simulator(ep, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
        
        for t in 1:T
            
            
            for m in 1:multi_count
                context = generate_context(context_dim, context_mean, context_sd, context_constant)
                contexts[m, :] = context
            end
            # coordinated action selection

            curr_actions = convert(Vector{Int64}, zeros(multi_count))
            max_actions = convert(Vector{Int64}, zeros(multi_count))
            max_gain = 0

            for n in 1:thompson_count

                thompson_samples = zeros(bandit_count, context_dim)
    
                for bandit in 1:bandit_count
	                thompson_samples[bandit, :] = rand(MvNormal((bandit_posterior_means[bandit,:]), (bandit_posterior_covs[bandit,:,:])))
                end

                for m in 1:multi_count
                    curr_actions[m] = findmax(thompson_samples * contexts[m, :])[2]
                end

                curr_gain = expected_entropy_gain(bandit_posterior_covs, contexts, curr_actions)
                print("\n")
                print(curr_gain)
                print("\n")

                flush(stdout)

                if curr_gain > max_gain
                    max_gain = curr_gain
                    max_actions = curr_actions
                end

            end
            

            actions .= max_actions

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
	return EPREWARDS, EPOPTREWARDS
end

function coord_thompson_2_ep_contextual_bandit_simulator(ep, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
        
        for t in 1:T
            
            
            for m in 1:multi_count
                context = generate_context(context_dim, context_mean, context_sd, context_constant)
                contexts[m, :] = context
            end
            # coordinated action selection

            thompson_samples = zeros(bandit_count, context_dim)
    
            for bandit in 1:bandit_count
	            thompson_samples[bandit, :] = rand(MvNormal((bandit_posterior_means[bandit,:]), (bandit_posterior_covs[bandit,:,:])))
            end

            actions = constrained_thompson_actions(100000, thompson_samples, bandit_posterior_means, bandit_posterior_covs, contexts)
            actions = convert(Vector{Int64}, actions)

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
	return EPREWARDS, EPOPTREWARDS
end
function coord_thompson_3_ep_contextual_bandit_simulator(ep, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
        
        for t in 1:T
            
            
            for m in 1:multi_count
                context = generate_context(context_dim, context_mean, context_sd, context_constant)
                contexts[m, :] = context
            end
            # coordinated action selection

            thompson_samples = zeros(multi_count, bandit_count, context_dim)
    
            for m in 1:multi_count
                for bandit in 1:bandit_count
	                thompson_samples[m, bandit, :] = rand(MvNormal((bandit_posterior_means[bandit,:]), (bandit_posterior_covs[bandit,:,:])))
                end
            end
            

            #actions = constrained_thompson_actions(100000, thompson_samples, bandit_posterior_means, bandit_posterior_covs, contexts)
            #actions = convert(Vector{Int64}, actions)
            for m in 1:multi_count
                actions[m] = findmax(thompson_samples[m, :, :] * contexts[m, :])[2]
            end
            
            thompson_expected_rewards = [dot(bandit_posterior_means[actions[m], :], contexts[m, :]) for m = 1:multi_count]
            restrictions = convert(Matrix{Int64}, zeros(multi_count, bandit_count))

            for m in 1:multi_count
                for b in 1: bandit_count
                    if dot(bandit_posterior_means[b, :], contexts[m, :]) >= thompson_expected_rewards[m]
                        restrictions[m, b] = 1
                    end
                end
            end
            if sum(restrictions) > multi_count                
            #alloc = KL_optimal_allocations_with_restrictions(bandit_posterior_covs, contexts, restrictions)
            #actions = [findmax(alloc[i, :])[2] for i = 1:multi_count]
                actions = sa_restricted(bandit_posterior_covs, contexts, actions, restrictions, 50000, 10, .99)
                actions = convert(Vector{Int64}, actions)
            end
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
	return EPREWARDS, EPOPTREWARDS
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
    
        EPREWARDS, EPOPTREWARDS, agree_count, base_count = approx_coord_greedy_ep_contextual_bandit_simulator(ep,T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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


function coord_thompson_contextual_bandit_simulator(T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
    
        EPREWARDS, EPOPTREWARDS = coord_thompson_ep_contextual_bandit_simulator(ep,T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                       context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
        ep_count += 1
	    REWARDS[:, ep] = EPREWARDS
	    OPTREWARDS[:, ep] = EPOPTREWARDS
        #tot_agree_count += agree_count
        #tot_base_count += base_count
    
	    println("Ep: ", ep, " of ", n_episodes, " for coord thompson")
        flush(stdout)
    end
    
    return REWARDS', OPTREWARDS'

end

function coord_thompson_2_contextual_bandit_simulator(T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
    
        EPREWARDS, EPOPTREWARDS = coord_thompson_2_ep_contextual_bandit_simulator(ep,T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                       context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
        ep_count += 1
	    REWARDS[:, ep] = EPREWARDS
	    OPTREWARDS[:, ep] = EPOPTREWARDS
        #tot_agree_count += agree_count
        #tot_base_count += base_count
    
	    println("Ep: ", ep, " of ", n_episodes, " for coord thompson")
        flush(stdout)
    end
    
    return REWARDS', OPTREWARDS'

end


function coord_thompson_3_contextual_bandit_simulator(T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
    
        EPREWARDS, EPOPTREWARDS = coord_thompson_3_ep_contextual_bandit_simulator(ep,T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                       context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
        ep_count += 1
	    REWARDS[:, ep] = EPREWARDS
	    OPTREWARDS[:, ep] = EPOPTREWARDS
        #tot_agree_count += agree_count
        #tot_base_count += base_count
    
	    println("Ep: ", ep, " of ", n_episodes, " for coord thompson")
        flush(stdout)
    end
    
    return REWARDS', OPTREWARDS'

end

#####################
# INFORMATION DIRECTED SAMPLING
# ####################


function coord_ids_ep_contextual_bandit_simulator(ep, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
        
    
        bandit_posterior_means = zeros(bandit_count, context_dim)
        bandit_posterior_covs = zeros(bandit_count, context_dim, context_dim)
    	bandit_param = copy(global_bandit_param)
        true_bandit_param = copy(global_bandit_param)
        EPREWARDS = zeros(T)
	    EPOPTREWARDS = zeros(T)
        
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
            actions = convert(Vector{Int64}, zeros(multi_count))
            for m in 1:multi_count
                context = generate_context(context_dim, context_mean, context_sd, context_constant)
                contexts[m, :] = context
            end

            sa_actions = sequential_max_entropy(bandit_posterior_covs, contexts, actions)

#            start_gain = expected_entropy_gain(bandit_posterior_covs, contexts, sa_actions)
#            start_reward = sum([dot(contexts[m, :], bandit_posterior_means[sa_actions[m], :]) for m = 1:multi_count])
#            print("\nStartActions:$(sa_actions), $(start_gain), $(start_reward)\n")            
            
            sa_actions = sa_ids(bandit_posterior_covs, bandit_posterior_means, contexts, sa_actions, 50000, 10, .99)

            actions = convert(Vector{Int64}, sa_actions)
#            end_gain = expected_entropy_gain(bandit_posterior_covs, contexts, actions)
#            end_reward = sum([dot(contexts[m, :], bandit_posterior_means[actions[m], :]) for m = 1:multi_count])
#            print("\nEndActions:$(actions), $(end_gain), $(end_reward)\n")            
            

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
function coord_ids_contextual_bandit_simulator(T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
    
        EPREWARDS, EPOPTREWARDS, agree_count, base_count = coord_ids_ep_contextual_bandit_simulator(ep,T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                       context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
        ep_count += 1
	    REWARDS[:, ep] = EPREWARDS
	    OPTREWARDS[:, ep] = EPOPTREWARDS
        tot_agree_count += agree_count
        tot_base_count += base_count
    
	    println("Ep: ", ep, " of ", n_episodes, " for coord ids")
        flush(stdout)
    end
    
    return REWARDS', OPTREWARDS', tot_agree_count, tot_base_count

end

#########################
# COORDINATED DUAL THOMPSON
# ########################
function coord_dual_thompson_ep_contextual_bandit_simulator(ep, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
        
        for t in 1:T
            
            
            for m in 1:multi_count
                context = generate_context(context_dim, context_mean, context_sd, context_constant)
                contexts[m, :] = context
            end
            # coordinated action selection

            thompson_samples = zeros(multi_count, bandit_count, context_dim)
    
            for m in 1:multi_count
                for bandit in 1:bandit_count
	                thompson_samples[m, bandit, :] = rand(MvNormal((bandit_posterior_means[bandit,:]), (bandit_posterior_covs[bandit,:,:])))
                end
            end
            

            #actions = constrained_thompson_actions(100000, thompson_samples, bandit_posterior_means, bandit_posterior_covs, contexts)
            #actions = convert(Vector{Int64}, actions)
            for m in 1:multi_count
                actions[m] = findmax(thompson_samples[m, :, :] * contexts[m, :])[2]
            end

            actions = sa_dual_thompson(bandit_posterior_covs, bandit_posterior_means, contexts, actions, 50000, 10, .99)
            actions = convert(Vector{Int64}, actions)


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
	return EPREWARDS, EPOPTREWARDS
end
function coord_dual_thompson_contextual_bandit_simulator(T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
    
        EPREWARDS, EPOPTREWARDS = coord_dual_thompson_ep_contextual_bandit_simulator(ep,T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                       context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
        ep_count += 1
	    REWARDS[:, ep] = EPREWARDS
	    OPTREWARDS[:, ep] = EPOPTREWARDS
        #tot_agree_count += agree_count
        #tot_base_count += base_count
    
	    println("Ep: ", ep, " of ", n_episodes, " for coord dual thompson")
        flush(stdout)
    end
    
    return REWARDS', OPTREWARDS'

end

#########################
# COORDINATED THOMPSON IDS
# #######################

function coord_thompson_ids_ep_contextual_bandit_simulator(ep, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
        
        for t in 1:T
            
            
            for m in 1:multi_count
                context = generate_context(context_dim, context_mean, context_sd, context_constant)
                contexts[m, :] = context
            end
            # coordinated action selection

            thompson_samples = zeros(multi_count, bandit_count, context_dim)
    
            for m in 1:multi_count
                for bandit in 1:bandit_count
	                thompson_samples[m, bandit, :] = rand(MvNormal((bandit_posterior_means[bandit,:]), (bandit_posterior_covs[bandit,:,:])))
                end
            end
            

            #actions = constrained_thompson_actions(100000, thompson_samples, bandit_posterior_means, bandit_posterior_covs, contexts)
            #actions = convert(Vector{Int64}, actions)
            for m in 1:multi_count
                actions[m] = findmax(thompson_samples[m, :, :] * contexts[m, :])[2]
            end
            
            thompson_expected_rewards = [dot(bandit_posterior_means[actions[m], :], contexts[m, :]) for m = 1:multi_count]
            restrictions = convert(Matrix{Int64}, zeros(multi_count, bandit_count))

            for m in 1:multi_count
                for b in 1: bandit_count
                    if dot(bandit_posterior_means[b, :], contexts[m, :]) >= thompson_expected_rewards[m]
                        restrictions[m, b] = 1
                    end
                end
            end
            if sum(restrictions) > multi_count                
            #alloc = KL_optimal_allocations_with_restrictions(bandit_posterior_covs, contexts, restrictions)
            #actions = [findmax(alloc[i, :])[2] for i = 1:multi_count]
                actions = sa_ids_restricted(bandit_posterior_covs, bandit_posterior_means, contexts, actions, restrictions, 50000, 10, .99)
                actions = convert(Vector{Int64}, actions)
            end
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
	return EPREWARDS, EPOPTREWARDS
end
function coord_thompson_ids_contextual_bandit_simulator(T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
    
        EPREWARDS, EPOPTREWARDS = coord_thompson_ids_ep_contextual_bandit_simulator(ep,T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                       context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
        ep_count += 1
	    REWARDS[:, ep] = EPREWARDS
	    OPTREWARDS[:, ep] = EPOPTREWARDS
        #tot_agree_count += agree_count
        #tot_base_count += base_count
    
	    println("Ep: ", ep, " of ", n_episodes, " for coord thompson")
        flush(stdout)
    end
    
    return REWARDS', OPTREWARDS'

end



#########################
# COORDINATED THOMPSON IDS 2
# #######################

# Take several thompson samples and select the one with the largest information ratio

function coord_thompson_ids_2_ep_contextual_bandit_simulator(ep, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
        
        for t in 1:T
            
            
            for m in 1:multi_count
                context = generate_context(context_dim, context_mean, context_sd, context_constant)
                contexts[m, :] = context
            end
            # coordinated action selection

            thompson_samples = zeros(multi_count, bandit_count, context_dim)
            regrets = expected_regrets(contexts, bandit_posterior_means, bandit_posterior_covs, 10000)
            min_actions = 0
            min_gain = 0

            for ct in 1:thompson_count
            
                for m in 1:multi_count
                    for bandit in 1:bandit_count
	                    thompson_samples[m, bandit, :] = rand(MvNormal((bandit_posterior_means[bandit,:]), (bandit_posterior_covs[bandit,:,:])))
                    end
                end
                

                #actions = constrained_thompson_actions(100000, thompson_samples, bandit_posterior_means, bandit_posterior_covs, contexts)
                #actions = convert(Vector{Int64}, actions)
                for m in 1:multi_count
                    actions[m] = findmax(thompson_samples[m, :, :] * contexts[m, :])[2]
                end
                
                ir = information_ratio(bandit_posterior_covs, contexts, actions, regrets)
                
                if ct == 1 || ir < min_gain
                    min_gain = ir
                    min_actions = copy(actions)
                end            
            end

            actions = min_actions

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
	return EPREWARDS, EPOPTREWARDS
end
function coord_thompson_ids_2_contextual_bandit_simulator(T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
    
        EPREWARDS, EPOPTREWARDS = coord_thompson_ids_2_ep_contextual_bandit_simulator(ep,T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                       context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
        ep_count += 1
	    REWARDS[:, ep] = EPREWARDS
	    OPTREWARDS[:, ep] = EPOPTREWARDS
        #tot_agree_count += agree_count
        #tot_base_count += base_count
    
	    println("Ep: ", ep, " of ", n_episodes, " for coord thompson")
        flush(stdout)
    end
    
    return REWARDS', OPTREWARDS'

end


function val_thompson_ids_rollout(thompson_count, T_remainder, rollout_length, contexts, use_context, lambda, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, discount, bandit_posterior_covs, bandit_posterior_means, bandit_param,
	true_expected_rewards, CovCon, old_cov, SigInvMu)

    disc_reward = 0
    #fill!(context, 0.0)
    fill!(true_expected_rewards, 0.0)
    fill!(CovCon, 0.0)
    fill!(old_cov,0.0)
    fill!(SigInvMu,0.0)
    
    truncation_length = T_remainder - min(T_remainder, rollout_length)

    bandit_param = copy(global_bandit_param)
    true_bandit_param = copy(global_bandit_param)
    
    actions = convert(Vector{Int64}, zeros(multi_count))
    obss = zeros(multi_count)
    for t in 1:min(T_remainder, rollout_length)


        if t > 1 || !use_context

            for m in 1:multi_count
                context = generate_context(context_dim, context_mean, context_sd, context_constant)
                contexts[m, :] = context
            end

        end


    ### NEW STUFF

        thompson_samples = zeros(multi_count, bandit_count, context_dim)
        regrets = expected_regrets(contexts, bandit_posterior_means, bandit_posterior_covs, 10000)
        min_actions = 0
        min_gain = 0

        for ct in 1:thompson_count
        
            for m in 1:multi_count
                for bandit in 1:bandit_count
                    thompson_samples[m, bandit, :] = rand(MvNormal((bandit_posterior_means[bandit,:]), (bandit_posterior_covs[bandit,:,:])))
                end
            end
            

            #actions = constrained_thompson_actions(100000, thompson_samples, bandit_posterior_means, bandit_posterior_covs, contexts)
            #actions = convert(Vector{Int64}, actions)
            for m in 1:multi_count
                actions[m] = findmax(thompson_samples[m, :, :] * contexts[m, :])[2]
            end
            
            ir = information_ratio(bandit_posterior_covs, contexts, actions, regrets)
            
            if ct == 1 || ir < min_gain
                min_gain = ir
                min_actions = copy(actions)
            end            
        end

        actions = min_actions

        for m in 1:multi_count
            #print(actions)
            #print("\n")
            action = actions[m]
            context = contexts[m, :]
            true_expected_reward = dot(bandit_posterior_means[action,:], context)
            disc_reward += true_expected_reward * discount^(t-1)
            #obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:],context)) + true_expected_reward
            #obs = randn() * obs_sd + true_expected_rewards[action]
            obss[m] = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:], context)) + true_expected_reward

        
        end


        for act in 1:bandit_count

            action_obss = obss[actions .== act]
            action_contexts = contexts[actions .== act, :]
            old_cov = bandit_posterior_covs[act, :, :]
            bandit_posterior_covs[act, :, :] = inv(inv(old_cov) + action_contexts' * action_contexts ./ obs_sd^2)
            bandit_posterior_covs[act, :, :] = ((bandit_posterior_covs[act,:,:]) + bandit_posterior_covs[act,:,:]')/2
            bandit_posterior_means[act, :, :] = bandit_posterior_covs[act, :, :] * (old_cov \ bandit_posterior_means[act, :] + action_contexts' * action_obss ./ obs_sd^2)

        end
    end

    if truncation_length > 0
        
        
        #### SWITCHED TO TRUE VALUE NEURAL NETWORK AS TEST
        
        #input_vec = Vector(vec(bandit_param'))
        #append!(input_vec, Vector(vec(bandit_posterior_means')))
        input_vec = Vector(vec(bandit_posterior_means'))
        append!(input_vec, vcat([upper_triangular_vec(bandit_posterior_covs[a, :, :]) for a in 1:bandit_count]...))
        
        scaled_input_vec = (input_vec .- scale_list[truncation_length][1:end .!= 1, 1]) ./ scale_list[truncation_length][1:end .!= 1, 2] 
        
        disc_reward += (discount^min(T_remainder, rollout_length)) * (multi_scale_list[truncation_length][1,2]*multi_neural_net_list[truncation_length](scaled_input_vec)[1] + multi_scale_list[truncation_length][1,1])
    
        
        # Evaluating parameters for sanity check
        
        #test_out = []
        #append!(test_out, input_vec)
        #append!(test_out, scaled_input_vec)
        #append!(test_out, (scale_list[truncation_length][1,2]*neural_net_list[truncation_length](scaled_input_vec)[1] + scale_list[truncation_length][1,1]))
        #append!(test_out, neural_net_list[truncation_length](scaled_input_vec)[1]) 
        #print(test_out) 
    end

    return disc_reward

end

function val_thompson_ids_policy(t, T, bandit_count, contexts, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    BANDIT_VALUES = zeros(bandit_count)
    predictive_rewards = bandit_posterior_means * context

## PREALLOCATION
    roll_context = zeros(context_dim)
    roll_true_expected_rewards = zeros(bandit_count)
    roll_CovCon = zeros(context_dim)
    roll_old_cov = zeros(context_dim, context_dim)
    roll_SigInvMu = zeros(context_dim)

    temp_post_means = zeros(bandit_count, context_dim)
    temp_post_covs = zeros(bandit_count, context_dim, context_dim)
    temp_bandit_mean = zeros(context_dim)
    temp_bandit_cov = zeros(context_dim, context_dim)

    bandit_param = zeros(bandit_count, context_dim)
    true_expected_rewards = zeros(bandit_count)
    grad_est = zeros(3)

    
    #policies = [greedy_policy, thompson_policy]
    thompson_values = []
    
    println("Context Start: ", context)
    flush(stdout)


    for tc in thompson_counts  

            MEAN_REWARD = 0

            for roll in 1:n_opt_rollouts
            
                copy!(temp_post_means, bandit_posterior_means)
                copy!(temp_post_covs, bandit_posterior_covs)
                #bandit_param = zeros(bandit_count, context_dim)
                for bandit in 1:bandit_count
                    copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
                    copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
                    bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
                end

                use_context = true
                
                rollout_value = val_thompson_ids_rollout(tc, T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
                    context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
                    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
                
                
                MEAN_REWARD = ((roll - 1) * MEAN_REWARD + rollout_value) / roll

            end
            
            push!(thompson_values, MEAN_REWARD)
        
    end
    
    println("Context Finish: ", context)
    flush(stdout)

    opt_index = findmax(epsilon_values)[2]
    opt_count = thompson_counts[opt_index]
    flush(stdout)

    # END OPTIMIZATION OF LAMBDA
    
    ######################
    # FIND ACTUAL ACTION
    # ####################
    thompson_samples = zeros(multi_count, bandit_count, context_dim)
    regrets = expected_regrets(contexts, bandit_posterior_means, bandit_posterior_covs, 10000)
    min_actions = 0
    min_gain = 0
    for ct in 1:opt_count
    
        for m in 1:multi_count
            for bandit in 1:bandit_count
                thompson_samples[m, bandit, :] = rand(MvNormal((bandit_posterior_means[bandit,:]), (bandit_posterior_covs[bandit,:,:])))
            end
        end
        

        #actions = constrained_thompson_actions(100000, thompson_samples, bandit_posterior_means, bandit_posterior_covs, contexts)
        #actions = convert(Vector{Int64}, actions)
        for m in 1:multi_count
            actions[m] = findmax(thompson_samples[m, :, :] * contexts[m, :])[2]
        end
        
        ir = information_ratio(bandit_posterior_covs, contexts, actions, regrets)
        
        if ct == 1 || ir < min_gain
            min_gain = ir
            min_actions = copy(actions)
        end            
    end

    actions = min_actions
    return actions
end

#############################
# GENERIC COORDINATED MULTI SIMULATOR
# ####################################


function coord_multi_ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
                context = generate_context(context_dim, context_mean, context_sd, context_constant)
                contexts[m, :] = context
            end
            actions = action_function(t, T, bandit_count, contexts, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
            for m in 1:multi_count
                context = contexts[m, :]
                true_expected_rewards = true_bandit_param * context
                #true_expected_rewards = bandit_posterior_means * context
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
function coord_multi_contextual_bandit_simulator(action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

    REWARDS = zeros(T, n_episodes)
    OPTREWARDS = zeros(T, n_episodes)
    ep_count = 1
    

    ## USING BAYESIAN FORMULATION FOR FAIR COMPARISON
    #global_bandit_param = [1 0; 0 1; 2 -1]
    global_bandit_param = rand(Normal(bandit_prior_mean, bandit_prior_sd), bandit_count, context_dim)
    
    
    #threadreps = zeros(Threads.nthreads())


    for ep in 1:n_episodes
    
        EPREWARDS, EPOPTREWARDS = coord_multi_ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                       context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
        ep_count += 1
	    REWARDS[:, ep] = EPREWARDS
	    OPTREWARDS[:, ep] = EPOPTREWARDS
    
    end
    
    return REWARDS', OPTREWARDS'

end



###################################
# Non Stationary Model
# #################################


function dlm_ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param, delta)
        
    
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
            context = generate_context(context_dim, context_mean, context_sd, context_constant)
            true_expected_rewards = true_bandit_param * context
            #true_expected_rewards = bandit_posterior_means * context
            action = action_function(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
            true_expected_reward = true_expected_rewards[action]
            EPREWARDS[t] = true_expected_reward
            EPOPTREWARDS[t] = maximum(true_expected_rewards)
            #obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:],context)) + true_expected_reward
            obs = randn() * obs_sd + true_expected_reward
	        old_cov = bandit_posterior_covs[action, :, :]
	        CovCon = old_cov * context ./ obs_sd
	        #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)
	        bandit_posterior_covs[action, :, :] = old_cov - CovCon * CovCon' ./ (1 + dot(context, old_cov, context) / obs_sd^2)
	        bandit_posterior_covs[action, :, :] = ((bandit_posterior_covs[action,:,:]) + bandit_posterior_covs[action,:,:]')/2
	        bandit_posterior_means[action, :] = (bandit_posterior_covs[action, :, :]) * (old_cov \ (bandit_posterior_means[action,:]) + context * obs / obs_sd^2)

            ## EVOLVE MODEL

            for i in 1:bandit_count

                #testoo =  MvNormal(zeros(context_dim), bandit_posterior_covs[i, :, :] .* ((1 - delta) / delta))
                #print(testoo)
                true_bandit_param[i, :] += rand(MvNormal(zeros(context_dim), bandit_posterior_covs[i, :, :] .* ((1 - delta) / delta)))
                bandit_posterior_covs[i, :, :] .*= (1 / delta)
            end

	        println("Ep: ", ep, " - ", t, " of ", T, " for ", String(Symbol(action_function)))
            flush(stdout)


        end
	return EPREWARDS, EPOPTREWARDS
end

function dlm_contextual_bandit_simulator(action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, delta)

    REWARDS = zeros(T, n_episodes)
    OPTREWARDS = zeros(T, n_episodes)
    ep_count = 1
    

    ## USING BAYESIAN FORMULATION FOR FAIR COMPARISON
    #global_bandit_param = [1 0; 0 1; 2 -1]
    
    
    #threadreps = zeros(Threads.nthreads())


    for ep in 1:n_episodes
    
        global_bandit_param = rand(Normal(bandit_prior_mean, bandit_prior_sd), bandit_count, context_dim)
        EPREWARDS, EPOPTREWARDS = dlm_ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                       context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param, delta)
        ep_count += 1
	    REWARDS[:, ep] = EPREWARDS
	    OPTREWARDS[:, ep] = EPOPTREWARDS
    
    end
    #print(threadreps)
    return REWARDS', OPTREWARDS'
end











## SIMULATIONS

# Sims


const discount_vector = discount .^ collect(0:(T-1))
gp_exp4_policy = exp4([gp_greedy_policy, gp_thompson_policy, gp_glm_ucb_policy, gp_ids_policy], [1, 1, 1, 1])
vb_exp4_policy = exp4([vb_greedy_policy, vb_thompson_policy, vb_glm_ucb_policy, vb_ids_policy], [1, 1, 1, 1])
mab_exp4_policy = exp4([mab_greedy_policy, mab_thompson_policy, mab_bayes_ucb_policy, mab_ids_policy], [1, 1, 1, 1])
lin_exp4_policy = exp4([greedy_policy, thompson_policy, bayes_ucb_policy, ids_policy], [1, 1, 1, 1])
#const run_policies = [greedy_policy, greedy_policy, thompson_policy, thompson_policy]
#const multi_ind = [true, false, true, false]

#const run_policies = [vftp2, vftp4, vftp8, vftp16, vftp32, vftp64, vegp2, vegp4, vegp8, vegp16, vegp32, vegp64]
#const multi_ind = [false, false, false, false, false, false, false, false, false, false, false, false]

#const run_policies = [greedy_policy, epsilon_greedy_policy] 
#const multi_ind = [true, true]


#const run_policies = [epsilon_greedy_policy, epsilon_greedy_policy, thompson_policy, thompson_policy, greedy_policy, greedy_policy, glm_ucb_policy, glm_ucb_policy, ids_policy, ids_policy, val_greedy_rollout_policy, val_greedy_thompson_ucb_ids_policy, val_better_grid_lambda_policy, greedy_rollout_policy, bernoulli_val_greedy_rollout_policy, bernoulli_val_greedy_thompson_ucb_ids_policy, bernoulli_val_better_grid_lambda_policy, bernoulli_greedy_rollout_policy]
#const multi_ind = [false for i in 1:18]
#const bern_ind = [true, false, true, false, true, false, true, false, true, false, false, false, false, false, true, true, true, true]

#const run_policies = [greedy_policy, thompson_policy, bayes_ucb_policy, ids_policy, val_greedy_thompson_ucb_ids_policy, mab_greedy_policy, mab_thompson_policy, mab_bayes_ucb_policy, mab_ids_policy, mab_val_greedy_thompson_ucb_ids_policy]
#const run_policies = [mab_greedy_policy, mab_thompson_policy, mab_bayes_ucb_policy, mab_ids_policy, mab_val_greedy_thompson_ucb_ids_q_policy]
#const run_policies = [greedy_policy, thompson_policy, bayes_ucb_policy, ids_policy, val_greedy_thompson_ucb_ids_policy]

#const run_policies = [gp_greedy_policy, gp_thompson_policy, gp_glm_ucb_policy, gp_ids_policy, gp_val_greedy_thompson_ucb_ids_policy]
#const run_policies = [gp_greedy_policy, gp_thompson_policy, gp_glm_ucb_policy, gp_ids_policy, gp_exp4_policy]
#const run_policies = [greedy_policy, thompson_policy, bayes_ucb_policy, ids_policy]
const run_policies = [greedy_policy, thompson_policy, bayes_ucb_policy, ids_policy, dp_greedy_thompson_ucb_ids_policy]
#const run_policies = [greedy_policy, thompson_policy, bayes_ucb_policy, ids_policy]
#const run_policies = [mab_ids_policy, mab_ids_1_policy, mab_ids_1_5_policy, mab_ids_2_5_policy, mab_ids_3_policy, mab_ids_multi_policy]

const multi_ind = repeat([false], 10)
const bern_ind = repeat([false], 10)
const dlm_ind = repeat([false], 10)
const mcmc_bern_ind = repeat([false], 10)
const vb_bern_ind = repeat([false], 10)
const gp_ind = repeat([false], 10)
const mab_ind = vcat(repeat([false], 5), repeat([true], 5))

const coord_epsilon_greedy = false
const coord_thompson = false
const coord_ids = false
const coord_dual_thompson = false
const coord_thompson_ids = false
const coord_thompson_ids_2 = false 
count = 0
thompson_count = 10
# don't modify
regret_header = []

cumulative_discounted_regrets = zeros(T, length(run_policies) + coord_epsilon_greedy + coord_thompson + coord_ids + coord_dual_thompson + coord_thompson_ids + coord_thompson_ids_2)

for pol in 1:length(run_policies)
    
    if multi_ind[pol]
        pol_rewards, pol_opt_rewards = multi_contextual_bandit_simulator(run_policies[pol], T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
        push!(regret_header, "$(run_policies[pol])_multi")
    
    elseif bern_ind[pol]
        pol_rewards, pol_opt_rewards = bernoulli_contextual_bandit_simulator(run_policies[pol], T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
        push!(regret_header, "$(run_policies[pol])_bern")
    elseif dlm_ind[pol] 
        pol_rewards, pol_opt_rewards = dlm_contextual_bandit_simulator(run_policies[pol], T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, delta)
        push!(regret_header, "$(run_policies[pol])")
    
    elseif mcmc_bern_ind[pol]
        pol_rewards, pol_opt_rewards = mcmc_bernoulli_contextual_bandit_simulator(run_policies[pol], T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean, context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
        push!(regret_header, "$(run_policies[pol])_mcmc_bern")
    
    elseif vb_bern_ind[pol]
        pol_rewards, pol_opt_rewards = vb_bernoulli_contextual_bandit_simulator(run_policies[pol], T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean, context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
        push!(regret_header, "$(run_policies[pol])_vb_bern")
    elseif gp_ind[pol]
        pol_rewards, pol_opt_rewards = gp_contextual_bandit_simulator(run_policies[pol], T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean, context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
        push!(regret_header, "$(run_policies[pol])")
    elseif mab_ind[pol]
        pol_rewards, pol_opt_rewards = multi_armed_bandit_simulator(run_policies[pol], T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, bandit_count, a_beta_prior, b_beta_prior, discount, epsilon)
        push!(regret_header, "$(run_policies[pol])")
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
    cumulative_discounted_regrets[:, 1+count+length(run_policies)] = cdr
    push!(regret_header, "Coord_Epsilon_Greedy")
    count += 1
    #CSV.write("/hpc/home/jml165/rl/arrayresults/agreeprop_$(idx).csv", Tables.table([agree_count base_count]))
end

if coord_thompson
    ct_rewards, ct_opt_rewards = coord_thompson_3_contextual_bandit_simulator(T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
    cdr = cumsum(discount_vector .* [mean(c) for c in eachcol(ct_opt_rewards - ct_rewards)])
    cumulative_discounted_regrets[:, 1+count+length(run_policies)] = cdr
    push!(regret_header, "Coord_Thompson")
    count += 1
end

if coord_ids
    cids_rewards, cids_opt_rewards = coord_ids_contextual_bandit_simulator(T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
    cdr = cumsum(discount_vector .* [mean(c) for c in eachcol(cids_opt_rewards - cids_rewards)])
    cumulative_discounted_regrets[:, 1+count+length(run_policies)] = cdr
    push!(regret_header, "Coord_IDS")
    count += 1
end

if coord_dual_thompson
    cdt_rewards, cdt_opt_rewards = coord_dual_thompson_contextual_bandit_simulator(T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
    cdr = cumsum(discount_vector .* [mean(c) for c in eachcol(cdt_opt_rewards - cdt_rewards)])
    cumulative_discounted_regrets[:, 1+count+length(run_policies)] = cdr
    push!(regret_header, "Coord_Dual_Thompson")
    count += 1
end
if coord_thompson_ids
    cdt_rewards, cdt_opt_rewards = coord_thompson_ids_contextual_bandit_simulator(T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
    cdr = cumsum(discount_vector .* [mean(c) for c in eachcol(cdt_opt_rewards - cdt_rewards)])
    cumulative_discounted_regrets[:, 1+count+length(run_policies)] = cdr
    push!(regret_header, "Coord_Thompson_IDS")
    count += 1
end
if coord_thompson_ids_2
    cdt_rewards, cdt_opt_rewards = coord_thompson_ids_2_contextual_bandit_simulator(T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
    cdr = cumsum(discount_vector .* [mean(c) for c in eachcol(cdt_opt_rewards - cdt_rewards)])
    cumulative_discounted_regrets[:, 1+count+length(run_policies)] = cdr
    push!(regret_header, "Coord_Thompson_IDS_2")
    count += 1
end


#const combReg = [greedyCumDiscReg epsgreedyCumDiscReg thompsonCumDiscReg bayesucbCumDiscReg squarecbCumDiscReg lambdaCumDiscReg lambdameanCumDiscReg greedyrolloutCumDiscReg betterlambdaCumDiscReg optlambdaCumDiscReg bayesoptlambdaCumDiscReg]

#const combReg = [greedyCumDiscReg thompsonCumDiscReg greedyrolloutCumDiscReg valgreedyrolloutCumDiscReg valbettergridlambdaCumDiscReg valgreedythompsonCumDiscReg]
CSV.write("/hpc/home/jml165/rl/arrayresults2/results_$(idx).csv", Tables.table(cumulative_discounted_regrets), header=regret_header)
#CSV.write("/hpc/home/jml165/rl/arrayresults/results_$(idx).csv",  Tables.table(combReg), header=["Greedy", "EpsGreedy", "Thompson", "BayesUCB", "SquareCB", "Lambda","LambdaMean","GreedyRollout", "BetterLambda","OptLambda", "BayesOptLambda"])
