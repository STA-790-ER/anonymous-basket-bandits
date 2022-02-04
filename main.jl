
using LinearAlgebra, Statistics, Plots, Distributions, StatsBase, DelimitedFiles, CSV, Tables, Random, BenchmarkTools

using BayesianOptimization, GaussianProcesses, Distributions
using Flux
using BSON: @load
BLAS.set_num_threads(1)

include("rollouts.jl")
include("policies.jl")


#cnst mts = MersenneTwister.(1:Threads.nthreads())
# Parameters
const context_dim = 2
const context_mean = 0
const context_sd = 1
const obs_sd = 1

const bandit_count = 3
const bandit_prior_mean = 0
const bandit_prior_sd = 10

# Multi Action
const multi_count = 5

# SIMULATION HORIZON
const T = 100

# NUMBER OF GLOBAL SIMULATION EPISODES (PER INDEX JOB)
const n_episodes = 10

# DISCOUNT PARAMETER
const discount = 1.

# PARAMETER FOR EPSILON GREEDY POLICY
const epsilon = .01

# PARAMETERS FOR ALL ROLLOUT METHODS
const rollout_length = 5
const n_rollouts = 50

# PARAMETERS FOR SPSA OPTIMIZATION METHOD
const n_opt_rollouts = 2500
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


## SIMULATIONS

# Sims

#print("\n")
#
#@time valgreedythompsonRewards, valgreedythompsonOptrewards = contextual_bandit_simulator(val_greedy_thompson_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
#                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
#
#print("\n")
#
#
#@time greedyRewards, greedyOptrewards = contextual_bandit_simulator(greedy_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
#                                          context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
#
#print("\n")
##@time epsgreedyRewards, epsgreedyOptrewards = contextual_bandit_simulator(epsilon_greedy_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
##                                          context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
#
#print("\n")
#@time thompsonRewards, thompsonOptrewards = contextual_bandit_simulator(thompson_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
#                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
#
##print("\n")
##@time bayesucbRewards, bayesucbOptrewards = contextual_bandit_simulator(bayes_ucb_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
##                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
#
##print("\n")
##@time squarecbRewards, squarecbOptrewards = contextual_bandit_simulator(squarecb_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
##                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
#
##print("\n")
##@time lambdaRewards, lambdaOptrewards = contextual_bandit_simulator(lambda_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
##                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
#
#
##print("\n")
#
##@time betterlambdaRewards, betterlambdaOptrewards = contextual_bandit_simulator(better_lambda_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
##                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
#
##print("\n")
#
##@time optlambdaRewards, optlambdaOptrewards = contextual_bandit_simulator(opt_lambda_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
##                                          context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
#
##print("\n")
#
##@time lambdameanRewards, lambdameanOptrewards = contextual_bandit_simulator(lambda_mean_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
##                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
##print("\n")
#
#
#@time greedyrolloutRewards, greedyrolloutOptrewards = contextual_bandit_simulator(greedy_rollout_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
#                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
#print("\n")
#
#
#@time valgreedyrolloutRewards, valgreedyrolloutOptrewards = contextual_bandit_simulator(val_greedy_rollout_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
#                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
#print("\n")
#
#
##@time bayesoptrolloutRewards, bayesoptrolloutOptrewards = contextual_bandit_simulator(bayesopt_lambda_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
##                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
##print("\n")
##@time gridlambdaRewards, gridlambdaOptrewards = contextual_bandit_simulator(grid_lambda_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
##                                           context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
##print("\n")
##@time entlambdaRewards, entlambdaOptrewards = contextual_bandit_simulator(ent_lambda_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
##                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
##print("\n")
##@time entoptlambdaRewards, entoptlambdaOptrewards = contextual_bandit_simulator(ent_opt_lambda_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
##                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
#
#
##print("\n")
#
##@time valbetteroptlambdaRewards, valbetteroptlambdaOptrewards = contextual_bandit_simulator(val_better_opt_lambda_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
##                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
#
##print("\n")
#
#
#@time valbettergridlambdaRewards, valbettergridlambdaOptrewards = contextual_bandit_simulator(val_better_grid_lambda_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
#                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
#
#print("\n")
#
## Plot
#
#const discount_vector = discount .^ collect(0:(T-1))
#
#const greedyCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(greedyOptrewards - greedyRewards)])
##const epsgreedyCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(epsgreedyOptrewards - epsgreedyRewards)])
#const thompsonCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(thompsonOptrewards - thompsonRewards)])
##const bayesucbCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(bayesucbOptrewards - bayesucbRewards)])
##const squarecbCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(squarecbOptrewards - squarecbRewards)])
##const lambdaCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(lambdaOptrewards - lambdaRewards)])
##const lambdameanCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(lambdameanOptrewards - lambdameanRewards)])
#const greedyrolloutCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(greedyrolloutOptrewards - greedyrolloutRewards)])
#const valgreedyrolloutCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(valgreedyrolloutOptrewards - valgreedyrolloutRewards)])
##const betterlambdaCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(betterlambdaOptrewards - betterlambdaRewards)])
##const optlambdaCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(optlambdaOptrewards - optlambdaRewards)])
##const bayesoptlambdaCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(bayesoptrolloutOptrewards - bayesoptrolloutRewards)])
##const gridlambdaCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(gridlambdaOptrewards - gridlambdaRewards)])
##const entlambdaCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(entlambdaOptrewards - entlambdaRewards)])
##const entoptlambdaCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(entoptlambdaOptrewards - entoptlambdaRewards)])
##const valbetteroptlambdaCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(valbetteroptlambdaOptrewards - valbetteroptlambdaRewards)])
#const valbettergridlambdaCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(valbettergridlambdaOptrewards - valbettergridlambdaRewards)])
#const valgreedythompsonCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(valgreedythompsonOptrewards - valgreedythompsonRewards)])
#

const discount_vector = discount .^ collect(0:(T-1))
#const run_policies = [greedy_policy, greedy_policy, thompson_policy, thompson_policy]
#const multi_ind = [true, false, true, false]

const run_policies = [vegp2, vegp4, vegp8, vegp16, vegp32, vegp64]
const multi_ind = [false, false, false, false, false, false]
regret_header = []
cumulative_discounted_regrets = zeros(T, length(run_policies))

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



#const combReg = [greedyCumDiscReg epsgreedyCumDiscReg thompsonCumDiscReg bayesucbCumDiscReg squarecbCumDiscReg lambdaCumDiscReg lambdameanCumDiscReg greedyrolloutCumDiscReg betterlambdaCumDiscReg optlambdaCumDiscReg bayesoptlambdaCumDiscReg]

#const combReg = [greedyCumDiscReg thompsonCumDiscReg greedyrolloutCumDiscReg valgreedyrolloutCumDiscReg valbettergridlambdaCumDiscReg valgreedythompsonCumDiscReg]
CSV.write("/hpc/home/jml165/rl/arrayresults/results_$(idx).csv", Tables.table(cumulative_discounted_regrets), header=regret_header)

#CSV.write("/hpc/home/jml165/rl/arrayresults/results_$(idx).csv",  Tables.table(combReg), header=["Greedy", "EpsGreedy", "Thompson", "BayesUCB", "SquareCB", "Lambda","LambdaMean","GreedyRollout", "BetterLambda","OptLambda", "BayesOptLambda"])
