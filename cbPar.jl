
using LinearAlgebra, Statistics, Plots, Distributions, StatsBase, DelimitedFiles, CSV, Tables, Random, BenchmarkTools

using BayesianOptimization, GaussianProcesses, Distributions
BLAS.set_num_threads(1)

#cnst mts = MersenneTwister.(1:Threads.nthreads())
# Parameters
const context_dim = 2
const context_mean = 0
const context_sd = 1
const obs_sd = 1

const bandit_count = 3
const bandit_prior_mean = 0
const bandit_prior_sd = 10

const T = 100
const n_episodes = 2

const discount = 1.
const epsilon = .01

const rollout_length = 100
const n_rollouts = 50000
const n_opt_rollouts = 10
const n_spsa_iter = 10

const n_grid_iter = 7
const grid_ratio = 2
const grid_num = 6
const int_length = 2
const n_grid_rollouts = 50
## SIMULATOR FUNCTION


const idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

function ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
    #for ep in 1:n_episodes
        #pct_done = round(100 * ep_count / n_episodes, digits=1)
	#print(pct_done)
	#threadreps[Threads.threadid()] += 1
##print("\rEpisode: $pct_done%")
       #bandit_param = randn(bandit_count, context_dim) * bandit_prior_sd .+ bandit_prior_mean
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
        #io = open("~/rl/monitor/monitor_$(idx).txt", "w")
	#write(io, 0)
	#close(io)
        for t in 1:T
            context = randn(context_dim) * context_sd .+ context_mean
            true_expected_rewards = true_bandit_param * context
            action = action_function(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
            true_expected_reward = true_expected_rewards[action]
            EPREWARDS[t] = true_expected_reward
            EPOPTREWARDS[t] = maximum(true_expected_rewards)
	    obs = randn() * obs_sd + true_expected_reward
            #old_cov = bandit_posterior_covs[action, :, :]
	    #print(bandit_posterior_covs[action,:,:])
	    #old_precision = inv(bandit_posterior_covs[action,:,:])
	    old_cov = bandit_posterior_covs[action, :, :]
	    CovCon = old_cov * context ./ obs_sd
	    #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)
	    bandit_posterior_covs[action, :, :] = old_cov - CovCon * CovCon' ./ (1 + dot(context, old_cov, context))
	    bandit_posterior_covs[action, :, :] = ((bandit_posterior_covs[action,:,:]) + (bandit_posterior_covs[action,:,:])')/2
	    bandit_posterior_means[action, :] = (bandit_posterior_covs[action, :, :]) * (old_cov \ (bandit_posterior_means[action,:]) + context * obs / obs_sd^2)

	    println("Ep: ", ep, " - ", t, " of ", T, " for ", String(Symbol(action_function)))
            flush(stdout)

	    #io = open("~/rl/monitor/monitor_$(idx).txt", "r")
	    #curr_prog = read(io, Int)
	    #close(io)
	    #io = open("~/rl/monitor/monitor_$(idx).txt", "w")
	    #write(io, curr_prog+1)
	    #close(io)

        end
	return EPREWARDS, EPOPTREWARDS
end

function contextual_bandit_simulator(action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

    REWARDS = zeros(T, n_episodes)
    OPTREWARDS = zeros(T, n_episodes)
    ep_count = 1
    global_bandit_param = [1 0; 0 1; 2 -1]
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

## POLICY FUNCTIONS

# Greedy
function greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    val, action = findmax(bandit_posterior_means * context)
    return action
end

# Epsilon Greedy
function epsilon_greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    epsilon_draw = rand()

    if epsilon_draw < epsilon
        return rand(1:bandit_count)
    else
        val, action = findmax(bandit_posterior_means * context)
        return(action)
    end

end

# Bayes UCB
function bayes_ucb_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    val, action = findmax(bandit_posterior_means * context)
    reward_means = bandit_posterior_means * context
    reward_sds = sqrt.(vec([context' * (@view bandit_posterior_covs[i,:,:]) * context for i=1:bandit_count]))
    ucbs = quantile.(Normal.(reward_means, reward_sds), 1-1/t)
    return findmax(ucbs)[2]
end

# Thompson
function thompson_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    thompson_samples = zeros(bandit_count, context_dim)

    for bandit in 1:bandit_count
	    thompson_samples[bandit, :] = rand(MvNormal((bandit_posterior_means[bandit,:]), (bandit_posterior_covs[bandit,:,:])))
    end

    return findmax(thompson_samples * context)[2]
end

# SquareCB
function squarecb_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    reward_means = bandit_posterior_means * context
    opt_mean, opt_id = findmax(reward_means)
    selection_probs = zeros(bandit_count)
    for bandit in 1:bandit_count
        selection_probs[bandit] = 1 / (bandit_count + sqrt(bandit_count*T) * (opt_mean - reward_means[bandit]))
    end

    selection_probs[opt_id] = 0
    selection_probs[opt_id] = 1 - sum(selection_probs)

    return sample(weights(selection_probs))
end

function lambda_rollout(T, rollout_length, lambda, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, discount, bandit_posterior_covs, bandit_posterior_means, bandit_param,
	context, true_expected_rewards, CovCon, old_cov, SigInvMu)

    #temp_post_covs = bandit_posterior_covs
    #temp_post_means = bandit_posterior_means
    disc_reward = 0
    fill!(context, 0.0)
    fill!(true_expected_rewards, 0.0)
    fill!(CovCon, 0.0)
    fill!(old_cov,0.0)
    fill!(SigInvMu,0.0)

    for t in 1:min(T, rollout_length)

        context_seed = rand(1:context_dim)
        fill!(context, zero(context[1]))
        context[context_seed] = 1
        mul!(true_expected_rewards, bandit_param, context)

        #mean_max_bandit = 1
        #mean_max_val = dot((@view bandit_posterior_means[1,:]), context)
        #
        #var_max_bandit = 1
        #var_max_val = dot(context, (@view bandit_posterior_covs[1,:,:]), context)
#
        #for i = 2:bandit_count
            #
            #new_val = dot((@view bandit_posterior_means[i,:]),context)
            #if new_val > mean_max_val
                #mean_max_val = new_val
                #mean_max_bandit = i
            #end
#
            #new_val = dot(context, (@view bandit_posterior_covs[i, :, :]), context)
#
            #if new_val > var_max_val
                #var_max_val = new_val
                #var_max_bandit = i
            #end
        #end
#
        #if mean_max_bandit == var_max_bandit
        #
            #action = mean_max_bandit
#
        #else
        # alt action find
        curr_ind = 1
        curr_max = dot((@view bandit_posterior_means[1,:]),context) * (1-lambda[t]) + lambda[t] * dot(context, (@view bandit_posterior_covs[1,:,:]), context)
        for i in 2:bandit_count
            candidate_max = dot((@view bandit_posterior_means[i,:]),context) * (1-lambda[t]) + lambda[t] * dot(context, (@view bandit_posterior_covs[i,:,:]), context)
            if candidate_max > curr_max
                curr_ind = i
                curr_max = candidate_max
            end
        end

        action = curr_ind
        #end

        true_expected_reward = true_expected_rewards[action]
        disc_reward += true_expected_reward * discount^(t-1)
        obs = randn() * obs_sd + true_expected_reward
        #old_cov = temp_post_covs[action, :, :]
        old_cov .= bandit_posterior_covs[action, :, :]
        #CovCon .= old_cov * context ./ obs_sd
        mul!(CovCon, old_cov, context)
        CovCon ./= obs_sd
        #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)

        dividy = 1 + dot(context, old_cov, context)
        for i in 1:context_dim
            for j in 1:context_dim
                bandit_posterior_covs[action,j,i] = old_cov[j,i] - CovCon[j]*CovCon[i] / dividy
            end
        end

        #bandit_posterior_covs[action, :, :] .= old_cov .- CovCon .* CovCon' ./ (1 + dot(context, old_cov, context))
        #bandit_posterior_covs[action, :, :] .= ((bandit_posterior_covs[action, :, :]) + (bandit_posterior_covs[action, :, :])') ./ 2

	    for i in 1:context_dim
		    for j in 1:(i-1)
			    bandit_posterior_covs[action,i,j] = (bandit_posterior_covs[action,i,j]+bandit_posterior_covs[action,j,i])/2
			    bandit_posterior_covs[action,j,i] = bandit_posterior_covs[action,i,j]
		    end
	    end

        SigInvMu .= old_cov \ @view bandit_posterior_means[action,:]
	    SigInvMu .+= context .* obs ./ obs_sd.^2
	    mul!((@view bandit_posterior_means[action,:]), (@view bandit_posterior_covs[action,:,:]), SigInvMu)
	    #bandit_posterior_means[action, :] .= (bandit_posterior_covs[action, :, :]) * (old_cov \ (bandit_posterior_means[action,:]) + context * obs / obs_sd^2)
    end

    return disc_reward

end

function lambda_mean_rollout(T_remainder, rollout_length, lambda, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, discount, bandit_posterior_covs, bandit_posterior_means, bandit_param,
	context, true_expected_rewards, CovCon, old_cov, SigInvMu)

    #temp_post_covs = bandit_posterior_covs
    #temp_post_means = bandit_posterior_means
    disc_reward = 0
    fill!(context, 0.0)
    fill!(true_expected_rewards, 0.0)
    fill!(CovCon, 0.0)
    fill!(old_cov,0.0)
    fill!(SigInvMu,0.0)

    for t in 1:min(T_remainder, rollout_length)

        context_seed = rand(1:context_dim)
        fill!(context, zero(context[1]))
        context[context_seed] = 1
        mul!(true_expected_rewards, bandit_param, context)

        #mean_max_bandit = 1
        #mean_max_val = dot((@view bandit_posterior_means[1,:]), context)
        #
        #var_max_bandit = 1
        #var_max_val = dot(context, (@view bandit_posterior_covs[1,:,:]), context)
#
        #for i = 2:bandit_count
            #
            #new_val = dot((@view bandit_posterior_means[i,:]),context)
            #if new_val > mean_max_val
                #mean_max_val = new_val
                #mean_max_bandit = i
            #end
#
            #new_val = dot(context, (@view bandit_posterior_covs[i, :, :]), context)
#
            #if new_val > var_max_val
                #var_max_val = new_val
                #var_max_bandit = i
            #end
        #end
#
        #if mean_max_bandit == var_max_bandit
        #
            #action = mean_max_bandit
#
        #else
        # alt action find
        curr_ind = 1
        curr_max = dot((@view bandit_posterior_means[1,:]),context) * (1-lambda[t]) + lambda[t] * max(1,T - T_remainder + t - 1) * dot(context, (@view bandit_posterior_covs[1,:,:]), context)
        for i in 2:bandit_count
            candidate_max = dot((@view bandit_posterior_means[i,:]),context) * (1-lambda[t]) + lambda[t] * max(1,T - T_remainder + t - 1) * dot(context, (@view bandit_posterior_covs[i,:,:]), context)
            if candidate_max > curr_max
                curr_ind = i
                curr_max = candidate_max
            end
        end

        action = curr_ind
        #end

        true_expected_reward = true_expected_rewards[action]
        disc_reward += true_expected_reward * discount^(t-1)
        obs = randn() * obs_sd + true_expected_reward
        #old_cov = temp_post_covs[action, :, :]
        old_cov .= bandit_posterior_covs[action, :, :]
        #CovCon .= old_cov * context ./ obs_sd
        mul!(CovCon, old_cov, context)
        CovCon ./= obs_sd
        #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)

        dividy = 1 + dot(context, old_cov, context)
        for i in 1:context_dim
            for j in 1:context_dim
                bandit_posterior_covs[action,j,i] = old_cov[j,i] - CovCon[j]*CovCon[i] / dividy
            end
        end

        #bandit_posterior_covs[action, :, :] .= old_cov .- CovCon .* CovCon' ./ (1 + dot(context, old_cov, context))
        #bandit_posterior_covs[action, :, :] .= ((bandit_posterior_covs[action, :, :]) + (bandit_posterior_covs[action, :, :])') ./ 2

	    for i in 1:context_dim
		    for j in 1:(i-1)
			    bandit_posterior_covs[action,i,j] = (bandit_posterior_covs[action,i,j]+bandit_posterior_covs[action,j,i])/2
			    bandit_posterior_covs[action,j,i] = bandit_posterior_covs[action,i,j]
		    end
	    end

        SigInvMu .= old_cov \ @view bandit_posterior_means[action,:]
	    SigInvMu .+= context .* obs ./ obs_sd.^2
	    mul!((@view bandit_posterior_means[action,:]), (@view bandit_posterior_covs[action,:,:]), SigInvMu)
	    #bandit_posterior_means[action, :] .= (bandit_posterior_covs[action, :, :]) * (old_cov \ (bandit_posterior_means[action,:]) + context * obs / obs_sd^2)
    end

    return disc_reward

end

function greedy_rollout(T_remainder, rollout_length, lambda, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, discount, bandit_posterior_covs, bandit_posterior_means, bandit_param,
	context, true_expected_rewards, CovCon, old_cov, SigInvMu)

    #temp_post_covs = bandit_posterior_covs
    #temp_post_means = bandit_posterior_means
    disc_reward = 0
    fill!(context, 0.0)
    fill!(true_expected_rewards, 0.0)
    fill!(CovCon, 0.0)
    fill!(old_cov,0.0)
    fill!(SigInvMu,0.0)

    for t in 1:min(T_remainder, rollout_length)

        context_seed = rand(1:context_dim)
        fill!(context, zero(context[1]))
        context[context_seed] = 1
        mul!(true_expected_rewards, bandit_param, context)

        #mean_max_bandit = 1
        #mean_max_val = dot((@view bandit_posterior_means[1,:]), context)
        #
        #var_max_bandit = 1
        #var_max_val = dot(context, (@view bandit_posterior_covs[1,:,:]), context)
#
        #for i = 2:bandit_count
            #
            #new_val = dot((@view bandit_posterior_means[i,:]),context)
            #if new_val > mean_max_val
                #mean_max_val = new_val
                #mean_max_bandit = i
            #end
#
            #new_val = dot(context, (@view bandit_posterior_covs[i, :, :]), context)
#
            #if new_val > var_max_val
                #var_max_val = new_val
                #var_max_bandit = i
            #end
        #end
#
        #if mean_max_bandit == var_max_bandit
        #
            #action = mean_max_bandit
#
        #else
        # alt action find
        curr_ind = 1
        curr_max = dot((@view bandit_posterior_means[1,:]),context)
        for i in 2:bandit_count
            candidate_max = dot((@view bandit_posterior_means[i,:]),context)
            if candidate_max > curr_max
                curr_ind = i
                curr_max = candidate_max
            end
        end

        action = curr_ind
        #end

        true_expected_reward = true_expected_rewards[action]
        disc_reward += true_expected_reward * discount^(t-1)
        obs = randn() * obs_sd + true_expected_reward
        #old_cov = temp_post_covs[action, :, :]
        old_cov .= bandit_posterior_covs[action, :, :]
        #CovCon .= old_cov * context ./ obs_sd
        mul!(CovCon, old_cov, context)
        CovCon ./= obs_sd
        #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)

        dividy = 1 + dot(context, old_cov, context)
        for i in 1:context_dim
            for j in 1:context_dim
                bandit_posterior_covs[action,j,i] = old_cov[j,i] - CovCon[j]*CovCon[i] / dividy
            end
        end

        #bandit_posterior_covs[action, :, :] .= old_cov .- CovCon .* CovCon' ./ (1 + dot(context, old_cov, context))
        #bandit_posterior_covs[action, :, :] .= ((bandit_posterior_covs[action, :, :]) + (bandit_posterior_covs[action, :, :])') ./ 2

	    for i in 1:context_dim
		    for j in 1:(i-1)
			    bandit_posterior_covs[action,i,j] = (bandit_posterior_covs[action,i,j]+bandit_posterior_covs[action,j,i])/2
			    bandit_posterior_covs[action,j,i] = bandit_posterior_covs[action,i,j]
		    end
	    end

        SigInvMu .= old_cov \ @view bandit_posterior_means[action,:]
	    SigInvMu .+= context .* obs ./ obs_sd.^2
	    mul!((@view bandit_posterior_means[action,:]), (@view bandit_posterior_covs[action,:,:]), SigInvMu)
	    #bandit_posterior_means[action, :] .= (bandit_posterior_covs[action, :, :]) * (old_cov \ (bandit_posterior_means[action,:]) + context * obs / obs_sd^2)
    end

    return disc_reward

end
function better_rollout(T, curr_t, rollout_length, lambda_param, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, discount, bandit_posterior_covs, bandit_posterior_means, bandit_param,
	context, true_expected_rewards, CovCon, old_cov, SigInvMu)

    #temp_post_covs = bandit_posterior_covs
    #temp_post_means = bandit_posterior_means
    disc_reward = 0
    fill!(context, 0.0)
    fill!(true_expected_rewards, 0.0)
    fill!(CovCon, 0.0)
    fill!(old_cov,0.0)
    fill!(SigInvMu,0.0)
    for t in 1:min(T, rollout_length)
        context_seed = rand(1:context_dim)
        fill!(context, zero(context[1]))
        context[context_seed] = 1
        mul!(true_expected_rewards, bandit_param, context)

        curr_ind = 1
        curr_max = dot((@view bandit_posterior_means[1,:]),context) + lambda_param[1]*(log(t+curr_t)^lambda_param[2]) * (1 - ((t+curr_t) / (curr_t+T))^lambda_param[3]) * dot(context, (@view bandit_posterior_covs[1,:,:]), context)
        for i in 2:bandit_count
            candidate_max = dot((@view bandit_posterior_means[i,:]),context) + lambda_param[1]*(log(t+curr_t)^lambda_param[2]) * (1 - ((t+curr_t) / (curr_t+T))^lambda_param[3]) * dot(context, (@view bandit_posterior_covs[i,:,:]), context)
            if candidate_max > curr_max
                curr_ind = i
                curr_max = candidate_max
            end
        end

        action = curr_ind


        true_expected_reward = true_expected_rewards[action]
	    disc_reward += true_expected_reward * discount^(t-1)
        obs = randn() * obs_sd + true_expected_reward
        old_cov .= bandit_posterior_covs[action, :, :]
	    mul!(CovCon, old_cov, context)
	    CovCon ./= obs_sd

	    dividy = 1 + dot(context, old_cov, context)
        for i in 1:context_dim
            for j in 1:context_dim
                bandit_posterior_covs[action,j,i] = old_cov[j,i] - CovCon[j]*CovCon[i] / dividy
            end
        end

        for i in 1:context_dim
            for j in 1:(i-1)
                bandit_posterior_covs[action,i,j] = (bandit_posterior_covs[action,i,j]+bandit_posterior_covs[action,j,i])/2
                bandit_posterior_covs[action,j,i] = bandit_posterior_covs[action,i,j]
            end
        end

        SigInvMu .+= context .* obs ./ obs_sd.^2
        SigInvMu .= old_cov \ @view bandit_posterior_means[action,:]
        mul!((@view bandit_posterior_means[action,:]), (@view bandit_posterior_covs[action,:,:]), SigInvMu)
    end

    return disc_reward

end

## LAMBDA POLICY

function lambda_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    predictive_rewards = bandit_posterior_means * context

    mean_max_bandit = findmax(vec(predictive_rewards))[2]
    var_max_bandit = findmax(vec([context' * (@view bandit_posterior_covs[i,:,:]) * context for i=1:bandit_count]))[2]

    if mean_max_bandit == var_max_bandit
    	return mean_max_bandit
    end

    BANDIT_VALUES = zeros(bandit_count)
    temp_param = zeros(context_dim)

    temp_post_means = zeros(bandit_count, context_dim)
    temp_post_covs = zeros(bandit_count, context_dim, context_dim)
    temp_bandit_mean = zeros(context_dim)
    temp_bandit_cov = zeros(context_dim, context_dim)

    bandit_param = zeros(bandit_count, context_dim)
    true_expected_rewards = zeros(bandit_count)

    lambda_param = [.9, .5]

    lambda = lambda_param[1] .* (lambda_param[2] .^ (0:(T-t)))
    roll_context = zeros(context_dim)
    roll_true_expected_rewards = zeros(bandit_count)
    roll_CovCon = zeros(context_dim)
    roll_old_cov = zeros(context_dim, context_dim)
    roll_SigInvMu = zeros(context_dim)

    # check for strictly dominating arm
    # selecting this arm without rollout will save lots of time and should be strictly better
    for bandit in 1:bandit_count

        BANDIT_REWARDS = 0
        for roll in 1:n_rollouts

	    fill!(roll_context, 0.0)
	    fill!(roll_true_expected_rewards, 0.0)
	    fill!(roll_CovCon, 0.0)
	    fill!(roll_old_cov, 0.0)
	    fill!(roll_SigInvMu, 0.0)


	    copy!(temp_post_means, bandit_posterior_means)
        copy!(temp_post_covs, bandit_posterior_covs)
        #bandit_param = zeros(bandit_count, context_dim)
        for bandit in 1:bandit_count
	        copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
	        copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
		    bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
        end
	    mul!(true_expected_rewards, bandit_param, context)
        true_expected_reward = true_expected_rewards[bandit]
        obs = randn() * obs_sd + true_expected_reward
        #old_cov = temp_post_covs[bandit, :, :]
	    #old_precision = inv(temp_post_covs[bandit, :, :])
	    copy!(roll_old_cov, (@view temp_post_covs[bandit, :, :]))
	    mul!(roll_CovCon, roll_old_cov, context)
	    roll_CovCon ./= obs_sd
            #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)


            # new implementation

	    dividy = 1 + dot(context, roll_old_cov, context)
        for i in 1:context_dim
	        for j in 1:context_dim
                temp_post_covs[bandit,j,i] = roll_old_cov[j,i] - roll_CovCon[j]*roll_CovCon[i] / dividy
	        end
	    end


            #bandit_posterior_covs[action, :, :] .= old_cov .- CovCon .* CovCon' ./ (1 + dot(context, old_cov, context))
            #bandit_posterior_covs[action, :, :] .= ((bandit_posterior_covs[action, :, :]) + (bandit_posterior_covs[action, :, :])') ./ 2

	    for i in 1:context_dim
	        for j in 1:(i-1)
		        temp_post_covs[bandit,i,j] = (temp_post_covs[bandit,i,j]+temp_post_covs[bandit,j,i])/2
		        temp_post_covs[bandit,j,i] = temp_post_covs[bandit,i,j]
		    end
	    end


        roll_SigInvMu .= roll_old_cov \ @view temp_post_means[bandit,:]
	    roll_SigInvMu .+= context .* obs ./ obs_sd.^2
	    mul!((@view temp_post_means[bandit,:]), (@view temp_post_covs[bandit,:,:]), roll_SigInvMu)


	    #temp_post_covs[bandit, :, :] = old_cov - CovCon * CovCon' ./ (1 + dot(context, old_cov, context))
            #temp_post_covs[bandit, :, :] = ((temp_post_covs[bandit,:,:]) + (temp_post_covs[bandit,:,:])')/2
            #temp_post_means[bandit, :] = (temp_post_covs[bandit, :, :]) * (old_cov \ (temp_post_means[bandit,:]) + context * obs / obs_sd^2)
            ##temp_post_covs[bandit, :, :] = inv(context * context' / obs_sd^2 + old_precision)
	    ##temp_post_covs[bandit, :, :] = ((temp_post_covs[bandit,:,:]) + (temp_post_covs[bandit,:,:])')/2
	    ##temp_post_means[bandit, :] = (temp_post_covs[bandit, :, :]) * (old_precision * (temp_post_means[bandit,:]) + context * obs / obs_sd^2)

        rollout_value = lambda_rollout(T-t, rollout_length, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end

function greedy_rollout_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    predictive_rewards = bandit_posterior_means * context

    mean_max_bandit = findmax(vec(predictive_rewards))[2]
    var_max_bandit = findmax(vec([context' * (@view bandit_posterior_covs[i,:,:]) * context for i=1:bandit_count]))[2]

    if mean_max_bandit == var_max_bandit
    	return mean_max_bandit
    end

    BANDIT_VALUES = zeros(bandit_count)
    temp_param = zeros(context_dim)

    temp_post_means = zeros(bandit_count, context_dim)
    temp_post_covs = zeros(bandit_count, context_dim, context_dim)
    temp_bandit_mean = zeros(context_dim)
    temp_bandit_cov = zeros(context_dim, context_dim)

    bandit_param = zeros(bandit_count, context_dim)
    true_expected_rewards = zeros(bandit_count)

    lambda_param = [.9, .5]

    lambda = lambda_param[1] .* (lambda_param[2] .^ (0:(T-t)))
    roll_context = zeros(context_dim)
    roll_true_expected_rewards = zeros(bandit_count)
    roll_CovCon = zeros(context_dim)
    roll_old_cov = zeros(context_dim, context_dim)
    roll_SigInvMu = zeros(context_dim)

    # check for strictly dominating arm
    # selecting this arm without rollout will save lots of time and should be strictly better
    for bandit in 1:bandit_count

        BANDIT_REWARDS = 0
        for roll in 1:n_rollouts

	    fill!(roll_context, 0.0)
	    fill!(roll_true_expected_rewards, 0.0)
	    fill!(roll_CovCon, 0.0)
	    fill!(roll_old_cov, 0.0)
	    fill!(roll_SigInvMu, 0.0)


	    copy!(temp_post_means, bandit_posterior_means)
        copy!(temp_post_covs, bandit_posterior_covs)
        #bandit_param = zeros(bandit_count, context_dim)
        for bandit in 1:bandit_count
	        copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
	        copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
		    bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
        end
	    mul!(true_expected_rewards, bandit_param, context)
        true_expected_reward = true_expected_rewards[bandit]
        obs = randn() * obs_sd + true_expected_reward
        #old_cov = temp_post_covs[bandit, :, :]
	    #old_precision = inv(temp_post_covs[bandit, :, :])
	    copy!(roll_old_cov, (@view temp_post_covs[bandit, :, :]))
	    mul!(roll_CovCon, roll_old_cov, context)
	    roll_CovCon ./= obs_sd
            #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)


            # new implementation

	    dividy = 1 + dot(context, roll_old_cov, context)
        for i in 1:context_dim
	        for j in 1:context_dim
                temp_post_covs[bandit,j,i] = roll_old_cov[j,i] - roll_CovCon[j]*roll_CovCon[i] / dividy
	        end
	    end


            #bandit_posterior_covs[action, :, :] .= old_cov .- CovCon .* CovCon' ./ (1 + dot(context, old_cov, context))
            #bandit_posterior_covs[action, :, :] .= ((bandit_posterior_covs[action, :, :]) + (bandit_posterior_covs[action, :, :])') ./ 2

	    for i in 1:context_dim
	        for j in 1:(i-1)
		        temp_post_covs[bandit,i,j] = (temp_post_covs[bandit,i,j]+temp_post_covs[bandit,j,i])/2
		        temp_post_covs[bandit,j,i] = temp_post_covs[bandit,i,j]
		    end
	    end


        roll_SigInvMu .= roll_old_cov \ @view temp_post_means[bandit,:]
	    roll_SigInvMu .+= context .* obs ./ obs_sd.^2
	    mul!((@view temp_post_means[bandit,:]), (@view temp_post_covs[bandit,:,:]), roll_SigInvMu)


	    #temp_post_covs[bandit, :, :] = old_cov - CovCon * CovCon' ./ (1 + dot(context, old_cov, context))
            #temp_post_covs[bandit, :, :] = ((temp_post_covs[bandit,:,:]) + (temp_post_covs[bandit,:,:])')/2
            #temp_post_means[bandit, :] = (temp_post_covs[bandit, :, :]) * (old_cov \ (temp_post_means[bandit,:]) + context * obs / obs_sd^2)
            ##temp_post_covs[bandit, :, :] = inv(context * context' / obs_sd^2 + old_precision)
	    ##temp_post_covs[bandit, :, :] = ((temp_post_covs[bandit,:,:]) + (temp_post_covs[bandit,:,:])')/2
	    ##temp_post_means[bandit, :] = (temp_post_covs[bandit, :, :]) * (old_precision * (temp_post_means[bandit,:]) + context * obs / obs_sd^2)

        rollout_value = greedy_rollout(T-t, rollout_length, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end
function lambda_mean_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    predictive_rewards = bandit_posterior_means * context

    mean_max_bandit = findmax(vec(predictive_rewards))[2]
    var_max_bandit = findmax(vec([context' * (@view bandit_posterior_covs[i,:,:]) * context for i=1:bandit_count]))[2]

    if mean_max_bandit == var_max_bandit
    	return mean_max_bandit
    end

    BANDIT_VALUES = zeros(bandit_count)
    temp_param = zeros(context_dim)

    temp_post_means = zeros(bandit_count, context_dim)
    temp_post_covs = zeros(bandit_count, context_dim, context_dim)
    temp_bandit_mean = zeros(context_dim)
    temp_bandit_cov = zeros(context_dim, context_dim)

    bandit_param = zeros(bandit_count, context_dim)
    true_expected_rewards = zeros(bandit_count)

    lambda_param = [.9, .5]

    lambda = lambda_param[1] .* (lambda_param[2] .^ (0:(T-t)))
    roll_context = zeros(context_dim)
    roll_true_expected_rewards = zeros(bandit_count)
    roll_CovCon = zeros(context_dim)
    roll_old_cov = zeros(context_dim, context_dim)
    roll_SigInvMu = zeros(context_dim)

    # check for strictly dominating arm
    # selecting this arm without rollout will save lots of time and should be strictly better
    for bandit in 1:bandit_count

        BANDIT_REWARDS = 0
        for roll in 1:n_rollouts

	    fill!(roll_context, 0.0)
	    fill!(roll_true_expected_rewards, 0.0)
	    fill!(roll_CovCon, 0.0)
	    fill!(roll_old_cov, 0.0)
	    fill!(roll_SigInvMu, 0.0)


	    copy!(temp_post_means, bandit_posterior_means)
        copy!(temp_post_covs, bandit_posterior_covs)
        #bandit_param = zeros(bandit_count, context_dim)
        for bandit in 1:bandit_count
	        copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
	        copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
		    bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
        end
	    mul!(true_expected_rewards, bandit_param, context)
        true_expected_reward = true_expected_rewards[bandit]
        obs = randn() * obs_sd + true_expected_reward
        #old_cov = temp_post_covs[bandit, :, :]
	    #old_precision = inv(temp_post_covs[bandit, :, :])
	    copy!(roll_old_cov, (@view temp_post_covs[bandit, :, :]))
	    mul!(roll_CovCon, roll_old_cov, context)
	    roll_CovCon ./= obs_sd
            #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)


            # new implementation

	    dividy = 1 + dot(context, roll_old_cov, context)
        for i in 1:context_dim
	        for j in 1:context_dim
                temp_post_covs[bandit,j,i] = roll_old_cov[j,i] - roll_CovCon[j]*roll_CovCon[i] / dividy
	        end
	    end


            #bandit_posterior_covs[action, :, :] .= old_cov .- CovCon .* CovCon' ./ (1 + dot(context, old_cov, context))
            #bandit_posterior_covs[action, :, :] .= ((bandit_posterior_covs[action, :, :]) + (bandit_posterior_covs[action, :, :])') ./ 2

	    for i in 1:context_dim
	        for j in 1:(i-1)
		        temp_post_covs[bandit,i,j] = (temp_post_covs[bandit,i,j]+temp_post_covs[bandit,j,i])/2
		        temp_post_covs[bandit,j,i] = temp_post_covs[bandit,i,j]
		    end
	    end


        roll_SigInvMu .= roll_old_cov \ @view temp_post_means[bandit,:]
	    roll_SigInvMu .+= context .* obs ./ obs_sd.^2
	    mul!((@view temp_post_means[bandit,:]), (@view temp_post_covs[bandit,:,:]), roll_SigInvMu)


	    #temp_post_covs[bandit, :, :] = old_cov - CovCon * CovCon' ./ (1 + dot(context, old_cov, context))
            #temp_post_covs[bandit, :, :] = ((temp_post_covs[bandit,:,:]) + (temp_post_covs[bandit,:,:])')/2
            #temp_post_means[bandit, :] = (temp_post_covs[bandit, :, :]) * (old_cov \ (temp_post_means[bandit,:]) + context * obs / obs_sd^2)
            ##temp_post_covs[bandit, :, :] = inv(context * context' / obs_sd^2 + old_precision)
	    ##temp_post_covs[bandit, :, :] = ((temp_post_covs[bandit,:,:]) + (temp_post_covs[bandit,:,:])')/2
	    ##temp_post_means[bandit, :] = (temp_post_covs[bandit, :, :]) * (old_precision * (temp_post_means[bandit,:]) + context * obs / obs_sd^2)

        rollout_value = lambda_mean_rollout(T-t, rollout_length, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end
## BETER LAMBDA POLICY

function better_lambda_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    BANDIT_VALUES = zeros(bandit_count)
    predictive_rewards = bandit_posterior_means * context

    temp_param = zeros(context_dim)


    temp_post_means = zeros(bandit_count, context_dim)
    temp_post_covs = zeros(bandit_count, context_dim, context_dim)
    temp_bandit_mean = zeros(context_dim)
    temp_bandit_cov = zeros(context_dim, context_dim)

    bandit_param = zeros(bandit_count, context_dim)
    true_expected_rewards = zeros(bandit_count)

    lambda_param = [.9,1.5,1]

    #lambda = lambda_param[1] .* (lambda_param[2] .^ (0:(T-t)))

    mean_max_bandit = findmax(vec(predictive_rewards))[2]
    var_max_bandit = findmax(vec([context' * (@view bandit_posterior_covs[i,:,:]) * context for i=1:bandit_count]))[2]

    roll_context = zeros(context_dim)
    roll_true_expected_rewards = zeros(bandit_count)
    roll_CovCon = zeros(context_dim)
    roll_old_cov = zeros(context_dim, context_dim)
    roll_SigInvMu = zeros(context_dim)

    # check for strictly dominating arm
    # selecting this arm without rollout will save lots of time and should be strictly better
    if mean_max_bandit == var_max_bandit
    	return mean_max_bandit
    end
    for bandit in 1:bandit_count

        BANDIT_REWARDS = 0
        for roll in 1:n_rollouts

	    fill!(roll_context, 0.0)
	    fill!(roll_true_expected_rewards, 0.0)
	    fill!(roll_CovCon, 0.0)
	    fill!(roll_old_cov, 0.0)
	    fill!(roll_SigInvMu, 0.0)


	    copy!(temp_post_means, bandit_posterior_means)
        copy!(temp_post_covs, bandit_posterior_covs)
            #bandit_param = zeros(bandit_count, context_dim)
        for bandit in 1:bandit_count
	        copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
	        copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
		    bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
        end
	    mul!(true_expected_rewards, bandit_param, context)
        true_expected_reward = true_expected_rewards[bandit]
        obs = randn() * obs_sd + true_expected_reward
            #old_cov = temp_post_covs[bandit, :, :]
	        #old_precision = inv(temp_post_covs[bandit, :, :])
	    copy!(roll_old_cov, (@view temp_post_covs[bandit, :, :]))
	    mul!(roll_CovCon, roll_old_cov, context)
	    roll_CovCon ./= obs_sd
            #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)


            # new implementation

	    dividy = 1 + dot(context, roll_old_cov, context)
        for i in 1:context_dim
	        for j in 1:context_dim
                temp_post_covs[bandit,j,i] = roll_old_cov[j,i] - roll_CovCon[j]*roll_CovCon[i] / dividy
	        end
	    end


            #bandit_posterior_covs[action, :, :] .= old_cov .- CovCon .* CovCon' ./ (1 + dot(context, old_cov, context))
            #bandit_posterior_covs[action, :, :] .= ((bandit_posterior_covs[action, :, :]) + (bandit_posterior_covs[action, :, :])') ./ 2

	    for i in 1:context_dim
	        for j in 1:(i-1)
		        temp_post_covs[bandit,i,j] = (temp_post_covs[bandit,i,j]+temp_post_covs[bandit,j,i])/2
		        temp_post_covs[bandit,j,i] = temp_post_covs[bandit,i,j]
		    end
	    end


        roll_SigInvMu .= roll_old_cov \ @view temp_post_means[bandit,:]
	    roll_SigInvMu .+= context .* obs ./ obs_sd.^2
	    mul!((@view temp_post_means[bandit,:]), (@view temp_post_covs[bandit,:,:]), roll_SigInvMu)


	    #temp_post_covs[bandit, :, :] = old_cov - CovCon * CovCon' ./ (1 + dot(context, old_cov, context))
            #temp_post_covs[bandit, :, :] = ((temp_post_covs[bandit,:,:]) + (temp_post_covs[bandit,:,:])')/2
            #temp_post_means[bandit, :] = (temp_post_covs[bandit, :, :]) * (old_cov \ (temp_post_means[bandit,:]) + context * obs / obs_sd^2)
            ##temp_post_covs[bandit, :, :] = inv(context * context' / obs_sd^2 + old_precision)
	    ##temp_post_covs[bandit, :, :] = ((temp_post_covs[bandit,:,:]) + (temp_post_covs[bandit,:,:])')/2
	    ##temp_post_means[bandit, :] = (temp_post_covs[bandit, :, :]) * (old_precision * (temp_post_means[bandit,:]) + context * obs / obs_sd^2)

        rollout_value = better_rollout(T-t, t, rollout_length, lambda_param, context_dim, context_mean,
            context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		    roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end

function opt_lambda_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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
    grad_est = zeros(2)
## END PREALLOCATION
    # check for strictly dominating arm
    # selecting this arm without rollout will save lots of time and should be strictly better
    mean_max_bandit = findmax(vec(predictive_rewards))[2]
    var_max_bandit = findmax(vec([context' * ((bandit_posterior_covs[i,:,:])) * context for i=1:bandit_count]))[2]

    if mean_max_bandit == var_max_bandit
    	return mean_max_bandit
    end
    lambda_param = [.9, .5]
    lambda_trans = log.(lambda_param ./ (1 .- lambda_param))
    BANDIT_REWARDS = zeros(n_rollouts)

    ################################################################
    ### LAMBDA OPTIMIZATION VIA SPSA
    ################################################################

    for iter in 1:n_spsa_iter

        delta_n = rand([-1, 1], 2)
        c_n = 1 / iter^(.3)
        a_n = 1 / iter
        lambda_trans_up = lambda_trans + c_n .* delta_n
        lambda_trans_dn = lambda_trans - c_n .* delta_n
        lambda_up = 1 ./ (1 .+ exp.(-lambda_trans_up))
        lambda_dn = 1 ./ (1 .+ exp.(-lambda_trans_dn))

        MEAN_UP_REWARD = 0
        MEAN_DOWN_REWARD = 0

        for roll in 1:n_opt_rollouts

            ###############
            #### UP ROLLOUT
            ###############

	        copy!(temp_post_means, bandit_posterior_means)
            copy!(temp_post_covs, bandit_posterior_covs)
            #bandit_param = zeros(bandit_count, context_dim)
            for bandit in 1:bandit_count
	            copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
	            copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
		        bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
            end

            lambda = lambda_up[1] .* (lambda_up[2] .^ (0:(T-t)))
            rollout_value = lambda_rollout(T-t, rollout_length+1, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
            # up_reward = predictive_rewards[bandit] + discount * rollout_value
            up_reward = rollout_value
            MEAN_UP_REWARD = ((roll - 1) * MEAN_UP_REWARD + up_reward) / roll

            ################
            #### DN Rollout
            ################

	        copy!(temp_post_means, bandit_posterior_means)
            copy!(temp_post_covs, bandit_posterior_covs)
            #bandit_param = zeros(bandit_count, context_dim)
            for bandit in 1:bandit_count
	            copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
	            copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
		        bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
            end

            lambda = lambda_dn[1] .* (lambda_dn[2] .^ (0:(T-t)))
            rollout_value = lambda_rollout(T-t, rollout_length+1, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
            # down_reward = predictive_rewards[bandit] + discount * rollout_value
            down_reward = rollout_value
            MEAN_DOWN_REWARD = ((roll - 1) * MEAN_DOWN_REWARD + down_reward) / roll

        end

        ###############
        #### UPDATE ###
        ###############

        grad_est .= (MEAN_UP_REWARD .- MEAN_DOWN_REWARD) ./ ((2 * c_n) .* delta_n)

        lambda_trans .+= (a_n .* grad_est)

    end



    # set final lambda parameter
    lambda_param = 1 ./ (1 .+ exp.(-lambda_trans))


    println(lambda_param[1],", ", lambda_param[2])
    flush(stdout)

    lambda = lambda_param[1] .* (lambda_param[2] .^ (0:(T-t)))
    # END OPTIMIZATION OF LAMBDA


    for bandit in 1:bandit_count
        BANDIT_REWARDS = 0
        for roll in 1:n_rollouts

	    fill!(roll_context, 0.0)
	    fill!(roll_true_expected_rewards, 0.0)
	    fill!(roll_CovCon, 0.0)
	    fill!(roll_old_cov, 0.0)
	    fill!(roll_SigInvMu, 0.0)


	    copy!(temp_post_means, bandit_posterior_means)
            copy!(temp_post_covs, bandit_posterior_covs)
            #bandit_param = zeros(bandit_count, context_dim)
            for bandit in 1:bandit_count
	        copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
	        copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
		bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
            end
	    mul!(true_expected_rewards, bandit_param, context)
            true_expected_reward = true_expected_rewards[bandit]
            obs = randn() * obs_sd + true_expected_reward
            #old_cov = temp_post_covs[bandit, :, :]
	    #old_precision = inv(temp_post_covs[bandit, :, :])
	    copy!(roll_old_cov, (@view temp_post_covs[bandit, :, :]))
	    mul!(roll_CovCon, roll_old_cov, context)
	    roll_CovCon ./= obs_sd
            #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)


            # new implementation

	    dividy = 1 + dot(context, roll_old_cov, context)
            for i in 1:context_dim
	        for j in 1:context_dim
                     temp_post_covs[bandit,j,i] = roll_old_cov[j,i] - roll_CovCon[j]*roll_CovCon[i] / dividy
	        end
	    end


            #bandit_posterior_covs[action, :, :] .= old_cov .- CovCon .* CovCon' ./ (1 + dot(context, old_cov, context))
            #bandit_posterior_covs[action, :, :] .= ((bandit_posterior_covs[action, :, :]) + (bandit_posterior_covs[action, :, :])') ./ 2

	    for i in 1:context_dim
	        for j in 1:(i-1)
		    temp_post_covs[bandit,i,j] = (temp_post_covs[bandit,i,j]+temp_post_covs[bandit,j,i])/2
		    temp_post_covs[bandit,j,i] = temp_post_covs[bandit,i,j]
		end
	    end


            roll_SigInvMu .= roll_old_cov \ @view temp_post_means[bandit,:]
	    roll_SigInvMu .+= context .* obs ./ obs_sd.^2
	    mul!((@view temp_post_means[bandit,:]), (@view temp_post_covs[bandit,:,:]), roll_SigInvMu)



            rollout_value = lambda_rollout(T-t, rollout_length, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end


function better_opt_lambda_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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
    grad_est = zeros(2)
## END PREALLOCATION
    # check for strictly dominating arm
    # selecting this arm without rollout will save lots of time and should be strictly better
    mean_max_bandit = findmax(vec(predictive_rewards))[2]
    var_max_bandit = findmax(vec([context' * ((bandit_posterior_covs[i,:,:])) * context for i=1:bandit_count]))[2]

    if mean_max_bandit == var_max_bandit
    	return mean_max_bandit
    end
    lambda_param = [.9, .5]
    lambda_trans = log.(lambda_param ./ (1 .- lambda_param))
    BANDIT_REWARDS = zeros(n_rollouts)

    ################################################################
    ### LAMBDA OPTIMIZATION VIA SPSA
    ################################################################

    for iter in 1:n_spsa_iter

        delta_n = rand([-1, 1], 2)
        c_n = 1 / iter^(.3)
        a_n = 1 / iter
        lambda_trans_up = lambda_trans + c_n .* delta_n
        lambda_trans_dn = lambda_trans - c_n .* delta_n
        lambda_up = 1 ./ (1 .+ exp.(-lambda_trans_up))
        lambda_dn = 1 ./ (1 .+ exp.(-lambda_trans_dn))

        MEAN_UP_REWARD = 0
        MEAN_DOWN_REWARD = 0

        for roll in 1:n_opt_rollouts

            ###############
            #### UP ROLLOUT
            ###############

	        copy!(temp_post_means, bandit_posterior_means)
            copy!(temp_post_covs, bandit_posterior_covs)
            #bandit_param = zeros(bandit_count, context_dim)
            for bandit in 1:bandit_count
	            copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
	            copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
		        bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
            end

            lambda = lambda_up[1] .* (lambda_up[2] .^ (0:(T-t)))
            rollout_value = lambda_rollout(T-t, rollout_length+1, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
            # up_reward = predictive_rewards[bandit] + discount * rollout_value
            up_reward = rollout_value
            MEAN_UP_REWARD = ((roll - 1) * MEAN_UP_REWARD + up_reward) / roll

            ################
            #### DN Rollout
            ################

	        copy!(temp_post_means, bandit_posterior_means)
            copy!(temp_post_covs, bandit_posterior_covs)
            #bandit_param = zeros(bandit_count, context_dim)
            for bandit in 1:bandit_count
	            copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
	            copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
		        bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
            end

            lambda = lambda_dn[1] .* (lambda_dn[2] .^ (0:(T-t)))
            rollout_value = lambda_rollout(T-t, rollout_length+1, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
            # down_reward = predictive_rewards[bandit] + discount * rollout_value
            down_reward = rollout_value
            MEAN_DOWN_REWARD = ((roll - 1) * MEAN_DOWN_REWARD + down_reward) / roll

        end

        ###############
        #### UPDATE ###
        ###############

        grad_est .= (MEAN_UP_REWARD .- MEAN_DOWN_REWARD) ./ ((2 * c_n) .* delta_n)

        lambda_trans .+= (a_n .* grad_est)

    end



    # set final lambda parameter
    lambda_param = 1 ./ (1 .+ exp.(-lambda_trans))


    println(lambda_param[1],", ", lambda_param[2])
    flush(stdout)

    lambda = lambda_param[1] .* (lambda_param[2] .^ (0:(T-t)))
    # END OPTIMIZATION OF LAMBDA


    for bandit in 1:bandit_count
        BANDIT_REWARDS = 0
        for roll in 1:n_rollouts

	    fill!(roll_context, 0.0)
	    fill!(roll_true_expected_rewards, 0.0)
	    fill!(roll_CovCon, 0.0)
	    fill!(roll_old_cov, 0.0)
	    fill!(roll_SigInvMu, 0.0)


	    copy!(temp_post_means, bandit_posterior_means)
            copy!(temp_post_covs, bandit_posterior_covs)
            #bandit_param = zeros(bandit_count, context_dim)
            for bandit in 1:bandit_count
	        copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
	        copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
		bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
            end
	    mul!(true_expected_rewards, bandit_param, context)
            true_expected_reward = true_expected_rewards[bandit]
            obs = randn() * obs_sd + true_expected_reward
            #old_cov = temp_post_covs[bandit, :, :]
	    #old_precision = inv(temp_post_covs[bandit, :, :])
	    copy!(roll_old_cov, (@view temp_post_covs[bandit, :, :]))
	    mul!(roll_CovCon, roll_old_cov, context)
	    roll_CovCon ./= obs_sd
            #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)


            # new implementation

	    dividy = 1 + dot(context, roll_old_cov, context)
            for i in 1:context_dim
	        for j in 1:context_dim
                     temp_post_covs[bandit,j,i] = roll_old_cov[j,i] - roll_CovCon[j]*roll_CovCon[i] / dividy
	        end
	    end


            #bandit_posterior_covs[action, :, :] .= old_cov .- CovCon .* CovCon' ./ (1 + dot(context, old_cov, context))
            #bandit_posterior_covs[action, :, :] .= ((bandit_posterior_covs[action, :, :]) + (bandit_posterior_covs[action, :, :])') ./ 2

	    for i in 1:context_dim
	        for j in 1:(i-1)
		    temp_post_covs[bandit,i,j] = (temp_post_covs[bandit,i,j]+temp_post_covs[bandit,j,i])/2
		    temp_post_covs[bandit,j,i] = temp_post_covs[bandit,i,j]
		end
	    end


            roll_SigInvMu .= roll_old_cov \ @view temp_post_means[bandit,:]
	    roll_SigInvMu .+= context .* obs ./ obs_sd.^2
	    mul!((@view temp_post_means[bandit,:]), (@view temp_post_covs[bandit,:,:]), roll_SigInvMu)



            rollout_value = lambda_rollout(T-t, rollout_length, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end

function bayesopt_lambda_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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
    grad_est = zeros(2)
    ## END PREALLOCATION
    mean_max_bandit = findmax(vec(predictive_rewards))[2]
    var_max_bandit = findmax(vec([context' * ((bandit_posterior_covs[i,:,:])) * context for i=1:bandit_count]))[2]

    if mean_max_bandit == var_max_bandit
    	return mean_max_bandit
    end
    lambda_param = [.9,2,2]

    function rolloutsim(lambda_param)

        copy!(temp_post_means, bandit_posterior_means)
        copy!(temp_post_covs, bandit_posterior_covs)
        for bandit in 1:bandit_count
            copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
            copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
            bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
        end

        rollout_value = better_rollout(T-t, t, rollout_length+1, lambda_param, context_dim, context_mean,
            context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
            roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
        return rollout_value
    end


    # Choose as a model an elastic GP with input dimensions 2.
    # The GP is called elastic, because data can be appended efficiently.
    model = ElasticGPE(3,                            # 2 input dimensions
                       mean = MeanConst(0.),
                       kernel = SEArd([0., 0., 0.], 5.),
                       logNoise = 0.,
                       capacity = 3000)              # the initial capacity of the GP is 3000 samples.

    # Optimize the hyperparameters of the GP using maximum a posteriori (MAP) estimates every 50 steps
    modeloptimizer = MAPGPOptimizer(every = 50, noisebounds = [-4, 3],       # bounds of the logNoise
                                    kernbounds = [[-1, -1, -1, 0], [4, 4, 4, 10]],  # bounds of the 3 parameters GaussianProcesses.get_param_names(model.kernel)
                                    maxeval = 40)
    opt = BOpt(rolloutsim,
               model,
               UpperConfidenceBound(),                   # type of acquisition
               modeloptimizer,
               [0, 1, 1], [1, 5, 5],                     # lowerbounds, upperbounds
               repetitions = 2000,                          # evaluate the function for each input 5 times
               maxiterations = 100,                      # evaluate at 100 input positions
               sense = Max,                              # minimize the function
               acquisitionoptions = (method = :LD_LBFGS, # run optimization of acquisition function with NLopts :LD_LBFGS method
                                     restarts = 5,       # run the NLopt method from 5 random initial conditions each time.
                                     maxtime = 0.1,      # run the NLopt method for at most 0.1 second each time
                                     maxeval = 1000),    # run the NLopt methods for at most 1000 iterations (for other options see https://github.com/JuliaOpt/NLopt.jl)
                verbosity = Silent)

    obs_opt, obs_optimizer, model_opt, lambda_param = boptimize!(opt)
    println(lambda_param[1],", ", lambda_param[2],", ",lambda_param[3])
    flush(stdout)

    # END OPTIMIZATION OF LAMBDA

    for bandit in 1:bandit_count

        BANDIT_REWARDS = 0
        for roll in 1:n_rollouts

	    fill!(roll_context, 0.0)
	    fill!(roll_true_expected_rewards, 0.0)
	    fill!(roll_CovCon, 0.0)
	    fill!(roll_old_cov, 0.0)
	    fill!(roll_SigInvMu, 0.0)

	    copy!(temp_post_means, bandit_posterior_means)
        copy!(temp_post_covs, bandit_posterior_covs)
        for bandit in 1:bandit_count
	        copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
	        copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
		    bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
        end
	    mul!(true_expected_rewards, bandit_param, context)
        true_expected_reward = true_expected_rewards[bandit]
        obs = randn() * obs_sd + true_expected_reward
	    copy!(roll_old_cov, (@view temp_post_covs[bandit, :, :]))
	    mul!(roll_CovCon, roll_old_cov, context)
	    roll_CovCon ./= obs_sd

	    dividy = 1 + dot(context, roll_old_cov, context)
        for i in 1:context_dim
	        for j in 1:context_dim
                temp_post_covs[bandit,j,i] = roll_old_cov[j,i] - roll_CovCon[j]*roll_CovCon[i] / dividy
	        end
	    end

	    for i in 1:context_dim
	        for j in 1:(i-1)
		        temp_post_covs[bandit,i,j] = (temp_post_covs[bandit,i,j]+temp_post_covs[bandit,j,i])/2
		        temp_post_covs[bandit,j,i] = temp_post_covs[bandit,i,j]
		    end
	    end

        roll_SigInvMu .= roll_old_cov \ @view temp_post_means[bandit,:]
	    roll_SigInvMu .+= context .* obs ./ obs_sd.^2
	    mul!((@view temp_post_means[bandit,:]), (@view temp_post_covs[bandit,:,:]), roll_SigInvMu)

        rollout_value = better_rollout(T-t, t, rollout_length, lambda_param, context_dim, context_mean,
            context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		    roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end

##### GRID SEARCH VERSION
function grid_lambda_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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
    grad_est = zeros(2)
## END PREALLOCATION
    # check for strictly dominating arm
    # selecting this arm without rollout will save lots of time and should be strictly better
    mean_max_bandit = findmax(vec(predictive_rewards))[2]
    var_max_bandit = findmax(vec([context' * ((bandit_posterior_covs[i,:,:])) * context for i=1:bandit_count]))[2]

    if mean_max_bandit == var_max_bandit
    	return mean_max_bandit
    end
    lambda_param = [.9, .5]
    lambda_trans = log.(lambda_param ./ (1 .- lambda_param))
    BANDIT_REWARDS = zeros(n_rollouts)


    curr_max = -10000000
    lambda1_curr = lambda_param[1]
    lambda2_curr = lambda_param[2]

    curr_length = int_length

    for iter in 1:n_grid_iter


        lower1 = max(0, lambda1_curr - curr_length)
	upper1 = min(1, lambda1_curr + curr_length)
	lower2 = max(0, lambda2_curr - curr_length)
	upper2 = min(1, lambda2_curr + curr_length)

	inc1 = (upper1 - lower1) / grid_num
	inc2 = (upper2 - lower2) / grid_num

	grid_iterator = Iterators.product(lower1:inc1:upper1, lower2:inc2:upper2)
        curr_length *= .5

	for grid_spot in grid_iterator

            MEAN_REWARD = 0

            for roll in 1:n_grid_rollouts

	        copy!(temp_post_means, bandit_posterior_means)
                copy!(temp_post_covs, bandit_posterior_covs)
                for bandit in 1:bandit_count
	            copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
	            copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
		    bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
                end

		lambda = grid_spot[1] .* (grid_spot[2] .^ (0:(T-t)))
                rollout_value = lambda_rollout(T-t, rollout_length+1, lambda, context_dim, context_mean,
                    context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		    roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
                MEAN_REWARD = ((roll - 1) * MEAN_REWARD + rollout_value) / roll

	    end

            if MEAN_REWARD > curr_max
	        curr_max = MEAN_REWARD
		lambda1_curr = grid_spot[1]
		lambda2_curr = grid_spot[2]
            end
        end
    end

    lambda_param[1] = lambda1_curr
    lambda_param[2] = lambda2_curr

    lambda = lambda_param[1] .* (lambda_param[2] .^ (0:(T-t)))
    # END OPTIMIZATION OF LAMBDA


    for bandit in 1:bandit_count
        BANDIT_REWARDS = 0
        for roll in 1:n_rollouts

	    fill!(roll_context, 0.0)
	    fill!(roll_true_expected_rewards, 0.0)
	    fill!(roll_CovCon, 0.0)
	    fill!(roll_old_cov, 0.0)
	    fill!(roll_SigInvMu, 0.0)


	    copy!(temp_post_means, bandit_posterior_means)
            copy!(temp_post_covs, bandit_posterior_covs)
            #bandit_param = zeros(bandit_count, context_dim)
            for bandit in 1:bandit_count
	        copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
	        copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
		bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
            end
	    mul!(true_expected_rewards, bandit_param, context)
            true_expected_reward = true_expected_rewards[bandit]
            obs = randn() * obs_sd + true_expected_reward
            #old_cov = temp_post_covs[bandit, :, :]
	    #old_precision = inv(temp_post_covs[bandit, :, :])
	    copy!(roll_old_cov, (@view temp_post_covs[bandit, :, :]))
	    mul!(roll_CovCon, roll_old_cov, context)
	    roll_CovCon ./= obs_sd
            #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)


            # new implementation

	    dividy = 1 + dot(context, roll_old_cov, context)
            for i in 1:context_dim
	        for j in 1:context_dim
                     temp_post_covs[bandit,j,i] = roll_old_cov[j,i] - roll_CovCon[j]*roll_CovCon[i] / dividy
	        end
	    end


	    for i in 1:context_dim
	        for j in 1:(i-1)
		    temp_post_covs[bandit,i,j] = (temp_post_covs[bandit,i,j]+temp_post_covs[bandit,j,i])/2
		    temp_post_covs[bandit,j,i] = temp_post_covs[bandit,i,j]
		end
	    end


            roll_SigInvMu .= roll_old_cov \ @view temp_post_means[bandit,:]
	    roll_SigInvMu .+= context .* obs ./ obs_sd.^2
	    mul!((@view temp_post_means[bandit,:]), (@view temp_post_covs[bandit,:,:]), roll_SigInvMu)

            rollout_value = lambda_rollout(T-t, rollout_length, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end


function ent_lambda_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    BANDIT_VALUES = zeros(bandit_count)
    predictive_rewards = bandit_posterior_means * context

    temp_param = zeros(context_dim)


    temp_post_means = zeros(bandit_count, context_dim)
    temp_post_covs = zeros(bandit_count, context_dim, context_dim)
    temp_bandit_mean = zeros(context_dim)
    temp_bandit_cov = zeros(context_dim, context_dim)

    bandit_param = zeros(bandit_count, context_dim)
    true_expected_rewards = zeros(bandit_count)

    lambda_param = [.9, .5]

    lambda = lambda_param[1] .* (lambda_param[2] .^ (0:(T-t)))

    mean_max_bandit = findmax(vec(predictive_rewards))[2]
    var_max_bandit = findmax(vec([log(1+context' * (@view bandit_posterior_covs[i,:,:]) * context/obs_sd^2) for i=1:bandit_count]))[2]

    roll_context = zeros(context_dim)
    roll_true_expected_rewards = zeros(bandit_count)
    roll_CovCon = zeros(context_dim)
    roll_old_cov = zeros(context_dim, context_dim)
    roll_SigInvMu = zeros(context_dim)

    # check for strictly dominating arm
    # selecting this arm without rollout will save lots of time and should be strictly better
    if mean_max_bandit == var_max_bandit
    	return mean_max_bandit
    end
    for bandit in 1:bandit_count

        BANDIT_REWARDS = 0
        for roll in 1:n_rollouts

	    fill!(roll_context, 0.0)
	    fill!(roll_true_expected_rewards, 0.0)
	    fill!(roll_CovCon, 0.0)
	    fill!(roll_old_cov, 0.0)
	    fill!(roll_SigInvMu, 0.0)


	    copy!(temp_post_means, bandit_posterior_means)
            copy!(temp_post_covs, bandit_posterior_covs)
            #bandit_param = zeros(bandit_count, context_dim)
            for bandit in 1:bandit_count
	        copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
	        copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
		bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
            end
	    mul!(true_expected_rewards, bandit_param, context)
            true_expected_reward = true_expected_rewards[bandit]
            obs = randn() * obs_sd + true_expected_reward
            #old_cov = temp_post_covs[bandit, :, :]
	    #old_precision = inv(temp_post_covs[bandit, :, :])
	    copy!(roll_old_cov, (@view temp_post_covs[bandit, :, :]))
	    mul!(roll_CovCon, roll_old_cov, context)
	    roll_CovCon ./= obs_sd
            #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)


            # new implementation

	    dividy = 1 + dot(context, roll_old_cov, context)
            for i in 1:context_dim
	        for j in 1:context_dim
                     temp_post_covs[bandit,j,i] = roll_old_cov[j,i] - roll_CovCon[j]*roll_CovCon[i] / dividy
	        end
	    end


            #bandit_posterior_covs[action, :, :] .= old_cov .- CovCon .* CovCon' ./ (1 + dot(context, old_cov, context))
            #bandit_posterior_covs[action, :, :] .= ((bandit_posterior_covs[action, :, :]) + (bandit_posterior_covs[action, :, :])') ./ 2

	    for i in 1:context_dim
	        for j in 1:(i-1)
		    temp_post_covs[bandit,i,j] = (temp_post_covs[bandit,i,j]+temp_post_covs[bandit,j,i])/2
		    temp_post_covs[bandit,j,i] = temp_post_covs[bandit,i,j]
		end
	    end


            roll_SigInvMu .= roll_old_cov \ @view temp_post_means[bandit,:]
	    roll_SigInvMu .+= context .* obs ./ obs_sd.^2
	    mul!((@view temp_post_means[bandit,:]), (@view temp_post_covs[bandit,:,:]), roll_SigInvMu)


            rollout_value = ent_lambda_rollout(T-t, rollout_length, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end

function ent_opt_lambda_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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
    grad_est = zeros(2)
## END PREALLOCATION
    # check for strictly dominating arm
    # selecting this arm without rollout will save lots of time and should be strictly better
    mean_max_bandit = findmax(vec(predictive_rewards))[2]
    var_max_bandit = findmax(vec([log(1+(context' * ((bandit_posterior_covs[i,:,:])) * context)/obs_sd^2) for i=1:bandit_count]))[2]

    if mean_max_bandit == var_max_bandit
    	return mean_max_bandit
    end
    lambda_param = [.9, .5]
    lambda_trans = log.(lambda_param ./ (1 .- lambda_param))
    BANDIT_REWARDS = zeros(n_rollouts)

    ################################################################
    ### LAMBDA OPTIMIZATION VIA SPSA
    ################################################################

    for iter in 1:n_spsa_iter

        delta_n = rand([-1, 1], 2)
        c_n = 1 / iter^(.3)
        a_n = 1 / iter
        lambda_trans_up = lambda_trans + c_n .* delta_n
        lambda_trans_dn = lambda_trans - c_n .* delta_n
        lambda_up = 1 ./ (1 .+ exp.(-lambda_trans_up))
        lambda_dn = 1 ./ (1 .+ exp.(-lambda_trans_dn))

        MEAN_UP_REWARD = 0
        MEAN_DOWN_REWARD = 0

        for roll in 1:n_opt_rollouts

            ###############
            #### UP ROLLOUT
            ###############

	    copy!(temp_post_means, bandit_posterior_means)
            copy!(temp_post_covs, bandit_posterior_covs)
            #bandit_param = zeros(bandit_count, context_dim)
            for bandit in 1:bandit_count
	        copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
	        copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
		bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
            end

            lambda = lambda_up[1] .* (lambda_up[2] .^ (0:(T-t)))
            rollout_value = ent_lambda_rollout(T-t, rollout_length+1, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
            # up_reward = predictive_rewards[bandit] + discount * rollout_value
            up_reward = rollout_value
            MEAN_UP_REWARD = ((roll - 1) * MEAN_UP_REWARD + up_reward) / roll

            ################
            #### DN Rollout
            ################

	    copy!(temp_post_means, bandit_posterior_means)
            copy!(temp_post_covs, bandit_posterior_covs)
            #bandit_param = zeros(bandit_count, context_dim)
            for bandit in 1:bandit_count
	        copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
	        copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
		bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
            end

            lambda = lambda_dn[1] .* (lambda_dn[2] .^ (0:(T-t)))
            rollout_value = ent_lambda_rollout(T-t, rollout_length+1, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
            # down_reward = predictive_rewards[bandit] + discount * rollout_value
            down_reward = rollout_value
            MEAN_DOWN_REWARD = ((roll - 1) * MEAN_DOWN_REWARD + down_reward) / roll

        end

        ###############
        #### UPDATE ###
        ###############

        grad_est .= (MEAN_UP_REWARD .- MEAN_DOWN_REWARD) ./ ((2 * c_n) .* delta_n)

        lambda_trans .+= (a_n .* grad_est)

    end

    # set final lambda parameter
    lambda_param = 1 ./ (1 .+ exp.(-lambda_trans))

    lambda = lambda_param[1] .* (lambda_param[2] .^ (0:(T-t)))
    # END OPTIMIZATION OF LAMBDA


    for bandit in 1:bandit_count
        BANDIT_REWARDS = 0
        for roll in 1:n_rollouts

	    fill!(roll_context, 0.0)
	    fill!(roll_true_expected_rewards, 0.0)
	    fill!(roll_CovCon, 0.0)
	    fill!(roll_old_cov, 0.0)
	    fill!(roll_SigInvMu, 0.0)


	    copy!(temp_post_means, bandit_posterior_means)
            copy!(temp_post_covs, bandit_posterior_covs)
            #bandit_param = zeros(bandit_count, context_dim)
            for bandit in 1:bandit_count
	        copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
	        copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
		bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
            end
	    mul!(true_expected_rewards, bandit_param, context)
            true_expected_reward = true_expected_rewards[bandit]
            obs = randn() * obs_sd + true_expected_reward
            #old_cov = temp_post_covs[bandit, :, :]
	    #old_precision = inv(temp_post_covs[bandit, :, :])
	    copy!(roll_old_cov, (@view temp_post_covs[bandit, :, :]))
	    mul!(roll_CovCon, roll_old_cov, context)
	    roll_CovCon ./= obs_sd
            #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)


            # new implementation

	    dividy = 1 + dot(context, roll_old_cov, context)
            for i in 1:context_dim
	        for j in 1:context_dim
                     temp_post_covs[bandit,j,i] = roll_old_cov[j,i] - roll_CovCon[j]*roll_CovCon[i] / dividy
	        end
	    end



	    for i in 1:context_dim
	        for j in 1:(i-1)
		    temp_post_covs[bandit,i,j] = (temp_post_covs[bandit,i,j]+temp_post_covs[bandit,j,i])/2
		    temp_post_covs[bandit,j,i] = temp_post_covs[bandit,i,j]
		end
	    end


            roll_SigInvMu .= roll_old_cov \ @view temp_post_means[bandit,:]
	    roll_SigInvMu .+= context .* obs ./ obs_sd.^2
	    mul!((@view temp_post_means[bandit,:]), (@view temp_post_covs[bandit,:,:]), roll_SigInvMu)


            rollout_value = ent_lambda_rollout(T-t, rollout_length, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end
## SIMULATIONS

# Sims

print("\n")

@time greedyRewards, greedyOptrewards = contextual_bandit_simulator(greedy_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                          context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

print("\n")
#@time epsgreedyRewards, epsgreedyOptrewards = contextual_bandit_simulator(epsilon_greedy_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
#                                          context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

#print("\n")
#@time thompsonRewards, thompsonOptrewards = contextual_bandit_simulator(thompson_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
#                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

#print("\n")
#@time bayesucbRewards, bayesucbOptrewards = contextual_bandit_simulator(bayes_ucb_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
#                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

#print("\n")
#@time squarecbRewards, squarecbOptrewards = contextual_bandit_simulator(squarecb_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
#                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

#print("\n")
#@time lambdaRewards, lambdaOptrewards = contextual_bandit_simulator(lambda_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
#                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)


#print("\n")

@time betterlambdaRewards, betterlambdaOptrewards = contextual_bandit_simulator(better_lambda_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

print("\n")

#@time optlambdaRewards, optlambdaOptrewards = contextual_bandit_simulator(opt_lambda_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
#                                          context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

#print("\n")

#@time lambdameanRewards, lambdameanOptrewards = contextual_bandit_simulator(lambda_mean_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
#                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
#print("\n")


@time greedyrolloutRewards, greedyrolloutOptrewards = contextual_bandit_simulator(greedy_rollout_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
print("\n")

#@time bayesoptrolloutRewards, bayesoptrolloutOptrewards = contextual_bandit_simulator(bayesopt_lambda_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
#                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
#print("\n")
#@time gridlambdaRewards, gridlambdaOptrewards = contextual_bandit_simulator(grid_lambda_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
#                                           context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
#print("\n")
#@time entlambdaRewards, entlambdaOptrewards = contextual_bandit_simulator(ent_lambda_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
#                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
#print("\n")
#@time entoptlambdaRewards, entoptlambdaOptrewards = contextual_bandit_simulator(ent_opt_lambda_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
#                                            context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)
# Plot

const discount_vector = discount .^ collect(0:(T-1))

const greedyCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(greedyOptrewards - greedyRewards)])
#const epsgreedyCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(epsgreedyOptrewards - epsgreedyRewards)])
#const thompsonCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(thompsonOptrewards - thompsonRewards)])
#const bayesucbCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(bayesucbOptrewards - bayesucbRewards)])
#const squarecbCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(squarecbOptrewards - squarecbRewards)])
#const lambdaCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(lambdaOptrewards - lambdaRewards)])
#const lambdameanCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(lambdameanOptrewards - lambdameanRewards)])
const greedyrolloutCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(greedyrolloutOptrewards - greedyrolloutRewards)])
const betterlambdaCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(betterlambdaOptrewards - betterlambdaRewards)])
#const optlambdaCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(optlambdaOptrewards - optlambdaRewards)])
#const bayesoptlambdaCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(bayesoptrolloutOptrewards - bayesoptrolloutRewards)])
#const gridlambdaCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(gridlambdaOptrewards - gridlambdaRewards)])
#const entlambdaCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(entlambdaOptrewards - entlambdaRewards)])
#const entoptlambdaCumDiscReg = cumsum(discount_vector .* [mean(c) for c in eachcol(entoptlambdaOptrewards - entoptlambdaRewards)])

#const combReg = [greedyCumDiscReg epsgreedyCumDiscReg thompsonCumDiscReg bayesucbCumDiscReg squarecbCumDiscReg lambdaCumDiscReg lambdameanCumDiscReg greedyrolloutCumDiscReg betterlambdaCumDiscReg optlambdaCumDiscReg bayesoptlambdaCumDiscReg]

const combReg = [greedyCumDiscReg greedyrolloutCumDiscReg betterlambdaCumDiscReg]
CSV.write("/hpc/home/jml165/rl/arrayresults/results_$(idx).csv", Tables.table(combReg), header=["Greedy", "GreedyRollout", "BetterLambda"])

#CSV.write("/hpc/home/jml165/rl/arrayresults/results_$(idx).csv",  Tables.table(combReg), header=["Greedy", "EpsGreedy", "Thompson", "BayesUCB", "SquareCB", "Lambda","LambdaMean","GreedyRollout", "BetterLambda","OptLambda", "BayesOptLambda"])
