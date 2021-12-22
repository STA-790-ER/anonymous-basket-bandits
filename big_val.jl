
using LinearAlgebra, Statistics, Plots, Distributions, StatsBase, DelimitedFiles, CSV, Tables, Random, BenchmarkTools

using BayesianOptimization, GaussianProcesses, Distributions
BLAS.set_num_threads(1)

using TimerOutputs

to = TimerOutput()

#const mts = MersenneTwister.(1:Threads.nthreads())
# Parameters
const context_dim = 2
const context_mean = 0
const context_sd = 1
const obs_sd = 1

const bandit_count = 3
const bandit_prior_mean = 0
const bandit_prior_sd = 10

const n_episodes = 1000

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

const n_points = 10000

## SIMULATOR FUNCTION



const idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])


const T = idx

function grid_contextual_bandit_simulator(n_points, action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

    means = zeros(n_points, bandit_count, context_dim)
    covs = zeros(n_points, bandit_count, context_dim, context_dim)

    GRID_REWARDS = zeros(n_episodes*n_points, T)
    GRID_POST_MEANS = zeros(n_episodes*n_points, T, bandit_count, context_dim)
    GRID_POST_COVS = zeros(n_episodes*n_points, T, bandit_count, context_dim, context_dim)
    MEAN_REWARDS = zeros(n_points, T)
    MEAN_POST_MEANS = zeros(n_points, T, bandit_count, context_dim)
    MEAN_POST_COVS = zeros(n_points, T, bandit_count, context_dim, context_dim)
    for i in 1:n_points
        for j in 1:bandit_count
            means[i, j, :] = rand(MvNormal(repeat([bandit_prior_mean], context_dim),
                                              Matrix((bandit_prior_sd^2)I,context_dim,context_dim)))
            covs[i, j, :, :] = rand(InverseWishart(context_dim + 2, Matrix((bandit_prior_sd^2)I, context_dim, context_dim)))
        end

    end


    for i in 1:n_points
        bandit_prior_means = means[i, :, :]
        bandit_prior_covs = covs[i, :, :, :]

        TOT_REWARDS, TOT_POST_MEANS, TOT_POST_COVS = contextual_bandit_simulator(action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
            context_sd, obs_sd, bandit_count, bandit_prior_means, bandit_prior_covs, discount, epsilon)
        #print(TOT_REWARDS)
        GRID_REWARDS[((i-1)*n_episodes+1):(i*n_episodes), :] = TOT_REWARDS
        MEAN_REWARDS[i, :] = Vector([mean(TOT_REWARDS[:, t]) for t in 1:T])
        MEAN_POST_MEANS[i, :, :, :] = TOT_POST_MEANS[1, :, :, :]
        MEAN_POST_COVS[i, :, :, :, :] = TOT_POST_COVS[1, :, :, :, :]
        GRID_POST_MEANS[((i-1)*n_episodes+1):(i*n_episodes), :, :, :] = TOT_POST_MEANS
        GRID_POST_COVS[((i-1)*n_episodes+1):(i*n_episodes), :, :, :, :] = TOT_POST_COVS


    end


    return GRID_REWARDS, GRID_POST_MEANS, GRID_POST_COVS, MEAN_REWARDS, MEAN_POST_MEANS, MEAN_POST_COVS



end


function ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_means, bandit_prior_covs, discount, epsilon, global_bandit_param)
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

        POST_MEANS = zeros(T, bandit_count, context_dim)
        POST_COVS = zeros(T, bandit_count, context_dim, context_dim)

        copy!(bandit_posterior_means, bandit_prior_means)
        copy!(bandit_posterior_covs, bandit_prior_covs)

        #io = open("~/rl/monitor/monitor_$(idx).txt", "w")
	#write(io, 0)
	#close(io)
        for t in 1:T

            POST_MEANS[t, :, :] = bandit_posterior_means
            POST_COVS[t, :, :, :] = bandit_posterior_covs

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

	    #println("Ep: ", ep, " - ", t, " of ", T, " for ", String(Symbol(action_function)))
        #    flush(stdout)

	    #io = open("~/rl/monitor/monitor_$(idx).txt", "r")
	    #curr_prog = read(io, Int)
	    #close(io)
	    #io = open("~/rl/monitor/monitor_$(idx).txt", "w")
	    #write(io, curr_prog+1)
	    #close(io)

        end
        if ep % 10 == 0 
	        println("Ep: ", ep, " for ", String(Symbol(action_function)))
            flush(stdout)
        end
	return EPREWARDS, POST_MEANS, POST_COVS
end

function contextual_bandit_simulator(action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_means, bandit_prior_covs, discount, epsilon)

    REWARDS = zeros(T, n_episodes)
    OPTREWARDS = zeros(T, n_episodes)
    TOT_REWARDS = zeros(n_episodes, T)
    TOT_POST_MEANS = zeros(n_episodes, T, bandit_count, context_dim)
    TOT_POST_COVS = zeros(n_episodes, T, bandit_count, context_dim, context_dim)
    ep_count = 1
    global_bandit_param = zeros(bandit_count, context_dim)
    #threadreps = zeros(Threads.nthreads())


    for ep in 1:n_episodes

        for bandit in 1:bandit_count
	        global_bandit_param[bandit, :] = rand(MvNormal((bandit_prior_means[bandit,:]), (bandit_prior_covs[bandit,:,:])))
        end

        EPREWARDS, POST_MEANS, POST_COVS = ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim,        context_mean,context_sd, obs_sd, bandit_count, bandit_prior_means, bandit_prior_covs, discount, epsilon, global_bandit_param)
        ep_count += 1

        TOT_REWARDS[ep, :] = EPREWARDS
        TOT_POST_MEANS[ep, :, :, :] = POST_MEANS
        TOT_POST_COVS[ep, :, :, :, :] = POST_COVS

    end
    #print(threadreps)
    return TOT_REWARDS, TOT_POST_MEANS, TOT_POST_COVS
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

## SIMULATIONS

# Sims

print("\n")

@time BIG_GRID_REWARDS, BIG_GRID_POST_MEANS, BIG_GRID_POST_COVS, GRID_REWARDS, GRID_POST_MEANS, GRID_POST_COVS = grid_contextual_bandit_simulator(n_points, greedy_policy, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

#print(GRID_REWARDS)


cov_vec_len = convert(Int64, (context_dim + 1) * context_dim / 2)

output = zeros(T, n_points*n_episodes, 1 + (context_dim + cov_vec_len) * bandit_count)

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

t = idx
cum_rewards = [sum(GRID_REWARDS[j, (T-t+1):T]) for j in 1:(n_points*1)]
output[t, :, 1] = cum_rewards
for bandit in 1:bandit_count

    output[t, :, (2+context_dim*(bandit-1)):(1+context_dim*bandit)] =
        GRID_POST_MEANS[:, (T-t+1), bandit, :]

    for j in 1:(n_points*1)

        #output[t, j, (2+context_dim*bandit_count+context_dim*(bandit-1)):(1+context_dim*bandit_count+context_dim*bandit)] = eigvals(GRID_POST_COVS[j, (T-t+1), bandit, :, :])
        output[t, j, (2+context_dim*bandit_count+cov_vec_len*(bandit-1)):(1+context_dim*bandit_count+cov_vec_len*bandit)] = upper_triangular_vec(GRID_POST_COVS[j, (T-t+1), bandit, :, :])

    end
end

CSV.write("/hpc/home/jml165/rl/valresults/results_$(idx).csv", Tables.table(output[t, :, :]))

