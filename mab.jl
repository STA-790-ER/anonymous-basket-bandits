using SpecialFunctions
using Distributions
using LinearAlgebra















function ep_multi_armed_bandit_simulator(ep, action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, bandit_count, a_beta, b_beta, discount, epsilon, global_bandit_param)
        
    
    	#bandit_param = copy(global_bandit_param)
        #true_bandit_param = copy(global_bandit_param)
        EPREWARDS = zeros(T)
	    EPOPTREWARDS = zeros(T)
        

        
        for t in 1:T

            action = action_function(ep, t, T, bandit_count, a_beta, b_beta, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts)
            
            
            
            EPREWARDS[t] = global_bandit_param[action]
            EPOPTREWARDS[t] = maximum(global_bandit_param)
            #obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:],context)) + true_expected_reward
            obs = 1 * (global_bandit_param[action] > rand())
            
            # the order of the below matters for training covariance inverses computation!!!!
            a_beta[action] += obs
            b_beta[action] += (1-obs)

	        println("Ep: ", ep, " - ", t, " of ", T, " for ", String(Symbol(action_function)))
            flush(stdout)

        end
	return EPREWARDS, EPOPTREWARDS
end

function multi_armed_bandit_simulator(action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, bandit_count, a_beta_prior, b_beta_prior, discount, epsilon)

    REWARDS = zeros(T, n_episodes)
    OPTREWARDS = zeros(T, n_episodes)
    ep_count = 1
    

    ## USING BAYESIAN FORMULATION FOR FAIR COMPARISON
    #global_bandit_param = [1 0; 0 1; 2 -1]
    
    
    #threadreps = zeros(Threads.nthreads())

    for ep in 1:n_episodes
        a_beta = repeat([a_beta_prior], bandit_count)
        b_beta = repeat([b_beta_prior], bandit_count)
        global_bandit_param = rand.(Beta.(a_beta, b_beta))
        EPREWARDS, EPOPTREWARDS = ep_multi_armed_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, bandit_count, a_beta, b_beta, discount, epsilon, global_bandit_param)
        ep_count += 1
	REWARDS[:, ep] = EPREWARDS
	OPTREWARDS[:, ep] = EPOPTREWARDS
    end
    #print(threadreps)
    return REWARDS', OPTREWARDS'
end


#########################################################################
# POLICIES
# #######################################################################



function mab_val_greedy_thompson_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, kernel_scale, kernel_bandwidth, obs_sd, expected_rewards, variance_rewards, training_covariance_inverses)

    #BANDIT_VALUES = zeros(bandit_count)
    #predictive_rewards = bandit_posterior_means * context

## PREALLOCATION
    #roll_context = zeros(context_dim)
    roll_true_expected_rewards = zeros(bandit_count)
    #roll_CovCon = zeros(context_dim)
    #roll_old_cov = zeros(context_dim, context_dim)
    #roll_SigInvMu = zeros(context_dim)

    #temp_post_means = zeros(bandit_count, context_dim)
    #temp_post_covs = zeros(bandit_count, context_dim, context_dim)
    #temp_bandit_mean = zeros(context_dim)
    #temp_bandit_cov = zeros(context_dim, context_dim)

    bandit_param = zeros(bandit_count, context_dim)
    #true_expected_rewards = zeros(bandit_count)
    #grad_est = zeros(3)

    
    policies = [gp_greedy_policy, gp_thompson_policy]
    policy_values = []
    
    println("Context Start: ", context)
    flush(stdout)

    lambda = [0, 0]

    for policy in policies  

            MEAN_REWARD = 0

            for roll in 1:n_opt_rollouts
            
                #copy!(temp_post_means, bandit_posterior_means)
                #copy!(temp_post_covs, bandit_posterior_covs)
                #bandit_param = zeros(bandit_count, context_dim)
                #for bandit in 1:bandit_count
                    #copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
                    #copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
                #    m, C = jj_posterior_batch(X[A .== bandit,:], y[A .== bandit], prior_mean, prior_cov, .001)
                #    bandit_param[bandit,:] = rand(MvNormal(m, C))
                #end

                use_context = true
                
                rollout_value = gp_val_rollout(ep, policy, T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
                    context_sd, obs_sd, bandit_count, discount, X, y, A, bandit_param,
                    roll_true_expected_rewards, kernel_scale, kernel_bandwidth, training_covariance_inverses)
                A[t:T] .= 0
                
                MEAN_REWARD = ((roll - 1) * MEAN_REWARD + rollout_value) / roll

            end
            
            push!(policy_values, MEAN_REWARD)
        
    end
    
    println("Context Finish: ", context)
    flush(stdout)

    opt_index = findmax(policy_values)[2]
    opt_policy = policies[opt_index]

    println("GREEDY: ", policy_values[1],", THOMPSON: ", policy_values[2])
    flush(stdout)

    # END OPTIMIZATION OF LAMBDA

    
    opt_act = opt_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, kernel_scale, kernel_bandwidth, obs_sd, expected_rewards, variance_rewards, training_covariance_inverses)

    println("Optimal Action: ",opt_act)
    flush(stdout)

    return opt_act

end
function mab_val_greedy_thompson_ucb_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, kernel_scale, kernel_bandwidth, obs_sd, expected_rewards, variance_rewards, training_covariance_inverses)

    #BANDIT_VALUES = zeros(bandit_count)
    #predictive_rewards = bandit_posterior_means * context

## PREALLOCATION
    #roll_context = zeros(context_dim)
    roll_true_expected_rewards = zeros(bandit_count)
    #roll_CovCon = zeros(context_dim)
    #roll_old_cov = zeros(context_dim, context_dim)
    #roll_SigInvMu = zeros(context_dim)

    #temp_post_means = zeros(bandit_count, context_dim)
    #temp_post_covs = zeros(bandit_count, context_dim, context_dim)
    #temp_bandit_mean = zeros(context_dim)
    #temp_bandit_cov = zeros(context_dim, context_dim)

    bandit_param = zeros(bandit_count, context_dim)
    #true_expected_rewards = zeros(bandit_count)
    #grad_est = zeros(3)

    
    policies = [gp_greedy_policy, gp_thompson_policy, gp_glm_ucb_policy]
    policy_values = []
    
    println("Context Start: ", context)
    flush(stdout)

    lambda = [0, 0]

    for policy in policies  

            MEAN_REWARD = 0

            for roll in 1:n_opt_rollouts
            
                #copy!(temp_post_means, bandit_posterior_means)
                #copy!(temp_post_covs, bandit_posterior_covs)
                #bandit_param = zeros(bandit_count, context_dim)
                #for bandit in 1:bandit_count
                    #copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
                    #copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
                #    m, C = jj_posterior_batch(X[A .== bandit,:], y[A .== bandit], prior_mean, prior_cov, .001)
                #    bandit_param[bandit,:] = rand(MvNormal(m, C))
                #end

                use_context = true
                
                rollout_value = gp_val_rollout(ep, policy, T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
                    context_sd, obs_sd, bandit_count, discount, X, y, A, bandit_param,
                    roll_true_expected_rewards, kernel_scale, kernel_bandwidth, training_covariance_inverses)
                A[t:T] .= 0
                
                MEAN_REWARD = ((roll - 1) * MEAN_REWARD + rollout_value) / roll

            end
            
            push!(policy_values, MEAN_REWARD)
        
    end
    
    println("Context Finish: ", context)
    flush(stdout)

    opt_index = findmax(policy_values)[2]
    opt_policy = policies[opt_index]

    println("GREEDY: ", policy_values[1],", THOMPSON: ", policy_values[2])
    flush(stdout)

    # END OPTIMIZATION OF LAMBDA

    
    opt_act = opt_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, kernel_scale, kernel_bandwidth, obs_sd, expected_rewards, variance_rewards, training_covariance_inverses)

    println("Optimal Action: ",opt_act)
    flush(stdout)

    return opt_act

end
function mab_val_greedy_thompson_ucb_ids_policy(ep, t, T, bandit_count, a_beta, b_beta, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts)

    if t == 1
        return rand(1:bandit_count)
    end
    #BANDIT_VALUES = zeros(bandit_count)
    #predictive_rewards = bandit_posterior_means * context

## PREALLOCATION
    #roll_context = zeros(context_dim)
    roll_true_expected_rewards = zeros(bandit_count)
    #roll_CovCon = zeros(context_dim)
    #roll_old_cov = zeros(context_dim, context_dim)
    #roll_SigInvMu = zeros(context_dim)

    #temp_post_means = zeros(bandit_count, context_dim)
    #temp_post_covs = zeros(bandit_count, context_dim, context_dim)
    #temp_bandit_mean = zeros(context_dim)
    #temp_bandit_cov = zeros(context_dim, context_dim)

    bandit_param = zeros(bandit_count, context_dim)
    #true_expected_rewards = zeros(bandit_count)
    #grad_est = zeros(3)

    
    policies = [mab_greedy_policy, mab_thompson_policy, mab_bayes_ucb_policy, mab_ids_policy]
    policy_values = []
    
    action_samps = [pol(ep, t, T, bandit_count, a_beta, b_beta, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts) for pol in policies]
    if length(unique(action_samps)) == 1
        println("Agreement Action: ", action_samps[1])
        flush(stdout)
        return action_samps[1]
    end

    lambda = [0, 0]
    bandit_param_samps = zeros(bandit_count, n_opt_rollouts)
    for k in 1:bandit_count
        bandit_param_samps[k, :] = rand(Beta(a_beta[k], b_beta[k]), n_opt_rollouts)
    end

    for policy in policies  
            MEAN_REWARD = 0
            for roll in 1:n_opt_rollouts
                bandit_param = bandit_param_samps[:, roll] 
                rollout_value = mab_val_rollout(ep, policy, T-t+1, rollout_length, bandit_count, discount, a_beta, b_beta, bandit_param)
                MEAN_REWARD = ((roll - 1) * MEAN_REWARD + rollout_value) / roll
            end
            push!(policy_values, MEAN_REWARD)
    end
    

    opt_index = findmax(policy_values)[2]
    opt_policy = policies[opt_index]

    println("GREEDY: ", policy_values[1],", THOMPSON: ", policy_values[2])
    flush(stdout)

    # END OPTIMIZATION OF LAMBDA

    
    opt_act = opt_policy(ep, t, T, bandit_count, a_beta, b_beta, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts)

    println("Optimal Action: ",opt_act)
    flush(stdout)

    return opt_act

end
function mab_val_greedy_thompson_ucb_ids_q_policy(ep, t, T, bandit_count, a_beta, b_beta, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts)

    if t == 1
        return rand(1:bandit_count)
    end
    roll_true_expected_rewards = zeros(bandit_count)

    bandit_param = zeros(bandit_count, context_dim)

    
    policies = [mab_greedy_policy, mab_thompson_policy, mab_bayes_ucb_policy, mab_ids_policy]
    policy_values = []
    
    action_samps = [pol(ep, t, T, bandit_count, a_beta, b_beta, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts) for pol in policies]
    #if length(unique(action_samps)) == 1
    #    println("Agreement Action: ", action_samps[1])
    #    flush(stdout)
    #    return action_samps[1]
    #end

    lambda = [0, 0]
    bandit_param_samps = zeros(bandit_count, n_opt_rollouts)
    for k in 1:bandit_count
        bandit_param_samps[k, :] = rand(Beta(a_beta[k], b_beta[k]), n_opt_rollouts)
    end
    policy_count = length(policies)
    a_beta_rollout = copy(a_beta)
    b_beta_rollout = copy(b_beta)

    
    REWARD_SAMPS = zeros(n_opt_rollouts, length(policies))
    halt_vec = repeat([false], length(policies))
    rollout_counts = repeat([0], length(policies))
    for roll in 1:n_opt_rollouts  

            MEAN_REWARD = 0

            for pol_ind in 1:policy_count
                
                if halt_vec[pol_ind]
                    continue
                end

                rollout_counts[pol_ind] += 1
                bandit_param = bandit_param_samps[:, roll]    
                
                rollout_value = mab_val_rollout(ep, policies[pol_ind], T-t+1, rollout_length, bandit_count, discount, a_beta_rollout, b_beta_rollout, bandit_param)
                
                
                REWARD_SAMPS[roll, pol_ind] = rollout_value
                
                for k in 1:bandit_count
                    a_beta_rollout[k] = a_beta[k]
                    b_beta_rollout[k] = b_beta[k]
                end
            end
        
            if roll % 100 == 0
                continue_inds = findall(halt_vec .== false)
                policy_means = [mean(REWARD_SAMPS[1:roll, p]) for p in continue_inds]
                policy_stds = [std(REWARD_SAMPS[1:roll, p] ./ sqrt(roll)) for p in continue_inds]
                max_mean, max_ind = findmax(policy_means)
                diff_means = max_mean .- policy_means
                diff_stds = sqrt.(policy_stds[max_ind]^2 .+ policy_stds.^2)
                pol_expected_regret_proportions = (diff_means .* cdf.(Normal(), -diff_means ./ diff_stds) .- diff_stds .* pdf.(Normal(), -diff_means ./ diff_stds)) ./ max_mean
                halt_vec[continue_inds] = (abs.(pol_expected_regret_proportions) .< expected_regret_thresh)
                halt_vec[continue_inds[max_ind]] = false
            end

            if sum(halt_vec .== false) == 1
                break
            end
    end
    
    continue_inds = findall(halt_vec .== false)
    
    policy_means = [mean(REWARD_SAMPS[1:rollout_counts[p], p]) for p in 1:policy_count]
    policy_stds = [std(REWARD_SAMPS[1:rollout_counts[p], p]) / sqrt(rollout_counts[p]) for p in 1:policy_count]

    opt_index = continue_inds[findmax(policy_means[continue_inds])[2]]
    opt_mean = policy_means[opt_index]
    opt_std = policy_stds[opt_index]
    println("POLICY VALUES: ", policy_means)
    println("POLICY STDS: ", policy_stds)
    println("OPT VALUE: ", opt_mean)
    println("OPT POLICY: ", String(Symbol(policies[opt_index])))
    println("POLICY ITERS: ", rollout_counts)
    flush(stdout)

    opt_policy = policies[opt_index]
    ACTION_MEAN_REWARDS = zeros(bandit_count)
    ACTION_STD_REWARDS = zeros(bandit_count)
    REWARD_SAMPS = zeros(n_opt_rollouts, bandit_count)
    halt_vec = repeat([false], bandit_count)
    rollout_counts = repeat([0], bandit_count)

    a_beta_rollout = copy(a_beta)
    b_beta_rollout = copy(b_beta)
    for roll in 1:n_opt_rollouts
        for action in 1:bandit_count 
            
            if halt_vec[action]
                continue
            end

            rollout_counts[action] += 1

            bandit_param = bandit_param_samps[:, roll]    
            
            obs = 1 * (bandit_param[action] > rand())
            
            a_beta_rollout[action] += obs
            b_beta_rollout[action] += (1-obs)

            
            rollout_value = mab_val_rollout(ep, opt_policy, T-t, rollout_length-1, bandit_count, discount, a_beta_rollout, b_beta_rollout, bandit_param)
            
            REWARD_SAMPS[roll, action] = bandit_param[action] + discount * rollout_value 

            for k in 1:bandit_count
                a_beta_rollout[k] = a_beta[k]
                b_beta_rollout[k] = b_beta[k]
            end
            
            if roll % 100 == 0
                continue_inds = findall(halt_vec .== false)
                action_means = [mean(REWARD_SAMPS[1:roll, p]) for p in continue_inds]
                action_stds = [std(REWARD_SAMPS[1:roll, p]) / sqrt(roll) for p in continue_inds]
                max_mean, max_ind = findmax(action_means)
                diff_means = max_mean .- action_means
                diff_stds = sqrt.(action_stds[max_ind]^2 .+ action_stds.^2)
                action_expected_regret_proportions = (diff_means .* cdf.(Normal(), -diff_means ./ diff_stds) .- diff_stds .* pdf.(Normal(), -diff_means ./ diff_stds)) ./ max_mean
                halt_vec[continue_inds] = (abs.(action_expected_regret_proportions) .< action_expected_regret_thresh)
                halt_vec[continue_inds[max_ind]] = false
                if (sum(halt_vec .== false) == 1)
                    break
                end
            end
        end
    end

    continue_inds = findall(halt_vec .== false)
    
    action_means = [mean(REWARD_SAMPS[1:rollout_counts[p], p]) for p in 1:bandit_count]
    action_stds = [std(REWARD_SAMPS[1:rollout_counts[p], p]) / sqrt(rollout_counts[p]) for p in 1:bandit_count]

    opt_action_index = continue_inds[findmax(action_means[continue_inds])[2]]
    opt_action_mean = action_means[opt_action_index]
    opt_action_std = action_stds[opt_action_index]
    println("ACTION VALUES: ", action_means)
    println("ACTION STDS: ", action_stds)
    println("OPT VALUE: ", opt_action_mean)
    println("OPT ACTION: ", opt_action_index)
    println("ACTION ITERS: ", rollout_counts)
    flush(stdout)

    return opt_action_index

end
function mab_val_greedy_thompson_ucb_ids_policy(ep, t, T, bandit_count, a_beta, b_beta, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts)

    if t == 1
        return rand(1:bandit_count)
    end
    roll_true_expected_rewards = zeros(bandit_count)

    bandit_param = zeros(bandit_count, context_dim)

    
    policies = [mab_greedy_policy, mab_thompson_policy, mab_bayes_ucb_policy, mab_ids_policy]
    policy_values = []
    
    action_samps = [pol(ep, t, T, bandit_count, a_beta, b_beta, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts) for pol in policies]
    if length(unique(action_samps)) == 1
        println("Agreement Action: ", action_samps[1])
        flush(stdout)
        return action_samps[1]
    end

    lambda = [0, 0]
    bandit_param_samps = zeros(bandit_count, n_opt_rollouts)
    for k in 1:bandit_count
        bandit_param_samps[k, :] = rand(Beta(a_beta[k], b_beta[k]), n_opt_rollouts)
    end
    policy_count = length(policies)
    a_beta_rollout = copy(a_beta)
    b_beta_rollout = copy(b_beta)

    
    REWARD_SAMPS = zeros(n_opt_rollouts, length(policies))
    halt_vec = repeat([false], length(policies))
    rollout_counts = repeat([0], length(policies))
    same_break = false
    for roll in 1:n_opt_rollouts  

            MEAN_REWARD = 0

            for pol_ind in 1:policy_count
                
                if halt_vec[pol_ind]
                    continue
                end

                rollout_counts[pol_ind] += 1
                bandit_param = bandit_param_samps[:, roll]    
                
                rollout_value = mab_val_rollout(ep, policies[pol_ind], T-t+1, rollout_length, bandit_count, discount, a_beta_rollout, b_beta_rollout, bandit_param)
                
                
                REWARD_SAMPS[roll, pol_ind] = rollout_value
                
                for k in 1:bandit_count
                    a_beta_rollout[k] = a_beta[k]
                    b_beta_rollout[k] = b_beta[k]
                end
            end
        
            if roll % 100 == 0
                continue_inds = findall(halt_vec .== false)
                policy_means = [mean(REWARD_SAMPS[1:roll, p]) for p in continue_inds]
                policy_stds = [std(REWARD_SAMPS[1:roll, p] ./ sqrt(roll)) for p in continue_inds]
                max_mean, max_ind = findmax(policy_means)
                diff_means = max_mean .- policy_means
                diff_stds = sqrt.(policy_stds[max_ind]^2 .+ policy_stds.^2)
                pol_expected_regret_proportions = (diff_means .* cdf.(Normal(), -diff_means ./ diff_stds) .- diff_stds .* pdf.(Normal(), -diff_means ./ diff_stds)) ./ max_mean
                halt_vec[continue_inds] = (abs.(pol_expected_regret_proportions) .< expected_regret_thresh)
                halt_vec[continue_inds[max_ind]] = false
                continue_inds = findall(halt_vec .== false)
                if length(unique(action_samps[continue_inds])) == 1
                    println("REMAINING POLICIES HAVE SAME ACTION")
                    same_break = true
                end
            end
            
            if same_break
                break
            end

            if sum(halt_vec .== false) == 1
                break
            end
    end
    
    continue_inds = findall(halt_vec .== false)
    
    policy_means = [mean(REWARD_SAMPS[1:rollout_counts[p], p]) for p in 1:policy_count]
    policy_stds = [std(REWARD_SAMPS[1:rollout_counts[p], p]) / sqrt(rollout_counts[p]) for p in 1:policy_count]

    opt_index = continue_inds[findmax(policy_means[continue_inds])[2]]
    opt_mean = policy_means[opt_index]
    opt_std = policy_stds[opt_index]
    println("POLICY VALUES: ", policy_means)
    println("POLICY STDS: ", policy_stds)
    println("OPT VALUE: ", opt_mean)
    println("OPT POLICY: ", String(Symbol(policies[opt_index])))
    println("POLICY ITERS: ", rollout_counts)
    flush(stdout)

    opt_policy = policies[opt_index]
    return opt_policy(ep, t, T, bandit_count, a_beta, b_beta, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts) 

    ACTION_MEAN_REWARDS = zeros(bandit_count)
    ACTION_STD_REWARDS = zeros(bandit_count)
    REWARD_SAMPS = zeros(n_opt_rollouts, bandit_count)
    halt_vec = repeat([false], bandit_count)
    rollout_counts = repeat([0], bandit_count)

    a_beta_rollout = copy(a_beta)
    b_beta_rollout = copy(b_beta)
    for roll in 1:n_opt_rollouts
        for action in 1:bandit_count 
            
            if halt_vec[action]
                continue
            end

            rollout_counts[action] += 1

            bandit_param = bandit_param_samps[:, roll]    
            
            obs = 1 * (bandit_param[action] > rand())
            
            a_beta_rollout[action] += obs
            b_beta_rollout[action] += (1-obs)

            
            rollout_value = mab_val_rollout(ep, opt_policy, T-t, rollout_length-1, bandit_count, discount, a_beta_rollout, b_beta_rollout, bandit_param)
            
            REWARD_SAMPS[roll, action] = bandit_param[action] + discount * rollout_value 

            for k in 1:bandit_count
                a_beta_rollout[k] = a_beta[k]
                b_beta_rollout[k] = b_beta[k]
            end
            
            if roll % 100 == 0
                continue_inds = findall(halt_vec .== false)
                action_means = [mean(REWARD_SAMPS[1:roll, p]) for p in continue_inds]
                action_stds = [std(REWARD_SAMPS[1:roll, p]) / sqrt(roll) for p in continue_inds]
                max_mean, max_ind = findmax(action_means)
                diff_means = max_mean .- action_means
                diff_stds = sqrt.(action_stds[max_ind]^2 .+ action_stds.^2)
                action_expected_regret_proportions = (diff_means .* cdf.(Normal(), -diff_means ./ diff_stds) .- diff_stds .* pdf.(Normal(), -diff_means ./ diff_stds)) ./ max_mean
                halt_vec[continue_inds] = (abs.(action_expected_regret_proportions) .< action_expected_regret_thresh)
                halt_vec[continue_inds[max_ind]] = false
                if (sum(halt_vec .== false) == 1)
                    break
                end
            end
        end
    end

    continue_inds = findall(halt_vec .== false)
    
    action_means = [mean(REWARD_SAMPS[1:rollout_counts[p], p]) for p in 1:bandit_count]
    action_stds = [std(REWARD_SAMPS[1:rollout_counts[p], p]) / sqrt(rollout_counts[p]) for p in 1:bandit_count]

    opt_action_index = continue_inds[findmax(action_means[continue_inds])[2]]
    opt_action_mean = action_means[opt_action_index]
    opt_action_std = action_stds[opt_action_index]
    println("ACTION VALUES: ", action_means)
    println("ACTION STDS: ", action_stds)
    println("OPT VALUE: ", opt_action_mean)
    println("OPT ACTION: ", opt_action_index)
    println("ACTION ITERS: ", rollout_counts)
    flush(stdout)

    return opt_action_index

end

function mab_val_rollout(ep, policy, T_remainder, rollout_length, bandit_count, discount, a_beta_rollout, b_beta_rollout, bandit_param)

    disc_reward = 0
    #fill!(context, 0.0)
    #fill!(CovCon, 0.0)
    #fill!(old_cov,0.0)
    #fill!(SigInvMu,0.0)
    truncation_length = T_remainder - min(T_remainder, rollout_length)
    t_curr = T - T_remainder + 1
    #rollout_training_covariance_inverses = copy(training_covariance_inverses)
    for t in 1:min(T_remainder, rollout_length)

        #context_seed = rand(1:context_dim)
        #fill!(context, zero(context[1]))
        #context[context_seed] = 1
        #mul!(true_expected_rewards, bandit_param, context)

        action = policy(ep, t_curr + t - 1, min(T_remainder, rollout_length), bandit_count, a_beta_rollout, b_beta_rollout, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts)


        # TRUE PARAM VERSION
        #true_expected_reward = true_expected_rewards[action]
        #disc_reward += true_expected_reward * discount^(t-1)
        #obs = randn() * obs_sd + true_expected_reward
        
        disc_reward += bandit_param[action] * discount^(t-1)

        obs = 1 * (bandit_param[action] > rand())
        
        # order of below matters for covariance inverses !!!
        a_beta_rollout[action] += obs
        b_beta_rollout[action] += (1-obs)
        
        # UNKNOWN PARAM VERSION
    end


    if truncation_length > 0
        t_trunc = t_curr + min(T_remainder, rollout_length)
        reg_est = 0
        for n in 1:1
            true_regs = findmax(bandit_param)[1] .- bandit_param
            if String(Symbol(policy)) == "mab_thompson_policy"
                for m in 1:500
                    action = policy(ep, t_trunc, min(T_remainder, rollout_length), bandit_count, a_beta_rollout, b_beta_rollout, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts)
                    reg_est *= ((n-1)*500 + m - 1) / ((n-1)*500 + m)
                    reg_est += true_regs[action] / ((n-1)*500 + m)
                end
            else
                action = policy(ep, t_trunc, min(T_remainder, rollout_length), bandit_count, a_beta_rollout, b_beta_rollout, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts)
                reg_est = true_regs[action]
            end
            #reg_est *= (n-1) / n
            #reg_est += (findmax(point_samps)[1] - point_samps[action]) / n
        end

        disc_reward -= .9 * discount^min(T_remainder, rollout_length) * sum(discount^(t - t_trunc) * reg_est * (T - t) / (T - t_trunc) for t in t_trunc:T)

    end

    return disc_reward
    end

# Greedy
function mab_greedy_policy(ep, t, T, bandit_count, a_beta, b_beta, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts)
    
    #bandit_expected_rewards = zeros(bandit_count)
    #for k in 1:bandit_count
    #    m, C = gp_posterior(context, X[A .== k, :], y[A .== k], kernel_scale, kernel_bandwidth, obs_sd)
    #    bandit_expected_rewards[k] = m
    #end
    
    #val, action = findmax(bandit_expected_rewards)
    return findmax(a_beta ./ (a_beta .+ b_beta))[2]
end

# Epsilon Greedy NOT DONE
function gp_epsilon_greedy_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    epsilon_draw = rand()
    
    if decreasing
        thresh = epsilon / t
    else
        thresh = epsilon
    end

    if epsilon_draw < thresh
        return rand(1:bandit_count)
    else
        val, action = findmax(bandit_posterior_means * context)
        return(action)
    end

end

# Bayes UCB  NOT DONE
function mab_bayes_ucb_policy(ep, t, T, bandit_count, a_beta, b_beta, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts)
    ucbs = quantile.(Beta.(a_beta, b_beta), 1-1/t)
    return findmax(ucbs)[2]
end

# Bayes UCB
function gp_glm_ucb_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, kernel_scale, kernel_bandwidth, obs_sd, expected_rewards, variance_rewards, training_covariance_inverses)
    #val, action = findmax(bandit_posterior_means * context)
    #reward_means = zeros(bandit_count)
    #reward_sds = zeros(bandit_count)
    #for k in 1:bandit_count
    #    m, C = gp_posterior(context, X[A .== k,:], y[A .== k], kernel_scale, kernel_bandwidth, obs_sd)
    #    reward_means[k] = m
    #    reward_sds[k] = sqrt(C)
    #end
    ucbs = expected_rewards .+ max(1,sqrt(log(t))) .* sqrt.(variance_rewards)
    return findmax(ucbs)[2]
end

# Thompson
function mab_thompson_policy(ep, t, T, bandit_count, a_beta, b_beta, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts)
    #thompson_means = zeros(bandit_count)
    #for k in 1:bandit_count
    #    m, C = gp_posterior(context, X[A .== k,:], y[A .== k], kernel_scale, kernel_bandwidth, obs_sd)
    #    thompson_means[k] = m + sqrt(C) * randn()
    #end


    return findmax(rand.(Beta.(a_beta, b_beta)))[2]
end




## IDS POLICY
# IDS NOT DONE
function mab_ids_expected_regrets(a_beta, b_beta, niter)

   
    reward_draws = mapreduce(permutedims, vcat, rand.(Beta.(a_beta, b_beta), niter))
    mean_rewards = dropdims(mean(reward_draws, dims = 2), dims = 2)

    mean_max_reward = 0
    for i in 1:niter
        mean_max_reward += findmax(reward_draws[:, i])[1] / niter
    end
    
    res = max.(0, mean_max_reward .- mean_rewards)

    return res
end

function beta_entropy(a, b)
    return logbeta(a, b) - (a - 1) * digamma(a) - (b - 1) * digamma(b) + (a + b - 2) * digamma(a + b)
end


function mab_ids_policy(ep, t, T, bandit_count, a_beta, b_beta, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts)
    
    ent_0 = beta_entropy.(a_beta, b_beta)
    ent_gains = ent_0 .- (a_beta .* beta_entropy.(a_beta .+ 1, b_beta) .+ b_beta .* beta_entropy.(a_beta, b_beta .+ 1)) ./ (a_beta .+ b_beta)
    
    
    mab_expected_regrets = mab_ids_expected_regrets(a_beta, b_beta, 1000)  
    return findmax(-1 .* mab_expected_regrets.^2 ./ ent_gains)[2]
end



