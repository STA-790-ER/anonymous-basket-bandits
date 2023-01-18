using SpecialFunctions
using Distributions
using LinearAlgebra








function gp_kernel_single(diff, kernel_scale, kernel_bandwidth)
    return kernel_scale * exp(-dot(diff, diff) / kernel_bandwidth)
end

function gp_kernel(X, kernel_scale, kernel_bandwidth) # assumed non-empty
    n = size(X)[1]
    out = zeros(n, n)
    for i in 1:n
        for j in 1:i
            k_ij = gp_kernel_single(X[i, :] .- X[j, :], kernel_scale, kernel_bandwidth)
            out[i, j] = k_ij
            out[j, i] = k_ij
        end
    end
    return out
end

function gp_new_kernel(x_new, X, kernel_scale, kernel_bandwidth) # assumed non-empty
    n = size(X)[1]
    return [gp_kernel_single(x_new .- X[i, :], kernel_scale, kernel_bandwidth) for i in 1:n]
end

function gp_posterior(x_new, X, y, kernel_scale, kernel_bandwidth, obs_sd)
    n = size(X)[1]
    if n == 0
        return 0, obs_sd^2
    end
    K_plus_noise = gp_kernel(X, kernel_scale, kernel_bandwidth) + diagm(repeat([obs_sd^2], n))
    K_star = gp_new_kernel(x_new, X, kernel_scale, kernel_bandwidth)

    return dot(K_star, K_plus_noise \ y), kernel_scale - dot(K_star, K_plus_noise \ K_star)
    
end

function gp_posterior_given_inverse(x_new, X, y, kernel_scale, kernel_bandwidth, obs_sd, training_cov_inverse)
    
    n = size(X)[1]
    if n == 0
        return 0, obs_sd^2
    end
    #K_plus_noise = gp_kernel(X, kernel_scale, kernel_bandwidth) + diagm(repeat([obs_sd^2], n))
    K_star = gp_new_kernel(x_new, X, kernel_scale, kernel_bandwidth)
    #print(K_star)
    #print(training_cov_inverse)
    #print(y)
    return dot(K_star, training_cov_inverse, y), kernel_scale - dot(K_star, training_cov_inverse, K_star)
     
end


function gp_get_context(gp_lower, gp_upper, gp_context_dim)
    return gp_lower .+ (gp_upper - gp_lower) .* rand(gp_context_dim)
end

function gp_ep_contextual_bandit_simulator(ep, action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param, kernel_scale, kernel_bandwidth)
        
    
        bandit_posterior_means = zeros(bandit_count, context_dim)
        bandit_posterior_covs = zeros(bandit_count, context_dim, context_dim)
    	#bandit_param = copy(global_bandit_param)
        #true_bandit_param = copy(global_bandit_param)
        EPREWARDS = zeros(T)
	    EPOPTREWARDS = zeros(T)
        
        X_tot = zeros(T, context_dim)
        Y_tot = zeros(T)
        A_tot = zeros(T)
        training_covariance_inverses = [zeros(0,0) for k in 1:bandit_count]

        expected_rewards = zeros(bandit_count)
        variance_rewards = zeros(bandit_count)
        for i in 1:bandit_count
            bandit_posterior_means[i, :] = repeat([bandit_prior_mean], context_dim)
            bandit_posterior_covs[i, :, :] = Diagonal(repeat([bandit_prior_sd^2], context_dim))
        end
        
        for t in 1:T
            context = randn(context_dim) * context_sd .+ context_mean
            #true_expected_rewards_logit = true_bandit_param * context
            #true_expected_rewards = exp.(true_expected_rewards_logit) ./ (1 .+ exp.(true_expected_rewards_logit))
            #true_expected_rewards = bandit_posterior_means * context
            #
            #
            #
            for k in 1:bandit_count
                expected_rewards[k], variance_rewards[k] = gp_posterior_given_inverse(context, X_tot[A_tot .== k, :], Y_tot[A_tot .== k], kernel_scale, kernel_bandwidth, obs_sd, training_covariance_inverses[k])
            end

            action = action_function(ep, t, T, bandit_count, context, X_tot, Y_tot, A_tot, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, kernel_scale, kernel_bandwidth, obs_sd, expected_rewards, variance_rewards, training_covariance_inverses)
            
            if t == 1
                true_expected_rewards = sqrt(kernel_scale) .* randn(bandit_count)
            else 
                true_moments = [gp_posterior(context, X_tot[1:(t-1), :], global_bandit_param[k, 1:(t-1)], kernel_scale, kernel_bandwidth, 0) for k in 1:bandit_count]   
                true_expected_rewards = [item[1] + sqrt(item[2]) * randn() for item in true_moments]
            end
            EPREWARDS[t] = true_expected_rewards[action]
            EPOPTREWARDS[t] = maximum(true_expected_rewards)
            #obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:],context)) + true_expected_reward
            obs = true_expected_rewards[action] + obs_sd * randn()
            
            # the order of the below matters for training covariance inverses computation!!!!
            training_covariance_inverses[action] = gp_block_inverse(training_covariance_inverses[action], gp_new_kernel(context, X_tot[A_tot .== action, :], kernel_scale, kernel_bandwidth), obs_sd^2 + kernel_scale)   
            Y_tot[t] = obs
            A_tot[t] = action
            X_tot[t, :] = context
            global_bandit_param[:, t] = true_expected_rewards


	        println("Ep: ", ep, " - ", t, " of ", T, " for ", String(Symbol(action_function)))
            flush(stdout)

        end
	return EPREWARDS, EPOPTREWARDS
end

function gp_contextual_bandit_simulator(action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

    REWARDS = zeros(T, n_episodes)
    OPTREWARDS = zeros(T, n_episodes)
    ep_count = 1
    

    ## USING BAYESIAN FORMULATION FOR FAIR COMPARISON
    #global_bandit_param = [1 0; 0 1; 2 -1]
    
    
    #threadreps = zeros(Threads.nthreads())


    for ep in 1:n_episodes
        global_bandit_param = zeros(bandit_count, T)
        EPREWARDS, EPOPTREWARDS = gp_ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                       context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param, kernel_scale, kernel_bandwidth)
        ep_count += 1
	REWARDS[:, ep] = EPREWARDS
	OPTREWARDS[:, ep] = EPOPTREWARDS
    end
    #print(threadreps)
    return REWARDS', OPTREWARDS'
end

function gp_block_inverse(A_inv, B, d)

    if size(A_inv)[1] == 0
        return diagm([1/d])
    end
    A_inv_B = A_inv * B
    LR = 1 / (d - dot(B, A_inv_B))
    UL = A_inv + A_inv_B * A_inv_B' .* LR
    UR = -A_inv_B .* LR
    n = size(A_inv)[1] + 1
    out = zeros(n, n)
    out[1:(n-1), 1:(n-1)] = UL
    out[1:(n-1), n] = UR
    out[n, 1:(n-1)] = UR
    out[n, n] = LR
    return out
end

#########################################################################
# POLICIES
# #######################################################################



function gp_val_greedy_thompson_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, kernel_scale, kernel_bandwidth, obs_sd, expected_rewards, variance_rewards, training_covariance_inverses)

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
function gp_val_greedy_thompson_ucb_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, kernel_scale, kernel_bandwidth, obs_sd, expected_rewards, variance_rewards, training_covariance_inverses)

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
function gp_val_greedy_thompson_ucb_ids_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, kernel_scale, kernel_bandwidth, obs_sd, expected_rewards, variance_rewards, training_covariance_inverses)


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

    
    policies = [gp_greedy_policy, gp_thompson_policy, gp_glm_ucb_policy, gp_ids_policy]
    policy_values = []
    action_samps = [pol(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, kernel_scale, kernel_bandwidth, obs_sd, expected_rewards, variance_rewards, training_covariance_inverses) for pol in policies]
    if length(unique(action_samps)) == 1
        println("Agreement Action: ", action_samps[1])
        flush(stdout)
        return action_samps[1]
    end
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

    # unnecessary given changes checking first for agreement
    opt_act = opt_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, kernel_scale, kernel_bandwidth, obs_sd, expected_rewards, variance_rewards, training_covariance_inverses)

    println("Optimal Action: ",opt_act)
    flush(stdout)

    return action_samps[opt_index]

end

function gp_val_rollout(ep, policy, T_remainder, rollout_length, context, use_context, lambda, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, discount, X, y, A, bandit_param,
	true_expected_rewards, kernel_scale, kernel_bandwidth, training_covariance_inverses)

    disc_reward = 0
    #fill!(context, 0.0)
    fill!(true_expected_rewards, 0.0)
    #fill!(CovCon, 0.0)
    #fill!(old_cov,0.0)
    #fill!(SigInvMu,0.0)
    expected_rewards = zeros(bandit_count)
    variance_rewards = zeros(bandit_count)
    truncation_length = T_remainder - min(T_remainder, rollout_length)
    
    t_curr = T - T_remainder + 1
    rollout_training_covariance_inverses = copy(training_covariance_inverses)
    for t in 1:min(T_remainder, rollout_length)

        #context_seed = rand(1:context_dim)
        #fill!(context, zero(context[1]))
        #context[context_seed] = 1
        #mul!(true_expected_rewards, bandit_param, context)

        if t > 1 || !use_context
            context = randn(context_dim) * context_sd .+ context_mean
        end
        
        for k in 1:bandit_count 
            expected_rewards[k], variance_rewards[k] = gp_posterior_given_inverse(context, X[A .== k, :], y[A .== k], kernel_scale, kernel_bandwidth, obs_sd, rollout_training_covariance_inverses[k])
        end
        action = policy(ep, t_curr + t - 1, min(T_remainder, rollout_length), bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, kernel_scale, kernel_bandwidth, obs_sd, expected_rewards, variance_rewards, rollout_training_covariance_inverses)


        # TRUE PARAM VERSION
        #true_expected_reward = true_expected_rewards[action]
        #disc_reward += true_expected_reward * discount^(t-1)
        #obs = randn() * obs_sd + true_expected_reward
        
        disc_reward += expected_rewards[action] * discount^(t-1)

        obs = expected_rewards[action] + obs_sd * randn()
        
        # order of below matters for covariance inverses !!!
        rollout_training_covariance_inverses[action] = gp_block_inverse(rollout_training_covariance_inverses[action], gp_new_kernel(context, X[A .== action, :], kernel_scale, kernel_bandwidth), obs_sd^2 + kernel_scale)   
        
        y[t_curr + t - 1] = obs
        A[t_curr + t - 1] = action
        X[(t_curr + t - 1), :] = context
        
        # UNKNOWN PARAM VERSION
    end


    if truncation_length > 0
        t_trunc = t_curr + min(T_remainder, rollout_length)
        reg_est = 0
        for n in 1:2
            context = randn(context_dim) .* context_sd .+ context_mean
            context_reg_est = 0
            for k in 1:bandit_count
                expected_rewards[k], variance_rewards[k] = gp_posterior_given_inverse(context, X[A .== k, :], y[A .== k], kernel_scale, kernel_bandwidth, obs_sd, rollout_training_covariance_inverses[k])
            end
            point_samps =  expected_rewards .+ sqrt.(variance_rewards) .* randn(bandit_count, 500)
            if String(Symbol(policy)) == "gp_thompson_policy"
                for m in 1:500
                    action = policy(ep, t_trunc, min(T_remainder, rollout_length), bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, kernel_scale, kernel_bandwidth, obs_sd, expected_rewards, variance_rewards, rollout_training_covariance_inverses)
                    reg_est *= ((n-1)*500 + m - 1) / ((n-1)*500 + m)
                    reg_est += (findmax(point_samps[:, m])[1] - point_samps[action, m]) / ((n-1)*500 + m)
                end
            else
                action = policy(ep, t_trunc, min(T_remainder, rollout_length), bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, kernel_scale, kernel_bandwidth, obs_sd, expected_rewards, variance_rewards, rollout_training_covariance_inverses)
                for m in 1:500
                    reg_est *= ((n-1)*500 + m - 1) / ((n-1)*500 + m)
                    reg_est += (findmax(point_samps[:, m])[1] - point_samps[action, m]) / ((n-1)*500 + m)
                end
            end
            #reg_est *= (n-1) / n
            #reg_est += (findmax(point_samps)[1] - point_samps[action]) / n
        end

        disc_reward -= .9 * discount^min(T_remainder, rollout_length) * sum(discount^(t - t_trunc) * reg_est * (1 + T - t) / (1 + T - t_trunc) for t in t_trunc:T)

    end

    return disc_reward
    end

# Greedy
function gp_greedy_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, kernel_scale, kernel_bandwidth, obs_sd, expected_rewards, variance_rewards, training_covariance_inverses)
    
    #bandit_expected_rewards = zeros(bandit_count)
    #for k in 1:bandit_count
    #    m, C = gp_posterior(context, X[A .== k, :], y[A .== k], kernel_scale, kernel_bandwidth, obs_sd)
    #    bandit_expected_rewards[k] = m
    #end
    
    #val, action = findmax(bandit_expected_rewards)
    return findmax(expected_rewards)[2]
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
function gp_bayes_ucb_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    val, action = findmax(bandit_posterior_means * context)
    reward_means = bandit_posterior_means * context
    reward_sds = sqrt.(vec([context' * (@view bandit_posterior_covs[i,:,:]) * context for i=1:bandit_count]))
    ucbs = quantile.(Normal.(reward_means, reward_sds), 1-1/t)
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
function gp_thompson_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, kernel_scale, kernel_bandwidth, obs_sd, expected_rewards, variance_rewards, training_covariance_inverses)
    #thompson_means = zeros(bandit_count)
    #for k in 1:bandit_count
    #    m, C = gp_posterior(context, X[A .== k,:], y[A .== k], kernel_scale, kernel_bandwidth, obs_sd)
    #    thompson_means[k] = m + sqrt(C) * randn()
    #end


    return findmax(expected_rewards .+ sqrt.(variance_rewards) .* randn(bandit_count))[2]
end




## IDS POLICY
# IDS NOT DONE
function gp_ids_expected_regrets(expected_rewards, variance_rewards, niter)

    #draws = zeros(bandit_count, context_dim, niter)
    reward_draws = zeros(bandit_count, niter)
    
    for b in 1:bandit_count
        #draws[b, :, :] = rand(MvNormal(bandit_posterior_means[b, :], bandit_posterior_covs[b, :, :]), niter)
        #for i in 1:niter
        #    reward_draws[b, i] = dot(context, draws[b, :, i])
        #end
        reward_draws[b, :] = expected_rewards[b] .+ sqrt(variance_rewards[b]) .* randn(niter)
    end
    
    mean_rewards = dropdims(mean(reward_draws, dims = 2), dims = 2)

    mean_max_reward = 0
    for i in 1:niter
        mean_max_reward += findmax(reward_draws[:, i])[1] / niter
    end
    
    res = max.(0, mean_max_reward .- mean_rewards)

    return res
end
function gp_ids_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, kernel_scale, kernel_bandwidth, obs_sd, expected_rewards, variance_rewards, training_covariance_inverses)
    
    gp_ent_gains = zeros(bandit_count)
    new_cov_invs = [gp_block_inverse(training_covariance_inverses[k], gp_new_kernel(context, X[A .== k, :], kernel_scale, kernel_bandwidth), obs_sd^2 + kernel_scale) for k in 1:bandit_count]
    X[t, :] = context 
    #X[t, :] = context
    for r in 1:50
        random_context = randn(context_dim) .* context_sd .+ context_mean
        for k in 1:bandit_count
            A[t] = k
            #print(new_cov_inv)
            #print(X[A .== k, :])
            m, C = gp_posterior_given_inverse(random_context, X[A .== k,:], y[A .== k], kernel_scale, kernel_bandwidth, obs_sd, new_cov_invs[k]) 
            gp_ent_gains[k] *= (r - 1) / r
            gp_ent_gains[k] += log(variance_rewards[k] / C) / r        
        end
    end
    A[t] = 0
    gp_expected_regrets = gp_ids_expected_regrets(expected_rewards, variance_rewards, 1000)  
    return findmax(-1 .* gp_expected_regrets.^2 ./ gp_ent_gains)[2]
end



