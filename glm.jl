using SpecialFunctions

# find beta prior parameters from normal parameters with f=mean and q=variance
function solve_beta_param(f, q, err)
    r_curr = min(10,(1 + exp(f)) / q)
    s_curr = min(10,(1 + exp(-f)) / q)
    #log_r_curr = log(r_curr)
    #log_s_curr = log(s_curr)
    
    f_gap = digamma(r_curr) - digamma(s_curr) - f
    q_gap = trigamma(r_curr) + trigamma(s_curr) - q
    while (abs(f_gap) + abs(q_gap) > err)
        #print("\n$(abs(f_gap) + abs(q_gap))\n")
        trigamma_r = trigamma(r_curr)
        trigamma_s = trigamma(s_curr)
        quadgamma_r = polygamma(2, r_curr)
        quadgamma_s = polygamma(2, s_curr)
        divisor = (trigamma_r * quadgamma_s + trigamma_s * quadgamma_r)
        r_step = (f_gap * quadgamma_s + q_gap * trigamma_s) / divisor
        s_step = (q_gap * trigamma_r - f_gap * quadgamma_r) / divisor
        
        while (r_step > r_curr)
            r_step /= 2
        end
        while (s_step > s_curr)
            s_step /= 2
        end
        
        #r_curr = abs(r_curr)
        #s_curr = abs(s_curr)
        #if (r_curr < 0)
        #    r_curr = 1
        #end
        #if (s_curr < 0)
        #    s_curr = 1
        #end
        r_curr -= r_step
        s_curr -= s_step
        
        f_gap = digamma(r_curr) - digamma(s_curr) - f
        q_gap = trigamma(r_curr) + trigamma(s_curr) - q
        #print("\nFGAP:$(f_gap),QGAP:$(q_gap)\n")
        #flush(stdout)
    end

    return [r_curr, s_curr]

end

function update_approx_bernoulli(y, a, R, F, err)

    f = dot(a, F)
    q = dot(F, R, F)

    r, s = solve_beta_param(f, q, err)

    r += y
    s += (1-y)

    f_star = digamma(r) - digamma(s)
    q_star = trigamma(r) + trigamma(s)
    #print("\nSTART\n")
    #print("$(y)\n$(a)\n$(f)\n$(q)\n$(r)\n$(s)$(f_star)\n$(q_star)\n")
    #print(q_star)
    #print(R)
    #print("\n")
    #print("$(F)")
    #print("\n")
    RF = R * F
    #print(RF * RF')
    #print("\n")
    m = a .+ RF .* (f_star - f) / q
    C = R .- RF * RF' .* (1 - q_star / q) ./ q
    C = (C + C') ./ 2
    #print(C)
    #flush(stdout)
    return m, C
end

function bernoulli_ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
            true_expected_rewards_logit = true_bandit_param * context
            true_expected_rewards = exp.(true_expected_rewards_logit) ./ (1 .+ exp.(true_expected_rewards_logit))
            #true_expected_rewards = bandit_posterior_means * context
            action = action_function(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
            true_expected_reward = true_expected_rewards[action]
            EPREWARDS[t] = true_expected_reward
            EPOPTREWARDS[t] = maximum(true_expected_rewards)
            #obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:],context)) + true_expected_reward
            obs = 1 * (rand() < true_expected_reward)

            m_t, C_t, = update_approx_bernoulli(obs, bandit_posterior_means[action, :], bandit_posterior_covs[action, :, :], context, .00001)
            
            bandit_posterior_means[action, :] = m_t
            bandit_posterior_covs[action, :, :] = C_t

	        println("Ep: ", ep, " - ", t, " of ", T, " for ", String(Symbol(action_function)))
            flush(stdout)

        end
	return EPREWARDS, EPOPTREWARDS
end

function bernoulli_contextual_bandit_simulator(action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

    REWARDS = zeros(T, n_episodes)
    OPTREWARDS = zeros(T, n_episodes)
    ep_count = 1
    

    ## USING BAYESIAN FORMULATION FOR FAIR COMPARISON
    #global_bandit_param = [1 0; 0 1; 2 -1]
    global_bandit_param = rand(Normal(bandit_prior_mean, bandit_prior_sd), bandit_count, context_dim)
    
    
    #threadreps = zeros(Threads.nthreads())


    for ep in 1:n_episodes
        EPREWARDS, EPOPTREWARDS = bernoulli_ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
                                       context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
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

function bernoulli_val_better_grid_lambda_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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

## END PREALLOCATION
    # check for strictly dominating arm
    # selecting this arm without rollout will save lots of time and should be strictly better
    mean_max_bandit = findmax(vec(predictive_rewards))[2]
    var_max_bandit = findmax(vec([context' * ((bandit_posterior_covs[i,:,:])) * context for i=1:bandit_count]))[2]

    if mean_max_bandit == var_max_bandit
    	return mean_max_bandit
    end
    
    ################################################################
    ### LAMBDA OPTIMIZATION VIA SPSA
    ################################################################


    lambdas = []
    lambda_values = [] 
    

    len1 = length(grid_margin_1)
    len2 = length(grid_margin_2)
    len3 = length(grid_margin_3)

    lambda_sds = zeros(len1, len2, len3)

    for i1 in 1:len1
        for i2 in 1:len2
            for i3 in 1:len3
                
                MEAN_REWARD = 0
                lambda_param = [grid_margin_1[i1], grid_margin_2[i2], grid_margin_3[i3]]

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
                    rollout_value = bernoulli_val_better_rollout(T-t+1, t, rollout_length, context, use_context, lambda_param, context_dim, context_mean,
                        context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
                        roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
                    # up_reward = predictive_rewards[bandit] + discount * rollout_value
                    
                    
                    MEAN_REWARD = ((roll - 1) * MEAN_REWARD + rollout_value) / roll



                end
                
                push!(lambdas, lambda_param)
                push!(lambda_values, MEAN_REWARD)
            
            end
        end
    end
    opt_index = findmax(lambda_values)[2]
    lambda_param = lambdas[opt_index]

    println(lambda_param[1],", ", lambda_param[2],", ", lambda_param[3])
    flush(stdout)

    # END OPTIMIZATION OF LAMBDA

    for bandit in 1:bandit_count
        
        BANDIT_REWARD = 0
        
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

            # TRUE PARAM VERSION 

            true_expected_rewards_logit = bandit_param * context
            true_expected_rewards = exp.(true_expected_rewards_logit) ./ (1 .+ exp.(true_expected_rewards_logit))
            true_expected_reward = true_expected_rewards[bandit]
            #obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:],context)) + true_expected_reward
            obs = 1 * (rand() < true_expected_reward)

            m_t, C_t, = update_approx_bernoulli(obs, temp_post_means[bandit, :], temp_post_covs[bandit, :, :], context, .00001)
            
            temp_post_means[bandit, :] = m_t
            temp_post_covs[bandit, :, :] = C_t



            use_context = false
            rollout_value = bernoulli_val_better_rollout(T-t, t, rollout_length, context, use_context, lambda_param, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
                roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)

            BANDIT_REWARD = (BANDIT_REWARD * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end

        BANDIT_VALUES[bandit] = BANDIT_REWARD
    
    end
    
    return findmax(BANDIT_VALUES)[2]

end


function bernoulli_val_greedy_thompson_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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

    
    policies = [greedy_policy, thompson_policy]
    policy_values = []
    
    println("Context Start: ", context)
    flush(stdout)

    lambda = [0, 0]

    for policy in policies  

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
                
                rollout_value = bernoulli_val_rollout(policy, T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
                    context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
                    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
                
                
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

    
    opt_act = opt_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    println("Optimal Action: ",opt_act)
    flush(stdout)

    return opt_act

end
function bernoulli_val_greedy_thompson_ucb_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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

    
    policies = [greedy_policy, thompson_policy, glm_ucb_policy]
    policy_values = []
    
    println("Context Start: ", context)
    flush(stdout)

    lambda = [0, 0]

    for policy in policies  

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
                
                rollout_value = bernoulli_val_rollout(policy, T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
                    context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
                    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
                
                
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

    
    opt_act = opt_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    println("Optimal Action: ",opt_act)
    flush(stdout)

    return opt_act

end


function bernoulli_val_greedy_rollout_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    predictive_rewards = bandit_posterior_means * context

    #mean_max_bandit = findmax(vec(predictive_rewards))[2]
    #var_max_bandit = findmax(vec([context' * (@view bandit_posterior_covs[i,:,:]) * context for i=1:bandit_count]))[2]

    #if mean_max_bandit == var_max_bandit
    #	return mean_max_bandit
    #end

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
	    #mul!(true_expected_rewards, bandit_param, context)
        
        
        
	    
        
        # TRUE PARAM VERSION
        #true_expected_reward = true_expected_rewards[action]
        #obs = randn() * obs_sd + true_expected_reward
        
        # UNKNOWN PARAM VERSION

        true_expected_rewards_logit = bandit_param * context
        true_expected_rewards = exp.(true_expected_rewards_logit) ./ (1 .+ exp.(true_expected_rewards_logit))
        true_expected_reward = true_expected_rewards[bandit]
        #obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:],context)) + true_expected_reward
        obs = 1 * (rand() < true_expected_reward)

        m_t, C_t, = update_approx_bernoulli(obs, temp_post_means[bandit, :], temp_post_covs[bandit, :, :], context, .00001)
        
        temp_post_means[bandit, :] = m_t
        temp_post_covs[bandit, :, :] = C_t


        rollout_value = bernoulli_val_greedy_rollout(T-t, rollout_length, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end


function bernoulli_greedy_rollout_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    predictive_rewards = bandit_posterior_means * context

    #mean_max_bandit = findmax(vec(predictive_rewards))[2]
    #var_max_bandit = findmax(vec([context' * (@view bandit_posterior_covs[i,:,:]) * context for i=1:bandit_count]))[2]

    #if mean_max_bandit == var_max_bandit
    #	return mean_max_bandit
    #end

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
        true_expected_rewards_logit = bandit_param * context
        true_expected_rewards = exp.(true_expected_rewards_logit) ./ (1 .+ exp.(true_expected_rewards_logit))
        true_expected_reward = true_expected_rewards[bandit]
        #obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:],context)) + true_expected_reward
        obs = 1 * (rand() < true_expected_reward)

        m_t, C_t, = update_approx_bernoulli(obs, temp_post_means[bandit, :], temp_post_covs[bandit, :, :], context, .00001)
        
        temp_post_means[bandit, :] = m_t
        temp_post_covs[bandit, :, :] = C_t

        rollout_value = bernoulli_greedy_rollout(T-t, rollout_length, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end
function bernoulli_greedy_rollout(T_remainder, rollout_length, lambda, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, discount, bandit_posterior_covs, bandit_posterior_means, bandit_param,
	context, true_expected_rewards, CovCon, old_cov, SigInvMu)

    disc_reward = 0
    fill!(context, 0.0)
    fill!(true_expected_rewards, 0.0)
    fill!(CovCon, 0.0)
    fill!(old_cov,0.0)
    fill!(SigInvMu,0.0)

    for t in 1:min(T_remainder, rollout_length)

        #context_seed = rand(1:context_dim)
        #fill!(context, zero(context[1]))
        #context[context_seed] = 1


        context = randn(context_dim) * context_sd .+ context_mean
        mul!(true_expected_rewards, bandit_param, context)


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

        true_expected_rewards_logit = bandit_param * context
        true_expected_rewards = exp.(true_expected_rewards_logit) ./ (1 .+ exp.(true_expected_rewards_logit))
        true_expected_reward = true_expected_rewards[action]
        #obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:],context)) + true_expected_reward
        obs = 1 * (rand() < true_expected_reward)

        m_t, C_t, = update_approx_bernoulli(obs, bandit_posterior_means[action, :], bandit_posterior_covs[action, :, :], context, .00001)
        
        bandit_posterior_means[action, :] = m_t
        bandit_posterior_covs[action, :, :] = C_t
        true_expected_reward = true_expected_rewards[action]
        disc_reward += true_expected_reward * discount^(t-1)

    end

    return disc_reward

end


function bernoulli_val_greedy_rollout(T_remainder, rollout_length, lambda, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, discount, bandit_posterior_covs, bandit_posterior_means, bandit_param,
	context, true_expected_rewards, CovCon, old_cov, SigInvMu)

    disc_reward = 0
    fill!(context, 0.0)
    fill!(true_expected_rewards, 0.0)
    fill!(CovCon, 0.0)
    fill!(old_cov,0.0)
    fill!(SigInvMu,0.0)
    
    truncation_length = T_remainder - min(T_remainder, rollout_length)

    for t in 1:min(T_remainder, rollout_length)

        context_seed = rand(1:context_dim)
        fill!(context, zero(context[1]))
        context[context_seed] = 1
        mul!(true_expected_rewards, bandit_param, context)


        context = randn(context_dim) * context_sd .+ context_mean



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




        # TRUE PARAM VERSION
        #true_expected_reward = true_expected_rewards[action]
        #disc_reward += true_expected_reward * discount^(t-1)
        #obs = randn() * obs_sd + true_expected_reward
        
        true_expected_reward_logit = dot(bandit_posterior_means[action,:], context)
        true_expected_reward = exp(true_expected_reward_logit) / (1 + exp(true_expected_reward_logit))
        disc_reward += true_expected_reward * discount^(t-1)

        obs = 1 * (rand() < true_expected_reward)

        m_t, C_t, = update_approx_bernoulli(obs, bandit_posterior_means[action, :], bandit_posterior_covs[action, :, :], context, .00001)
        
        bandit_posterior_means[action, :] = m_t
        bandit_posterior_covs[action, :, :] = C_t
        # UNKNOWN PARAM VERSION
    end


    if truncation_length > 0
        
        
        #### SWITCHED TO TRUE VALUE NEURAL NETWORK AS TEST
        
        #input_vec = Vector(vec(bandit_param'))
        #append!(input_vec, Vector(vec(bandit_posterior_means')))
        input_vec = Vector(vec(bandit_posterior_means'))
        append!(input_vec, vcat([upper_triangular_vec(bandit_posterior_covs[a, :, :]) for a in 1:bandit_count]...))
        
        scaled_input_vec = (input_vec .- bern_scale_list[truncation_length][1:end .!= 1, 1]) ./ bern_scale_list[truncation_length][1:end .!= 1, 2] 
        
        disc_reward += (discount^min(T_remainder, rollout_length)) * (bern_scale_list[truncation_length][1,2]*bern_neural_net_list[truncation_length](scaled_input_vec)[1] + bern_scale_list[truncation_length][1,1])
    
        
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
function bernoulli_val_rollout(policy, T_remainder, rollout_length, context, use_context, lambda, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, discount, bandit_posterior_covs, bandit_posterior_means, bandit_param,
	true_expected_rewards, CovCon, old_cov, SigInvMu)

    disc_reward = 0
    #fill!(context, 0.0)
    fill!(true_expected_rewards, 0.0)
    fill!(CovCon, 0.0)
    fill!(old_cov,0.0)
    fill!(SigInvMu,0.0)
    
    truncation_length = T_remainder - min(T_remainder, rollout_length)

    for t in 1:min(T_remainder, rollout_length)

        #context_seed = rand(1:context_dim)
        #fill!(context, zero(context[1]))
        #context[context_seed] = 1
        #mul!(true_expected_rewards, bandit_param, context)

        if t > 1 || !use_context
            context = randn(context_dim) * context_sd .+ context_mean
        end

        action = policy(t, min(T_remainder, rollout_length), bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)


        # TRUE PARAM VERSION
        #true_expected_reward = true_expected_rewards[action]
        #disc_reward += true_expected_reward * discount^(t-1)
        #obs = randn() * obs_sd + true_expected_reward
        
        true_expected_reward_logit = dot(bandit_posterior_means[action,:], context)
        true_expected_reward = exp(true_expected_reward_logit) / (1 + exp(true_expected_reward_logit))
        disc_reward += true_expected_reward * discount^(t-1)

        obs = 1 * (rand() < true_expected_reward)

        m_t, C_t, = update_approx_bernoulli(obs, bandit_posterior_means[action, :], bandit_posterior_covs[action, :, :], context, .00001)
        
        bandit_posterior_means[action, :] = m_t
        bandit_posterior_covs[action, :, :] = C_t
        # UNKNOWN PARAM VERSION
    end


    if truncation_length > 0
        
        
        #### SWITCHED TO TRUE VALUE NEURAL NETWORK AS TEST
        
        #input_vec = Vector(vec(bandit_param'))
        #append!(input_vec, Vector(vec(bandit_posterior_means')))
        input_vec = Vector(vec(bandit_posterior_means'))
        append!(input_vec, vcat([upper_triangular_vec(bandit_posterior_covs[a, :, :]) for a in 1:bandit_count]...))
        
        scaled_input_vec = (input_vec .- bern_scale_list[truncation_length][1:end .!= 1, 1]) ./ bern_scale_list[truncation_length][1:end .!= 1, 2] 
        
        disc_reward += (discount^min(T_remainder, rollout_length)) * (bern_scale_list[truncation_length][1,2]*bern_neural_net_list[truncation_length](scaled_input_vec)[1] + bern_scale_list[truncation_length][1,1])
    
        
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
function bernoulli_val_better_rollout(T_remainder, curr_t, rollout_length, context, use_context, lambda_param, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, discount, bandit_posterior_covs, bandit_posterior_means, bandit_param,
	true_expected_rewards, CovCon, old_cov, SigInvMu)

    

    #temp_post_covs = bandit_posterior_covs
    #temp_post_means = bandit_posterior_means
    disc_reward = 0
    #fill!(context, 0.0)
    fill!(true_expected_rewards, 0.0)
    fill!(CovCon, 0.0)
    fill!(old_cov,0.0)
    fill!(SigInvMu,0.0)

    truncation_length = T_remainder - min(T_remainder, rollout_length)

    for t in 1:min(T_remainder, rollout_length)
        #context_seed = rand(1:context_dim)
        #fill!(context, zero(context[1]))
        #context[context_seed] = 1


        if (t > 1) || !use_context
            context = randn(context_dim) * context_sd .+ context_mean
        end
        mul!(true_expected_rewards, bandit_param, context)
        #context = randn(context_dim) * context_sd .+ context_mean


        curr_ind = 1
        curr_max = dot((@view bandit_posterior_means[1,:]),context) + lambda_param[1]*(log(t+curr_t)^(1+lambda_param[2])) * (1 - ((t+curr_t) / (curr_t+T_remainder))^lambda_param[3]) * dot(context, (@view bandit_posterior_covs[1,:,:]), context)
        for i in 2:bandit_count
            candidate_max = dot((@view bandit_posterior_means[i,:]),context) + lambda_param[1]*(log(t+curr_t)^(1+lambda_param[2])) * (1 - ((t+curr_t) / (curr_t+T_remainder))^lambda_param[3]) * dot(context, (@view bandit_posterior_covs[i,:,:]), context)
            if candidate_max > curr_max
                curr_ind = i
                curr_max = candidate_max
            end
        end

        action = curr_ind

        # TRUE PARAM VERSION
        #true_expected_reward = true_expected_rewards[action]
        #disc_reward += true_expected_reward * discount^(t-1)
        #obs = randn() * obs_sd + true_expected_reward
        
        # UNKNOWN PARAM VERSION
        true_expected_reward_logit = true_expected_rewards[action]
        true_expected_reward = exp(true_expected_reward_logit) / (1 + exp(true_expected_reward_logit))
        disc_reward += true_expected_reward * discount^(t-1)

        obs = 1 * (rand() < true_expected_reward)

        m_t, C_t, = update_approx_bernoulli(obs, bandit_posterior_means[action, :], bandit_posterior_covs[action, :, :], context, .00001)
        
        bandit_posterior_means[action, :] = m_t
        bandit_posterior_covs[action, :, :] = C_t
    end

    if truncation_length > 0
        
        
        #### SWITCHED TO TRUE VALUE NEURAL NETWORK AS TEST
        
        #input_vec = Vector(vec(bandit_param'))
        #append!(input_vec, Vector(vec(bandit_posterior_means')))
        input_vec = Vector(vec(bandit_posterior_means'))
        append!(input_vec, vcat([upper_triangular_vec(bandit_posterior_covs[a, :, :]) for a in 1:bandit_count]...))
        
        scaled_input_vec = (input_vec .- bern_scale_list[truncation_length][1:end .!= 1, 1]) ./ bern_scale_list[truncation_length][1:end .!= 1, 2] 
        
        disc_reward += (discount^min(T_remainder, rollout_length)) * (bern_scale_list[truncation_length][1,2]*bern_neural_net_list[truncation_length](scaled_input_vec)[1] + bern_scale_list[truncation_length][1,1])
    
        
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
