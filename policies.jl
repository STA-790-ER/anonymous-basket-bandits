
# Greedy
function greedy_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    val, action = findmax(bandit_posterior_means * context)
    return action
end

# Epsilon Greedy
function epsilon_greedy_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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

# Bayes UCB
function bayes_ucb_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    val, action = findmax(bandit_posterior_means * context)
    reward_means = bandit_posterior_means * context
    reward_sds = sqrt.(vec([context' * (@view bandit_posterior_covs[i,:,:]) * context for i=1:bandit_count]))
    ucbs = quantile.(Normal.(reward_means, reward_sds), 1-1/t)
    return findmax(ucbs)[2]
end

# Bayes UCB
function glm_ucb_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    #val, action = findmax(bandit_posterior_means * context)
    reward_means = bandit_posterior_means * context
    reward_sds = sqrt.(vec([context' * (@view bandit_posterior_covs[i,:,:]) * context for i=1:bandit_count]))
    ucbs = reward_means .+ max(1,sqrt(log(t))) .* reward_sds
    return findmax(ucbs)[2]
end

# Thompson
function thompson_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    thompson_samples = zeros(bandit_count, context_dim)

    for bandit in 1:bandit_count
	    thompson_samples[bandit, :] = rand(MvNormal((bandit_posterior_means[bandit,:]), (bandit_posterior_covs[bandit,:,:])))
    end

    return findmax(thompson_samples * context)[2]
end

# FRACTIONAL THOMPSON (SCALE COVARIANCE)

function thompson_policy_2(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    thompson_samples = zeros(bandit_count, context_dim)

    for bandit in 1:bandit_count
	    thompson_samples[bandit, :] = rand(MvNormal((bandit_posterior_means[bandit,:]), (2 .* bandit_posterior_covs[bandit,:,:])))
    end

    return findmax(thompson_samples * context)[2]
end
function thompson_policy_4(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    thompson_samples = zeros(bandit_count, context_dim)

    for bandit in 1:bandit_count
	    thompson_samples[bandit, :] = rand(MvNormal((bandit_posterior_means[bandit,:]), (4 .* bandit_posterior_covs[bandit,:,:])))
    end

    return findmax(thompson_samples * context)[2]
end
function thompson_policy_0_5(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    thompson_samples = zeros(bandit_count, context_dim)

    for bandit in 1:bandit_count
	    thompson_samples[bandit, :] = rand(MvNormal((bandit_posterior_means[bandit,:]), (.5 .* bandit_posterior_covs[bandit,:,:])))
    end

    return findmax(thompson_samples * context)[2]
end
function thompson_policy_0_25(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    thompson_samples = zeros(bandit_count, context_dim)

    for bandit in 1:bandit_count
	    thompson_samples[bandit, :] = rand(MvNormal((bandit_posterior_means[bandit,:]), (.25 .* bandit_posterior_covs[bandit,:,:])))
    end

    return findmax(thompson_samples * context)[2]
end

# SquareCB
function squarecb_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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
## LAMBDA POLICY

function lambda_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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

        dividy = 1 + dot(context ./ obs_sd, roll_old_cov, context ./ obs_sd)
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
        use_context = false
        rollout_value = lambda_rollout(T-t, rollout_length, context, use_context, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end

function greedy_rollout_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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

        dividy = 1 + dot(context ./ obs_sd, roll_old_cov, context ./ obs_sd)
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


function val_greedy_rollout_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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
	    mul!(true_expected_rewards, bandit_param, context)
        
        
        
	    
        
        # TRUE PARAM VERSION
        #true_expected_reward = true_expected_rewards[action]
        #obs = randn() * obs_sd + true_expected_reward
        
        # UNKNOWN PARAM VERSION
        true_expected_reward = dot(bandit_posterior_means[bandit,:], context)
        obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[bandit,:,:], context)) + true_expected_reward
        
        
        copy!(roll_old_cov, (@view temp_post_covs[bandit, :, :]))
	    mul!(roll_CovCon, roll_old_cov, context)
	    roll_CovCon ./= obs_sd



        dividy = 1 + dot(context ./ obs_sd, roll_old_cov, context ./ obs_sd)
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



        rollout_value = val_greedy_rollout(T-t, rollout_length, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end



function lambda_mean_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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

        dividy = 1 + dot(context ./ obs_sd, roll_old_cov, context ./ obs_sd)
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
        use_context = false
        rollout_value = lambda_mean_rollout(T-t, rollout_length, context, use_context, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end
## BETER LAMBDA POLICY

function better_lambda_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
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

        dividy = 1 + dot(context ./ obs_sd, roll_old_cov, context ./ obs_sd)
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
        use_context = false
        rollout_value = better_rollout(T-t, t, rollout_length, context, use_context, lambda_param, context_dim, context_mean,
            context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end


function val_better_lambda_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
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

        dividy = 1 + dot(context ./ obs_sd, roll_old_cov, context ./ obs_sd)
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
        use_context = false
        rollout_value = val_better_rollout(T-t, t, rollout_length, context, use_context, lambda_param, context_dim, context_mean,
            context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end


function opt_lambda_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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
            use_context = true
            rollout_value = lambda_rollout(T-t+1, rollout_length+1, context, use_context, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
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
            use_context = true
            rollout_value = lambda_rollout(T-t+1, rollout_length+1, context, use_context, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
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

        dividy = 1 + dot(context ./ obs_sd, roll_old_cov, context ./ obs_sd)
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


        use_context = false
        rollout_value = lambda_rollout(T-t, rollout_length, context, use_context, lambda, context_dim, context_mean,
            context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end


function better_opt_lambda_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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
    lambda_param = [1, 1, 2]
    lambda_trans = log.(lambda_param)
    BANDIT_REWARDS = zeros(n_rollouts)

    ################################################################
    ### LAMBDA OPTIMIZATION VIA SPSA
    ################################################################

    for iter in 1:n_spsa_iter

        delta_n = rand([-1, 1], 3)
        c_n = 1 / iter^(.3)
        a_n = 1 / iter
        lambda_trans_up = lambda_trans + c_n .* delta_n
        lambda_trans_dn = lambda_trans - c_n .* delta_n
        lambda_up = exp.(lambda_trans_up)
        lambda_dn = exp.(lambda_trans_dn)

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

            lambda_param = lambda_up
            use_context = true
            rollout_value = better_rollout(T-t+1, t, rollout_length, context, use_context, lambda_param, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
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

            lambda_param = lambda_dn
            use_context = true
            rollout_value = better_rollout(T-t+1, t, rollout_length, context, use_context, lambda_param, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
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
    lambda_param = exp.(lambda_trans)


    println(lambda_param[1],", ", lambda_param[2],", ", lambda_param[3])
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
            #bandit_param = zeros(bandit_count, context_dim)
            for bandit in 1:bandit_count
                copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
                copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
                bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
            end
            # TRUE PARAM VERSION 
            
            #mul!(true_expected_rewards, bandit_param, context)
            #true_expected_reward = true_expected_rewards[bandit]
            #disc_reward += true_expected_reward * discount^(t-1)
            #obs = randn() * obs_sd + true_expected_reward
            
            # UNKNOWN PARAM VERSION
            true_expected_reward = dot(bandit_posterior_means[bandit,:], context)
            disc_reward += true_expected_reward * discount^(t-1)
            obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[bandit,:,:], context)) + true_expected_reward
            
            
            
            copy!(roll_old_cov, (@view temp_post_covs[bandit, :, :]))
            mul!(roll_CovCon, roll_old_cov, context)
            roll_CovCon ./= obs_sd

            dividy = 1 + dot(context ./ obs_sd, roll_old_cov, context ./ obs_sd)
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

            use_context = false
            rollout_value = better_rollout(T-t, t, rollout_length, context, use_context, lambda_param, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
                roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)

            BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end

        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    
    end
    
    return findmax(BANDIT_VALUES)[2]

end


function val_better_opt_lambda_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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
    lambda_param = [1, 1, 2]
    lambda_trans = log.(lambda_param)
    BANDIT_REWARDS = zeros(n_rollouts)

    ################################################################
    ### LAMBDA OPTIMIZATION VIA SPSA
    ################################################################

    for iter in 1:n_spsa_iter

        delta_n = rand([-1, 1], 3)
        c_n = 0.1 / iter^(.3)
        a_n = 0.1 / iter
        lambda_trans_up = lambda_trans + c_n .* delta_n
        lambda_trans_dn = lambda_trans - c_n .* delta_n
        lambda_up = exp.(lambda_trans_up)
        lambda_dn = exp.(lambda_trans_dn)

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

            lambda_param = lambda_up
            use_context = true
            rollout_value = val_better_rollout(T-t+1, t, rollout_length, context, use_context, lambda_param, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
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

            lambda_param = lambda_dn
            use_context = true
            rollout_value = val_better_rollout(T-t+1, t, rollout_length, context, use_context, lambda_param, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
            # down_reward = predictive_rewards[bandit] + discount * rollout_value
            down_reward = rollout_value
            MEAN_DOWN_REWARD = ((roll - 1) * MEAN_DOWN_REWARD + down_reward) / roll

        end

        ###############
        #### UPDATE ###
        ###############
        #print("\n$(MEAN_UP_REWARD), $(MEAN_DOWN_REWARD), $(c_n), $(delta_n)\n")
        grad_est .= (MEAN_UP_REWARD .- MEAN_DOWN_REWARD) ./ ((2 * c_n) .* delta_n)

        lambda_trans .+= (a_n .* grad_est)

    end



    # set final lambda parameter
    lambda_param = exp.(lambda_trans)


    println(lambda_param[1],", ", lambda_param[2],", ", lambda_param[3])
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
            #bandit_param = zeros(bandit_count, context_dim)
            for bandit in 1:bandit_count
                copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
                copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
                bandit_param[bandit,:] .= rand(MvNormal(temp_bandit_mean, temp_bandit_cov))
            end
            # TRUE PARAM VERSION 
            
            #mul!(true_expected_rewards, bandit_param, context)
            #true_expected_reward = true_expected_rewards[bandit]
            #obs = randn() * obs_sd + true_expected_reward
            
            # UNKNOWN PARAM VERSION
            true_expected_reward = dot(bandit_posterior_means[bandit,:], context)
            obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[bandit,:,:], context)) + true_expected_reward
            
            
            
            copy!(roll_old_cov, (@view temp_post_covs[bandit, :, :]))
            mul!(roll_CovCon, roll_old_cov, context)
            roll_CovCon ./= obs_sd

            dividy = 1 + dot(context ./ obs_sd, roll_old_cov, context ./ obs_sd)
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

            use_context = false
            rollout_value = val_better_rollout(T-t, t, rollout_length, context, use_context, lambda_param, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
                roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)

            BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end

        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    
    end
    
    return findmax(BANDIT_VALUES)[2]

end

function val_better_grid_lambda_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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
                    rollout_value = val_better_rollout(T-t+1, t, rollout_length, context, use_context, lambda_param, context_dim, context_mean,
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
    if (t == 1) && (ep == 1)
        CSV.write("/hpc/home/jml165/rl/arrayselectionresults/selectionresults_$(idx).csv", Tables.table(vcat(t, lambda_param)'), header=["time", "param1","param2", "param3"])
    else
        CSV.write("/hpc/home/jml165/rl/arrayselectionresults/selectionresults_$(idx).csv", Tables.table(vcat(t, lambda_param)'), append=true)
    end
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
            
            #mul!(true_expected_rewards, bandit_param, context)
            #true_expected_reward = true_expected_rewards[bandit]
            #obs = randn() * obs_sd + true_expected_reward
            
            # UNKNOWN PARAM VERSION
            true_expected_reward = dot(bandit_posterior_means[bandit,:], context)
            obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[bandit,:,:], context)) + true_expected_reward
            
            
            
            copy!(roll_old_cov, (@view temp_post_covs[bandit, :, :]))
            mul!(roll_CovCon, roll_old_cov, context)
            roll_CovCon ./= obs_sd

            dividy = 1 + dot(context ./ obs_sd, roll_old_cov, context ./ obs_sd)
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

            use_context = false
            rollout_value = val_better_rollout(T-t, t, rollout_length, context, use_context, lambda_param, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
                roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)

            BANDIT_REWARD = (BANDIT_REWARD * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end

        BANDIT_VALUES[bandit] = BANDIT_REWARD
    
    end
    
    return findmax(BANDIT_VALUES)[2]

end


function val_greedy_thompson_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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
                
                rollout_value = val_rollout(ep, policy, T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
                    context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
                    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
                
                #if roll == 1
                #    MEAN_REWARD = rollout_value
                #end
                MEAN_REWARD = ((roll - 1) * MEAN_REWARD + rollout_value) / roll

            end
            
            push!(policy_values, MEAN_REWARD)
        
    end
    
    println("Context Finish: ", context)
    flush(stdout)

    opt_index = findmax(policy_values)[2]
    opt_policy = policies[opt_index]

    if (t == 1) && (ep == 1)
        CSV.write("/hpc/home/jml165/rl/arrayselectionresults/selectionresults_$(idx).csv", Tables.table([t String(Symbol(opt_policy))]), header=["time", "policy"])
    else
        CSV.write("/hpc/home/jml165/rl/arrayselectionresults/selectionresults_$(idx).csv", Tables.table([t String(Symbol(opt_policy))]), append=true)
    end
    println("GREEDY: ", policy_values[1],", THOMPSON: ", policy_values[2])
    flush(stdout)

    # END OPTIMIZATION OF LAMBDA

    
    opt_act = opt_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    println("Optimal Action: ",opt_act)
    flush(stdout)

    return opt_act

end

function val_greedy_thompson_ucb_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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
                
                rollout_value = val_rollout(policy, T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
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
function val_greedy_thompson_ucb_ids_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    if t == 1
        return rand(1:bandit_count)
    end
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

    
    policies = [greedy_policy, thompson_policy, bayes_ucb_policy, ids_policy]
    policy_values = []
    
    action_samps = [pol(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim) for pol in policies]
    if length(unique(action_samps)) == 1
        println("Agreement Action: ", action_samps[1])
        flush(stdout)
        return action_samps[1]
    end
    println("Context Start: ", context)
    flush(stdout)

    lambda = [0, 0]
    policy_count = length(policies)
    REWARD_SAMPS = zeros(n_opt_rollouts, length(policies))
    halt_vec = repeat([false], length(policies))
    rollout_counts = repeat([0], length(policies))


    BANDIT_PARAM_SAMPS = zeros(bandit_count, context_dim, n_opt_rollouts)
    for k in 1:bandit_count
        BANDIT_PARAM_SAMPS[k, :, :] = rand(MvNormal(bandit_posterior_means[k, :], bandit_posterior_covs[k, :, :]), n_opt_rollouts)
    end
    same_break = false
    for roll in 1:n_opt_rollouts  

            MEAN_REWARD = 0
            
            bandit_param = BANDIT_PARAM_SAMPS[:, :, roll]

            for pol_ind in 1:policy_count
                
                if halt_vec[pol_ind]
                    continue
                end

                rollout_counts[pol_ind] += 1
            
                copy!(temp_post_means, bandit_posterior_means)
                copy!(temp_post_covs, bandit_posterior_covs)
                #bandit_param = zeros(bandit_count, context_dim)
                use_context = true
                
                rollout_value = val_rollout(ep, policies[pol_ind], T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
                    context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
                    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
                
                REWARD_SAMPS[roll, pol_ind] = rollout_value     

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
                    println("REMAINING POLICIES HAVE SAME ACTION (SAMPLE)")
                    same_break = true
                end
            end
            
            if same_break
                break
            end

            if (sum(halt_vec .== false) == 1) 
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
    
    return opt_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    
    ## SETUP FOR OPTIMAL ACTION
    ACTION_MEAN_REWARDS = zeros(bandit_count)
    ACTION_STD_REWARDS = zeros(bandit_count)
    REWARD_SAMPS = zeros(n_opt_rollouts, bandit_count)
    halt_vec = repeat([false], bandit_count)
    rollout_counts = repeat([0], bandit_count)
    
    for roll in 1:n_opt_rollouts  

            bandit_param = BANDIT_PARAM_SAMPS[:, :, roll]

            for action in 1:bandit_count
                if halt_vec[action]
                    continue
                end

                rollout_counts[action] += 1
            

                use_context = false

                copy!(temp_post_means, bandit_posterior_means)
                copy!(temp_post_covs, bandit_posterior_covs)
                obs = dot(bandit_param[action, :], context) + obs_sd * randn()
                
                old_cov = temp_post_covs[action, :, :]
                CovCon = old_cov * context
                CovCon ./= obs_sd
                #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)

                dividy = 1 + dot(context ./ obs_sd, old_cov, context ./ obs_sd)
                for i in 1:context_dim
                    for j in 1:context_dim
                        temp_post_covs[action,j,i] = old_cov[j,i] - CovCon[j]*CovCon[i] / dividy
                    end
                end

                for i in 1:context_dim
                    for j in 1:(i-1)
                        temp_post_covs[action,i,j] = (temp_post_covs[action,i,j]+temp_post_covs[action,j,i])/2
                        temp_post_covs[action,j,i] = temp_post_covs[action,i,j]
                    end
                end

                SigInvMu = old_cov \ temp_post_means[action,:]
                SigInvMu += context .* obs ./ obs_sd.^2
                temp_post_means[action,:] = temp_post_covs[action,:,:] * SigInvMu

                rollout_value = val_rollout(ep, opt_policy, T-t, rollout_length-1, context, use_context, lambda, context_dim, context_mean,
                    context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
                    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
                
                REWARD_SAMPS[roll, pol_ind] = dot(bandit_param[action, :], context) + discount * rollout_value     
                
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
function val_greedy_thompson_ucb_ids_q_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    if t == 1
        return rand(1:bandit_count)
    end
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

    
    policies = [greedy_policy, thompson_policy, bayes_ucb_policy, ids_policy]
    policy_values = []
    
    action_samps = [pol(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim) for pol in policies]
    #if length(unique(action_samps)) == 1
    #    println("Agreement Action: ", action_samps[1])
    #    flush(stdout)
    #    return action_samps[1]
    #end
    println("Context Start: ", context)
    flush(stdout)

    lambda = [0, 0]
    policy_count = length(policies)
    REWARD_SAMPS = zeros(n_opt_rollouts, length(policies))
    halt_vec = repeat([false], length(policies))
    rollout_counts = repeat([0], length(policies))


    BANDIT_PARAM_SAMPS = zeros(bandit_count, context_dim, n_opt_rollouts)
    for k in 1:bandit_count
        BANDIT_PARAM_SAMPS[k, :, :] = rand(MvNormal(bandit_posterior_means[k, :], bandit_posterior_covs[k, :, :]), n_opt_rollouts)
    end

    for roll in 1:n_opt_rollouts  

            MEAN_REWARD = 0
            
            bandit_param = BANDIT_PARAM_SAMPS[:, :, roll]

            for pol_ind in 1:policy_count
                
                if halt_vec[pol_ind]
                    continue
                end

                rollout_counts[pol_ind] += 1
            
                copy!(temp_post_means, bandit_posterior_means)
                copy!(temp_post_covs, bandit_posterior_covs)
                #bandit_param = zeros(bandit_count, context_dim)
                use_context = true
                
                rollout_value = val_rollout(ep, policies[pol_ind], T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
                    context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
                    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
                
                REWARD_SAMPS[roll, pol_ind] = rollout_value     

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
                #continue_inds = findall(halt_vec .== false)
                #if length(unique(action_samps[continue_inds])) == 1
                #    println("REMAINING POLICIES HAVE SAME ACTION (SAMPLE)")
                #    break
                #end
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
    
    
    ## SETUP FOR OPTIMAL ACTION
    ACTION_MEAN_REWARDS = zeros(bandit_count)
    ACTION_STD_REWARDS = zeros(bandit_count)
    REWARD_SAMPS = zeros(n_opt_rollouts, bandit_count)
    halt_vec = repeat([false], bandit_count)
    rollout_counts = repeat([0], bandit_count)
    
    for roll in 1:n_opt_rollouts  

            bandit_param = BANDIT_PARAM_SAMPS[:, :, roll]

            for action in 1:bandit_count
                if halt_vec[action]
                    continue
                end

                rollout_counts[action] += 1
            

                use_context = false

                copy!(temp_post_means, bandit_posterior_means)
                copy!(temp_post_covs, bandit_posterior_covs)
                obs = dot(bandit_param[action, :], context) + obs_sd * randn()
                
                old_cov = temp_post_covs[action, :, :]
                CovCon = old_cov * context
                CovCon ./= obs_sd
                #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)

                dividy = 1 + dot(context ./ obs_sd, old_cov, context ./ obs_sd)
                for i in 1:context_dim
                    for j in 1:context_dim
                        temp_post_covs[action,j,i] = old_cov[j,i] - CovCon[j]*CovCon[i] / dividy
                    end
                end

                for i in 1:context_dim
                    for j in 1:(i-1)
                        temp_post_covs[action,i,j] = (temp_post_covs[action,i,j]+temp_post_covs[action,j,i])/2
                        temp_post_covs[action,j,i] = temp_post_covs[action,i,j]
                    end
                end

                SigInvMu = old_cov \ temp_post_means[action,:]
                SigInvMu += context .* obs ./ obs_sd.^2
                temp_post_means[action,:] = temp_post_covs[action,:,:] * SigInvMu

                rollout_value = val_rollout(ep, opt_policy, T-t, rollout_length-1, context, use_context, lambda, context_dim, context_mean,
                    context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
                    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
                
                REWARD_SAMPS[roll, pol_ind] = dot(bandit_param[action, :], context) + discount * rollout_value     
                
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
function dp_greedy_thompson_ucb_ids_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    if t == 1
        return rand(1:bandit_count)
    end
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

    
    policies = [greedy_policy, thompson_policy, bayes_ucb_policy, ids_policy]
    policy_values = []
    
    action_samps = [pol(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim) for pol in policies]
    #if length(unique(action_samps)) == 1
    #    println("Agreement Action: ", action_samps[1])
    #    flush(stdout)
    #    return action_samps[1]
    #end
    println("Context Start: ", context)
    flush(stdout)

    lambda = [0, 0]
    policy_count = length(policies)
    REWARD_SAMPS = zeros(n_opt_rollouts, length(policies))
    halt_vec = repeat([false], length(policies))
    rollout_counts = repeat([0], length(policies))


    BANDIT_PARAM_SAMPS = zeros(bandit_count, context_dim, n_opt_rollouts)
    for k in 1:bandit_count
        BANDIT_PARAM_SAMPS[k, :, :] = rand(MvNormal(bandit_posterior_means[k, :], bandit_posterior_covs[k, :, :]), n_opt_rollouts)
    end
    same_break = false
    for roll in 1:n_opt_rollouts  

            MEAN_REWARD = 0
            
            bandit_param = BANDIT_PARAM_SAMPS[:, :, roll]

            for pol_ind in 1:policy_count
                
                if halt_vec[pol_ind]
                    continue
                end

                rollout_counts[pol_ind] += 1
            
                copy!(temp_post_means, bandit_posterior_means)
                copy!(temp_post_covs, bandit_posterior_covs)
                #bandit_param = zeros(bandit_count, context_dim)
                use_context = true
                
                rollout_value = val_rollout(ep, policies[pol_ind], T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
                    context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
                    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
                
                REWARD_SAMPS[roll, pol_ind] = rollout_value     

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
                #if length(unique(action_samps[continue_inds])) == 1
                #    println("REMAINING POLICIES HAVE SAME ACTION (SAMPLE)")
                #    same_break = true
                #end
            end
            
            if same_break
                break
            end

            if (sum(halt_vec .== false) == 1) 
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
    use_context = true
    opt_policy = policies[opt_index]
    MEM = [[[] for b in 1:bandit_count] for tt in 1:min(T-t+1,dp_length)]
    BANDIT_PARAM_SAMPS = zeros(bandit_count, context_dim, dp_rollout_count)
    for k in 1:bandit_count
        BANDIT_PARAM_SAMPS[k, :, :] = rand(MvNormal(bandit_posterior_means[k, :], bandit_posterior_covs[k, :, :]), dp_rollout_count)
    end
    for tt in 1:dp_rollout_count
        dp_actions = sample(1:bandit_count, dp_length)
        RESULTS = dp_rollout(ep, opt_policy, T-t+1, rollout_length, dp_length, dp_actions, context, use_context, lambda, context_dim, context_mean,
                             context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, BANDIT_PARAM_SAMPS[:, :, tt],
            roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
        for i in eachindex(RESULTS)
            if i == 1
                push!(MEM[i][dp_actions[1]], RESULTS[i])
            else
                push!(MEM[i][RESULTS[i][2]], RESULTS[i])
            end
        end
    end
    M = [Vector{Any}(undef, bandit_count) for tt in 1:min(T-t+1, dp_length)]
    for s in 1:min(dp_length, T-t+1)
        tt = min(dp_length, T-t+1) - s + 1
        for b in 1:bandit_count
            Y = [u[3] + discount * (u[4] == 0 ? maximum([XGBoost.predict(M[tt+1][bb], reshape(u[5], 1, length(u[5])))[1] for bb in 1:bandit_count]) : 0) for u in MEM[tt][b]]
            X = reduce(vcat, transpose.([u[1] for u in MEM[tt][b]]))
            if tt > 1
                M[tt][b] = xgboost_cv(X, Y, 3)
            else
                M[tt][b] = mean(Y)
            end
        end
    end
    return findmax(M[1])[2]

    ## SETUP FOR OPTIMAL ACTION
    ACTION_MEAN_REWARDS = zeros(bandit_count)
    ACTION_STD_REWARDS = zeros(bandit_count)
    REWARD_SAMPS = zeros(n_opt_rollouts, bandit_count)
    halt_vec = repeat([false], bandit_count)
    rollout_counts = repeat([0], bandit_count)
    
    for roll in 1:n_opt_rollouts  

            bandit_param = BANDIT_PARAM_SAMPS[:, :, roll]

            for action in 1:bandit_count
                if halt_vec[action]
                    continue
                end

                rollout_counts[action] += 1
            

                use_context = false

                copy!(temp_post_means, bandit_posterior_means)
                copy!(temp_post_covs, bandit_posterior_covs)
                obs = dot(bandit_param[action, :], context) + obs_sd * randn()
                
                old_cov = temp_post_covs[action, :, :]
                CovCon = old_cov * context
                CovCon ./= obs_sd
                #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)

                dividy = 1 + dot(context ./ obs_sd, old_cov, context ./ obs_sd)
                for i in 1:context_dim
                    for j in 1:context_dim
                        temp_post_covs[action,j,i] = old_cov[j,i] - CovCon[j]*CovCon[i] / dividy
                    end
                end

                for i in 1:context_dim
                    for j in 1:(i-1)
                        temp_post_covs[action,i,j] = (temp_post_covs[action,i,j]+temp_post_covs[action,j,i])/2
                        temp_post_covs[action,j,i] = temp_post_covs[action,i,j]
                    end
                end

                SigInvMu = old_cov \ temp_post_means[action,:]
                SigInvMu += context .* obs ./ obs_sd.^2
                temp_post_means[action,:] = temp_post_covs[action,:,:] * SigInvMu

                rollout_value = val_rollout(ep, opt_policy, T-t, rollout_length-1, context, use_context, lambda, context_dim, context_mean,
                    context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
                    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
                
                REWARD_SAMPS[roll, pol_ind] = dot(bandit_param[action, :], context) + discount * rollout_value     
                
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

function greedy_thompson_ucb_ids_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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

    
    policies = [greedy_policy, thompson_policy, bayes_ucb_policy, ids_policy]
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
                
                rollout_value = rollout(ep, policy, T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
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

    
    opt_act = opt_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    println("Optimal Action: ",opt_act)
    flush(stdout)

    return opt_act

end
function val_generalized_ids_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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

    
    policies = [ids_policy_0_5, ids_policy_4, ids_policy]
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
                
                rollout_value = val_rollout(policy, T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
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
function val_epsilon_greedy_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, n_eps)

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
    epsilon_values = []
    
    println("Context Start: ", context)
    flush(stdout)

    lambda = [0, 0]

    for eps in 1:n_eps  

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
                
                rollout_value = val_epsilon_greedy_rollout((eps-1)/n_eps, T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
                    context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
                    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
                
                
                MEAN_REWARD = ((roll - 1) * MEAN_REWARD + rollout_value) / roll

            end
            
            push!(epsilon_values, MEAN_REWARD)
        
    end
    
    println("Context Finish: ", context)
    flush(stdout)

    opt_index = findmax(epsilon_values)[2]
    opt_epsilon = (opt_index-1) / n_eps

    println("EPSILON: ", opt_epsilon)
    flush(stdout)

    # END OPTIMIZATION OF LAMBDA

    
    if rand() > opt_epsilon
        return findmax(predictive_rewards)[2]
    else
        return rand(1:bandit_count)
    end

end

function vegp2(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_epsilon_greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 2))

end
function vegp4(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_epsilon_greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 4))

end
function vegp8(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_epsilon_greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 8))

end
function vegp16(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_epsilon_greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 16))

end
function vegp32(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_epsilon_greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 32))

end
function vegp64(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_epsilon_greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 64))

end

function val_fractional_thompson_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, n_thom)

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

    lambda = [0, 0]

    for thom in 1:n_thom  

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
                
                rollout_value = val_fractional_thompson_rollout((thom-1)/n_thom, T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
                    context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
                    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
                
                
                MEAN_REWARD = ((roll - 1) * MEAN_REWARD + rollout_value) / roll

            end
            
            push!(thompson_values, MEAN_REWARD)
        
    end
    
    println("Context Finish: ", context)
    flush(stdout)

    opt_index = findmax(thompson_values)[2]
    opt_thom = (opt_index-1) / n_thom

    println("EPSILON: ", opt_thom)
    flush(stdout)

    # END OPTIMIZATION OF LAMBDA
    if opt_thom == 0
        return greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    else
        frac = 1 / opt_thom - 1
        thompson_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs ./ frac, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    end

end

function vftp2(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_fractional_thompson_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 2))

end
function vftp4(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_fractional_thompson_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 4))

end
function vftp8(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_fractional_thompson_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 8))

end
function vftp16(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_fractional_thompson_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 16))

end
function vftp32(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_fractional_thompson_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 32))

end
function vftp64(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_fractional_thompson_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 64))

end

function bayesopt_lambda_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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

        use_context = true  
        rollout_value = better_rollout(T-t+1, t, rollout_length+1, context, use_context, lambda_param, context_dim, context_mean,
            context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
            roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
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

	    dividy = 1 + dot(context ./ obs_sd, roll_old_cov, context ./ obs_sd)
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

        use_context = false
        rollout_value = better_rollout(T-t, t, rollout_length, context, use_context, lambda_param, context_dim, context_mean,
            context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end

## GRID SEARCH VERSION
function grid_lambda_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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
            
            use_context = true
            rollout_value = lambda_rollout(T-t+1, rollout_length+1, context, use_context, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
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

	    dividy = 1 + dot(context ./ obs_sd, roll_old_cov, context ./ obs_sd)
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

        use_context = false
        rollout_value = lambda_rollout(T-t, rollout_length, context, use_context, lambda, context_dim, context_mean,
            context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end


function ent_lambda_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
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

            dividy = 1 + dot(context ./ obs_sd, roll_old_cov, context ./ obs_sd)
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

        use_context = false
        rollout_value = ent_lambda_rollout(T-t, rollout_length, context, use_context, lambda, context_dim, context_mean,
            context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end

function ent_opt_lambda_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)


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
            
            use_context = true
            rollout_value = ent_lambda_rollout(T-t+1, rollout_length+1, context, use_context, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
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
            
            use_context = true
            rollout_value = ent_lambda_rollout(T-t+1, rollout_length+1, context, use_context, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
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

	    dividy = 1 + dot(context ./ obs_sd, roll_old_cov, context ./ obs_sd)
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

        use_context = false
        rollout_value = ent_lambda_rollout(T-t, rollout_length, context, use_context, lambda, context_dim, context_mean,
            context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		    roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end



## IDS POLICY

function ids_expected_regrets(context, bandit_posterior_means, bandit_posterior_covs, niter)

    #draws = zeros(bandit_count, context_dim, niter)
    reward_draws = zeros(bandit_count, niter)
    
    for b in 1:bandit_count
        #draws[b, :, :] = rand(MvNormal(bandit_posterior_means[b, :], bandit_posterior_covs[b, :, :]), niter)
        #for i in 1:niter
        #    reward_draws[b, i] = dot(context, draws[b, :, i])
        #end
        reward_draws[b, :] = dot(context, bandit_posterior_means[b, :]) .+ sqrt(dot(context, bandit_posterior_covs[b, :, :], context)) .* randn(niter)
    end
    
    mean_rewards = dropdims(mean(reward_draws, dims = 2), dims = 2)

    mean_max_reward = 0
    for i in 1:niter
        mean_max_reward += findmax(reward_draws[:, i])[1] / niter
    end
    
    res = max.(0, mean_max_reward .- mean_rewards)

    return res
end



function ids_information_ratio(bandit_posterior_covs, context, action, regrets)
    gain = ids_expected_entropy_gain(bandit_posterior_covs[action, :, :], context)
    return -1*regrets[action]^2 / gain
end
function ids_information_ratio_0_5(bandit_posterior_covs, context, action, regrets)
    gain = ids_expected_entropy_gain(bandit_posterior_covs[action, :, :], context)
    #print(regrets[action])
    return -1*regrets[action]^.5 / gain
end
function ids_information_ratio_4(bandit_posterior_covs, context, action, regrets)
    gain = ids_expected_entropy_gain(bandit_posterior_covs[action, :, :], context)
    return -1*regrets[action]^4 / gain
end


function ids_expected_entropy_gain(cov, context)
    return log(det(Matrix(1.0I, context_dim, context_dim) + cov * context * context' ./ obs_sd.^2))
end

# IDS
function ids_policy_0_5(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    regrets = ids_expected_regrets(context, bandit_posterior_means, bandit_posterior_covs, 1000)
    

    return findmax([ids_information_ratio_0_5(bandit_posterior_covs, context, act, regrets) for act=1:bandit_count])[2]
end
function ids_policy_4(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    regrets = ids_expected_regrets(context, bandit_posterior_means, bandit_posterior_covs, 1000)
    

    return findmax([ids_information_ratio_4(bandit_posterior_covs, context, act, regrets) for act=1:bandit_count])[2]
end
function ids_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    regrets = ids_expected_regrets(context, bandit_posterior_means, bandit_posterior_covs, 1000)
    

    return findmax([ids_information_ratio(bandit_posterior_covs, context, act, regrets) for act=1:bandit_count])[2]
end


function xgboost_cv(X, Y, nfold)
    best_loss = Inf
    best_param = []
    for num_round in 1:10
        for eta in [.1,.2,.3,.4,.5]
            for max_depth in 1:5
            
                cv_param = ["max_depth" => max_depth,
                         "eta" => eta,
                         "objective" => "reg:squarederror",
                         "verbosity" => 0]
                curr_loss = 0
                for ff in 0:(nfold-1)
                    X_train = X[(1:end) .% nfold .!= ff, :]
                    X_test = X[(1:end) .% nfold .== ff, :]
                    Y_train = Y[(1:end) .% nfold .!= ff]
                    Y_test = Y[(1:end) .% nfold .== ff]
                    #println(size(X_train))
                    #println(size(X_test))
                    #println(size(Y_train))
                    #println(size(Y_test))
                    local m
                    @suppress begin
                        m = xgboost(X_train, num_round, label = Y_train, param = cv_param, metrics = metrics)
                    end
                    Y_pred = XGBoost.predict(m, X_test)
                    #println(size(Y_pred))
                    #println(Y_pred[1])
                    curr_loss += mean((Y_pred .- Y_test) .^ 2)
                end
                #println(curr_loss)
                if curr_loss < best_loss
                    #println("yay")
                    best_loss = curr_loss
                    best_param = [num_round, cv_param]
                end
            end
        end
    end
    return xgboost(X, best_param[1], label = Y, param = best_param[2], metrics = metrics)
end
function ep_contextual_bandit_simulator(ep,action_function::exp4, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
        # track eta/gamma
        E_star_t = 0
        for t in 1:T
            context = generate_context(context_dim, context_mean, context_sd, context_constant)
            true_expected_rewards = true_bandit_param * context
            #true_expected_rewards = bandit_posterior_means * context
            action_probs_matrix = lin_get_action_probs(action_function.policy_list, ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
            action_probs = action_probs_matrix' * action_function.policy_probs
            action = sample(1:bandit_count, Weights(action_probs))
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

            E_star_t += sum(maximum(action_probs_matrix, dims = 1))
            eta = sqrt(log(length(action_function.policy_list)) / E_star_t)
            gamma = eta / 2

            ## UPDATE POLICY PROBS
            obs_weighted = obs / (action_probs[action] + gamma)
            policy_rewards = obs_weighted .* action_probs_matrix[:, action]

            action_function.policy_probs .*= exp.(eta .* policy_rewards)
            action_function.policy_probs ./= sum(action_function.policy_probs)
	        println("Ep: ", ep, " - ", t, " of ", T, " for ", String(Symbol(action_function)))
            flush(stdout)


        end
	return EPREWARDS, EPOPTREWARDS
end

function lin_get_action_probs(policy_list, ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    out = zeros(length(policy_list), bandit_count)
    for i in eachindex(policy_list)
        pol = policy_list[i]
        if pol == thompson_policy
            for j in 1:1000
                out[i, pol(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)] += 1
            end
            out[i, :] ./= 1000
        else
            out[i, pol(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)] += 1
        end
    end

    return out
end

