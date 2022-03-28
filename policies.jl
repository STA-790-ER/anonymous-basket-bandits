
# Greedy
function greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    val, action = findmax(bandit_posterior_means * context)
    return action
end

# Epsilon Greedy
function epsilon_greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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

function greedy_rollout_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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


function val_greedy_rollout_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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


function val_better_lambda_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
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


function val_better_opt_lambda_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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

function val_better_grid_lambda_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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


function val_greedy_thompson_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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

function val_epsilon_greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, n_eps)

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

function vegp2(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_epsilon_greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 2))

end
function vegp4(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_epsilon_greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 4))

end
function vegp8(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_epsilon_greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 8))

end
function vegp16(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_epsilon_greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 16))

end
function vegp32(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_epsilon_greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 32))

end
function vegp64(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_epsilon_greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 64))

end

function val_fractional_thompson_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, n_thom)

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

function vftp2(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_fractional_thompson_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 2))

end
function vftp4(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_fractional_thompson_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 4))

end
function vftp8(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_fractional_thompson_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 8))

end
function vftp16(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_fractional_thompson_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 16))

end
function vftp32(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_fractional_thompson_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 32))

end
function vftp64(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return(val_fractional_thompson_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, 64))

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
