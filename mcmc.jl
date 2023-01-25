using SpecialFunctions
using Distributions

function logit_mcmc_update_check(theta_old, theta_new, X, y, prior_mean, prior_cov)

    dist = MvNormal(prior_mean, prior_cov)

    probs_old = 1 ./ (1 .+ exp.(-1 .* X * theta_old))
    probs_new = 1 ./ (1 .+ exp.(-1 .* X * theta_new))

    L_old = prod(probs_old .^ y) * prod((1 .- probs_old) .^ (1 .- y))
    L_new = prod(probs_new .^ y) * prod((1 .- probs_new) .^ (1 .- y))

    L_prior_old = pdf(dist, theta_old)
    L_prior_new = pdf(dist, theta_new)

    L_tot_old = L_old * L_prior_old
    L_tot_new = L_new * L_prior_new

    accept_prob = min(1, L_tot_new / L_tot_old)

    u = rand()

    if u < accept_prob
        return theta_new
    else
        return theta_old
    end



end
function logit_mcmc_update_check_vb(theta_old, theta_new, X, y, prior_mean, prior_cov, approx_post)

    dist = MvNormal(prior_mean, prior_cov)

    ll_old = sum(-log.(1 .+ exp.(X * theta_old)) .+ y .* (X * theta_old))
    ll_new = sum(-log.(1 .+ exp.(X * theta_new)) .+ y .* (X * theta_new))


    ll_prior_old = logpdf(dist, theta_old)
    ll_prior_new = logpdf(dist, theta_new)

    ll_tot_old = ll_old + ll_prior_old
    ll_tot_new = ll_new + ll_prior_new

    ll_prop_old = logpdf(approx_post, theta_old)
    ll_prop_new = logpdf(approx_post, theta_new)

    accept_prob = min(1, exp(ll_tot_new + ll_prop_old - ll_tot_old - ll_prop_new))

    u = rand()

    if u < accept_prob
        return theta_new, 1
    else
        return theta_old, 0
    end



end



function logit_mcmc_update(theta_old, X, y, prior_mean, prior_cov, proposal_sd)

    theta_new = theta_old .+ proposal_sd .* randn(length(theta_old))

    return logit_mcmc_update_check(theta_old, theta_new, X, y, prior_mean, prior_cov)

end

function sample_mcmc(X, y, prior_mean, prior_cov, proposal_sd, n_samp, n_burn)

    if (size(X)[1] == 0)
        return rand(MvNormal(prior_mean, prior_cov), n_samp)'
    end
    
    theta_curr = zeros(context_dim)

    THETA = zeros((n_samp+n_burn), context_dim)

    for i in 1:(n_samp+n_burn)

        theta_curr = logit_mcmc_update(theta_curr, X, y, prior_mean, prior_cov, proposal_sd)
        THETA[i, :] = theta_curr
    end

    return THETA[(n_burn+1):(n_burn+n_samp), :]
end
function sample_mcmc_vb(X, y, prior_mean, prior_cov, m_vb, C_vb, n_samp, n_burn, n_dup)

    if (size(X)[1] == 0)
        return rand(MvNormal(prior_mean, prior_cov), n_samp)'
    end
    approx_post = MvNormal(m_vb, C_vb)
    PROPOSALS = rand(approx_post, n_burn+n_samp*n_dup) 
    theta_curr = rand(approx_post)

    THETA = zeros(context_dim, (n_samp*n_dup+n_burn))
    ACC = 0
    for i in 1:(n_samp*n_dup+n_burn)

        theta_curr, acc = logit_mcmc_update_check_vb(theta_curr, PROPOSALS[:, i], X, y, prior_mean, prior_cov, approx_post)
        THETA[:, i] = theta_curr
        ACC = ACC * (i - 1) / i + acc / i
    end

    return THETA[:, (n_burn+1):n_dup:(n_burn+n_samp*n_dup)], ACC
end



# find beta prior parameters from normal parameters with f=mean and q=variance


function mcmc_bernoulli_ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon, global_bandit_param)
        
    
        bandit_posterior_means = zeros(bandit_count, context_dim)
        bandit_posterior_covs = zeros(bandit_count, context_dim, context_dim)
    	bandit_param = copy(global_bandit_param)
        true_bandit_param = copy(global_bandit_param)
        EPREWARDS = zeros(T)
	    EPOPTREWARDS = zeros(T)
        
        X_tot = zeros(T, context_dim)
        Y_tot = zeros(T)
        A_tot = zeros(T)

        for i in 1:bandit_count
            bandit_posterior_means[i, :] = repeat([bandit_prior_mean], context_dim)
            bandit_posterior_covs[i, :, :] = Diagonal(repeat([bandit_prior_sd^2], context_dim))
        end
        
        for t in 1:T
            context = generate_context(context_dim, context_mean, context_sd, context_constant)
            true_expected_rewards_logit = true_bandit_param * context
            true_expected_rewards = exp.(true_expected_rewards_logit) ./ (1 .+ exp.(true_expected_rewards_logit))
            #true_expected_rewards = bandit_posterior_means * context
            action = action_function(ep, t, T, bandit_count, context, X_tot, Y_tot, A_tot, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
            true_expected_reward = true_expected_rewards[action]
            EPREWARDS[t] = true_expected_reward
            EPOPTREWARDS[t] = maximum(true_expected_rewards)
            #obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:],context)) + true_expected_reward
            obs = 1 * (rand() < true_expected_reward)
            Y_tot[t] = obs
            A_tot[t] = action
            X_tot[t, :] = context
            

	        println("Ep: ", ep, " - ", t, " of ", T, " for ", String(Symbol(action_function)))
            flush(stdout)

        end
	return EPREWARDS, EPOPTREWARDS
end

function mcmc_bernoulli_contextual_bandit_simulator(action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

    REWARDS = zeros(T, n_episodes)
    OPTREWARDS = zeros(T, n_episodes)
    ep_count = 1
    

    ## USING BAYESIAN FORMULATION FOR FAIR COMPARISON
    #global_bandit_param = [1 0; 0 1; 2 -1]
    
    
    #threadreps = zeros(Threads.nthreads())


    for ep in 1:n_episodes
        global_bandit_param = rand(Normal(bandit_prior_mean, bandit_prior_sd), bandit_count, context_dim)
        EPREWARDS, EPOPTREWARDS = mcmc_bernoulli_ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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

function mcmc_bernoulli_val_better_grid_lambda_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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
                    rollout_value = bernoulli_val_better_rollout(ep, T-t+1, t, rollout_length, context, use_context, lambda_param, context_dim, context_mean,
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


function mcmc_bernoulli_val_greedy_thompson_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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

    
    policies = [mcmc_greedy_policy, mcmc_thompson_policy]
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
                for bandit in 1:bandit_count
                    #copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
                    #copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
                    samp = sample_mcmc(X[A .== bandit,:], y[A .== bandit], prior_mean, prior_cov, proposal_sd, 1, 100)
                    bandit_param[bandit,:] = samp
                end

                use_context = true
                
                rollout_value = mcmc_bernoulli_val_rollout(ep, policy, T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
                    context_sd, obs_sd, bandit_count, discount, X, y, A, bandit_param,
                    roll_true_expected_rewards)
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

    
    opt_act = opt_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    println("Optimal Action: ",opt_act)
    flush(stdout)

    return opt_act

end
function mcmc_bernoulli_val_greedy_thompson_ucb_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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
                
                rollout_value = bernoulli_val_rollout(ep, policy, T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
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
function mcmc_bernoulli_val_greedy_thompson_ucb_ids_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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

    
    policies = [greedy_policy, thompson_policy, glm_ucb_policy, ids_policy]
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
                
                rollout_value = bernoulli_val_rollout(ep, policy, T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
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


function mcmc_bernoulli_val_greedy_rollout_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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


        rollout_value = bernoulli_val_greedy_rollout(ep, T-t, rollout_length, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end


function mcmc_bernoulli_greedy_rollout_policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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

        rollout_value = bernoulli_greedy_rollout(ep, T-t, rollout_length, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, temp_post_covs, temp_post_means, bandit_param,
		        roll_context, roll_true_expected_rewards, roll_CovCon, roll_old_cov, roll_SigInvMu)
	    BANDIT_REWARDS = (BANDIT_REWARDS * (roll-1) + predictive_rewards[bandit] + discount * rollout_value) / roll

        end
        BANDIT_VALUES[bandit] = BANDIT_REWARDS
    end
    return findmax(BANDIT_VALUES)[2]
end
function mcmc_bernoulli_greedy_rollout(ep, T_remainder, rollout_length, lambda, context_dim, context_mean,
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


        context = generate_context(context_dim, context_mean, context_sd, context_constant)
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


function mcmc_bernoulli_val_greedy_rollout(ep, T_remainder, rollout_length, lambda, context_dim, context_mean,
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


        context = generate_context(context_dim, context_mean, context_sd, context_constant)



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
function mcmc_bernoulli_val_rollout(ep, policy, T_remainder, rollout_length, context, use_context, lambda, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, discount, X, y, A, bandit_param,
	true_expected_rewards)

    disc_reward = 0
    #fill!(context, 0.0)
    fill!(true_expected_rewards, 0.0)
    #fill!(CovCon, 0.0)
    #fill!(old_cov,0.0)
    #fill!(SigInvMu,0.0)
    
    truncation_length = T_remainder - min(T_remainder, rollout_length)
    
    t_curr = T - T_remainder + 1

    for t in 1:min(T_remainder, rollout_length)

        #context_seed = rand(1:context_dim)
        #fill!(context, zero(context[1]))
        #context[context_seed] = 1
        #mul!(true_expected_rewards, bandit_param, context)

        if t > 1 || !use_context
            context = generate_context(context_dim, context_mean, context_sd, context_constant)
        end

        action = policy(ep, t_curr, min(T_remainder, rollout_length), bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)


        # TRUE PARAM VERSION
        #true_expected_reward = true_expected_rewards[action]
        #disc_reward += true_expected_reward * discount^(t-1)
        #obs = randn() * obs_sd + true_expected_reward
        
        true_expected_reward_logit = dot(bandit_param[action,:], context)
        true_expected_reward = exp(true_expected_reward_logit) / (1 + exp(true_expected_reward_logit))
        disc_reward += true_expected_reward * discount^(t-1)

        obs = 1 * (rand() < true_expected_reward)
        
        y[t_curr + t - 1] = obs
        A[t_curr + t - 1] = action
        X[(t_curr + t - 1), :] = context
        
        # UNKNOWN PARAM VERSION
    end


    if truncation_length > 0
        
        trunc_means = zeros(bandit_count, context_dim)
        trunc_covs = zeros(bandit_count, context_dim, context_dim)
        
        for k in 1:bandit_count
            samp = sample_mcmc(X[A .== k,:], y[A .== k], prior_mean, prior_cov, proposal_sd, 1000, 100)
            trunc_means[k, :] = mean(samp, dims = 1)
            trunc_covs[k, :, :] = cov(samp)
        end
        #### SWITCHED TO TRUE VALUE NEURAL NETWORK AS TEST
        
        #input_vec = Vector(vec(bandit_param'))
        #append!(input_vec, Vector(vec(bandit_posterior_means')))
        input_vec = Vector(vec(trunc_means'))
        append!(input_vec, vcat([upper_triangular_vec(trunc_covs[a, :, :]) for a in 1:bandit_count]...))
        
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

function mcmc_bernoulli_val_better_rollout(ep, T_remainder, curr_t, rollout_length, context, use_context, lambda_param, context_dim, context_mean,
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
            context = generate_context(context_dim, context_mean, context_sd, context_constant)
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



# Greedy
function mcmc_greedy_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    
    bandit_expected_rewards = zeros(bandit_count)
    for k in 1:bandit_count
        samp = sample_mcmc(X[A .== k,:], y[A .== k], prior_mean, prior_cov, proposal_sd, 1000, 100)
        bandit_expected_rewards[k] = mean(samp * context)
    end
    
    val, action = findmax(bandit_expected_rewards)
    return action
end

# Epsilon Greedy
function mcmc_epsilon_greedy_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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
function mcmc_bayes_ucb_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    val, action = findmax(bandit_posterior_means * context)
    reward_means = bandit_posterior_means * context
    reward_sds = sqrt.(vec([context' * (@view bandit_posterior_covs[i,:,:]) * context for i=1:bandit_count]))
    ucbs = quantile.(Normal.(reward_means, reward_sds), 1-1/t)
    return findmax(ucbs)[2]
end

# Bayes UCB
function mcmc_glm_ucb_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    #val, action = findmax(bandit_posterior_means * context)
    reward_means = zeros(bandit_count)
    reward_sds = zeros(bandit_count)
    for k in 1:bandit_count
        samp = sample_mcmc(X[A .== k,:], y[A .== k], prior_mean, prior_cov, proposal_sd, 1000, 100)
        samp_means = samp * context
        reward_means[k] = mean(samp_means)
        reward_sds[k] = sqrt(var(samp_means))
    end
    ucbs = reward_means .+ max(1,sqrt(log(t))) .* reward_sds
    return findmax(ucbs)[2]
end

# Thompson
function mcmc_thompson_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    thompson_means = zeros(bandit_count)
    for k in 1:bandit_count
        samp = sample_mcmc(X[A .== k,:], y[A .== k], prior_mean, prior_cov, proposal_sd, 1, 100)
        thompson_means[k] = dot(samp, context)
    end


    return findmax(thompson_means)[2]
end




## IDS POLICY

function mcmc_ids_expected_regrets(context, draws, niter)

    reward_draws = zeros(bandit_count, niter)
    
    for b in 1:bandit_count
        for i in 1:niter
            reward_draws[b, i] = dot(context, draws[b, i, :])
        end
    end
    
    mean_rewards = dropdims(mean(reward_draws, dims = 2), dims = 2)

    mean_max_reward = 0
    for i in 1:niter
        mean_max_reward += findmax(reward_draws[:, i])[1] / niter
    end
    
    res = max.(0, mean_max_reward .- mean_rewards)

    return res
end



function mcmc_ids_information_ratio(bandit_posterior_covs, context, action, regrets)
    gain = ids_expected_entropy_gain(bandit_posterior_covs[action, :, :], context)
    return -1*regrets[action]^2 / gain
end



# IDS
function mcmc_ids_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    n_burn = 100
    n_samp = 1000
    SAMP = zeros(bandit_count, n_samp, context_dim)
    ENTDIFFS = zeros(bandit_count)
    for k in 1:bandit_count
        samp = sample_mcmc(X[A .== k,:], y[A .== k], prior_mean, prior_cov, proposal_sd, 1000, 100)
        SAMP[k, :, :] = samp
        erls = samp * context
        probs = exp.(erls) ./ (1 .+ exp.(erls))
        obs = 1 .* (rand(n_samp) .< probs)
        fraction1 = length(obs[obs .== 1]) / n_samp
        cov_prior = cov(samp)

        rat1 = n_samp / sum(probs)
        mean1 = mean(samp .* probs, dims = 1) .* rat1
        
        samp_norm1 = samp .* sqrt.(probs)
        mean_norm1 = mean(samp_norm1, dims = 1)
        cov1 = (cov(samp_norm1) .+ (mean_norm1' * mean_norm1)) .* rat1 .- (mean1' * mean1)
        
        rat2 = n_samp / sum(1 .- probs)
        mean2 = mean(samp .* (1 .- probs), dims = 1) .* rat2
        
        samp_norm2 = samp .* sqrt.(1 .- probs)
        mean_norm2 = mean(samp_norm2, dims = 1)
        cov2 = (cov(samp_norm2) .+ (mean_norm2' * mean_norm2)) .* rat2 .- (mean2' * mean2)

        entdiff1 = .5 * log(det(cholesky(cov1) \ cov_prior))
        entdiff2 = .5 * log(det(cholesky(cov2) \ cov_prior))
        entdiff = fraction1 * entdiff1  + (1 - fraction1) * entdiff2
        ENTDIFFS[k] = entdiff
    
    end

    regs = mcmc_ids_expected_regrets(context, SAMP, n_samp)
    
    return findmax(-1 .* regs .^ 2 ./ ENTDIFFS)[2]
end



