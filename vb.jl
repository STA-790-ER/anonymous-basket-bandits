using SpecialFunctions
using Distributions


function expit(x)
    return 1 ./ (1 .+ exp(-1 .* x))
end

function lambda_jj(xi)
    return (.5 .- expit(x)) ./ (2 .* x)
end

function update_xi(X, mu, Sigma)
    return [sqrt(sum(x .* (Sigma * x)) + sum(x .* mu)^2) for x in eachrow(X)]
end

function update_Sigma(X, xi, Sigma)
    return inv(inv(Sigma) + 2 .* (X' * (X .* xi)))
end

function update_mu(X, Y, mu, Sigma_pre, Sigma_post)
    return Sigma_post * (Sigma_pre \ mu + X' * (Y .- .5))
end

function jj_posterior_batch(X, Y, mu, Sigma, tol)

    if length(Y) == 0
        return mu, Sigma
    end
    err = tol + 1
    mu_old = mu
    Sigma_old = Sigma

    while err > tol
        xi = update_xi(X, mu_old, Sigma_old)
        Sigma_new = update_Sigma(X, xi, prior_cov)
        mu_new = update_mu(X, Y, prior_mean, prior_cov, Sigma_new)
        err = sqrt(sum((mu_new .- mu_old).^2) + sum((Sigma_new .- Sigma_old).^2))
        mu_old = mu_new
        Sigma_old = Sigma_new
    end
    Sigma_old = (Sigma_old .+ Sigma_old') ./ 2
    return mu_old, Sigma_old
end


function vb_bernoulli_ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
            context = randn(context_dim) * context_sd .+ context_mean
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

function vb_bernoulli_contextual_bandit_simulator(action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, bandit_prior_mean, bandit_prior_sd, discount, epsilon)

    REWARDS = zeros(T, n_episodes)
    OPTREWARDS = zeros(T, n_episodes)
    ep_count = 1
    

    ## USING BAYESIAN FORMULATION FOR FAIR COMPARISON
    #global_bandit_param = [1 0; 0 1; 2 -1]
    
    
    #threadreps = zeros(Threads.nthreads())


    for ep in 1:n_episodes
        global_bandit_param = rand(Normal(bandit_prior_mean, bandit_prior_sd), bandit_count, context_dim)
        EPREWARDS, EPOPTREWARDS = vb_bernoulli_ep_contextual_bandit_simulator(ep,action_function, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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



function vb_bernoulli_val_greedy_thompson_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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

    
    policies = [vb_greedy_policy, vb_thompson_policy]
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
                    m, C = jj_posterior_batch(X[A .== bandit,:], y[A .== bandit], prior_mean, prior_cov, .001)
                    bandit_param[bandit,:] = rand(MvNormal(m, C))
                end

                use_context = true
                
                rollout_value = vb_bernoulli_val_rollout(ep, policy, T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
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
function vb_bernoulli_val_greedy_thompson_ucb_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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

    
    policies = [vb_greedy_policy, vb_thompson_policy, vb_glm_ucb_policy]
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
                    m, C = jj_posterior_batch(X[A .== bandit,:], y[A .== bandit], prior_mean, prior_cov, .001)
                    bandit_param[bandit,:] = rand(MvNormal(m, C))
                end

                use_context = true
                
                rollout_value = vb_bernoulli_val_rollout(ep, policy, T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
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

function vb_bernoulli_val_greedy_thompson_ucb_ids_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)


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

    
    policies = [vb_greedy_policy, vb_thompson_policy, vb_glm_ucb_policy, vb_ids_policy]
    policy_values = []
    action_samps = [pol(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim) for pol in policies]
    if length(unique(action_samps)) == 1
        println("Agreement Action: ", action_samps[1])
        flush(stdout)
        return action_samps[1]
    end
    
    println("Context Start: ", context)
    flush(stdout)

    lambda = [0, 0]
            
    ms_Cs = [jj_posterior_batch(X[A .== bandit,:], y[A .== bandit], prior_mean, prior_cov, .000001) for bandit in 1:bandit_count]
    rollout_params = zeros(context_dim, n_opt_rollouts, bandit_count)
    if true
        for k in 1:bandit_count
            if sum(A .== k) == 0
                rollout_params[:, :, k] = rand(MvNormal(ms_Cs[k][1], ms_Cs[k][2]), n_opt_rollouts)
            else
                rollout_params[:, :, k], ACC = sample_mcmc_vb(X[A .== k, :], y[A .== k, :], prior_mean, prior_cov, ms_Cs[k][1], ms_Cs[k][2], n_opt_rollouts, 200, 2)
                println("Acceptance Prob: ", ACC)
                flush(stdout)
            end
        end
    else
        for k in 1:bandit_count
            rollout_params[:, :, k] = rand(MvNormal(ms_Cs[k][1], ms_Cs[k][2]), n_opt_rollouts)
        end
    end

    for policy in policies  

            MEAN_REWARD = 0

            for roll in 1:n_opt_rollouts
            
                #copy!(temp_post_means, bandit_posterior_means)
                #copy!(temp_post_covs, bandit_posterior_covs)
                #bandit_param = zeros(bandit_count, context_dim)
                #for bandit in 1:bandit_count
                    #copy!(temp_bandit_mean, (@view bandit_posterior_means[bandit,:]))
                    #copy!(temp_bandit_cov, (@view bandit_posterior_covs[bandit,:,:]))
                #   bandit_param[bandit,:] = rand(MvNormal(ms_Cs[bandit][1], ms_Cs[bandit][2]))
                #end
                bandit_param = rollout_params[:, roll, :]'
                use_context = true
                
                rollout_value = vb_bernoulli_val_rollout(ep, policy, T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
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

    # Unnecessary given samples above 
    opt_act = opt_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

    println("Optimal Action: ",opt_act)
    flush(stdout)

    return action_samps[opt_index]

end

function vb_bernoulli_val_rollout(ep, policy, T_remainder, rollout_length, context, use_context, lambda, context_dim, context_mean,
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
            context = randn(context_dim) * context_sd .+ context_mean
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
        if true 
            t_trunc = t_curr + min(T_remainder, rollout_length)
            reg_est = 0

            ms_Cs = [jj_posterior_batch(X[A .== k,:], y[A .== k], prior_mean, prior_cov, .005) for k in 1:bandit_count]
            for n in 1:2
                context = randn(context_dim) .* context_sd .+ context_mean
                true_expected_reward_logits = bandit_param * context
                true_expected_rewards = exp.(true_expected_reward_logits) ./ (1 .+ exp.(true_expected_reward_logits))
                true_regs = findmax(true_expected_rewards)[1] .- true_expected_rewards
                #point_samps = 1 .* (rand(bandit_count, 500) .< true_expected_rewards)
                if String(Symbol(policy)) == "vb_thompson_policy"        
                    for m in 1:500
                        action = vb_thompson_policy_given_mC(ep, t_trunc, min(T_remainder, rollout_length), bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, ms_Cs)
                        reg_est *= ((n-1)*500 + m - 1) / ((n-1)*500 + m)
                        reg_est += (true_regs[action]) / ((n-1)*500 + m)
                    end
                else
                    trunc_policy = getfield(Main, Symbol(String(Symbol(policy)) * "_given_mC"))
                    action = trunc_policy(ep, t_trunc, min(T_remainder, rollout_length), bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, ms_Cs)
                    reg_est *= (n - 1) / n
                    reg_est += true_regs[action] / n
                    #for m in 1:500
                    #    reg_est *= ((n-1)*500 + m - 1) / ((n-1)*500 + m)
                    #    reg_est += (findmax(point_samps[:, m])[1] - point_samps[action, m]) / ((n-1)*500 + m)
                    #end
                end
                #reg_est *= (n-1) / n
                #reg_est += (findmax(point_samps)[1] - point_samps[action]) / n
            end

            #disc_reward -= .9 * discount^t_trunc * sum(discount^(t - t_trunc) * reg_est * (T - t) / (T - t_trunc) for t in t_trunc:T)
            disc_reward -= .9 * discount^min(T_remainder, rollout_length) * sum(discount^(t - t_trunc) * reg_est * (1 + T - t) / (1 + T - t_trunc) for t in t_trunc:T)
        else
            trunc_means = zeros(bandit_count, context_dim)
            trunc_covs = zeros(bandit_count, context_dim, context_dim)
            
            for k in 1:bandit_count
                m, C = jj_posterior_batch(X[A .== k,:], y[A .== k], prior_mean, prior_cov, .001)
                trunc_means[k, :] = m
                trunc_covs[k, :, :] = C
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
    end

    return disc_reward
    end

# Greedy
function vb_greedy_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    
    bandit_expected_rewards = zeros(bandit_count)
    for k in 1:bandit_count
        m, C = jj_posterior_batch(X[A .== k, :], y[A .== k], prior_mean, prior_cov, .005)
        bandit_expected_rewards[k] = sum(m .* context)
    end
    
    val, action = findmax(bandit_expected_rewards)
    return action
end
function vb_greedy_policy_given_mC(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, ms_Cs)
    
    bandit_expected_rewards = zeros(bandit_count)
    for k in 1:bandit_count
        bandit_expected_rewards[k] = dot(ms_Cs[k][1], context)
    end
    
    val, action = findmax(bandit_expected_rewards)
    return action
end

# Epsilon Greedy
function vb_epsilon_greedy_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

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
function vb_bayes_ucb_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    val, action = findmax(bandit_posterior_means * context)
    reward_means = bandit_posterior_means * context
    reward_sds = sqrt.(vec([context' * (@view bandit_posterior_covs[i,:,:]) * context for i=1:bandit_count]))
    ucbs = quantile.(Normal.(reward_means, reward_sds), 1-1/t)
    return findmax(ucbs)[2]
end

# Bayes UCB
function vb_glm_ucb_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    #val, action = findmax(bandit_posterior_means * context)
    reward_means = zeros(bandit_count)
    reward_sds = zeros(bandit_count)
    for k in 1:bandit_count
        m, C = jj_posterior_batch(X[A .== k,:], y[A .== k], prior_mean, prior_cov, .005)
        reward_means[k] = sum(m .* context)
        reward_sds[k] = sqrt(dot(context, C, context))
    end
    ucbs = reward_means .+ max(1,sqrt(log(t))) .* reward_sds
    return findmax(ucbs)[2]
end
function vb_glm_ucb_policy_given_mC(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, ms_Cs)
    #val, action = findmax(bandit_posterior_means * context)
    reward_means = zeros(bandit_count)
    reward_sds = zeros(bandit_count)
    for k in 1:bandit_count
        reward_means[k] = dot(ms_Cs[k][1], context)
        reward_sds[k] = sqrt(dot(context, ms_Cs[k][2], context))
    end
    ucbs = reward_means .+ max(1,sqrt(log(t))) .* reward_sds
    return findmax(ucbs)[2]
end

# Thompson
function vb_thompson_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    thompson_means = zeros(bandit_count)
    for k in 1:bandit_count
        m, C = jj_posterior_batch(X[A .== k,:], y[A .== k], prior_mean, prior_cov, .005)
        thompson_means[k] = dot(m, context) + sqrt(dot(context, C, context)) * randn()
    end


    return findmax(thompson_means)[2]
end
function vb_thompson_policy_given_mC(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, ms_Cs)
    thompson_means = zeros(bandit_count)
    for k in 1:bandit_count
        #m, C = jj_posterior_batch(X[A .== k,:], y[A .== k], prior_mean, prior_cov, .001)
        thompson_means[k] = dot(ms_Cs[k][1], context) + sqrt(dot(context, ms_Cs[k][2], context)) * randn()
    end


    return findmax(thompson_means)[2]
end




## IDS POLICY
# IDS
function vb_ids_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    
    vb_post_means = zeros(bandit_count, context_dim)
    vb_post_covs = zeros(bandit_count, context_dim, context_dim)
    
    for k in 1:bandit_count
        m, C = jj_posterior_batch(X[A .== k,:], y[A .== k], prior_mean, prior_cov, .005)
        vb_post_means[k, :] = m
        vb_post_covs[k, :, :] = C
    end
    
    return ids_policy(ep, t, T, bandit_count, context, vb_post_means, vb_post_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

end
function vb_ids_policy_given_mC(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, ms_Cs)
    
    vb_post_means = zeros(bandit_count, context_dim)
    vb_post_covs = zeros(bandit_count, context_dim, context_dim)
    
    for k in 1:bandit_count
        vb_post_means[k, :] = ms_Cs[k][1]
        vb_post_covs[k, :, :] = ms_Cs[k][2]
    end
    
    return ids_policy(ep, t, T, bandit_count, context, vb_post_means, vb_post_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

end



