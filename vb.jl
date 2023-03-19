using SpecialFunctions
using Distributions


function expit(x)
    return 1 ./ (1 .+ exp.(-1 .* x))
end

function lambda_jj(xi)
    return (expit(xi) .- .5) ./ (2 .* xi)
end

function update_xi(X, mu, Sigma)
    return [sqrt(sum(x .* (Sigma * x)) + sum(x .* mu)^2) for x in eachrow(X)]
end

function update_Sigma(X, xi, Sigma)
    return inv(inv(Sigma) + 2 .* (X' * (X .* abs.(lambda_jj(xi)))))
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
        err = sqrt((sum((mu_new .- mu_old).^2) + sum((Sigma_new .- Sigma_old).^2)) / (length(mu_old) + length(Sigma_old)))
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
    policy_stds = []
    action_samps = [pol(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim) for pol in policies]
    if length(unique(action_samps)) == 1
        println("Agreement Action: ", action_samps[1])
        flush(stdout)
        return action_samps[1]
    end
    policy_count = length(policies)    
    println("Context Start: ", context)
    flush(stdout)

    lambda = [0, 0]
            
    ms_Cs = [jj_posterior_batch(X[A .== bandit,:], y[A .== bandit], prior_mean, prior_cov, vb_rollout_tol) for bandit in 1:bandit_count]
    rollout_params = zeros(context_dim, n_opt_rollouts, bandit_count)
    if true
        for k in 1:bandit_count
            if sum(A .== k) == 0
                rollout_params[:, :, k] = rand(MvNormal(ms_Cs[k][1], ms_Cs[k][2]), n_opt_rollouts)
            else
                rollout_params[:, :, k], ACC = sample_mcmc_vb(X[A .== k, :], y[A .== k, :], prior_mean, prior_cov, ms_Cs[k][1], ms_Cs[k][2], n_opt_rollouts, n_burn, n_dup)
                println("Acceptance Prob: ", ACC)
                flush(stdout)
            end
        end
    else
        for k in 1:bandit_count
            rollout_params[:, :, k] = rand(MvNormal(ms_Cs[k][1], ms_Cs[k][2]), n_opt_rollouts)
        end
    end

    REWARD_SAMPS = zeros(n_opt_rollouts * vb_rollout_mult, length(policies))
    halt_vec = repeat([false], length(policies))
    rollout_counts = repeat([0], length(policies))
    same_break = false
    for roll in 1:(n_opt_rollouts*vb_rollout_mult)

        MEAN_REWARD = 0
        for pol_ind in 1:length(policies)
           
            if halt_vec[pol_ind]
                continue
            end

            rollout_counts[pol_ind] += 1
            bandit_param = rollout_params[:, 1 + ((roll - 1) % n_opt_rollouts), :]'
            use_context = true
            
            rollout_value = vb_bernoulli_val_rollout(ep, policies[pol_ind], T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, X, y, A, bandit_param,
                roll_true_expected_rewards, ms_Cs)
            A[t:T] .= 0
            REWARD_SAMPS[roll, pol_ind] = rollout_value
            #MEAN_REWARD = ((roll - 1) * MEAN_REWARD + rollout_value) / roll

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
    opt_act = opt_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return opt_act

    ACTION_MEAN_REWARDS = zeros(bandit_count)
    ACTION_STD_REWARDS = zeros(bandit_count)
    REWARD_SAMPS = zeros(n_opt_rollouts * vb_rollout_mult, bandit_count)
    halt_vec = repeat([false], bandit_count)
    rollout_counts = repeat([0], bandit_count)
    for roll in 1:(n_opt_rollouts*vb_rollout_mult)
        bandit_param = rollout_params[:, 1 + ((roll - 1) % n_opt_rollouts), :]'
        for action in 1:bandit_count 
            
            if halt_vec[action]
                continue
            end

            rollout_counts[action] += 1
            use_context = false
            true_expected_reward_logit = dot(bandit_param[action,:], context)
            true_expected_reward = exp(true_expected_reward_logit) / (1 + exp(true_expected_reward_logit))

            obs = 1 * (rand() < true_expected_reward)
            
            y[t] = obs
            A[t] = action
            X[t, :] = context
            
            rollout_value = vb_bernoulli_val_rollout(ep, opt_policy, T-t, rollout_length-1, context, use_context, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, X, y, A, bandit_param,
                roll_true_expected_rewards, ms_Cs)
            A[t:T] .= 0
            
            REWARD_SAMPS[roll, action] = true_expected_reward + discount * rollout_value
        end
        
        #if roll % 100 == 0
        #    continue_inds = findall(halt_vec .== false)
        #    action_means = [mean(REWARD_SAMPS[1:roll, p]) for p in continue_inds]
        #    action_stds = [std(REWARD_SAMPS[1:roll, p]) / sqrt(roll) for p in continue_inds]
        #    max_mean, max_ind = findmax(action_means)
        #    diff_means = max_mean .- action_means
        #    diff_stds = sqrt.(action_stds[max_ind]^2 .+ action_stds.^2)
        #    action_expected_regret_proportions = (diff_means .* cdf.(Normal(), -diff_means ./ diff_stds) .- diff_stds .* pdf.(Normal(), -diff_means ./ diff_stds)) ./ max_mean
        #    action_pol_diff_means = opt_mean .- action_means
        #    action_pol_diff_stds = sqrt.(opt_std^2 .+ action_stds.^2)
        #    action_pol_expected_regret_proportions = (action_pol_diff_means .* cdf.(Normal(), -action_pol_diff_means ./ action_pol_diff_stds) .- action_pol_diff_stds .* pdf.(Normal(), -action_pol_diff_means ./ action_pol_diff_stds)) ./ opt_mean
        #    halt_vec[continue_inds] = (abs.(action_expected_regret_proportions) .< expected_regret_thresh) .| ((abs.(action_pol_expected_regret_proportions) .< expected_regret_thresh) .& (action_pol_diff_means .> 0))
        #    halt_vec[continue_inds[max_ind]] = (abs(action_pol_expected_regret_proportions[max_ind]) < expected_regret_thresh) & (action_pol_diff_means[max_ind] > 0)
        #    if (sum(halt_vec .== false) == 0)
        #        halt_vec[continue_inds[max_ind]] = false
        #        break
        #    end
        #    halt_vec[continue_inds[max_ind]] = false
        #end
        
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

    if opt_mean >= opt_action_mean
        opt_act = opt_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
        println("Default to Policy, Action: ", opt_act)
        flush(stdout)
        return opt_act
    end

    m = opt_action_mean - opt_mean
    s = sqrt(opt_std^2 + opt_action_std^2)
    erp = (m * cdf(Normal(), -m/s) - s * pdf(Normal(), -m/s)) / opt_action_mean
    if abs(erp) < expected_regret_thresh
        println("Policy Improvement, Action: ", opt_action_index)
        flush(stdout)
        return opt_action_index
    else
        opt_act = opt_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
        println("Insufficient Improvement, Action: ", opt_act)
        flush(stdout)
        return opt_act
    end

    # END OPTIMIZATION OF LAMBDA

    # Unnecessary given samples above 
    #opt_act = opt_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    #opt_act = findmax(ACTION_MEAN_REWARDS)[2]
    #println("Optimal Action: ", opt_act)
    #flush(stdout)

    #return opt_act

end
function vb_bernoulli_val_greedy_thompson_ucb_ids_q_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)


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

    
    policies = [vb_greedy_policy, vb_thompson_policy, vb_glm_ucb_policy, vb_ids_policy, vb_ids_4_policy]
    policy_values = []
    policy_stds = []
    action_samps = [pol(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim) for pol in policies]
    #if length(unique(action_samps)) == 1
    #    println("Agreement Action: ", action_samps[1])
    #    flush(stdout)
    #    return action_samps[1]
    #end
    policy_count = length(policies)    
    println("Context Start: ", context)
    flush(stdout)

    lambda = [0, 0]
            
    ms_Cs = [jj_posterior_batch(X[A .== bandit,:], y[A .== bandit], prior_mean, prior_cov, vb_rollout_tol) for bandit in 1:bandit_count]
    rollout_params = zeros(context_dim, n_opt_rollouts, bandit_count)
    if true
        for k in 1:bandit_count
            if sum(A .== k) == 0
                rollout_params[:, :, k] = rand(MvNormal(ms_Cs[k][1], ms_Cs[k][2]), n_opt_rollouts)
            else
                rollout_params[:, :, k], ACC = sample_mcmc_vb(X[A .== k, :], y[A .== k, :], prior_mean, prior_cov, ms_Cs[k][1], ms_Cs[k][2], n_opt_rollouts, n_burn, n_dup)
                println("Acceptance Prob: ", ACC)
                flush(stdout)
            end
        end
    else
        for k in 1:bandit_count
            rollout_params[:, :, k] = rand(MvNormal(ms_Cs[k][1], ms_Cs[k][2]), n_opt_rollouts)
        end
    end

    REWARD_SAMPS = zeros(n_opt_rollouts * vb_rollout_mult, length(policies))
    halt_vec = repeat([false], length(policies))
    rollout_counts = repeat([0], length(policies))
    for roll in 1:(n_opt_rollouts*vb_rollout_mult)

        MEAN_REWARD = 0
        for pol_ind in 1:length(policies)
           
            if halt_vec[pol_ind]
                continue
            end

            rollout_counts[pol_ind] += 1
            bandit_param = rollout_params[:, 1 + ((roll - 1) % n_opt_rollouts), :]'
            use_context = true
            
            rollout_value = vb_bernoulli_val_rollout(ep, policies[pol_ind], T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, X, y, A, bandit_param,
                roll_true_expected_rewards, ms_Cs)
            A[t:T] .= 0
            REWARD_SAMPS[roll, pol_ind] = rollout_value
            #MEAN_REWARD = ((roll - 1) * MEAN_REWARD + rollout_value) / roll

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
    REWARD_SAMPS = zeros(n_opt_rollouts * vb_rollout_mult, bandit_count)
    halt_vec = repeat([false], bandit_count)
    rollout_counts = repeat([0], bandit_count)
    for roll in 1:(n_opt_rollouts*vb_rollout_mult)
        bandit_param = rollout_params[:, 1 + ((roll - 1) % n_opt_rollouts), :]'
        for action in 1:bandit_count 
            
            if halt_vec[action]
                continue
            end

            rollout_counts[action] += 1
            use_context = false
            true_expected_reward_logit = dot(bandit_param[action,:], context)
            true_expected_reward = exp(true_expected_reward_logit) / (1 + exp(true_expected_reward_logit))

            obs = 1 * (rand() < true_expected_reward)
            
            y[t] = obs
            A[t] = action
            X[t, :] = context
            
            rollout_value = vb_bernoulli_val_rollout(ep, opt_policy, T-t, rollout_length-1, context, use_context, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, X, y, A, bandit_param,
                roll_true_expected_rewards, ms_Cs)
            A[t:T] .= 0
            
            REWARD_SAMPS[roll, action] = true_expected_reward + discount * rollout_value
        end
        
        #if roll % 100 == 0
        #    continue_inds = findall(halt_vec .== false)
        #    action_means = [mean(REWARD_SAMPS[1:roll, p]) for p in continue_inds]
        #    action_stds = [std(REWARD_SAMPS[1:roll, p]) / sqrt(roll) for p in continue_inds]
        #    max_mean, max_ind = findmax(action_means)
        #    diff_means = max_mean .- action_means
        #    diff_stds = sqrt.(action_stds[max_ind]^2 .+ action_stds.^2)
        #    action_expected_regret_proportions = (diff_means .* cdf.(Normal(), -diff_means ./ diff_stds) .- diff_stds .* pdf.(Normal(), -diff_means ./ diff_stds)) ./ max_mean
        #    action_pol_diff_means = opt_mean .- action_means
        #    action_pol_diff_stds = sqrt.(opt_std^2 .+ action_stds.^2)
        #    action_pol_expected_regret_proportions = (action_pol_diff_means .* cdf.(Normal(), -action_pol_diff_means ./ action_pol_diff_stds) .- action_pol_diff_stds .* pdf.(Normal(), -action_pol_diff_means ./ action_pol_diff_stds)) ./ opt_mean
        #    halt_vec[continue_inds] = (abs.(action_expected_regret_proportions) .< expected_regret_thresh) .| ((abs.(action_pol_expected_regret_proportions) .< expected_regret_thresh) .& (action_pol_diff_means .> 0))
        #    halt_vec[continue_inds[max_ind]] = (abs(action_pol_expected_regret_proportions[max_ind]) < expected_regret_thresh) & (action_pol_diff_means[max_ind] > 0)
        #    if (sum(halt_vec .== false) == 0)
        #        halt_vec[continue_inds[max_ind]] = false
        #        break
        #    end
        #    halt_vec[continue_inds[max_ind]] = false
        #end
        
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

    if opt_mean >= opt_action_mean
        opt_act = opt_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
        println("Default to Policy, Action: ", opt_act)
        flush(stdout)
        return opt_act
    end

    m = opt_action_mean - opt_mean
    s = sqrt(opt_std^2 + opt_action_std^2)
    erp = (m * cdf(Normal(), -m/s) - s * pdf(Normal(), -m/s)) / opt_action_mean
    if abs(erp) < expected_regret_thresh
        println("Policy Improvement, Action: ", opt_action_index)
        flush(stdout)
        return opt_action_index
    else
        opt_act = opt_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
        println("Insufficient Improvement, Action: ", opt_act)
        flush(stdout)
        return opt_act
    end

    # END OPTIMIZATION OF LAMBDA

    # Unnecessary given samples above 
    #opt_act = opt_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    #opt_act = findmax(ACTION_MEAN_REWARDS)[2]
    #println("Optimal Action: ", opt_act)
    #flush(stdout)

    #return opt_act

end
function dp_vb_bernoulli_val_greedy_thompson_ucb_ids_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)


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
    policy_stds = []
    action_samps = [pol(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim) for pol in policies]
    #if length(unique(action_samps)) == 1
    #    println("Agreement Action: ", action_samps[1])
    #    flush(stdout)
    #    return action_samps[1]
    #end
    policy_count = length(policies)    
    println("Context Start: ", context)
    flush(stdout)

    lambda = [0, 0]
            
    ms_Cs = [jj_posterior_batch(X[A .== bandit,:], y[A .== bandit], prior_mean, prior_cov, vb_rollout_tol) for bandit in 1:bandit_count]
    rollout_params = zeros(context_dim, n_opt_rollouts, bandit_count)
    if true
        for k in 1:bandit_count
            if sum(A .== k) == 0
                rollout_params[:, :, k] = rand(MvNormal(ms_Cs[k][1], ms_Cs[k][2]), n_opt_rollouts)
            else
                rollout_params[:, :, k], ACC = sample_mcmc_vb(X[A .== k, :], y[A .== k, :], prior_mean, prior_cov, ms_Cs[k][1], ms_Cs[k][2], n_opt_rollouts, n_burn, n_dup)
                println("Acceptance Prob: ", ACC)
                flush(stdout)
            end
        end
    else
        for k in 1:bandit_count
            rollout_params[:, :, k] = rand(MvNormal(ms_Cs[k][1], ms_Cs[k][2]), n_opt_rollouts)
        end
    end

    REWARD_SAMPS = zeros(n_opt_rollouts * vb_rollout_mult, length(policies))
    halt_vec = repeat([false], length(policies))
    rollout_counts = repeat([0], length(policies))
    same_break = false
    use_context = true
    for roll in 1:(n_opt_rollouts*vb_rollout_mult)

        MEAN_REWARD = 0
        for pol_ind in 1:length(policies)
           
            if halt_vec[pol_ind]
                continue
            end

            rollout_counts[pol_ind] += 1
            bandit_param = rollout_params[:, 1 + ((roll - 1) % n_opt_rollouts), :]'
            
            rollout_value = vb_bernoulli_val_rollout(ep, policies[pol_ind], T-t+1, rollout_length, context, use_context, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, X, y, A, bandit_param,
                roll_true_expected_rewards, ms_Cs)
            A[t:T] .= 0
            REWARD_SAMPS[roll, pol_ind] = rollout_value
            #MEAN_REWARD = ((roll - 1) * MEAN_REWARD + rollout_value) / roll

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
            #    println("REMAINING POLICIES HAVE SAME ACTION")
            #    same_break = true
            #end
        end
        
        if same_break
            break
        end

        if sum(halt_vec .== false) == 1
            break
        end

        
    end
    rollout_params = nothing
    #REWARD_SAMPS = nothing
    continue_inds = findall(halt_vec .== false)
    
    policy_means = [mean(REWARD_SAMPS[1:rollout_counts[p], p]) for p in 1:policy_count]
    policy_stds = [std(REWARD_SAMPS[1:rollout_counts[p], p]) / sqrt(rollout_counts[p]) for p in 1:policy_count]
    REWARD_SAMPS = nothing
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
    
    MEM = [[] for tt in 1:min(T-t+1,dp_length)]
    MEM[1] = [[] for b in 1:bandit_count]
    BANDIT_PARAM_SAMPS = zeros(bandit_count, context_dim, dp_rollout_count)
    if true
        for k in 1:bandit_count
            if sum(A .== k) == 0
                BANDIT_PARAM_SAMPS[k, :, :] = rand(MvNormal(ms_Cs[k][1], ms_Cs[k][2]), dp_rollout_count)
            else
                BANDIT_PARAM_SAMPS[k, :, :], ACC = sample_mcmc_vb(X[A .== k, :], y[A .== k, :], prior_mean, prior_cov, ms_Cs[k][1], ms_Cs[k][2], dp_rollout_count, n_burn, n_dup)
                println("Acceptance Prob: ", ACC)
                flush(stdout)
            end
        end
    else
        for k in 1:bandit_count
            BANDIT_PARAM_SAMPS[:, :, k] = rand(MvNormal(ms_Cs[k][1], ms_Cs[k][2]), dp_rollout_count)
        end
    end
    for tt in 1:dp_rollout_count
        dp_actions = sample(1:bandit_count, dp_length)
        #copy!(temp_post_means, bandit_posterior_means)
        #copy!(temp_post_covs, bandit_posterior_covs)
        RESULTS = dp_vb_rollout(ep, opt_policy, T-t+1, rollout_length, dp_length, dp_actions, context, use_context, lambda, context_dim, context_mean,
                             context_sd, obs_sd, bandit_count, discount, X, y, A, BANDIT_PARAM_SAMPS[:, :, tt], roll_true_expected_rewards, ms_Cs)
        A[t:T] .= 0
        for i in eachindex(RESULTS)
            if i == 1
                push!(MEM[i][dp_actions[1]], RESULTS[i])
            else
                push!(MEM[i], RESULTS[i])
            end
        end
    end
    #M = [Vector{Any}(undef, bandit_count) for tt in 1:min(T-t+1, dp_length)]
    BANDIT_PARAM_SAMPS = nothing
    M = Vector{Any}(undef, min(T-t+1, dp_length))
    M[1] = zeros(bandit_count)
    for s in 1:min(dp_length, T-t+1)
        tt = min(dp_length, T-t+1) - s + 1
        if tt>1
            Y = [u[3] + discount * (u[4] == 0 ? maximum([XGBoost.predict(M[tt+1], reshape(reduce_features_vb(u[5], bb), 1, length(u[5])-(bandit_count-1)*3))[1] for bb in 1:bandit_count]) : 0) for u in MEM[tt]]
            X = reduce(vcat, transpose.([u[1] for u in MEM[tt]]))
            M[tt] = xgboost_cv(X, Y, 10)
        else
            for b in 1:bandit_count
                Y = [u[3] + discount * (u[4] == 0 ? maximum([XGBoost.predict(M[tt+1], reshape(reduce_features_vb(u[5], bb), 1, length(u[5])-(bandit_count-1)*3))[1] for bb in 1:bandit_count]) : 0) for u in MEM[tt][b]]
                M[tt][b] = mean(Y)
            end
        end
    end
    return findmax(M[1])[2]
    opt_act = opt_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    return opt_act

    ACTION_MEAN_REWARDS = zeros(bandit_count)
    ACTION_STD_REWARDS = zeros(bandit_count)
    REWARD_SAMPS = zeros(n_opt_rollouts * vb_rollout_mult, bandit_count)
    halt_vec = repeat([false], bandit_count)
    rollout_counts = repeat([0], bandit_count)
    for roll in 1:(n_opt_rollouts*vb_rollout_mult)
        bandit_param = rollout_params[:, 1 + ((roll - 1) % n_opt_rollouts), :]'
        for action in 1:bandit_count 
            
            if halt_vec[action]
                continue
            end

            rollout_counts[action] += 1
            use_context = false
            true_expected_reward_logit = dot(bandit_param[action,:], context)
            true_expected_reward = exp(true_expected_reward_logit) / (1 + exp(true_expected_reward_logit))

            obs = 1 * (rand() < true_expected_reward)
            
            y[t] = obs
            A[t] = action
            X[t, :] = context
            
            rollout_value = vb_bernoulli_val_rollout(ep, opt_policy, T-t, rollout_length-1, context, use_context, lambda, context_dim, context_mean,
                context_sd, obs_sd, bandit_count, discount, X, y, A, bandit_param,
                roll_true_expected_rewards, ms_Cs)
            A[t:T] .= 0
            
            REWARD_SAMPS[roll, action] = true_expected_reward + discount * rollout_value
        end
        
        #if roll % 100 == 0
        #    continue_inds = findall(halt_vec .== false)
        #    action_means = [mean(REWARD_SAMPS[1:roll, p]) for p in continue_inds]
        #    action_stds = [std(REWARD_SAMPS[1:roll, p]) / sqrt(roll) for p in continue_inds]
        #    max_mean, max_ind = findmax(action_means)
        #    diff_means = max_mean .- action_means
        #    diff_stds = sqrt.(action_stds[max_ind]^2 .+ action_stds.^2)
        #    action_expected_regret_proportions = (diff_means .* cdf.(Normal(), -diff_means ./ diff_stds) .- diff_stds .* pdf.(Normal(), -diff_means ./ diff_stds)) ./ max_mean
        #    action_pol_diff_means = opt_mean .- action_means
        #    action_pol_diff_stds = sqrt.(opt_std^2 .+ action_stds.^2)
        #    action_pol_expected_regret_proportions = (action_pol_diff_means .* cdf.(Normal(), -action_pol_diff_means ./ action_pol_diff_stds) .- action_pol_diff_stds .* pdf.(Normal(), -action_pol_diff_means ./ action_pol_diff_stds)) ./ opt_mean
        #    halt_vec[continue_inds] = (abs.(action_expected_regret_proportions) .< expected_regret_thresh) .| ((abs.(action_pol_expected_regret_proportions) .< expected_regret_thresh) .& (action_pol_diff_means .> 0))
        #    halt_vec[continue_inds[max_ind]] = (abs(action_pol_expected_regret_proportions[max_ind]) < expected_regret_thresh) & (action_pol_diff_means[max_ind] > 0)
        #    if (sum(halt_vec .== false) == 0)
        #        halt_vec[continue_inds[max_ind]] = false
        #        break
        #    end
        #    halt_vec[continue_inds[max_ind]] = false
        #end
        
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

    if opt_mean >= opt_action_mean
        opt_act = opt_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
        println("Default to Policy, Action: ", opt_act)
        flush(stdout)
        return opt_act
    end

    m = opt_action_mean - opt_mean
    s = sqrt(opt_std^2 + opt_action_std^2)
    erp = (m * cdf(Normal(), -m/s) - s * pdf(Normal(), -m/s)) / opt_action_mean
    if abs(erp) < expected_regret_thresh
        println("Policy Improvement, Action: ", opt_action_index)
        flush(stdout)
        return opt_action_index
    else
        opt_act = opt_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
        println("Insufficient Improvement, Action: ", opt_act)
        flush(stdout)
        return opt_act
    end

    # END OPTIMIZATION OF LAMBDA

    # Unnecessary given samples above 
    #opt_act = opt_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    #opt_act = findmax(ACTION_MEAN_REWARDS)[2]
    #println("Optimal Action: ", opt_act)
    #flush(stdout)

    #return opt_act

end

function vb_bernoulli_val_rollout(ep, policy, T_remainder, rollout_length, context, use_context, lambda, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, discount, X, y, A, bandit_param,
	true_expected_rewards, ms_Cs)

    disc_reward = 0
    #fill!(context, 0.0)
    fill!(true_expected_rewards, 0.0)
    #fill!(CovCon, 0.0)
    #fill!(old_cov,0.0)
    #fill!(SigInvMu,0.0)
    
    truncation_length = T_remainder - min(T_remainder, rollout_length)
    
    t_curr = T - T_remainder + 1
    

    ms_Cs_policy = getfield(Main, Symbol(String(Symbol(policy)) * "_given_mC"))

    for t in 1:min(T_remainder, rollout_length)

        #context_seed = rand(1:context_dim)
        #fill!(context, zero(context[1]))
        #context[context_seed] = 1
        #mul!(true_expected_rewards, bandit_param, context)

        if t > 1 || !use_context
            context = generate_context(context_dim, context_mean, context_sd, context_constant)
        end

        action = ms_Cs_policy(ep, t_curr, min(T_remainder, rollout_length), bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, ms_Cs)


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
        ms_Cs = [jj_posterior_batch(X[A .== bandit,:], y[A .== bandit], ms_Cs[bandit][1], ms_Cs[bandit][2], vb_policy_tol) for bandit in 1:bandit_count]
        
        # UNKNOWN PARAM VERSION
    end


    if truncation_length > 0
        if true 
            t_trunc = t_curr + min(T_remainder, rollout_length)
            reg_est = 0

            #ms_Cs = [jj_posterior_batch(X[A .== k,:], y[A .== k], prior_mean, prior_cov, vb_policy_tol) for k in 1:bandit_count]
            for n in 1:2
                context = generate_context(context_dim, context_mean, context_sd, context_constant)
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
function dp_vb_rollout(ep, policy, T_remainder, rollout_length, dp_length, dp_actions, context, use_context, lambda, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, discount, X, y, A, bandit_param,
	true_expected_rewards, ms_Cs)

    disc_reward = 0
    #fill!(context, 0.0)
    fill!(true_expected_rewards, 0.0)
    #fill!(CovCon, 0.0)
    #fill!(old_cov,0.0)
    #fill!(SigInvMu,0.0)
    
    truncation_length = T_remainder - min(T_remainder, rollout_length)
    
    t_curr = T - T_remainder + 1
    

    ms_Cs_policy = getfield(Main, Symbol(String(Symbol(policy)) * "_given_mC"))

    min_remainder_dp_length = min(T_remainder, dp_length)
    
    RESULTS = [[] for i in 1:min_remainder_dp_length]
    
    for t in 1:min_remainder_dp_length

        #context_seed = rand(1:context_dim)
        #fill!(context, zero(context[1]))
        #context[context_seed] = 1
        #mul!(true_expected_rewards, bandit_param, context)

        if t > 1 || !use_context
            context = generate_context(context_dim, context_mean, context_sd, context_constant)
        end

        #action = ms_Cs_policy(ep, t_curr, min(T_remainder, rollout_length), bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, ms_Cs)

        old_features = get_features_vb(context, ms_Cs)

        action = dp_actions[t]
        #action_ind = findfirst(arm_order .== action)
         
        old_features = reduce_features_vb(old_features, action)
        push!(RESULTS[t], old_features)
        push!(RESULTS[t], action)

        # TRUE PARAM VERSION
        #true_expected_reward = true_expected_rewards[action]
        #disc_reward += true_expected_reward * discount^(t-1)
        #obs = randn() * obs_sd + true_expected_reward
        
        true_expected_reward_logit = dot(bandit_param[action,:], context)
        true_expected_reward = exp(true_expected_reward_logit) / (1 + exp(true_expected_reward_logit))
        push!(RESULTS[t], true_expected_reward)
        push!(RESULTS[t], 1 * (t == min_remainder_dp_length))
        disc_reward += true_expected_reward * discount^(t-1)

        obs = 1 * (rand() < true_expected_reward)
        
        y[t_curr + t - 1] = obs
        A[t_curr + t - 1] = action
        X[(t_curr + t - 1), :] = context
        ms_Cs = [jj_posterior_batch(X[A .== bandit,:], y[A .== bandit], ms_Cs[bandit][1], ms_Cs[bandit][2], vb_policy_tol) for bandit in 1:bandit_count]
        
        new_features = get_features_vb(context, ms_Cs)
	    push!(RESULTS[t], new_features)  
        # UNKNOWN PARAM VERSION
    end

    for t in (1+min(T_remainder, dp_length)):min(T_remainder, rollout_length)

        #context_seed = rand(1:context_dim)
        #fill!(context, zero(context[1]))
        #context[context_seed] = 1
        #mul!(true_expected_rewards, bandit_param, context)

        context = generate_context(context_dim, context_mean, context_sd, context_constant)

        action = ms_Cs_policy(ep, t_curr, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, ms_Cs)


        #action_ind = findfirst(arm_order .== action)
         

        # TRUE PARAM VERSION
        #true_expected_reward = true_expected_rewards[action]
        #disc_reward += true_expected_reward * discount^(t-1)
        #obs = randn() * obs_sd + true_expected_reward
        
        true_expected_reward_logit = dot(bandit_param[action,:], context)
        true_expected_reward = exp(true_expected_reward_logit) / (1 + exp(true_expected_reward_logit))
        RESULTS[min_remainder_dp_length][3] += true_expected_reward * discount^(t - min(T_remainder, dp_length))
        disc_reward += true_expected_reward * discount^(t-1)

        obs = 1 * (rand() < true_expected_reward)
        
        y[t_curr + t - 1] = obs
        A[t_curr + t - 1] = action
        X[(t_curr + t - 1), :] = context
        ms_Cs = [jj_posterior_batch(X[A .== bandit,:], y[A .== bandit], ms_Cs[bandit][1], ms_Cs[bandit][2], vb_policy_tol) for bandit in 1:bandit_count]
        
        # UNKNOWN PARAM VERSION
    end

    if truncation_length > 0
        if true 
            t_trunc = t_curr + min(T_remainder, rollout_length)
            reg_est = 0

            #ms_Cs = [jj_posterior_batch(X[A .== k,:], y[A .== k], prior_mean, prior_cov, vb_policy_tol) for k in 1:bandit_count]
            for n in 1:2
                context = generate_context(context_dim, context_mean, context_sd, context_constant)
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
            RESULTS[min_remainder_dp_length][3] -= .9 * discount^(1+min(T_remainder, rollout_length)-min(T_remainder, dp_length)) * sum(discount^(t - t_trunc) * reg_est * (1 + T - t) / (1 + T - t_trunc) for t in t_trunc:T)
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

    return RESULTS
    end

function get_features_vb(context, ms_Cs)
    input_vec = [dot(post[1],context) for post in ms_Cs]
    append!(input_vec, [sqrt(dot(context, a[2], context)) for a in ms_Cs])
    append!(input_vec, [sqrt(dot(context, a[2], context)) for a in ms_Cs] .* input_vec[1:bandit_count])
    append!(input_vec, Vector(vcat([a[1] for a in ms_Cs]...)))
    append!(input_vec, vcat([upper_triangular_vec(a[2]) for a in ms_Cs]...))
    #append!(input_vec, context)
    return input_vec
end

function reduce_features_vb(features, action)
    return features[vcat([action, bandit_count + action, 2 * bandit_count + action], (3 * bandit_count + 1):end)]
end

# Greedy
function vb_greedy_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    
    bandit_expected_rewards = zeros(bandit_count)
    for k in 1:bandit_count
        m, C = jj_posterior_batch(X[A .== k, :], y[A .== k], prior_mean, prior_cov, vb_policy_tol)
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
        m, C = jj_posterior_batch(X[A .== k,:], y[A .== k], prior_mean, prior_cov, vb_policy_tol)
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
        m, C = jj_posterior_batch(X[A .== k,:], y[A .== k], prior_mean, prior_cov, vb_policy_tol)
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
function vb_ids_expected_regrets(bandit_count, context, ms_Cs, niter)

    #draws = zeros(bandit_count, context_dim, niter)
    reward_draws = zeros(bandit_count, niter)
    
    for b in 1:bandit_count
        #draws[b, :, :] = rand(MvNormal(bandit_posterior_means[b, :], bandit_posterior_covs[b, :, :]), niter)
        #for i in 1:niter
        #    reward_draws[b, i] = dot(context, draws[b, :, i])
        #end
        reward_draws[b, :] = expit(dot(context, ms_Cs[b][1]) .+ sqrt(dot(context, ms_Cs[b][2], context)) .* randn(niter))
    end
    
    mean_rewards = dropdims(mean(reward_draws, dims = 2), dims = 2)

    mean_max_reward = 0
    for i in 1:niter
        mean_max_reward += findmax(reward_draws[:, i])[1] / niter
    end
    
    res = max.(0, mean_max_reward .- mean_rewards)

    return res
end
function vb_ids_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    
    #vb_post_means = zeros(bandit_count, context_dim)
    #vb_post_covs = zeros(bandit_count, context_dim, context_dim)
    X[t, :] = context
    
    ent_gains = zeros(bandit_count)
    ms_Cs = [jj_posterior_batch(X[A .== k, :], y[A .== k], prior_mean, prior_cov, vb_policy_tol) for k in 1:bandit_count] 
    for k in 1:bandit_count
        m, C = ms_Cs[k]
        
        p1 = expit(dot(m, context))
        A[t] = k

        y[t] = 1
        m1, C1 = jj_posterior_batch(X[A .== k, :], y[A .== k], prior_mean, prior_cov, vb_policy_tol)

        y[t] = 0
        m0, C0 = jj_posterior_batch(X[A .== k, :], y[A .== k], prior_mean, prior_cov, vb_policy_tol)

        ent_gains[k] = logdet(C1 \ C) * p1 + logdet(C0 \ C) * (1 - p1)

    end
    A[t] = 0

    expected_regrets = vb_ids_expected_regrets(bandit_count, context, ms_Cs, 1000)

    return findmax(-((expected_regrets.^2) ./ ent_gains))[2]

end
function vb_ids_4_policy(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    
    #vb_post_means = zeros(bandit_count, context_dim)
    #vb_post_covs = zeros(bandit_count, context_dim, context_dim)
    X[t, :] = context
    
    ent_gains = zeros(bandit_count)
    ms_Cs = [jj_posterior_batch(X[A .== k, :], y[A .== k], prior_mean, prior_cov, vb_policy_tol) for k in 1:bandit_count] 
    for k in 1:bandit_count
        m, C = ms_Cs[k]
        
        p1 = expit(dot(m, context))
        A[t] = k

        y[t] = 1
        m1, C1 = jj_posterior_batch(X[A .== k, :], y[A .== k], prior_mean, prior_cov, vb_policy_tol)

        y[t] = 0
        m0, C0 = jj_posterior_batch(X[A .== k, :], y[A .== k], prior_mean, prior_cov, vb_policy_tol)

        ent_gains[k] = logdet(C1 \ C) * p1 + logdet(C0 \ C) * (1 - p1)

    end
    A[t] = 0

    expected_regrets = vb_ids_expected_regrets(bandit_count, context, ms_Cs, 1000)

    return findmax(-((expected_regrets.^4) ./ ent_gains))[2]

end
function vb_ids_policy_given_mC(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, ms_Cs)
    
    #vb_post_means = zeros(bandit_count, context_dim)
    #vb_post_covs = zeros(bandit_count, context_dim, context_dim)
    X[t, :] = context
    
    ent_gains = zeros(bandit_count)
    for k in 1:bandit_count
        m, C = ms_Cs[k]
        
        p1 = expit(dot(m, context))
        A[t] = k

        y[t] = 1
        m1, C1 = jj_posterior_batch(X[A .== k, :], y[A .== k], m, C, vb_policy_tol)

        y[t] = 0
        m0, C0 = jj_posterior_batch(X[A .== k, :], y[A .== k], m, C, vb_policy_tol)

        ent_gains[k] = logdet(C1 \ C) * p1 + logdet(C0 \ C) * (1 - p1)

    end
    A[t] = 0

    expected_regrets = vb_ids_expected_regrets(bandit_count, context, ms_Cs, 1000)

    return findmax(-((expected_regrets.^2) ./ ent_gains))[2]

end
function vb_ids_4_policy_given_mC(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim, ms_Cs)
    
    #vb_post_means = zeros(bandit_count, context_dim)
    #vb_post_covs = zeros(bandit_count, context_dim, context_dim)
    X[t, :] = context
    
    ent_gains = zeros(bandit_count)
    for k in 1:bandit_count
        m, C = ms_Cs[k]
        
        p1 = expit(dot(m, context))
        A[t] = k

        y[t] = 1
        m1, C1 = jj_posterior_batch(X[A .== k, :], y[A .== k], m, C, vb_policy_tol)

        y[t] = 0
        m0, C0 = jj_posterior_batch(X[A .== k, :], y[A .== k], m, C, vb_policy_tol)

        ent_gains[k] = logdet(C1 \ C) * p1 + logdet(C0 \ C) * (1 - p1)

    end
    A[t] = 0

    expected_regrets = vb_ids_expected_regrets(bandit_count, context, ms_Cs, 1000)

    return findmax(-((expected_regrets.^4) ./ ent_gains))[2]

end



function vb_get_action_probs(policy_list, ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
    out = zeros(length(policy_list), bandit_count)
    for i in eachindex(policy_list)
        pol = policy_list[i]
        if pol == vb_thompson_policy
            for j in 1:1000
                out[i, pol(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)] += 1
            end
            out[i, :] ./= 1000
        else
            out[i, pol(ep, t, T, bandit_count, context, X, y, A, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)] += 1
        end
    end

    return out
end

function vb_bernoulli_ep_contextual_bandit_simulator(ep,action_function::exp4, T, rollout_length, n_episodes, n_rollouts, n_opt_rollouts, context_dim, context_mean,
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
        E_star_t = 0        
        for t in 1:T
            context = generate_context(context_dim, context_mean, context_sd, context_constant)
            true_expected_rewards_logit = true_bandit_param * context
            true_expected_rewards = exp.(true_expected_rewards_logit) ./ (1 .+ exp.(true_expected_rewards_logit))
            #true_expected_rewards = bandit_posterior_means * context
            action_probs_matrix = vb_get_action_probs(action_function.policy_list, ep, t, T, bandit_count, context, X_tot, Y_tot, A_tot, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
            action_probs = action_probs_matrix' * action_function.policy_probs
            action = sample(1:bandit_count, Weights(action_probs))
            true_expected_reward = true_expected_rewards[action]
            EPREWARDS[t] = true_expected_reward
            EPOPTREWARDS[t] = maximum(true_expected_rewards)
            #obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:],context)) + true_expected_reward
            obs = 1 * (rand() < true_expected_reward)
            Y_tot[t] = obs
            A_tot[t] = action
            X_tot[t, :] = context
            
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
