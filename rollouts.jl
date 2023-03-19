
function lambda_rollout(T, rollout_length, context, use_context, lambda, context_dim, context_mean,
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

    for t in 1:min(T, rollout_length)

        #context_seed = rand(1:context_dim)
        #fill!(context, zero(context[1]))
        #context[context_seed] = 1

        if (t > 1) || !use_context
            context = generate_context(context_dim, context_mean, context_sd, context_constant)
        end

        mul!(true_expected_rewards, bandit_param, context)
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

        dividy = 1 + dot(context ./ obs_sd, old_cov, context ./ obs_sd)
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

function lambda_mean_rollout(T_remainder, rollout_length, context, use_context, lambda, context_dim, context_mean,
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

    for t in 1:min(T_remainder, rollout_length)

        #context_seed = rand(1:context_dim)
        #fill!(context, zero(context[1]))
        #context[context_seed] = 1


        if (t > 1) || !use_context
            context = generate_context(context_dim, context_mean, context_sd, context_constant)
        end
        mul!(true_expected_rewards, bandit_param, context)
        #context = randn(context_dim) * context_sd .+ context_mean


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

        dividy = 1 + dot(context ./ obs_sd, old_cov, context ./ obs_sd)
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

        true_expected_reward = true_expected_rewards[action]
        disc_reward += true_expected_reward * discount^(t-1)
        obs = randn() * obs_sd + true_expected_reward
        #old_cov = temp_post_covs[action, :, :]
        old_cov .= bandit_posterior_covs[action, :, :]
        #CovCon .= old_cov * context ./ obs_sd
        mul!(CovCon, old_cov, context)
        CovCon ./= obs_sd
        #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)

        dividy = 1 + dot(context ./ obs_sd, old_cov, context ./ obs_sd)
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


function val_greedy_rollout(T_remainder, rollout_length, lambda, context_dim, context_mean,
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
        
        # UNKNOWN PARAM VERSION
        true_expected_reward = dot(bandit_posterior_means[action,:], context)
        disc_reward += true_expected_reward * discount^(t-1)
        obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:], context)) + true_expected_reward
        
        
        
        #old_cov = temp_post_covs[action, :, :]
        old_cov .= bandit_posterior_covs[action, :, :]
        

        #print("Context: $(context)\n")
        #print("Obs: $(obs)\n")
        #print("Old Cov: $(old_cov)\n")
        #print("Old Mean: $(bandit_posterior_means[action, :])\n")


        #CovCon .= old_cov * context ./ obs_sd
        mul!(CovCon, old_cov, context)
        CovCon ./= obs_sd
        #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)

        dividy = 1 + dot(context ./ obs_sd, old_cov, context ./ obs_sd)
        for i in 1:context_dim
            for j in 1:context_dim
                bandit_posterior_covs[action,j,i] = old_cov[j,i] - CovCon[j]*CovCon[i] / dividy
            end
        end


        #Checking that computations are correct.
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
	    
        

        #Checking that computations are correct
        #print("New Cov: $(bandit_posterior_covs[action,:,:])\n")
        #print("New Mean: $(bandit_posterior_means[action, :, :])\n")
        
        #bandit_posterior_means[action, :] .= (bandit_posterior_covs[action, :, :]) * (old_cov \ (bandit_posterior_means[action,:]) + context * obs / obs_sd^2)
    end


    if truncation_length > 0
        
        
        #### SWITCHED TO TRUE VALUE NEURAL NETWORK AS TEST
        
        #input_vec = Vector(vec(bandit_param'))
        #append!(input_vec, Vector(vec(bandit_posterior_means')))
        input_vec = Vector(vec(bandit_posterior_means'))
        append!(input_vec, vcat([upper_triangular_vec(bandit_posterior_covs[a, :, :]) for a in 1:bandit_count]...))
        
        scaled_input_vec = (input_vec .- scale_list[truncation_length][1:end .!= 1, 1]) ./ scale_list[truncation_length][1:end .!= 1, 2] 
        
        disc_reward += (discount^min(T_remainder, rollout_length)) * (scale_list[truncation_length][1,2]*neural_net_list[truncation_length](scaled_input_vec)[1] + scale_list[truncation_length][1,1])
    
        
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

function val_epsilon_greedy_rollout(epsilon, T_remainder, rollout_length, context, use_context, lambda, context_dim, context_mean,
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


        if t > 1 || !use_context

            context = generate_context(context_dim, context_mean, context_sd, context_constant)

        end

        mul!(true_expected_rewards, bandit_param, context)

        if rand() > epsilon
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
        else
            action = rand(1:bandit_count)
        end




        # TRUE PARAM VERSION
        #true_expected_reward = true_expected_rewards[action]
        #disc_reward += true_expected_reward * discount^(t-1)
        #obs = randn() * obs_sd + true_expected_reward
        
        # UNKNOWN PARAM VERSION
        true_expected_reward = dot(bandit_posterior_means[action,:], context)
        disc_reward += true_expected_reward * discount^(t-1)
        obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:], context)) + true_expected_reward
        
        
        
        #old_cov = temp_post_covs[action, :, :]
        old_cov .= bandit_posterior_covs[action, :, :]
        

        #print("Context: $(context)\n")
        #print("Obs: $(obs)\n")
        #print("Old Cov: $(old_cov)\n")
        #print("Old Mean: $(bandit_posterior_means[action, :])\n")


        #CovCon .= old_cov * context ./ obs_sd
        mul!(CovCon, old_cov, context)
        CovCon ./= obs_sd
        #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)

        dividy = 1 + dot(context ./ obs_sd, old_cov, context ./ obs_sd)
        for i in 1:context_dim
            for j in 1:context_dim
                bandit_posterior_covs[action,j,i] = old_cov[j,i] - CovCon[j]*CovCon[i] / dividy
            end
        end


        #Checking that computations are correct.
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
	    
        

        #Checking that computations are correct
        #print("New Cov: $(bandit_posterior_covs[action,:,:])\n")
        #print("New Mean: $(bandit_posterior_means[action, :, :])\n")
        
        #bandit_posterior_means[action, :] .= (bandit_posterior_covs[action, :, :]) * (old_cov \ (bandit_posterior_means[action,:]) + context * obs / obs_sd^2)
    end


    if truncation_length > 0
        
        
        #### SWITCHED TO TRUE VALUE NEURAL NETWORK AS TEST
        
        #input_vec = Vector(vec(bandit_param'))
        #append!(input_vec, Vector(vec(bandit_posterior_means')))
        input_vec = Vector(vec(bandit_posterior_means'))
        append!(input_vec, vcat([upper_triangular_vec(bandit_posterior_covs[a, :, :]) for a in 1:bandit_count]...))
        
        scaled_input_vec = (input_vec .- scale_list[truncation_length][1:end .!= 1, 1]) ./ scale_list[truncation_length][1:end .!= 1, 2] 
        
        disc_reward += (discount^min(T_remainder, rollout_length)) * (scale_list[truncation_length][1,2]*neural_net_list[truncation_length](scaled_input_vec)[1] + scale_list[truncation_length][1,1])
    
        
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

function val_fractional_thompson_rollout(coarse, T_remainder, rollout_length, context, use_context, lambda, context_dim, context_mean,
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


        if t > 1 || !use_context

            context = generate_context(context_dim, context_mean, context_sd, context_constant)

        end

        mul!(true_expected_rewards, bandit_param, context)



        if coarse == 0
            action = greedy_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
        else
            frac = 1 / coarse - 1
            action = thompson_policy(t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs ./ frac, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
        end
    


        # TRUE PARAM VERSION
        #true_expected_reward = true_expected_rewards[action]
        #disc_reward += true_expected_reward * discount^(t-1)
        #obs = randn() * obs_sd + true_expected_reward
        
        # UNKNOWN PARAM VERSION
        true_expected_reward = dot(bandit_posterior_means[action,:], context)
        disc_reward += true_expected_reward * discount^(t-1)
        obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:], context)) + true_expected_reward
        
        
        
        #old_cov = temp_post_covs[action, :, :]
        old_cov .= bandit_posterior_covs[action, :, :]
        

        #print("Context: $(context)\n")
        #print("Obs: $(obs)\n")
        #print("Old Cov: $(old_cov)\n")
        #print("Old Mean: $(bandit_posterior_means[action, :])\n")


        #CovCon .= old_cov * context ./ obs_sd
        mul!(CovCon, old_cov, context)
        CovCon ./= obs_sd
        #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)

        dividy = 1 + dot(context ./ obs_sd, old_cov, context ./ obs_sd)
        for i in 1:context_dim
            for j in 1:context_dim
                bandit_posterior_covs[action,j,i] = old_cov[j,i] - CovCon[j]*CovCon[i] / dividy
            end
        end


        #Checking that computations are correct.
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
	    
        

        #Checking that computations are correct
        #print("New Cov: $(bandit_posterior_covs[action,:,:])\n")
        #print("New Mean: $(bandit_posterior_means[action, :, :])\n")
        
        #bandit_posterior_means[action, :] .= (bandit_posterior_covs[action, :, :]) * (old_cov \ (bandit_posterior_means[action,:]) + context * obs / obs_sd^2)
    end


    if truncation_length > 0
        
        
        #### SWITCHED TO TRUE VALUE NEURAL NETWORK AS TEST
        
        #input_vec = Vector(vec(bandit_param'))
        #append!(input_vec, Vector(vec(bandit_posterior_means')))
        input_vec = Vector(vec(bandit_posterior_means'))
        append!(input_vec, vcat([upper_triangular_vec(bandit_posterior_covs[a, :, :]) for a in 1:bandit_count]...))
        
        scaled_input_vec = (input_vec .- scale_list[truncation_length][1:end .!= 1, 1]) ./ scale_list[truncation_length][1:end .!= 1, 2] 
        
        disc_reward += (discount^min(T_remainder, rollout_length)) * (scale_list[truncation_length][1,2]*neural_net_list[truncation_length](scaled_input_vec)[1] + scale_list[truncation_length][1,1])
    
        
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

function val_rollout(ep, policy, T_remainder, rollout_length, context, use_context, lambda, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, discount, bandit_posterior_covs, bandit_posterior_means, bandit_param,
	true_expected_rewards, CovCon, old_cov, SigInvMu)

    disc_reward = 0
    #fill!(context, 0.0)
    fill!(true_expected_rewards, 0.0)
    fill!(CovCon, 0.0)
    fill!(old_cov,0.0)
    fill!(SigInvMu,0.0)
    
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

        action = policy(ep, t, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)


        #TRUE PARAM VERSION
        mr = maximum(bandit_param * context)
        true_expected_reward = dot(bandit_param[action, :], context)
        disc_reward += (true_expected_reward-mr) * discount^(t-1)
        obs = randn() * obs_sd + true_expected_reward
        
        # UNKNOWN PARAM VERSION
        #true_expected_reward = dot(bandit_posterior_means[action,:], context)
        #disc_reward += true_expected_reward * discount^(t-1)
        #obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:], context)) + true_expected_reward
        
        
        
        #old_cov = temp_post_covs[action, :, :]
        old_cov .= bandit_posterior_covs[action, :, :]
        

        #print("Context: $(context)\n")
        #print("Obs: $(obs)\n")
        #print("Old Cov: $(old_cov)\n")
        #print("Old Mean: $(bandit_posterior_means[action, :])\n")


        #CovCon .= old_cov * context ./ obs_sd
        mul!(CovCon, old_cov, context)
        CovCon ./= obs_sd
        #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)

        dividy = 1 + dot(context ./ obs_sd, old_cov, context ./ obs_sd)
        for i in 1:context_dim
            for j in 1:context_dim
                bandit_posterior_covs[action,j,i] = old_cov[j,i] - CovCon[j]*CovCon[i] / dividy
            end
        end


        #Checking that computations are correct.
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
	    
        

        #Checking that computations are correct
        #print("New Cov: $(bandit_posterior_covs[action,:,:])\n")
        #print("New Mean: $(bandit_posterior_means[action, :, :])\n")
        
        #bandit_posterior_means[action, :] .= (bandit_posterior_covs[action, :, :]) * (old_cov \ (bandit_posterior_means[action,:]) + context * obs / obs_sd^2)
    end


    if truncation_length > 0
        
        if true 
            t_trunc = t_curr + min(T_remainder, rollout_length)
            reg_est = 0
            for n in 1:2
                context = generate_context(context_dim, context_mean, context_sd, context_constant)
                context_reg_est = 0
                expected_rewards = [dot(context, bandit_posterior_means[k, :]) for k in 1:bandit_count]
                variance_rewards = [dot(context, bandit_posterior_covs[k, :, :], context) for k in 1:bandit_count]
                #point_samps =  expected_rewards .+ sqrt.(variance_rewards) .* randn(bandit_count, 500)
                #expected_rewards = bandit_param * context
                max_reward = maximum(expected_rewards)
                if String(Symbol(policy)) == "thompson_policy"
                    for m in 1:500
                        action = policy(ep, t_trunc, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
                        reg_est *= ((n-1)*500 + m - 1) / ((n-1)*500 + m)
                        reg_est += (max_reward - expected_rewards[action]) / ((n-1)*500 + m)
                    end
                else
                    action = policy(ep, t_trunc, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
                    reg_est += (max_reward - expected_rewards[action]) / 2
                    #for m in 1:500
                    #    reg_est *= ((n-1)*500 + m - 1) / ((n-1)*500 + m)
                    #    reg_est += (findmax(point_samps[:, m])[1] - point_samps[action, m]) / ((n-1)*500 + m)
                    #end
                end
                #reg_est *= (n-1) / n
                #reg_est += (findmax(point_samps)[1] - point_samps[action]) / n
            end

            disc_reward -= .9 * discount^min(T_remainder, rollout_length) * sum(discount^(t - t_trunc) * reg_est * (1 + T - t) / (1 + T - t_trunc) for t in t_trunc:T)
        
        else
            #### SWITCHED TO TRUE VALUE NEURAL NETWORK AS TEST
            
            #input_vec = Vector(vec(bandit_param'))
            #append!(input_vec, Vector(vec(bandit_posterior_means')))
            input_vec = Vector(vec(bandit_posterior_means'))
            append!(input_vec, vcat([upper_triangular_vec(bandit_posterior_covs[a, :, :]) for a in 1:bandit_count]...))
            
            scaled_input_vec = (input_vec .- scale_list[truncation_length][1:end .!= 1, 1]) ./ scale_list[truncation_length][1:end .!= 1, 2] 
            
            disc_reward += (discount^min(T_remainder, rollout_length)) * (scale_list[truncation_length][1,2]*neural_net_list[truncation_length](scaled_input_vec)[1] + scale_list[truncation_length][1,1])
        
            
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

function rollout(policy, T_remainder, rollout_length, context, use_context, lambda, context_dim, context_mean,
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
            context = generate_context(context_dim, context_mean, context_sd, context_constant)
        end

        action = policy(t, min(T_remainder, rollout_length), bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)


        # TRUE PARAM VERSION
        #true_expected_reward = true_expected_rewards[action]
        #disc_reward += true_expected_reward * discount^(t-1)
        #obs = randn() * obs_sd + true_expected_reward
        
        # UNKNOWN PARAM VERSION
        true_expected_reward = dot(bandit_posterior_means[action,:], context)
        disc_reward += true_expected_reward * discount^(t-1)
        obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:], context)) + true_expected_reward
        
        
        
        #old_cov = temp_post_covs[action, :, :]
        old_cov .= bandit_posterior_covs[action, :, :]
        

        #print("Context: $(context)\n")
        #print("Obs: $(obs)\n")
        #print("Old Cov: $(old_cov)\n")
        #print("Old Mean: $(bandit_posterior_means[action, :])\n")


        #CovCon .= old_cov * context ./ obs_sd
        mul!(CovCon, old_cov, context)
        CovCon ./= obs_sd
        #bandit_posterior_covs[action, :, :] = inv(context * context' / obs_sd^2 + old_precision)

        dividy = 1 + dot(context ./ obs_sd, old_cov, context ./ obs_sd)
        for i in 1:context_dim
            for j in 1:context_dim
                bandit_posterior_covs[action,j,i] = old_cov[j,i] - CovCon[j]*CovCon[i] / dividy
            end
        end


        #Checking that computations are correct.
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
	    
        

        #Checking that computations are correct
        #print("New Cov: $(bandit_posterior_covs[action,:,:])\n")
        #print("New Mean: $(bandit_posterior_means[action, :, :])\n")
        
        #bandit_posterior_means[action, :] .= (bandit_posterior_covs[action, :, :]) * (old_cov \ (bandit_posterior_means[action,:]) + context * obs / obs_sd^2)
    end


    if truncation_length > 0
        
        
        #### SWITCHED TO TRUE VALUE NEURAL NETWORK AS TEST
        
        #input_vec = Vector(vec(bandit_param'))
        #append!(input_vec, Vector(vec(bandit_posterior_means')))
        
        # BEGIN REAL
        #input_vec = Vector(vec(bandit_posterior_means'))
        #append!(input_vec, vcat([upper_triangular_vec(bandit_posterior_covs[a, :, :]) for a in 1:bandit_count]...))
        
        #scaled_input_vec = (input_vec .- scale_list[truncation_length][1:end .!= 1, 1]) ./ scale_list[truncation_length][1:end .!= 1, 2] 
        
        #disc_reward += (discount^min(T_remainder, rollout_length)) * (scale_list[truncation_length][1,2]*neural_net_list[truncation_length](scaled_input_vec)[1] + scale_list[truncation_length][1,1])
    
        # END REAL
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

function better_rollout(T, curr_t, rollout_length, context, use_context, lambda_param, context_dim, context_mean,
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
    for t in 1:min(T, rollout_length)
        #context_seed = rand(1:context_dim)
        #fill!(context, zero(context[1]))
        #context[context_seed] = 1


        if (t > 1) || !use_context
            context = generate_context(context_dim, context_mean, context_sd, context_constant)
        end
        mul!(true_expected_rewards, bandit_param, context)
        #context = randn(context_dim) * context_sd .+ context_mean


        curr_ind = 1
        curr_max = dot((@view bandit_posterior_means[1,:]),context) + lambda_param[1]*(log(t+curr_t)^(1+lambda_param[2])) * (1 - ((t+curr_t) / (curr_t+T))^lambda_param[3]) * dot(context, (@view bandit_posterior_covs[1,:,:]), context)
        for i in 2:bandit_count
            candidate_max = dot((@view bandit_posterior_means[i,:]),context) + lambda_param[1]*(log(t+curr_t)^(1+lambda_param[2])) * (1 - ((t+curr_t) / (curr_t+T))^lambda_param[3]) * dot(context, (@view bandit_posterior_covs[i,:,:]), context)
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
        true_expected_reward = dot(bandit_posterior_means[action,:], context)
        disc_reward += true_expected_reward * discount^(t-1)
        obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:], context)) + true_expected_reward

        old_cov .= bandit_posterior_covs[action, :, :]
	    mul!(CovCon, old_cov, context)
	    CovCon ./= obs_sd

        dividy = 1 + dot(context ./ obs_sd, old_cov, context ./ obs_sd)
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

        SigInvMu .= old_cov \ @view bandit_posterior_means[action,:]
        SigInvMu .+= context .* obs ./ obs_sd.^2
        mul!((@view bandit_posterior_means[action,:]), (@view bandit_posterior_covs[action,:,:]), SigInvMu)
    end

    return disc_reward

end

function val_better_rollout(T_remainder, curr_t, rollout_length, context, use_context, lambda_param, context_dim, context_mean,
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
        true_expected_reward = dot(bandit_posterior_means[action,:], context)
        disc_reward += true_expected_reward * discount^(t-1)
        obs = randn() * sqrt(obs_sd^2 + dot(context, bandit_posterior_covs[action,:,:], context)) + true_expected_reward

        old_cov .= bandit_posterior_covs[action, :, :]
	    mul!(CovCon, old_cov, context)
	    CovCon ./= obs_sd

        dividy = 1 + dot(context ./ obs_sd, old_cov, context ./ obs_sd)
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

        SigInvMu .= old_cov \ @view bandit_posterior_means[action,:]
        SigInvMu .+= context .* obs ./ obs_sd.^2
        mul!((@view bandit_posterior_means[action,:]), (@view bandit_posterior_covs[action,:,:]), SigInvMu)
    end

    if truncation_length > 0
        
        
        #### SWITCHED TO TRUE VALUE NEURAL NETWORK AS TEST
        
        #input_vec = Vector(vec(bandit_param'))
        #append!(input_vec, Vector(vec(bandit_posterior_means')))
        input_vec = Vector(vec(bandit_posterior_means'))
        append!(input_vec, vcat([upper_triangular_vec(bandit_posterior_covs[a, :, :]) for a in 1:bandit_count]...))
        
        scaled_input_vec = (input_vec .- scale_list[truncation_length][1:end .!= 1, 1]) ./ scale_list[truncation_length][1:end .!= 1, 2] 
        
        disc_reward += (discount^min(T_remainder, rollout_length)) * (scale_list[truncation_length][1,2]*neural_net_list[truncation_length](scaled_input_vec)[1] + scale_list[truncation_length][1,1])
    
        
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

function get_features(context, bandit_posterior_means, bandit_posterior_covs)
    sorted_arms = Integer.(sortslices(hcat((bandit_posterior_means * context) .+ sqrt.([dot(context, bandit_posterior_covs[a,:,:], context) for a in 1:bandit_count]), 1:bandit_count), rev = true, dims = 1)[:, 2])
    input_vec = Vector(vec(bandit_posterior_means[sorted_arms, :]'))
    append!(input_vec, vcat([upper_triangular_vec(bandit_posterior_covs[a, :, :]) for a in sorted_arms]...))
    append!(input_vec, (bandit_posterior_means * context)[sorted_arms])
    append!(input_vec, [sqrt(dot(context, bandit_posterior_covs[a,:,:], context)) for a in sorted_arms])
    append!(input_vec, [sqrt(dot(context, bandit_posterior_covs[a,:,:], context)) for a in sorted_arms] .* (bandit_posterior_means * context)[sorted_arms])
    #append!(input_vec, context)
    return input_vec, sorted_arms
end
function get_features_2(context, bandit_posterior_means, bandit_posterior_covs)
    input_vec = bandit_posterior_means * context
    append!(input_vec, [sqrt(dot(context, bandit_posterior_covs[a,:,:], context)) for a in 1:bandit_count])
    append!(input_vec, [sqrt(dot(context, bandit_posterior_covs[a,:,:], context)) for a in 1:bandit_count] .* (bandit_posterior_means * context))
    append!(input_vec, Vector(vec(bandit_posterior_means')))
    append!(input_vec, vcat([upper_triangular_vec(bandit_posterior_covs[a, :, :]) for a in 1:bandit_count]...))
    append!(input_vec, context)
    return input_vec
end

function reduce_features(features, action)
    triang_count = div((context_dim+1)*context_dim, 2)
    top_features = [action, bandit_count + action, 2 * bandit_count + action]
    mid_features = (3 * bandit_count + 1 + (action - 1) * context_dim):(3 * bandit_count + action * context_dim)
    bottom_features = (3 * bandit_count + context_dim * bandit_count + 1 + (action - 1) * triang_count):(3 * bandit_count + context_dim * bandit_count + action * triang_count)
    action_features = vcat(top_features, mid_features, bottom_features)
    rem_features = collect(1:length(features))[setdiff((3*bandit_count + 1):end, action_features)] 
    return features[vcat(action_features, rem_features)] 
    #return features[vcat([action, bandit_count + action, 2 * bandit_count + action], (3 * bandit_count + 1):end)]
end

function dp_rollout(ep, policy, T_remainder, rollout_length, dp_length, dp_actions, context, use_context, lambda, context_dim, context_mean,
    context_sd, obs_sd, bandit_count, discount, bandit_posterior_covs, bandit_posterior_means, bandit_param,
	true_expected_rewards, CovCon, old_cov, SigInvMu)

    disc_reward = 0
    #fill!(context, 0.0)
    fill!(true_expected_rewards, 0.0)
    fill!(CovCon, 0.0)
    fill!(old_cov,0.0)
    fill!(SigInvMu,0.0)
     
    truncation_length = T_remainder - min(T_remainder, rollout_length)
    t_curr = T - T_remainder + 1
    min_remainder_dp_length = min(T_remainder, dp_length)
    
    RESULTS = [[] for i in 1:min_remainder_dp_length]
    
    for t in 1:min_remainder_dp_length


        if (t == 1) && !use_context
            context = generate_context(context_dim, context_mean, context_sd, context_constant)
        end
        old_features = get_features_2(context, bandit_posterior_means, bandit_posterior_covs)

        action = dp_actions[t]
        #action_ind = findfirst(arm_order .== action)
         
        old_features = reduce_features(old_features, action)
        push!(RESULTS[t], old_features)
        push!(RESULTS[t], action)

        
        #TRUE PARAM VERSION
        mr = maximum(bandit_param * context)
        true_expected_reward = dot(bandit_param[action, :], context)
        push!(RESULTS[t], true_expected_reward - mr)
        push!(RESULTS[t], 1 * (t == min_remainder_dp_length))
        disc_reward += (true_expected_reward-mr) * discount^(t-1)
        obs = randn() * obs_sd + true_expected_reward
        
        old_cov .= bandit_posterior_covs[action, :, :]
        
        mul!(CovCon, old_cov, context)
        CovCon ./= obs_sd

        dividy = 1 + dot(context ./ obs_sd, old_cov, context ./ obs_sd)
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

        SigInvMu .= old_cov \ @view bandit_posterior_means[action,:]
	    SigInvMu .+= context .* obs ./ obs_sd.^2
	    mul!((@view bandit_posterior_means[action,:]), (@view bandit_posterior_covs[action,:,:]), SigInvMu)
       

        context = generate_context(context_dim, context_mean, context_sd, context_constant)
        new_features = get_features_2(context, bandit_posterior_means, bandit_posterior_covs)
	    push!(RESULTS[t], new_features)  

    end
    old_bandit_posterior_means = copy(bandit_posterior_means)
    old_bandit_posterior_covs = copy(bandit_posterior_covs)
    for ii in 1:dp_within_rollout_count
        bandit_posterior_means .= old_bandit_posterior_means
        bandit_posterior_covs .= old_bandit_posterior_covs
        for t in (1+min(T_remainder, dp_length)):min(T_remainder, rollout_length)

            context = generate_context(context_dim, context_mean, context_sd, context_constant)

            action = policy(ep, t_curr+t-1, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)

            #TRUE PARAM VERSION
            true_expected_reward = dot(bandit_param[action, :], context)
            RESULTS[min_remainder_dp_length][3] += (true_expected_reward - maximum(bandit_param * context)) * discount^(t - min(T_remainder, dp_length)) / dp_within_rollout_count
            obs = randn() * obs_sd + true_expected_reward
            old_cov .= bandit_posterior_covs[action, :, :]
            mul!(CovCon, old_cov, context)
            CovCon ./= obs_sd

            dividy = 1 + dot(context ./ obs_sd, old_cov, context ./ obs_sd)
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

            SigInvMu .= old_cov \ @view bandit_posterior_means[action,:]
            SigInvMu .+= context .* obs ./ obs_sd.^2
            mul!((@view bandit_posterior_means[action,:]), (@view bandit_posterior_covs[action,:,:]), SigInvMu)
            
        end

        if truncation_length > 0
            
            if true 
                t_trunc = t_curr + min(T_remainder, rollout_length)
                reg_est = 0
                for n in 1:2
                    context = generate_context(context_dim, context_mean, context_sd, context_constant)
                    context_reg_est = 0
                    #expected_rewards = [dot(context, bandit_posterior_means[k, :]) for k in 1:bandit_count]
                    variance_rewards = [dot(context, bandit_posterior_covs[k, :, :], context) for k in 1:bandit_count]
                    #point_samps =  expected_rewards .+ sqrt.(variance_rewards) .* randn(bandit_count, 500)
                    expected_rewards = bandit_param * context
                    max_reward = maximum(expected_rewards)
                    if String(Symbol(policy)) == "thompson_policy"
                        for m in 1:500
                            action = policy(ep, t_trunc, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
                            reg_est *= ((n-1)*500 + m - 1) / ((n-1)*500 + m)
                            reg_est += (max_reward - expected_rewards[action]) / ((n-1)*500 + m)
                        end
                    else
                        action = policy(ep, t_trunc, T, bandit_count, context, bandit_posterior_means, bandit_posterior_covs, discount, epsilon, rollout_length, n_rollouts, n_opt_rollouts, context_dim)
                        #for m in 1:500
                        #reg_est *= ((n-1)*500 + m - 1) / ((n-1)*500 + m)
                        reg_est += (max_reward - expected_rewards[action]) / 2
                        #end
                    end
                    #reg_est *= (n-1) / n
                    #reg_est += (findmax(point_samps)[1] - point_samps[action]) / n
                end
                RESULTS[min_remainder_dp_length][3] -= .9 * discount^(1+min(T_remainder, rollout_length)-min(T_remainder, dp_length)) * sum(discount^(t - t_trunc) * reg_est * (1 + T - t) / (1 + T - t_trunc) for t in t_trunc:T) / dp_within_rollout_count
            
            else
                #### SWITCHED TO TRUE VALUE NEURAL NETWORK AS TEST
                
                #input_vec = Vector(vec(bandit_param'))
                #append!(input_vec, Vector(vec(bandit_posterior_means')))
                input_vec = Vector(vec(bandit_posterior_means'))
                append!(input_vec, vcat([upper_triangular_vec(bandit_posterior_covs[a, :, :]) for a in 1:bandit_count]...))
                
                scaled_input_vec = (input_vec .- scale_list[truncation_length][1:end .!= 1, 1]) ./ scale_list[truncation_length][1:end .!= 1, 2] 
                
                disc_reward += (discount^min(T_remainder, rollout_length)) * (scale_list[truncation_length][1,2]*neural_net_list[truncation_length](scaled_input_vec)[1] + scale_list[truncation_length][1,1])
            
                
                # Evaluating parameters for sanity check
                
                #test_out = []
                #append!(test_out, input_vec)
                #append!(test_out, scaled_input_vec)
                #append!(test_out, (scale_list[truncation_length][1,2]*neural_net_list[truncation_length](scaled_input_vec)[1] + scale_list[truncation_length][1,1]))
                #append!(test_out, neural_net_list[truncation_length](scaled_input_vec)[1]) 
                #print(test_out) 
            end
        end
    end
    return RESULTS

end
