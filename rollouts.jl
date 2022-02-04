
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
            context = randn(context_dim) * context_sd .+ context_mean
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
            context = randn(context_dim) * context_sd .+ context_mean
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
    fill!(context, 0.0)
    fill!(true_expected_rewards, 0.0)
    fill!(CovCon, 0.0)
    fill!(old_cov,0.0)
    fill!(SigInvMu,0.0)
    
    truncation_length = T_remainder - min(T_remainder, rollout_length)

    for t in 1:min(T_remainder, rollout_length)


        if t > 1 || !use_context

            context = randn(context_dim) * context_sd .+ context_mean

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

function val_rollout(policy, T_remainder, rollout_length, context, use_context, lambda, context_dim, context_mean,
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
            context = randn(context_dim) * context_sd .+ context_mean
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
