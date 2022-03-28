
function mahalanobis_distances(thompson_samples, bandit_posterior_means, bandit_posterior_covs)
    
    distances = zeros(bandit_count)

    for i in 1:bandit_count

        distances[i] = sqrt((thompson_samples[i, :] - bandit_posterior_means[i, :])' * (bandit_posterior_covs[i, :, :] \ (thompson_samples[i, :] - bandit_posterior_means[i, :])))

    end

    return distances

end


function rand_nball_unif(nsamp, distances)

    samps = zeros(nsamp, bandit_count, context_dim)

    norms = randn(nsamp, bandit_count, context_dim)
    unifs = rand(nsamp, bandit_count)

    for i in 1:length(distances)
        distance = distances[i]
        for j in 1:nsamp
            samps[j, i, :] = (norms[j, i, :] / sqrt(sum(norms[j, i, :].^2))) * unifs[j, i]^(1 / context_dim) * distance
        end
    end

    return samps

end

function constrained_thompson_actions(nsamp, thompson_samples, bandit_posterior_means, bandit_posterior_covs, contexts)

    distances = mahalanobis_distances(thompson_samples, bandit_posterior_means, bandit_posterior_covs)
    samps = rand_nball_unif(nsamp, distances)
    action_samps = zeros(nsamp, multi_count)

    chols = zeros(bandit_count, context_dim, context_dim)
    for i in 1:bandit_count
        chols[i, :, :] = cholesky(bandit_posterior_covs[i, :, :]).L
    end


    # convert samples to parameter space
    for i in 1:bandit_count
        for j in 1:nsamp
            samps[j, i, :] = chols[i, :, :] * samps[j, i, :] + bandit_posterior_means[i, :]
        end
    end

    for i in 1:nsamp
        for j in 1:multi_count
            action_samps[i, j] = findmax(samps[i, :, :] * contexts[j, :])[2]
        end
    end

    context_outers = zeros(multi_count, context_dim, context_dim)
    for i in 1:multi_count
        context_outers[i, :, :] = contexts[i, :] * contexts[i, :]'
    end

    kl_gains = zeros(nsamp)

    for i in 1:nsamp
        actions = action_samps[i, :]
        kl_gains[i] = sum([logdet(I + bandit_posterior_covs[j, :, :] * dropdims(sum(context_outers[actions .== j, :, :], dims = 1), dims = 1) / obs_sd^2) for j = 1:bandit_count])
    end

    return action_samps[findmax(kl_gains)[2], :]

    end

function line_segment_thompson_actions(nsamp, thompson_samples, bandit_posterior_means, bandit_posterior_covs, contexts)

    samps = rand(nsamp, bandit_count)
    lincomb_samps = zeros(nsamp, bandit_count, context_dim)
    action_samps = zeros(nsamp, multi_count)

    for n in 1:nsamp
        for b in 1:bandit_count
            lincomb_samps[n, b, :] = bandit_posterior_means[b, :] + samps[n, b] * (thompson_samples[b, :] - bandit_posterior_means[b, :])
        end
    end


    for i in 1:nsamp
        for j in 1:multi_count
            action_samps[i, j] = findmax(lincomb_samps[i, :, :] * contexts[j, :])[2]
        end
    end

    context_outers = zeros(multi_count, context_dim, context_dim)
    for i in 1:multi_count
        context_outers[i, :, :] = contexts[i, :] * contexts[i, :]'
    end

    kl_gains = zeros(nsamp)

    for i in 1:nsamp
        actions = action_samps[i, :]
        kl_gains[i] = sum([logdet(I + bandit_posterior_covs[j, :, :] * dropdims(sum(context_outers[actions .== j, :, :], dims = 1), dims = 1) / obs_sd^2) for j = 1:bandit_count])
    end

    return action_samps[findmax(kl_gains)[2], :]

end

function same_line_segment_thompson_actions(nsamp, thompson_samples, bandit_posterior_means, bandit_posterior_covs, contexts)

    samps = rand(nsamp)
    lincomb_samps = zeros(nsamp, bandit_count, context_dim)
    action_samps = zeros(nsamp, multi_count)

    for n in 1:nsamp
        for b in 1:bandit_count
            lincomb_samps[n, b, :] = bandit_posterior_means[b, :] + samps[n] * (thompson_samples[b, :] - bandit_posterior_means[b, :])
        end
    end


    for i in 1:nsamp
        for j in 1:multi_count
            action_samps[i, j] = findmax(lincomb_samps[i, :, :] * contexts[j, :])[2]
        end
    end

    context_outers = zeros(multi_count, context_dim, context_dim)
    for i in 1:multi_count
        context_outers[i, :, :] = contexts[i, :] * contexts[i, :]'
    end

    kl_gains = zeros(nsamp)

    for i in 1:nsamp
        actions = action_samps[i, :]
        kl_gains[i] = sum([logdet(I + bandit_posterior_covs[j, :, :] * dropdims(sum(context_outers[actions .== j, :, :], dims = 1), dims = 1) / obs_sd^2) for j = 1:bandit_count])
    end

    return action_samps[findmax(kl_gains)[2], :]

end




function single_expected_entropy_gain(cov, contexts)
    return log(det(Matrix(1.0I, context_dim, context_dim) + cov * contexts' * contexts ./ obs_sd.^2))
end


function get_adjacent(actions, remains, bandit_count)
    adj_mat = convert(Matrix{Int64}, zeros(length(remains) * (bandit_count-1), length(actions)))
    count = 0
    for i in 1:length(remains)
        
        for j in 1:bandit_count
            if actions[remains[i]] != j
                count += 1
                adj_mat[count, :] = actions
                adj_mat[count, remains[i]] = j
            end
        end
    end
    return adj_mat
end

function get_adjacent_restricted(actions, restrictions, bandit_count)
    adj_mat = convert(Matrix{Int64}, zeros(sum(restrictions) - multi_count, length(actions)))
    count = 0
    for i in 1:multi_count
        
        for j in 1:bandit_count
            if actions[i] != j && restrictions[i, j] == 1
                count += 1
                adj_mat[count, :] = actions
                adj_mat[count, i] = j
            end
        end
    end
    return adj_mat
end
function expected_entropy_gain(bandit_posterior_covs, contexts, actions)
    gain = 0
    for b in 1:bandit_count
       gain += single_expected_entropy_gain(@view(bandit_posterior_covs[b, :, :]), @view(contexts[actions .== b, :]))
    end
    return gain
end


function sa_step(bandit_posterior_covs, contexts, actions, remains, bandit_count, temperature)
    adj_mat = get_adjacent(actions, remains, bandit_count)
    m = size(adj_mat)[1]
    candidate = sample(1:m)
    candidate_score = expected_entropy_gain(bandit_posterior_covs, contexts, adj_mat[candidate, :])
    current_score = expected_entropy_gain(bandit_posterior_covs, contexts, actions)
    acceptance_prob = min(1, exp((candidate_score - current_score) / temperature))
    if rand() < acceptance_prob
        return adj_mat[candidate, :], candidate_score
    else
        return actions, current_score
    end
end
    
function sa(bandit_posterior_covs, contexts, actions, remains, niter, initial_temperature, cooling_parameter)
    

    max_actions = actions
    max_score = expected_entropy_gain(bandit_posterior_covs, contexts, actions)
    temperature = initial_temperature
    for i in 1:niter
        
        actions, actions_score = sa_step(bandit_posterior_covs, contexts, actions, remains, bandit_count, temperature)

        if actions_score > max_score
            max_actions = actions
            max_score = actions_score
        end
        temperature *= cooling_parameter
        #print("\n Done With Iteration \n")
    end

    return max_actions
end

function sa_step_restricted(bandit_posterior_covs, contexts, actions, restrictions, bandit_count, temperature)
    adj_mat = get_adjacent_restricted(actions, restrictions, bandit_count)
    m = size(adj_mat)[1]
    candidate = sample(1:m)
    candidate_score = expected_entropy_gain(bandit_posterior_covs, contexts, adj_mat[candidate, :])
    current_score = expected_entropy_gain(bandit_posterior_covs, contexts, actions)
    acceptance_prob = min(1, exp((candidate_score - current_score) / temperature))
    if rand() < acceptance_prob
        return adj_mat[candidate, :], candidate_score
    else
        return actions, current_score
    end
end
    
function sa_restricted(bandit_posterior_covs, contexts, actions, restrictions, niter, initial_temperature, cooling_parameter)
    

    max_actions = actions
    max_score = expected_entropy_gain(bandit_posterior_covs, contexts, actions)
    temperature = initial_temperature
    for i in 1:niter
        
        actions, actions_score = sa_step_restricted(bandit_posterior_covs, contexts, actions, restrictions, bandit_count, temperature)

        if actions_score > max_score
            max_actions = actions
            max_score = actions_score
        end
        temperature *= cooling_parameter
    end

    return max_actions
end
# generate a first pass estimate of the max entropy action set by greedily sequentially constructing the action set
function sequential_max_entropy_step(bandit_posterior_covs, contexts, actions)

    max_score = 0
    max_actions = actions

    for i in 1:length(actions)
        if actions[i] == 0
            temp_actions = copy(max_actions)
            for j in 1:bandit_count
                temp_actions[i] = j
                eeg = expected_entropy_gain(bandit_posterior_covs, contexts[temp_actions .!= 0, :], temp_actions[temp_actions .!= 0])
                if eeg > max_score
                    max_score = eeg
                    max_actions = temp_actions
                end
            end
        end
    end

    return max_actions
end

function sequential_max_entropy(bandit_posterior_covs, contexts, actions)
    seq_actions = copy(actions)

    while 0 in seq_actions
        seq_actions = sequential_max_entropy_step(bandit_posterior_covs, contexts, seq_actions)
    end

    return seq_actions


end


### INFORMATION DIRECTED SAMPLING

function expected_regrets_2(contexts, bandit_posterior_means, bandit_posterior_covs, niter)
    res = zeros(multi_count, bandit_count)
    for m in 1:multi_count
        means = [dot(bandit_posterior_means[b, :], contexts[m, :]) for b = 1:bandit_count]
        vars = [dot(contexts[m, :], bandit_posterior_covs[b, :, :], contexts[m, :]) for b = 1:bandit_count]
        draws = rand(MvNormal(means, diagm(vars)), niter)

        for i in 1:niter
            draws[:, i] = findmax(draws[:, i])[1] .- draws[:, i]
        end
        for b in 1:bandit_count
            res[m, b] = mean(draws[b, :])
        end
    end
    return res
end

function expected_regrets(contexts, bandit_posterior_means, bandit_posterior_covs, niter)

    draws = zeros(bandit_count, context_dim, niter)
    reward_draws = zeros(multi_count, bandit_count, niter)
    
    for b in 1:bandit_count
        draws[b, :, :] = rand(MvNormal(bandit_posterior_means[b, :], bandit_posterior_covs[b, :, :]), niter)
        for i in 1:niter
            reward_draws[:, b, i] = contexts * draws[b, :, i]
        end
    end
    
    mean_rewards = dropdims(mean(reward_draws, dims = 3), dims = 3)

    mean_max_reward = 0
    for i in 1:niter
        mean_max_reward += sum([findmax(reward_draws[m, :, i])[1] for m = 1:multi_count]) / niter
    end
    
    res = mean_max_reward/multi_count .- mean_rewards

    return res
end



function information_ratio(bandit_posterior_covs, contexts, actions, regrets)
    gain = expected_entropy_gain(bandit_posterior_covs, contexts, actions)
    ret = 0
    for m in 1:multi_count
        ret += regrets[m, actions[m]]
    end
    return ret^2 / gain
end

function sa_ids(bandit_posterior_covs, bandit_posterior_means, contexts, actions, niter, initial_temperature, cooling_parameter)
    
    min_actions = actions
    regrets = expected_regrets(contexts, bandit_posterior_means, bandit_posterior_covs, 10000)
    min_score = information_ratio(bandit_posterior_covs, contexts, actions, regrets)
    temperature = initial_temperature
    for i in 1:niter
        
        actions, actions_score = sa_ids_step(bandit_posterior_covs, contexts, actions, regrets, bandit_count, temperature)

        if actions_score < min_score
            min_actions = actions
            min_score = actions_score
        end
        temperature *= cooling_parameter
        #print("\n Done With Iteration \n")
    end
    return min_actions
end

function sa_ids_step(bandit_posterior_covs, contexts, actions, regrets, bandit_count, temperature)
    candidate_subject = sample(1:multi_count)
    candidate_bandit = sample(1:(bandit_count-1))
    if candidate_bandit >= actions[candidate_subject]
        candidate_bandit += 1
    end
    candidate_actions = copy(actions)
    candidate_actions[candidate_subject] = candidate_bandit

    candidate_score = information_ratio(bandit_posterior_covs, contexts, candidate_actions, regrets)
    current_score = information_ratio(bandit_posterior_covs, contexts, actions, regrets)
    acceptance_prob = min(1, exp((current_score - candidate_score) / temperature))
    if rand() < acceptance_prob
        return candidate_actions, candidate_score
    else
        return actions, current_score
    end
end


## DUAL THOMPSON

function sa_dual_thompson(bandit_posterior_covs, bandit_posterior_means, contexts, actions, niter, initial_temperature, cooling_parameter)
    
    max_actions = actions
    min_eeg = expected_entropy_gain(bandit_posterior_covs, contexts, actions)
    max_score = sum([dot(bandit_posterior_means[actions[m], :], contexts[m, :]) for m = 1:multi_count])
    temperature = initial_temperature
    i=1
    infeasible_counter = 1
    
    max_niter = 5 * niter
    max_i = 1
    while i <= niter
        if max_i > max_niter
            break
        end
        actions, actions_score, infeasible_counter = sa_dual_thompson_step(bandit_posterior_covs, bandit_posterior_means, contexts, actions, bandit_count, temperature, infeasible_counter, min_eeg)

        if actions_score > max_score
            max_actions = actions
            max_score = actions_score
        end
        temperature *= cooling_parameter
        #print("\n Done With Iteration \n")
        
        if infeasible_counter == 1
            i += 1
        end
        max_i += 1
    end
    return max_actions
end


function sa_dual_thompson_step(bandit_posterior_covs, bandit_posterior_means, contexts, actions, bandit_count, temperature, infeasible_counter, min_eeg)
    
    candidate_subject = sample(1:multi_count)
    candidate_bandit = sample(1:(bandit_count-1))
    if candidate_bandit >= actions[candidate_subject]
        candidate_bandit += 1
    end
    candidate_actions = copy(actions)
    candidate_actions[candidate_subject] = candidate_bandit

    candidate_eeg = expected_entropy_gain(bandit_posterior_covs, contexts, candidate_actions)
    current_eeg = expected_entropy_gain(bandit_posterior_covs, contexts, actions)

    # ESCALATING PENALTY FOR VIOLATING FEASIBILITY
    candidate_score = sum([dot(bandit_posterior_means[candidate_actions[m], :], contexts[m, :]) for m = 1:multi_count]) - infeasible_counter * max(0, min_eeg - candidate_eeg)
    current_score = sum([dot(bandit_posterior_means[actions[m], :], contexts[m, :]) for m = 1:multi_count]) - infeasible_counter * max(0, min_eeg - current_eeg)
    acceptance_prob = min(1, exp((candidate_score - current_score) / temperature))
    if rand() < acceptance_prob

        if candidate_eeg < min_eeg
            infeasible_counter += 1
        else
            infeasible_counter = 1
        end

        return candidate_actions, candidate_score, infeasible_counter
    else

        if current_eeg < min_eeg
            infeasible_counter += 1
        else
            infeasible_counter = 1
        end

        return actions, current_score, infeasible_counter
    end
end

# THOMPSON IDS

function sa_ids_step_restricted(bandit_posterior_covs, contexts, actions, regrets, restrictions, bandit_count, temperature)
    adj_mat = get_adjacent_restricted(actions, restrictions, bandit_count)
    m = size(adj_mat)[1]
    candidate = sample(1:m)
    candidate_score = information_ratio(bandit_posterior_covs, contexts, adj_mat[candidate, :], regrets)
    current_score = information_ratio(bandit_posterior_covs, contexts, actions, regrets)
    acceptance_prob = min(1, exp((current_score - candidate_score) / temperature))
    if rand() < acceptance_prob
        return adj_mat[candidate, :], candidate_score
    else
        return actions, current_score
    end
end
    
function sa_ids_restricted(bandit_posterior_covs, bandit_posterior_means, contexts, actions, restrictions, niter, initial_temperature, cooling_parameter)
    
    min_actions = actions
    regrets = expected_regrets(contexts, bandit_posterior_means, bandit_posterior_covs, 10000)
    min_score = information_ratio(bandit_posterior_covs, contexts, actions, regrets)

    temperature = initial_temperature
    for i in 1:niter
        
        actions, actions_score = sa_ids_step_restricted(bandit_posterior_covs, contexts, actions, regrets, restrictions, bandit_count, temperature)

        if actions_score < min_score
            min_actions = actions
            min_score = actions_score
        end
        temperature *= cooling_parameter
    end

    return min_actions
end







