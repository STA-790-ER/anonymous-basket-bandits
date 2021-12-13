# lawson-bandits

cbPar.jl - main script for running bandit simulations
* A bandit simulation run is performed by calling contextual_bandit_simulator function, which takes global parameters specifying the bandit problem, as well as parameters determining the policy used and associated parameters/hyper-parameters. It iterates over a specified number of episodes (for the given cluster core on which the function is called)
* For each episode in contextual_bandit_simulator, a function ep_contextual_bandit_simulator is called which executes a single episode of the bandit simulation.
* Available policy functions:
  * greedy_policy - selects at each time step the arm with maximal mean reward for the given context
  * epsilon_greedy_policy - with probability 1-&epsilon; follows the greedy policy and otherwise behaves randomly
  * Bayes-UCB - computes for each arm the 1-1/[t * log(n)^c] contextual reward quantile, and selects the arm witht he largest such quantile (Kaufmann et al 2012)
  * Thompson - selects each arm with the posterior probability that it is optimal for a given context
  * Lambda - select the arm which maximizes a convex combination of the contextual posterior mean and covariance of rewards
  * Optmised Lambda - Executes the Lambda policy but optimizes the convex combination hyperparameter at each time step
  * Rollout Optimised Lambda - estimates the time t Q function under the Optimised Lambda policy and chooses the arm that maximizes the Q function
  * Truncated Rollout/Value Net Policies - Performs rollout as above, but does not roll out to the end of the simulation horizon, and the end of the truncated rollout period adds 
    adds
