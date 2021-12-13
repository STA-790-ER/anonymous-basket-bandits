# lawson-bandits

cbPar.jl - main script for running bandit simulations
* A bandit simulation run is performed by calling contextual_bandit_simulator function, which takes global parameters specifying the bandit problem, as well as parameters determining the policy used and associated parameters/hyper-parameters. It iterates over a specified number of episodes (for the given cluster core on which the function is called)
* For each episode in contextual_bandit_simulator, a function ep_contextual_bandit_simulator is called which executes a single episode of the bandit simulation.
* Available policy functions:
  * fdas
