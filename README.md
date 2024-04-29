# AASMA group project 2024
## Enviroment description [(link)](https://pettingzoo.farama.org/environments/mpe/simple_spread/)
This environment has N agents, N landmarks (default N=3). At a high level, agents must learn to cover all the landmarks while avoiding collisions.

More specifically, all agents are globally rewarded based on how far the closest agent is to each landmark (sum of the minimum distances). Locally, the agents are penalized if they collide with other agents (-1 for each collision). The relative weights of these rewards can be controlled with the local_ratio parameter.

Agent observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`


## Instalation 
`pip install 'pettingzoo[mpe]`

`