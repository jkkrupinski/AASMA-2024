# AASMA group project 2024
## Enviroment [(Simple Spread)](https://pettingzoo.farama.org/environments/mpe/simple_spread/)
This environment has `N` agents, `N` landmarks `(default N=3)`. At a high level, agents must learn to cover all the landmarks while avoiding collisions.

More specifically, all agents are globally rewarded based on how far the closest agent is to each landmark (sum of the minimum distances). Locally, the agents are penalized if they collide with other agents (-1 for each collision). The relative weights of these rewards can be controlled with the `local_ratio` parameter.

Agent observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

## Training
To start training use `train.py` script. Adjust hyperparameters as well as num of episodes to learn.

## Testing 
To test trained model use `test.py` script. Make sure models are named correctly and in the right folder.
Repository contains pre-trained models to see possible results named `agent_X.pt`, X is the agent number.

## Policies
- ### Independent DQL
    Agents were trained using Deep Q-Learning (DQL) with 3 sets of convolutional neural networks (CNN), one set for each agent. 
- ### Simple policy
    Agents...
- ### Advanced policy
    Agents...


## Requirements 
- Pettingzoo environment -  `pip install 'pettingzoo[mpe]' == 1.24.3` 
- PyTorch - `pip install torch == 2.2.2`
