from pettingzoo.mpe import simple_spread_v3
import numpy as np
from scipy.spatial.distance import cdist

EPSILON = 0.2

def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# 0 - no movement, 1 - left, 2 - right, 3 - down, 4 - up
def choose_action(position, target_pos):

    direction = target_pos - position

    if abs(direction[0]) < EPSILON and abs(direction[1]) < EPSILON:
        return 0  # No movement

    if abs(direction[0]) > abs(direction[1]):
        if direction[0] > 0:
            return 2  # Move right
        else:
            return 1  # Move left
    else:
        if direction[1] > 0:
            return 4  # Move up
        else:
            return 3  # Move down
        

env = simple_spread_v3.parallel_env(N=3, render_mode="human", max_cycles=50, local_ratio = 0.1)
observations, infos = env.reset(seed=42)

agents_targets = {agent: None for agent in env.agents}  # Track target for each agent

obs = observations["agent_0"]
self_pos = obs[2:4]
agent_1_position = obs[10:12] + self_pos
agent_2_position = obs[12:14] + self_pos
landmark_rel_positions = obs[4:10]
landmarks = np.array([landmark_rel_positions[0:2] + self_pos, landmark_rel_positions[2:4] + self_pos, landmark_rel_positions[4:6]+self_pos])

agents = np.array([self_pos, agent_1_position, agent_2_position])
landmarks = np.array([landmarks[i] for i in range(3)])

distances = cdist(agents, landmarks)

pairs = []
used_agents = np.zeros(len(agents), dtype=bool)

for landmark_index in range(len(landmarks)):
    min_dist = np.inf
    min_agent_index = None

    for agent_index in range(len(agents)):
        if used_agents[agent_index]:
            continue
        if distances[agent_index, landmark_index] < min_dist:
            min_dist = distances[agent_index, landmark_index]
            min_agent_index = agent_index

    if min_agent_index is not None:
        pairs.append((min_agent_index, landmark_index))
        used_agents[min_agent_index] = True

for i, (agent_index, landmark_index) in enumerate(pairs):
    agent = "agent_" + str(agent_index)
    agents_targets[agent] = landmarks[landmark_index]


while env.agents:
    actions = {}

    # Execute actions to move towards the chosen targets
    for agent in env.agents:
        target = agents_targets[agent]
        observation = observations[agent]
        position = observation[2:4]

        if target is not None:
            action = choose_action(position, target) 
        else:
            action = 0
        actions[agent] = action

    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(f'Rewards \n {rewards}')

env.close()