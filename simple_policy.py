from pettingzoo.mpe import simple_spread_v3
import numpy as np

EPSILON = 0.1


def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


# 0 - no movement, 1 - left, 2 - right, 3 - down, 4 - up
def choose_action(observations):

    obs = observations[agent]
    self_pos = obs[2:4]
    landmark_rel_positions = obs[4:10]
    landmarks = np.array(
        [
            landmark_rel_positions[0:2],
            landmark_rel_positions[2:4],
            landmark_rel_positions[4:6],
        ]
    )

    distances = []
    for landmark in landmarks:
        distances.append(distance(landmark, self_pos))

    target_pos = landmarks[distances.index(min(distances))]

    print(f"Agent: {agent}, Position: {self_pos}, Nearest Landmark: {target_pos}")
    print()

    if abs(target_pos[0]) < EPSILON and abs(target_pos[1]) < EPSILON:
        return 0

    if abs(target_pos[0]) > abs(target_pos[1]):
        if target_pos[0] > EPSILON:
            return 2
        else:
            return 1

    else:
        if target_pos[1] > EPSILON:
            return 4
        else:
            return 3


env = simple_spread_v3.parallel_env(N=3, render_mode="human", max_cycles=100)
observations, infos = env.reset(seed=None)  # 42

while env.agents:
    actions = {}

    for agent in env.agents:

        actions[agent] = choose_action(observations)

    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(f"Rewards: {rewards}")


env.close()
