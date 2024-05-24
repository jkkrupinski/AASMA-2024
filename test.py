from pettingzoo.mpe import simple_spread_v3
import random
import torch
import argparse
import numpy as np
from scipy.spatial.distance import cdist

from train import DQN


# Actions
NOTHING = 0
LEFT = 1
RIGHT = 2
DOWN = 3
UP = 4


class simplePolicy:
    def __init__(self, agent):
        self.agent = agent
        self.epsilon = 0.1

    def distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def choose_target(self, observation):
        self.position = observation[2:4]

        landmark_rel_positions = observation[4:10]
        landmarks = np.array(
            [
                landmark_rel_positions[0:2],
                landmark_rel_positions[2:4],
                landmark_rel_positions[4:6],
            ]
        )

        distances = []
        for landmark in landmarks:
            distances.append(self.distance(landmark, self.position))

        self.target_pos = landmarks[distances.index(min(distances))]

    def choose_action(self, observations):

        observation = observations[self.agent]
        self.choose_target(observation)

        if (
            abs(self.target_pos[0]) < self.epsilon
            and abs(self.target_pos[1]) < self.epsilon
        ):
            return NOTHING

        if abs(self.target_pos[0]) > abs(self.target_pos[1]):
            if self.target_pos[0] > self.epsilon:
                return RIGHT
            else:
                return LEFT

        else:
            if self.target_pos[1] > self.epsilon:
                return UP
            else:
                return DOWN


class complexPolicy:
    def __init__(self, agent):
        self.agent = agent
        self.epsilon = 0.2

    def choose_action(self, observations):

        agents_targets = self.choose_targets(observations)

        target = agents_targets[self.agent]
        observation = observations[self.agent]
        position = observation[2:4]

        direction = target - position

        if abs(direction[0]) < self.epsilon and abs(direction[1]) < self.epsilon:
            return NOTHING

        if abs(direction[0]) > abs(direction[1]):
            if direction[0] > 0:
                return RIGHT
            else:
                return LEFT
        else:
            if direction[1] > 0:
                return UP
            else:
                return DOWN

    def choose_targets(self, observations):
        agents_targets = {}

        obs = observations["agent_0"]

        self_pos = obs[2:4]
        agent_1_position = obs[10:12] + self_pos
        agent_2_position = obs[12:14] + self_pos

        landmark_rel_positions = obs[4:10]
        landmarks = np.array(
            [
                landmark_rel_positions[0:2] + self_pos,
                landmark_rel_positions[2:4] + self_pos,
                landmark_rel_positions[4:6] + self_pos,
            ]
        )

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

        for agent_index, landmark_index in pairs:
            agent = "agent_" + str(agent_index)
            agents_targets[agent] = landmarks[landmark_index]

        return agents_targets


class DQLPolicy:
    def __init__(self, agent):
        self.agent = agent
        self.load_models()

    def load_models(self):
        input_shape = 18
        output_actions = 5

        self.model = DQN(input_shape, output_actions)
        self.model.load_state_dict(torch.load("models/" + self.agent + ".pt"))

    def choose_action(self, observations):
        return self.model(torch.from_numpy(observations[self.agent])).argmax().item()


def init_policies(policies):
    agents = ["agent_0", "agent_1", "agent_2"]
    agent_policies = {}

    for i, agent in enumerate(agents):
        if policies[i] == "rl":
            agent_policies[agent] = DQLPolicy(agent)
        elif policies[i] == "sp":
            agent_policies[agent] = simplePolicy(agent)
        elif policies[i] == "cp":
            agent_policies[agent] = complexPolicy(agent)

    return agent_policies


def choose_seed(seeds, run_num):
    if seeds:
        return seeds[run_num]
    else:
        return random.randint(0, 1000)


def print_rewards(chosen_seed, average_rewards, rewards):
    print(
        "AVERAGE REWARDS [",
        chosen_seed,
        "]: agent_0:",
        round(average_rewards["agent_0"], 2),
        ", agent_1:",
        round(average_rewards["agent_1"], 2),
        ", agent_2:",
        round(average_rewards["agent_2"], 2),
    )
    print(
        "FINAL REWARDS [",
        chosen_seed,
        "]: agent_0:",
        round(rewards["agent_0"], 2),
        ", agent_1:",
        round(rewards["agent_1"], 2),
        ", agent_2:",
        round(rewards["agent_2"], 2),
    )
    print()


# seeds
# 73, 69! 45  40 24 85 10!


def test(policies, num_of_runs, seeds=None, DEBUG=False):

    if DEBUG:
        print(f"Policies: {policies}")
        print(f"Number of runs: {num_of_runs}")

        if seeds:
            print(f"Seeds: {seeds}")
        else:
            print("No seeds provided")

    NUM_CYCLES = 25
    LOCAL_RATIO = 0.5

    env = simple_spread_v3.parallel_env(
        render_mode="human", local_ratio=LOCAL_RATIO, max_cycles=NUM_CYCLES
    )

    agent_policies = init_policies(policies)

    for run_num in range(num_of_runs):

        chosen_seed = choose_seed(seeds, run_num)
        observations, _ = env.reset(seed=chosen_seed)

        average_rewards = {"agent_0": 0, "agent_1": 0, "agent_2": 0}
        actions = {}

        while env.agents:

            for agent in env.agents:
                actions[agent] = agent_policies[agent].choose_action(observations)

            observations, rewards, _, _, _ = env.step(actions)

            for agent in env.agents:
                average_rewards[agent] += rewards[agent] / NUM_CYCLES

        print_rewards(chosen_seed, average_rewards, rewards)

    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Test different policies for agents for certain number of runs with choosing the seed values."
    )

    parser.add_argument(
        "policies",
        nargs=3,
        choices=["rl", "sp", "cp"],
        help="Policies for agent_0, agent_1, agent_2. Possible options are: 'rl' - reinforement learning, 'sp' - simple policy, 'cp' - complex policy.",
    )
    parser.add_argument(
        "num_of_runs", type=int, help="Number of runs to test policies."
    )
    parser.add_argument(
        "seeds", type=int, nargs="*", help="Optional seed values for each of the runs."
    )

    args = parser.parse_args()

    if args.seeds and len(args.seeds) != args.num_of_runs:
        parser.error("The number of seeds provided must match number of runs.")

    test(args.policies, args.num_of_runs, args.seeds)
