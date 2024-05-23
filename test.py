from pettingzoo.mpe import simple_spread_v3
import random
import torch
import argparse

from train import DQN


class simplePolicy:
    pass


class complexPolicy:
    pass


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
        return random.randint(0, 100)


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
