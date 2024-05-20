from pettingzoo.mpe import simple_spread_v3
import random
import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, input_shape, output_actions):
        super().__init__()

        self.model = self.create_model(input_shape, output_actions)

    def create_model(self, input_shape, output_actions):
        layers = [
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_actions),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def load_models():
    input_shape = 18
    output_actions = 5

    model_0 = DQN(input_shape, output_actions)
    model_0.load_state_dict(torch.load("models/agent_0.pt"))

    model_1 = DQN(input_shape, output_actions)
    model_1.load_state_dict(torch.load("models/agent_1.pt"))

    model_2 = DQN(input_shape, output_actions)
    model_2.load_state_dict(torch.load("models/agent_2.pt"))

    models = {"agent_0": model_0, "agent_1": model_1, "agent_2": model_2}
    return models


# seeds
# 73, 69! 45  40 24 85 10!


def test():

    NUM_CYCLES = 25

    env = simple_spread_v3.parallel_env(
        render_mode="human", local_ratio=0.5, max_cycles=NUM_CYCLES
    )

    seed = random.randint(0, 100)
    print("Seed: ", seed)
    observations, _ = env.reset(seed=seed)

    models = load_models()
    avg_rewards = {"agent_0": 0, "agent_1": 0, "agent_2": 0}

    while env.agents:

        actions = {
            agent: models[agent](torch.from_numpy(observations[agent])).argmax().item()
            for agent in env.agents
        }
        observations, rewards, _, _, _ = env.step(actions)

        for agent in env.agents:
            avg_rewards[agent] += rewards[agent] / NUM_CYCLES

    print(
        "AVG REWARDS: agent_0:",
        round(avg_rewards["agent_0"], 2),
        ", agent_1:",
        round(avg_rewards["agent_1"], 2),
        ", agent_2:",
        round(avg_rewards["agent_2"], 2),
    )

    env.close()


if __name__ == "__main__":
    test()
