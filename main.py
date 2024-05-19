from pettingzoo.mpe import simple_spread_v3
import tensorflow as tf
import numpy as np
import random
import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, input_shape, output_actions):
        super().__init__()

        self.input_shape = input_shape
        self.output_actions = output_actions

        self.model = self.create_model()

    def create_model(self):
        layers = [
            nn.Linear(self.input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_actions),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 48 300
# 83 25

env = simple_spread_v3.parallel_env(render_mode="human", local_ratio=0.5, max_cycles=25)
seed = random.randint(0, 100)
print("Seed: ", seed)
observations, infos = env.reset(seed=seed)

# pytorch
model0 = DQN(18, 5)
model0.load_state_dict(torch.load("models/10000ep/agent_0_simple_spread_dql_cnn.pt"))
model1 = DQN(18, 5)
model1.load_state_dict(torch.load("models/10000ep/agent_1_simple_spread_dql_cnn.pt"))
model2 = DQN(18, 5)
model2.load_state_dict(torch.load("models/10000ep/agent_2_simple_spread_dql_cnn.pt"))

# tensorflow
# model_episode = "fin"
# model0 = tf.keras.models.load_model(
#     "models/model-" + model_episode + "-0.keras"
# )
# model1 = tf.keras.models.load_model(
#     "models/model-" + model_episode + "-1.keras"
# )
# model2 = tf.keras.models.load_model(
#     "models/model-" + model_episode + "-2.keras"
# )

models = {"agent_0": model0, "agent_1": model1, "agent_2": model2}
sum_rewards = {"agent_0": 0, "agent_1": 0, "agent_2": 0}

while env.agents:

    actions = {
        agent: models[agent](torch.from_numpy(observations[agent])).argmax().item()
        for agent in env.agents
    }
    observations, rewards, terminations, truncations, infos = env.step(actions)

    for agent in env.agents:
        sum_rewards[agent] += rewards[agent] / 25.0


print(
    "AVG REWARDS: agent_0:",
    round(sum_rewards["agent_0"], 2),
    ", agent_1:",
    round(sum_rewards["agent_1"], 2),
    ", agent_2:",
    round(sum_rewards["agent_2"], 2),
)

env.close()
