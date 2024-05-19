from pettingzoo.mpe import simple_spread_v3
import tensorflow as tf
import numpy as np
import random

# 48 300
# 83 25

env = simple_spread_v3.parallel_env(render_mode="human", local_ratio=0.5, max_cycles=25)
seed = random.randint(0,100)
print("Seed: ", seed) 
observations, infos = env.reset(seed=42)

model_episode = "fin"

model0 = tf.keras.models.load_model(
    "models/model-" + model_episode + "-0.keras"
)
model1 = tf.keras.models.load_model(
    "models/model-" + model_episode + "-1.keras"
)
model2 = tf.keras.models.load_model(
    "models/model-" + model_episode + "-2.keras"
)

models = {"agent_0": model0, "agent_1": model1, "agent_2": model2}
sum_rewards = {"agent_0": 0, "agent_1": 0, "agent_2": 0}


while env.agents:

    actions = {
        agent: np.argmax(models[agent](tf.expand_dims(observations[agent], 0)))
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
