from pettingzoo.mpe import simple_spread_v3
import tensorflow as tf
import numpy as np

env = simple_spread_v3.env(render_mode="human")
env.reset(seed=42)


model0 = tf.keras.models.load_model("models/S0.keras")
model1 = tf.keras.models.load_model("models/S1.keras")
model2 = tf.keras.models.load_model("models/S2.keras")

models = {"agent_0": model0, "agent_1": model1, "agent_2": model2}


for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = np.argmax(models[agent](tf.expand_dims(observation,0)))    

    env.step(action)
env.close()
