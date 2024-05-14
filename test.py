from pettingzoo.mpe import simple_spread_v3
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


env = simple_spread_v3.parallel_env()


class ActionValueNetwork:
    def __init__(self, network_config):
        self.state_dim = network_config.get("state_dim")
        self.num_actions = network_config.get("num_actions")
        self.step_size = network_config.get("step_size")

    def create_model(self):
        i = Input(shape=self.state_dim)
        x = Dense(256, activation="relu")(i)
        x = Dense(128, activation="relu")(x)
        x = Dense(self.num_actions, activation="linear")(x)
        model = Model(i, x)
        model.compile(optimizer=Adam(learning_rate=self.step_size), loss="mse")
        return model


epsilon = 1
EPSILON_DECAY = 0.998
MIN_EPSILON = 0.01


class ReplayBuffer:
    def __init__(self, size, minibatch_size, seed):

        self.buffer = []
        self.minibatch_size = minibatch_size
        self.rand_generator = np.random.RandomState(seed)
        self.max_size = size

    def append(self, state, action, reward, terminal, next_state):

        if len(self.buffer) == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, reward, terminal, next_state])

    def sample(self):

        idxs = self.rand_generator.choice(
            np.arange(len(self.buffer)), size=self.minibatch_size
        )
        return [self.buffer[idx] for idx in idxs]

    def size(self):
        return len(self.buffer)


agent_info = {
    "network_config": {"state_dim": (18,), "num_actions": 5, "step_size": 1e-3},
    "replay_buffer_size": 256,
    "minibatch_sz": 32,
    "num_replay_updates_per_step": 2,
    "gamma": 0.99,
    "seed": 0,
}


class Agent:
    def __init__(self, agent_config):
        replay_buffer0 = ReplayBuffer(
            agent_config["replay_buffer_size"],
            agent_config["minibatch_sz"],
            agent_config.get("seed"),
        )
        replay_buffer1 = ReplayBuffer(
            agent_config["replay_buffer_size"],
            agent_config["minibatch_sz"],
            agent_config.get("seed"),
        )
        replay_buffer2 = ReplayBuffer(
            agent_config["replay_buffer_size"],
            agent_config["minibatch_sz"],
            agent_config.get("seed"),
        )

        self.replay_buffer = {
            "agent_0": replay_buffer0,
            "agent_1": replay_buffer1,
            "agent_2": replay_buffer2,
        }

        self.network = ActionValueNetwork(agent_config["network_config"])

        model_0 = self.network.create_model()
        target_model_0 = self.network.create_model()

        model_1 = self.network.create_model()
        target_model_1 = self.network.create_model()

        model_2 = self.network.create_model()
        target_model_2 = self.network.create_model()

        self.models = {"agent_0": model_0, "agent_1": model_1, "agent_2": model_2}
        self.target_models = {
            "agent_0": target_model_0,
            "agent_1": target_model_1,
            "agent_2": target_model_2,
        }

        self.num_actions = agent_config["network_config"]["num_actions"]

        self.num_replay = agent_config["num_replay_updates_per_step"]
        self.discount = agent_config["gamma"]

        self.rand_generator = np.random.RandomState(agent_config.get("seed"))
        self.last_states = None
        self.actions = None
        self.epsilon = epsilon
        self.sum_rewards = {"agent_0": 0, "agent_1": 0, "agent_2": 0}
        self.episode_steps = 0

    def policy(self, agent, state):
        action_values = self.models[agent].predict(state, verbose=0)
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, env.action_spaces[agent].n)
        else:
            action = np.argmax(action_values)
        return int(action)

    def agent_start(self):
        self.sum_rewards = {"agent_0": 0, "agent_1": 0, "agent_2": 0}
        self.episode_steps = 0

        double_dict = (
            env.reset()
        )  # it returns a tuple of 2 dict second one empty (why?)
        self.last_states = double_dict[0]

        for agent in env.agents:
            self.last_states[agent] = np.array([self.last_states[agent]])

        self.actions = {
            agent: self.policy(agent, self.last_states[agent]) for agent in env.agents
        }
        actions = self.actions

        return actions

    def agent_step(self, states, rewards, terminals):
        # self.sum_rewards = {
        #     agent: (self.sum_rewards[agent] + rewards[agent]) for agent in rewards
        # }
        for agent in rewards:
            self.sum_rewards[agent] += rewards[agent]

        self.episode_steps += 1
        for i in states:
            states[i] = np.array([states[i]])

        for agent in states:
            state = states[agent]
            last_state = self.last_states[agent]
            action = self.actions[agent]
            reward = rewards[agent]
            terminal = terminals[agent]
            self.replay_buffer[agent].append(
                last_state, action, reward, terminal, state
            )

        for agent in self.replay_buffer:
            if (
                self.replay_buffer[agent].size()
                > self.replay_buffer[agent].minibatch_size
            ):
                self.target_models[agent].set_weights(self.models[agent].get_weights())
                for _ in range(self.num_replay):
                    experiences = self.replay_buffer[agent].sample()
                    self.agent_train(experiences, agent)

        self.last_states = states
        self.actions = {
            agent: self.policy(agent, self.last_states[agent]) for agent in env.agents
        }
        actions = self.actions
        return actions

    def agent_train(self, experiences, agent):

        states, actions, rewards, terminals, next_states = map(list, zip(*experiences))

        states = np.concatenate(states)
        next_states = np.concatenate(next_states)

        rewards = np.array(rewards)
        terminals = np.array(terminals)
        batch_size1 = states.shape[0]

        q_next_mat = self.target_models[agent].predict(next_states, verbose=0)

        v_next_vec = np.max(q_next_mat, axis=1) * (1 - terminals)

        target_vec = rewards + self.discount * v_next_vec
        q_mat = self.models[agent].predict(states, verbose=0)
        batch_indices = np.arange(q_mat.shape[0])
        X = states
        q_mat[batch_indices, actions] = target_vec
        self.models[agent].fit(
            X, q_mat, batch_size=batch_size1, verbose=0, shuffle=False
        )


dqn = Agent(agent_info)

episode_rewards = []
episode_steps = []
episode_epsilon = []
num_episodes = 2

for episode in range(0, num_episodes):

    actions = dqn.agent_start()

    for step in range(100):

        next_states, rewards, dones, infos, _ = env.step(actions)
        terminals = {agent: 1 if dones[agent] == True else 0 for agent in dones}
        actions = dqn.agent_step(next_states, rewards, terminals)

        if dqn.epsilon > MIN_EPSILON:

            dqn.epsilon *= EPSILON_DECAY
            dqn.epsilon = max(MIN_EPSILON, dqn.epsilon)
    print(
        "EPISODE", episode, " Epsilon = ", dqn.epsilon, " STEPS = ", dqn.episode_steps
    )
    print("REWARD", dqn.sum_rewards)

    episode_rewards.append(dqn.sum_rewards)
    episode_steps.append(dqn.episode_steps)
    episode_epsilon.append(dqn.epsilon)


avg_rewards_0 = []
avg_rewards_1 = []
avg_rewards_2 = []
rewards_0 = []
rewards_1 = []
rewards_2 = []

for m in episode_rewards:
    for agent in m:
        if agent == "agent_0":
            rewards_0.append(m[agent])

        if agent == "agent_1":
            rewards_1.append(m[agent])

        if agent == "agent_2":
            rewards_2.append(m[agent])
for i in range(0, num_episodes):
    k = np.mean(rewards_0[i : i + 100])
    k = np.round(k, 3)
    avg_rewards_0.append(k)
for i in range(0, num_episodes):
    k = np.mean(rewards_1[i : i + 100])
    k = np.round(k, 3)
    avg_rewards_1.append(k)
for i in range(0, num_episodes):
    k = np.mean(rewards_2[i : i + 100])
    k = np.round(k, 3)
    avg_rewards_2.append(k)
import matplotlib.pyplot as plt

no_episodes = []
for i in range(0, num_episodes):
    no_episodes.append(i)
np.mean(rewards_0) + np.mean(rewards_1) + np.mean(rewards_2)


plt.figure(figsize=(10, 5))
plt.plot(no_episodes, rewards_0, color="red", linestyle="-", label="Total Reward0")

plt.plot(
    no_episodes,
    avg_rewards_0,
    color="midnightblue",
    linestyle="--",
    label="Episodes Reward Average0",
)


#plt.grid(b=True, which="major", axis="y", linestyle="--")
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Reward", fontsize=12)
plt.title("Total Reward per Testing Episode", fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.show()


plt.plot(no_episodes, rewards_1, color="blue", linestyle="-", label="Total Reward1")
plt.plot(
    no_episodes,
    avg_rewards_1,
    color="black",
    linestyle="--",
    label="Episodes Reward Average1",
)

#plt.grid(b=True, which="major", axis="y", linestyle="--")
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Reward", fontsize=12)
plt.title("Total Reward per Testing Episode", fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.show()


plt.plot(no_episodes, rewards_2, color="green", linestyle="-", label="Total Reward2")
plt.plot(
    no_episodes,
    avg_rewards_2,
    color="pink",
    linestyle="--",
    label="Episodes Reward Average2",
)

#plt.grid(b=True, which="major", axis="y", linestyle="--")
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Reward", fontsize=12)
plt.title("Total Reward per Testing Episode", fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.show()


total_avg_reward = []
for i in range(0, num_episodes):
    k = np.mean(rewards_0[i] + rewards_1[i] + rewards_2[i])
    k = np.round(k, 3)
    total_avg_reward.append(k)
avg_total = []
for i in range(0, num_episodes):
    k = np.mean(total_avg_reward[i : i + 100])
    k = np.round(k, 3)
    avg_total.append(k)
plt.plot(
    no_episodes,
    total_avg_reward,
    color="blue",
    linestyle="-",
    label="MEAN REWARD OF 3AGENTS",
)
plt.plot(
    no_episodes,
    avg_total,
    color="white",
    linestyle="--",
    label="Episodes Reward Average FOR 3 AGENTS",
)

#plt.grid(b=True, which="major", axis="y", linestyle="--")
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Reward", fontsize=12)
plt.title("AVERAGE_REWARD", fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.show()


dqn.models["agent_0"].save("S0.h5")
dqn.models["agent_1"].save("S1.h5")
dqn.models["agent_2"].save("S2.h5")
