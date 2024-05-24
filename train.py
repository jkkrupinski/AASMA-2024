import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn as nn

from pettingzoo.mpe import simple_spread_v3


# Define model
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


# Define memory for Experience Replay
class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class SimpleSpreadDQL:

    def __init__(self):
        self._init_parameters()
        self._init_loss_functions()
        self._init_optimizers()

    def _init_parameters(self):
        # Hyperparameters (adjustable)
        self.learning_rate = 0.001  # learning rate (alpha)
        self.discount_factor_g = 0.9  # discount rate (gamma)
        self.network_sync_rate = 10  # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = 1000  # size of replay memory
        self.mini_batch_size = (
            32  # size of the training data set sampled from the replay memory
        )

    def _init_loss_functions(self):
        loss_fn_0 = nn.MSELoss()
        loss_fn_1 = nn.MSELoss()
        loss_fn_2 = nn.MSELoss()

        self.loss_fns = {
            "agent_0": loss_fn_0,
            "agent_1": loss_fn_1,
            "agent_2": loss_fn_2,
        }

    def _init_optimizers(self):
        optimizer_0 = None
        optimizer_1 = None
        optimizer_2 = None
        self.optimizers = {
            "agent_0": optimizer_0,
            "agent_1": optimizer_1,
            "agent_2": optimizer_2,
        }

    def _init_memories(self):
        memory_0 = ReplayMemory(self.replay_memory_size)
        memory_1 = ReplayMemory(self.replay_memory_size)
        memory_2 = ReplayMemory(self.replay_memory_size)

        self.memories = {"agent_0": memory_0, "agent_1": memory_1, "agent_2": memory_2}

    def _init_dqns(self):
        input_shape = 18
        output_actions = 5

        dqn_0 = DQN(input_shape=input_shape, output_actions=output_actions)
        dqn_1 = DQN(input_shape=input_shape, output_actions=output_actions)
        dqn_2 = DQN(input_shape=input_shape, output_actions=output_actions)

        dqns = {
            "agent_0": dqn_0,
            "agent_1": dqn_1,
            "agent_2": dqn_2,
        }

        return dqns

    def copy_dqn_weights(self, agent):
        self.target_dqns[agent].load_state_dict(self.policy_dqns[agent].state_dict())

    def set_dqn_optimizer(self, agent):
        self.optimizers[agent] = torch.optim.Adam(
            self.policy_dqns[agent].parameters(), lr=self.learning_rate
        )

    def choose_action(self, agent, states):
        if random.random() < self.epsilon:
            action = np.random.randint(0, self.env.action_spaces[agent].n)

        else:
            with torch.no_grad():
                action = (
                    self.policy_dqns[agent](torch.from_numpy(states[agent]))
                    .argmax()
                    .item()
                )

        return action

    def decay_epsilon(self, episodes):
        num_of_agents = 3  # decay is called for each agent, thats why we divide by (episodes * num_of_agents)
        self.epsilon = max(self.epsilon - 1 / (episodes * num_of_agents), 0)

    def save_policy(self, agent, prefix):
        torch.save(
            self.policy_dqns[agent].state_dict(),
            prefix + agent + ".pt",
        )

    # Train on the SimpleSpread environment
    def train(self, episodes):

        # Create SimpleSpread instance
        self.env = simple_spread_v3.parallel_env(local_ratio=0.5)
        agents = ["agent_0", "agent_1", "agent_2"]
        self.env.reset()

        self._init_memories()

        # Create policy and target networks.
        self.policy_dqns = self._init_dqns()
        self.target_dqns = self._init_dqns()

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        for agent in agents:
            self.copy_dqn_weights(agent)

        # Policy network optimizers.
        for agent in agents:
            self.set_dqn_optimizer(agent)

        episode_rewards = []
        self.epsilon = 1

        for episode in range(episodes):
            start = time.time()

            states = self.env.reset()[0]

            actions = {"agent_0": 0, "agent_1": 0, "agent_2": 0}
            avg_rewards = {"agent_0": 0, "agent_1": 0, "agent_2": 0}

            for step in range(NUM_STEPS):

                for agent in agents:
                    actions[agent] = self.choose_action(agent, states)

                # Execute actions
                new_states, rewards, terminations, _, _ = self.env.step(actions)

                for agent in agents:
                    # Save average rewards per episode
                    avg_rewards[agent] += rewards[agent] / NUM_STEPS

                    # Save experience into memory
                    self.memories[agent].append(
                        (
                            states[agent],
                            actions[agent],
                            new_states[agent],
                            rewards[agent],
                            terminations[agent],
                        )
                    )

                # Move to the next state
                states = new_states

            # Check if enough experience has been collected
            for agent in agents:

                memory = self.memories[agent]
                if memory.__len__() > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(
                        mini_batch,
                        self.policy_dqns[agent],
                        self.target_dqns[agent],
                        agent,
                    )

                    self.decay_epsilon(episodes)

                    # Copy policy network to target network after a certain number of steps
                    self.copy_dqn_weights(agent)

            end = time.time()
            self.print_rewards(avg_rewards, start, end, episode)
            episode_rewards.append(avg_rewards)

            # Save policy every X episodes
            if episode % SAVE_MODEL_EVERY == 0 and episode != 0:
                for agent in agents:
                    prefix = str(episode) + "_"
                    self.save_policy(agent, prefix)

        # Close environment
        self.env.close()

        # Save policy
        for agent in agents:
            prefix = "fin_"
            self.save_policy(agent, prefix)

        return episode_rewards

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn, agent):

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, _ in mini_batch:

            # Calculate target q value
            with torch.no_grad():
                target = torch.FloatTensor(
                    reward
                    + self.discount_factor_g
                    * target_dqn(torch.from_numpy(new_state)).max()
                )

            # Get the current set of Q values
            current_q = policy_dqn(torch.from_numpy(state))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(torch.from_numpy(state))

            # Adjust the specific action to the target that was just calculated.
            target_q[action] = target
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = self.loss_fns[agent](
            torch.stack(current_q_list), torch.stack(target_q_list)
        )

        # Optimize the model
        self.optimizers[agent].zero_grad()
        loss.backward()
        self.optimizers[agent].step()

    def plot_results(self, episode_rewards, NUM_EPISODES):
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

        for i in range(0, NUM_EPISODES):
            k = np.mean(rewards_0[i : i + 100])
            k = np.round(k, 3)
            avg_rewards_0.append(k)
        for i in range(0, NUM_EPISODES):
            k = np.mean(rewards_1[i : i + 100])
            k = np.round(k, 3)
            avg_rewards_1.append(k)
        for i in range(0, NUM_EPISODES):
            k = np.mean(rewards_2[i : i + 100])
            k = np.round(k, 3)
            avg_rewards_2.append(k)

        no_episodes = []
        for i in range(0, NUM_EPISODES):
            no_episodes.append(i)
        np.mean(rewards_0) + np.mean(rewards_1) + np.mean(rewards_2)

        plt.figure(figsize=(10, 5))
        plt.plot(
            no_episodes, rewards_0, color="red", linestyle="-", label="Total Reward0"
        )

        plt.plot(
            no_episodes,
            avg_rewards_0,
            color="midnightblue",
            linestyle="--",
            label="Episodes Reward Average0",
        )

        # plt.grid(b=True, which="major", axis="y", linestyle="--")
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Reward", fontsize=12)
        plt.title("Total Reward per Testing Episode", fontsize=12)
        plt.legend(loc="lower right", fontsize=12)
        plt.show()

        plt.plot(
            no_episodes, rewards_1, color="blue", linestyle="-", label="Total Reward1"
        )
        plt.plot(
            no_episodes,
            avg_rewards_1,
            color="black",
            linestyle="--",
            label="Episodes Reward Average1",
        )

        # plt.grid(b=True, which="major", axis="y", linestyle="--")
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Reward", fontsize=12)
        plt.title("Total Reward per Testing Episode", fontsize=12)
        plt.legend(loc="lower right", fontsize=12)
        plt.show()

        plt.plot(
            no_episodes, rewards_2, color="green", linestyle="-", label="Total Reward2"
        )
        plt.plot(
            no_episodes,
            avg_rewards_2,
            color="pink",
            linestyle="--",
            label="Episodes Reward Average2",
        )

        # plt.grid(b=True, which="major", axis="y", linestyle="--")
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Reward", fontsize=12)
        plt.title("Total Reward per Testing Episode", fontsize=12)
        plt.legend(loc="lower right", fontsize=12)
        plt.show()

        total_avg_reward = []
        for i in range(0, NUM_EPISODES):
            k = np.mean(rewards_0[i] + rewards_1[i] + rewards_2[i])
            k = np.round(k, 3)
            total_avg_reward.append(k)
        avg_total = []
        for i in range(0, NUM_EPISODES):
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

        # plt.grid(b=True, which="major", axis="y", linestyle="--")
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Reward", fontsize=12)
        plt.title("AVERAGE_REWARD", fontsize=12)
        plt.legend(loc="lower right", fontsize=12)
        plt.show()

    def print_rewards(self, avg_rewards, start, end, episode):
        print(
            "EPISODE",
            episode,
            " Epsilon = ",
            round(self.epsilon, 2),
            "TIME = ",
            round(end - start, 2),
            "s",
        )
        print(
            "AVG REWARDS: agent_0:",
            round(avg_rewards["agent_0"], 2),
            ", agent_1:",
            round(avg_rewards["agent_1"], 2),
            ", agent_2:",
            round(avg_rewards["agent_2"], 2),
        )
        print()


if __name__ == "__main__":

    NUM_STEPS = 25
    NUM_EPISODES = 100
    SAVE_MODEL_EVERY = 100

    dql = SimpleSpreadDQL()
    episode_rewards = dql.train(NUM_EPISODES)
    dql.plot_results(episode_rewards, NUM_EPISODES)
