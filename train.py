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
    # Hyperparameters (adjustable)
    learning_rate_a = 0.001  # learning rate (alpha)
    discount_factor_g = 0.9  # discount rate (gamma)
    network_sync_rate = 10  # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000  # size of replay memory
    mini_batch_size = 32  # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn_0 = nn.MSELoss()
    loss_fn_1 = nn.MSELoss()
    loss_fn_2 = nn.MSELoss()
    loss_fns = {"agent_0": loss_fn_0, "agent_1": loss_fn_1, "agent_2": loss_fn_2}

    optimizer_0 = None
    optimizer_1 = None
    optimizer_2 = None
    optimizers = {
        "agent_0": optimizer_0,
        "agent_1": optimizer_1,
        "agent_2": optimizer_2,
    }

    ACTIONS = ["N", "L", "R", "D", "U"]

    # Train the SimpleSpread environment
    def train(self, episodes):

        episode_rewards = []

        # Create SimpleSpread instance
        env = simple_spread_v3.parallel_env(local_ratio=0.5)
        env.reset()

        input_shape = 18
        output_actions = 5

        agents = ["agent_0", "agent_1", "agent_2"]

        epsilon = 1  # 1 = 100% random actions

        memory_0 = ReplayMemory(self.replay_memory_size)
        memory_1 = ReplayMemory(self.replay_memory_size)
        memory_2 = ReplayMemory(self.replay_memory_size)
        memories = {"agent_0": memory_0, "agent_1": memory_1, "agent_2": memory_2}

        # Create policy and target network.
        policy_dqn_0 = DQN(input_shape=input_shape, output_actions=output_actions)
        policy_dqn_1 = DQN(input_shape=input_shape, output_actions=output_actions)
        policy_dqn_2 = DQN(input_shape=input_shape, output_actions=output_actions)
        policy_dqns = {
            "agent_0": policy_dqn_0,
            "agent_1": policy_dqn_1,
            "agent_2": policy_dqn_2,
        }

        target_dqn_0 = DQN(input_shape=input_shape, output_actions=output_actions)
        target_dqn_1 = DQN(input_shape=input_shape, output_actions=output_actions)
        target_dqn_2 = DQN(input_shape=input_shape, output_actions=output_actions)
        target_dqns = {
            "agent_0": target_dqn_0,
            "agent_1": target_dqn_1,
            "agent_2": target_dqn_2,
        }

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        for agent in agents:
            target_dqns[agent].load_state_dict(policy_dqns[agent].state_dict())

        # Policy network optimizer. "Adam" optimizer can be swapped to something else.
        for agent in agents:
            self.optimizers[agent] = torch.optim.Adam(
                policy_dqns[agent].parameters(), lr=self.learning_rate_a
            )

        for episode in range(episodes):
            start = time.time()
            states = env.reset()[0]  # Initialize to state 0
            actions = {"agent_0": 0, "agent_1": 0, "agent_2": 0}
            avg_rewards = {"agent_0": 0, "agent_1": 0, "agent_2": 0}

            for step in range(NUM_STEPS):

                for agent in agents:

                    if random.random() < epsilon:
                        action = np.random.randint(0, env.action_spaces[agent].n)

                    else:
                        with torch.no_grad():
                            action = (
                                policy_dqns[agent](torch.from_numpy(states[agent]))
                                .argmax()
                                .item()
                            )
                            actions[agent] = action

                # Execute action
                new_states, rewards, terminations, _, _ = env.step(actions)

                for agent in agents:
                    # Save average rewards per episode
                    avg_rewards[agent] += rewards[agent] / NUM_STEPS

                    # Save experience into memory
                    memories[agent].append(
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

                memory = memories[agent]
                if memory.__len__() > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(
                        mini_batch, policy_dqns[agent], target_dqns[agent], agent
                    )

                    # Decay epsilon
                    epsilon = max(epsilon - 1 / (episodes * 3), 0)

                    # Copy policy network to target network after a certain number of steps
                    target_dqns[agent].load_state_dict(policy_dqns[agent].state_dict())

            end = time.time()

            print(
                "EPISODE",
                episode,
                " Epsilon = ",
                round(epsilon, 2),
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

            episode_rewards.append(avg_rewards)

            # Save policy every x episodes
            if episode % SAVE_MODEL_EVERY == 0 and episode != 0:
                for agent in agents:
                    torch.save(
                        policy_dqns[agent].state_dict(),
                        str(episode) + "_" + agent + "_simple_spread_dql_cnn.pt",
                    )

        # Close environment
        env.close()

        # Save policy
        for agent in agents:
            torch.save(
                policy_dqns[agent].state_dict(),
                "fin_" + agent + "_simple_spread_dql_cnn.pt",
            )

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


if __name__ == "__main__":

    NUM_STEPS = 25
    NUM_EPISODES = 100_000
    SAVE_MODEL_EVERY = 1_000

    dql = SimpleSpreadDQL()
    episode_rewards = dql.train(NUM_EPISODES)
    dql.plot_results(episode_rewards, NUM_EPISODES)
