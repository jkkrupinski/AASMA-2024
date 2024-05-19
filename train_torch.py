import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import time

import torch.nn as nn
import torch.optim as optim

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
    def train(self, episodes, render=False):
        # Create SimpleSpread instance
        env = simple_spread_v3.parallel_env(local_ratio=0.5)
        env.reset()
        num_actions = 5

        agents = ["agent_0", "agent_1", "agent_2"]

        epsilon = 1  # 1 = 100% random actions

        memory_0 = ReplayMemory(self.replay_memory_size)
        memory_1 = ReplayMemory(self.replay_memory_size)
        memory_2 = ReplayMemory(self.replay_memory_size)
        memories = {"agent_0": memory_0, "agent_1": memory_1, "agent_2": memory_2}

        # Create policy and target network.
        policy_dqn_0 = DQN(input_shape=18, output_actions=num_actions)
        policy_dqn_1 = DQN(input_shape=18, output_actions=num_actions)
        policy_dqn_2 = DQN(input_shape=18, output_actions=num_actions)
        policy_dqns = {
            "agent_0": policy_dqn_0,
            "agent_1": policy_dqn_1,
            "agent_2": policy_dqn_2,
        }

        target_dqn_0 = DQN(input_shape=18, output_actions=num_actions)
        target_dqn_1 = DQN(input_shape=18, output_actions=num_actions)
        target_dqn_2 = DQN(input_shape=18, output_actions=num_actions)
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

        # List to keep track of rewards collected per episode. Initialize list to 0's.

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
                    epsilon = max(epsilon - 1 / (episodes*3), 0)

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

        # Close environment
        env.close()

        # Save policy
        for agent in agents:
            torch.save(
                policy_dqns[agent].state_dict(), agent + "_simple_spread_dql_cnn.pt"
            )

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn, agent):

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

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
            # Target_q[batch][action], hardcode batch to 0 because there is only 1 batch.
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

    # Run the SimpleSpread environment with the learned policy
    def test(self, episodes):
        # Create SimpleSpread instance
        env = gym.make(
            "FrozenLake-v1",
            map_name="4x4",
            render_mode="human",
        )
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = DQN(input_shape=3, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("frozen_lake_dql_cnn.pt"))
        policy_dqn.eval()  # switch model to evaluation mode

        print("Policy (trained):")
        self.print_dqn(policy_dqn)

        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False  # True when agent falls in hole or reached goal
            truncated = False  # True when agent takes more than 200 actions

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while not terminated and not truncated:
                # Select best action
                with torch.no_grad():
                    action = policy_dqn(state).argmax().item()

                # Execute action
                state, reward, terminated, truncated, _ = env.step(action)

        env.close()


if __name__ == "__main__":

    NUM_STEPS = 25

    simple_spread = SimpleSpreadDQL()
    simple_spread.train(10_000)
    # frozen_lake.test(10)
