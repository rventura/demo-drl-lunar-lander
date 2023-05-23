#! /usr/bin/env python3

import sys
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm


class DQN:
    BATCH_SIZE = 100
    N_EPISODES = 250
    LEARNING_RATE = 0.001
    DISCOUNT_FACTOR = 0.9
    EPSILON_INITIAL = 1
    EPSILON_DECAY = 0.99
    D = 50

    def __init__(self):
        self.model = self.model_factory()
        self.model_hat = self.model_factory()
        self.copy_parameters(self.model, self.model_hat)
        self.loss = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.LEARNING_RATE)

    def model_factory(self):
        return nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            # nn.Linear(16, 16),
            # nn.ReLU(),
            nn.Linear(16, 4))

    def copy_parameters(self, src, dst):
        dst.load_state_dict(src.state_dict())

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)

    def optimal_policy(self, observation):
        output = self.model(torch.from_numpy(observation))
        return int(output.argmax())
        
    def epsilon_policy(self, observation):
        if random.random()<self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.optimal_policy(observation)

    def train(self):
        self.dataset = []
        stats_rewards = []
        stats_epsilon = []
        stats_loss = []
        self.iteration = 1
        self.epsilon = self.EPSILON_INITIAL
        self.env = gym.make("LunarLander-v2")
        for i in tqdm(range(self.N_EPISODES)):
            observation, info = self.env.reset()
            episode_rewards = []
            episode_loss = []
            while True:
                action = self.epsilon_policy(observation)
                next_observation, reward, terminated, truncated, next_info = self.env.step(action)
                episode_rewards.append(reward)
                if terminated or truncated:
                    self.dataset.append((observation, action, reward, next_observation, True))
                    break
                else:
                    self.dataset.append((observation, action, reward, next_observation, False))
                loss = self.train_step()
                if loss is not None:
                    episode_loss.append(loss)
                self.iteration += 1
                observation = next_observation
            if len(episode_rewards)>0:
                stats_rewards.append(np.mean(episode_rewards))
            if len(episode_loss)>0:
                stats_loss.append(np.mean(episode_loss))
            stats_epsilon.append(self.epsilon)
            self.epsilon *= self.EPSILON_DECAY
        plt.subplot(311)
        plt.plot(stats_rewards)
        plt.subplot(312)
        plt.plot(stats_loss)
        plt.subplot(313)
        plt.plot(stats_epsilon)
        plt.show()
        print(f"Statistics:\n  final epsilon = {self.epsilon}\n  # of steps = {self.iteration}\n  # of episodes = {self.N_EPISODES}")
        
    def train_step(self):
        if len(self.dataset)>=self.BATCH_SIZE:
            minibatch = random.sample(self.dataset, self.BATCH_SIZE)
            next_states = torch.from_numpy(np.array([step[3] for step in minibatch]))
            # NOTE: this should be done with frozen weights
            next_q = self.model_hat(next_states)
            max_q = torch.amax(next_q, dim=1)
            y = torch.tensor([step[2] for step in minibatch])  # rewards
            non_terminals = torch.tensor([not step[4] for step in minibatch])
            y[non_terminals] += self.DISCOUNT_FACTOR*max_q[non_terminals]
            #
            self.optimizer.zero_grad()
            inputs  = torch.from_numpy(np.array([step[0] for step in minibatch]))
            actions = torch.from_numpy(np.array([step[1] for step in minibatch]))
            outputs = self.model(inputs)
            loss = self.loss(y.to(outputs.dtype), outputs[range(len(actions)),actions])
            loss.backward()
            self.optimizer.step()
            #
            if self.iteration%self.D==0:
                self.copy_parameters(self.model, self.model_hat)
            return float(loss)

    def demo(self):
        self.env = gym.make("LunarLander-v2", render_mode="human")
        while True:
            observation, info = self.env.reset()
            while True:
                action = self.optimal_policy(observation)
                next_observation, reward, terminated, truncated, next_info = self.env.step(action)
                if terminated or truncated:
                    break
                observation = next_observation


def main(argv):
    if len(argv)<2 or argv[1]=='help':
        print(f"Usage: {argv[0]} <cmd> [<args>*]\n  <cmd> = help | train | demo ")
    elif argv[1]=='train':
        agent = DQN()
        agent.train()
        agent.save(argv[2] if len(argv)>2 else "dqn.model")
    elif argv[1]=='demo':
        agent = DQN()
        agent.load(argv[2] if len(argv)>2 else "dqn.model")
        agent.demo()
    else:
        print(f"*** Invalid command; use \"{argv[0]} help\" for help.")

    
if __name__=='__main__':
    main(sys.argv)

# EOF
