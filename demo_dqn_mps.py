
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
    N_EPISODES = 1000
    LEARNING_RATE = 0.001
    DISCOUNT_FACTOR = 0.9
    EPSILON_INITIAL = 0.5
    EPSILON_DECAY = 0.99
    D = 50

    def __init__(self):
        if torch.backends.mps.is_available():
            self.dev = torch.device("mps")
        else:
            self.dev = torch.device("cpu")
        #
        self.model = self.model_factory().to(self.dev)
        self.model_hat = self.model_factory().to(self.dev)
        self.copy_parameters(self.model, self.model_hat)
        self.loss = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.LEARNING_RATE)

    def model_factory(self):
        return nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4))

    def copy_parameters(self, src, dst):
        dst.load_state_dict(src.state_dict())

    def optimal_policy(self, observation):
        output = self.model(torch.from_numpy(observation).to(self.dev))
        return int(output.argmax())
        
    def epsilon_policy(self, observation):
        if random.random()<self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.optimal_policy(observation)

    def train(self):
        self.pool = []
        average_rewards = []
        self.iteration = 1
        self.epsilon = self.EPSILON_INITIAL
        self.env = gym.make("LunarLander-v2")
        for i in tqdm(range(self.N_EPISODES)):
            observation, info = self.env.reset()
            episode_rewards = []
            while True:
                action = self.epsilon_policy(observation)
                next_observation, reward, terminated, truncated, next_info = self.env.step(action)
                episode_rewards.append(reward)
                if terminated or truncated:
                    self.pool.append((observation, action, reward, next_observation, True))
                    break
                else:
                    self.pool.append((observation, action, reward, next_observation, False))
                self.train_step()
                self.iteration += 1
                observation = next_observation
            average_rewards.append(np.mean(episode_rewards))
            self.epsilon *= self.EPSILON_DECAY
        plt.plot(average_rewards)
        plt.show(block=False)
        print(f"Statistics:\n  final epsilon = {self.epsilon}\n  # of steps = {self.iteration}\n  # of episodes = {self.N_EPISODES}")
        
    def train_step(self):
        if len(self.pool)>=self.BATCH_SIZE:
            minibatch = random.sample(self.pool, self.BATCH_SIZE)
            next_states = torch.from_numpy(np.array([step[3] for step in minibatch]))
            # NOTE: this should be done with frozen weights
            next_q = self.model_hat(next_states.to(self.dev))
            max_q = torch.amax(next_q, dim=1)
            y = torch.tensor([step[2] for step in minibatch]).to(self.dev)  # rewards
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
        

def main():
    agent = DQN()
    agent.train()
    agent.demo()

    
if __name__=='__main__':
    main()

# EOF
