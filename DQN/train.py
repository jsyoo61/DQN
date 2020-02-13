import gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from preprocess import Preprocessor
from model import Agent

# Hyperparameters
game_name = 'Breakout-v0'
million = 1000000
n_training_frames = 50 * million

env = gym.make(game_name)
n_action_space = env.action_space.n

m = 4

# Create instance of Agent
agent = Agent(n_action_space = n_action_space, n_training_frames = n_training_frames, m = m)

n_episode = 0

# Start episodes
while(agent.train_complete is False):
    # Count episodes
    n_episode += 1

    # observation: [row, column, RGB]
    observation = env.reset()
    agent.reset(observation)

    done = False
    timestep = 0

    print('Episode start: %s'%(episode))

    # Play game
    while(done is False):
        env.render()

        action = agent.act()
        observation, reward, done, info = env.step(action)
        agent.train(observation, reward, done)

        timestep += 1

    print('Episode finished after timestep: %s'%(timestep))

env.close()
print('Training complete after episode: %s'%(n_episode))




# plt.matshow(observation[:100, 20:80 ,0])
# plt.matshow(observation[:,:,2])
# plt.matshow(observation[:,:,1])
ob = env.reset()
observation.shape
help(env)

reward = []
ob = env.reset()
import time
for i in range(600):
    env.render()
    time.sleep(1/60)
    ob, re, do, info = env.step(env.action_space.sample())
    reward.append(re)
    # info

env.action_space.sample()

env.render()
info
do
reward
env.close()

n_action_space = env.action_space.n
n_action_space
