import numpy as np
import copy
from collections import deque
from preprocess import Preprocessor
from model import DQN


class Agent():

    def __init__(self, n_action_space, n_training_frames = 50 * 1000000, replay_memory_size = 1000000, k = 4, m = 4):

        # Hyperparameters - dynamic
        self.n_action_space = n_action_space
        self.n_training_frames = n_training_frames
        self.replay_memory_size = replay_memory_size
        self.replay_memory = deque(maxlen = self.replay_memory_size)
        self.m = m
        self.k = k

        # Hyperparameters - static
        self.epsilon = 1.0
        self.minibatch_size = 32
        self.C = 10000
        self.gamma = 0.99
        self.update_frequency

        self.epsilon_initial = 1.0
        self.epsilon_final = 0.1
        self.exploration_frame = 1000000 # 1 million
        self.epsilon_decay = (self.epsilon_initial - self.epsilon_final) / self.exploration_frame

        self.replay_start_frame = 50000

        # Parameters - etc
        self.action = None
        self.timestep = 0

        # Modules
        self.preprocessor = Preprocessor(m = self.m)
        self.Q = DQN()
        self.Q_hat = copy.deepcopy(self.Q)

        # Operations
        self.mode('train')


    def reset(self, observation):
        self.preprocessor.reset()
        self.processed_observation = self.preprocessor.preprocess(observation)

    def mode(self, mode):

        self.mode = mode
        if self.mode == 'train':
            self.epsilon = self.epsilon_initial - self.timestep * self.
        elif self.mode == 'test':
            self.epsilon = 0.05
        else:
            assert False, 'mode not specified'

    def epsilon_greedy(self):
        if np.random.random() < self.epsilon:
            return True
        else:
            return False

    def apply_epsilon_decay(self):
        self.epsilon = self.epsilon - self.epsilon_decay

    def act(self):
        # See & Select actions every kth frame. Modify ations every kth frame
        # Otherwise, skip frame (use same action)
        if self.timestep % self.k == 0:

            # 1. Epsilon greedy
            if self.epsilon_greedy():
                # Return random action
                self.action = np.random.randint(self.n_action_space)
                return
            else:
                # Return DQN result
                self.action = self.Q
                pass

        return self.action

    def train(self, observation, reward, done):

        # 1. Preprocess new observation
        new_processed_observation = self.preprocessor.preprocess(observation)

        # 2. Store transition in replay memory
        # If memory size exceeds, the oldest memory is popped (deque property)
        self.replay_memory.append( (self.observation, self.action, reward, new_processed_observation) )

        # 3. Training process
        # After
        # Sample random minibatch of transitions from replay memory

        # 4.
