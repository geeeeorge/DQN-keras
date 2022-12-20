import numpy as np
import random
from collections import deque


class RandomMemory:
    def __init__(self, limit):
        self.experiences = deque(maxlen=limit)

    def sample(self, batch_size):
        assert batch_size > 1, "batch_size must be positive integer"

        batch_size = min(batch_size, len(self.experiences))
        mini_batch = random.sample(self.experiences, batch_size)
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        for state, action, reward, next_state, done in mini_batch:
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            terminal_batch.append(0. if done else 1.)

        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        terminal_batch = np.array(terminal_batch)

        assert len(state_batch) == batch_size

        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def append(self, state, action, reward, next_state, terminal=False):
        assert state is not None
        assert action is not None
        assert reward is not None
        assert next_state is not None
        assert terminal is not None

        self.experiences.append((state, action, reward, next_state, terminal))
