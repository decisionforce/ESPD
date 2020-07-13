import random
import numpy as np
from collections import deque


class ReplayBuffer_imitation(object):
    def __init__(self, capacity):
        self.buffer = {'1step': deque(maxlen=capacity)}
        self.capacity = capacity

    def push(self, state, action, step_num):
        try:
            self.buffer[step_num]
        except:
            self.buffer[step_num] = deque(maxlen=self.capacity)
        self.buffer[step_num].append((state, action))

    def sample(self, batch_size, step_num):
        state, action = zip(*random.sample(self.buffer[step_num], batch_size))
        return np.stack(state), action

    def lenth(self, step_num):
        try:
            self.buffer[step_num]
        except:
            return 0
        return len(self.buffer[step_num])

    def __len__(self, step_num):
        try:
            self.buffer[step_num]
        except:
            return 0
        return len(self.buffer[step_num])
