"""
Simple tabular Q-learning agent.
Q-table shape: (n_states, n_actions)
"""
import numpy as np
import random
from typing import Optional

class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1, gamma: float = 0.99, epsilon: float = 0.2, seed: Optional[int]=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.RandomState(seed)
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def select_action(self, state: int, greedy: bool = False) -> int:
        if not greedy and self.rng.rand() < self.epsilon:
            return int(self.rng.randint(self.n_actions))
        else:
            return int(self._argmax(self.Q[state]))

    def _argmax(self, arr):
        # ties broken randomly
        maxv = np.max(arr)
        candidates = np.flatnonzero(arr == maxv)
        return self.rng.choice(candidates)

    def update(self, state, action, reward, next_state, done: bool):
        q = self.Q[state, action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] = q + self.lr * (target - q)

    def save(self, path: str):
        np.save(path, self.Q)

    def load(self, path: str):
        self.Q = np.load(path)
