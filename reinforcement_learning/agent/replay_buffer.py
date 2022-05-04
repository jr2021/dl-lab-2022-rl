from collections import namedtuple
import numpy as np
import os
import gzip
import pickle
import torch

class ReplayBuffer:

    # TODO: implement a capacity for the replay buffer (FIFO, capacity:    1e5 - 1e6)

    # Replay buffer for experience replay. Stores transitions.
    def __init__(self):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], dones=[])

    def add_transition(self, state, action, next_state, reward, done):
        """
        This method adds a transition to the replay buffer.
        """
        # TODO: check capacity and remove oldest
        self._data.states[:1e5]
        self._data.actions[:1e5]
        self._data.next_states[:1e5]
        self._data.rewards[:1e5]
        self._data.dones[:1e5]

        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.dones.append(done)

    def next_batch(self, batch_size):
        """
        This method samples a batch of transitions.
        """
        # TODO: transform to Tensor
        batch_indices = torch.Tensor(np.random.choice(len(self._data.states), batch_size))
        batch_states = torch.Tensor(np.array([self._data.states[i] for i in batch_indices]))
        batch_actions = torch.Tensor(np.array([self._data.actions[i] for i in batch_indices]))
        batch_next_states = torch.Tensor(np.array([self._data.next_states[i] for i in batch_indices]))
        batch_rewards = torch.Tensor(np.array([self._data.rewards[i] for i in batch_indices]))
        batch_dones = torch.Tensor(np.array([self._data.dones[i] for i in batch_indices]))
        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones
