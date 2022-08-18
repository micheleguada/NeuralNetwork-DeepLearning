# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset, IterableDataset

# python imports
import numpy as np
import random
from collections import deque

# additional libraries
import pytorch_lightning as pl



# replay memory
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity) # Define a queue with maxlen "capacity"

    def push(self, state, action, next_state, reward, done):
        #  Add the tuple (state, action, next_state, reward) to the queue
        self.memory.append( (state, action, next_state, reward, done) )

    def sample_one(self):
        # Randomly select 1 sample
        idx = np.random.choice(len(self))
        return self.memory[idx]  
    
    def sample(self, batch_size):
        batch_size = min(batch_size, len(self))   
        # Randomly select "batch_size" samples
        indices = np.random.choice(len(self.memory), batch_size, replace=False) 
        states, actions, next_states, rewards, dones = zip(*[self.memory[idx] for idx in indices])
        return ( states, actions, next_states, np.array(rewards, dtype=np.float32), dones )

    def __len__(self):
        return len(self.memory) # Return the number of samples currently stored in the memory
    
   
# pytorch dataset class
class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the expereinces
    which will be updated during training
    """
    def __init__(self,
                 memory: ReplayMemory, 
                 sample_size: int = 512,
                ):
        self.memory = memory
        self.sample_size = sample_size

    def __iter__(self):
        states, actions, next_states, rewards, dones = self.memory.sample(self.sample_size)
        for i in range(len(actions)):
            yield states[i], actions[i], next_states[i], rewards[i], dones[i]
    
