# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# python imports
import numpy as np
import random
from collections import deque

# additional libraries
import pytorch_lightning as pl



# behaviour class
class SoftmaxBehaviour(object):
    """
    Class that implements agent's action selection with a softmax behaviour with exponential
    temperature profile.
    """
    def __init__(self, n_iterations, params=None):
                
        if params is None:
            initial_temperature     = 3.  
            decay_const_in_interval = 8
        else:
            initial_temperature     = params["initial_temperature"] 
            decay_const_in_interval = params["decay_const_in_interval"] 
                
        self.exploration_profile = self._compute_exploration_profile(initial_temperature,
                                                                     decay_const_in_interval,
                                                                     n_iterations,
                                                                    )
        self.min_temp = 1e-8
        

    def _compute_exploration_profile(self, temp_init, k, n_iters):
        # y = C * exp(-t / tau), where:
        #   C    = initial value
        #   k    = number of characteristic length to be represented in the interval
        #   t    = step 
        #   tau  = characteristic length     -> (n_iters / k)
        tau = n_iters / k
        exploration_profile = [temp_init * np.exp(-ii/tau) for ii in range(n_iters)]
        
        return exploration_profile
                                                                           
    
    def choose_action(self, net, state, temperature):
        
        ### evaluate network output
        net_out = self._evaluate_net_output(net, state)

        ### apply choice        
        if temperature < 0:
            raise Exception('The temperature value must be greater than or equal to 0 ')

        # if the temperature is 0, just select the best action
        if temperature == 0.:
            best_action = int(net_out.argmax())
            return best_action

        # apply softmax with temp
        temperature = max(temperature, self.min_temp) # set a minimum to the temperature for numerical stability
        softmax_out = nn.functional.softmax(net_out.squeeze(0) / temperature, dim=0).numpy()  
        
        # sample the action using softmax output as mass pdf
        all_possible_actions = np.arange(0, softmax_out.shape[-1])
        
        # this samples a random element from "all_possible_actions" with the probability 
        #      distribution p (softmax_out in this case)
        action = np.random.choice(all_possible_actions, p=softmax_out) 

        return action, net_out.numpy() 

    
    def choose_optimal_action(self, net, state):
        
        # evaluate network output
        net_out = self._evaluate_net_output(net, state)
        # apply choice
        action = int(net_out.argmax())
        
        return action, net_out.numpy()  
    
    @torch.no_grad()
    def _evaluate_net_output(self, net, state):
        # Evaluate the network output from the current state
        net.eval()
        if isinstance(state, torch.Tensor):        # state is the env rendering (pixels)
            net_out = net(state.unsqueeze(0))
        else:
            wrapped = torch.tensor(state, dtype=torch.float32)   # state is the env output
            net_out = net(wrapped)
        return net_out
    

