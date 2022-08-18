# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
import torchvision.transforms as transforms

# python imports
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from scipy import ndimage
from tqdm.notebook import tqdm
import os

# additional libraries
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

# RL libraries
import gym

# custom modules 
from .callbacks import RLResults, MaxEpisodesStop
from .components import BaseHPS, BasePLModule, PolicyNetwork
from .utilities import get_activation, conv_output_shape
from .memory import ReplayMemory, RLDataset
from .behaviour import SoftmaxBehaviour



####################### DQN Agent ###########################################################################

class DQNAgent(BasePLModule):
    """
    DQN agent class to be used within PyTorch-Lightning framework.
    """
    def __init__(self,
                 env_name        : str    = "CartPole-v1",
                 N_episodes      : int    = 100,
                 memory_class    : object = ReplayMemory,
                 mem_capacity    : int    = 1024,
                 behaviour_class : object = SoftmaxBehaviour,
                 behaviour_params: dict   = None,
                 policy_class    : object = PolicyNetwork,
                 policy_params   : dict   = None,
                 target_sync_rate: int    = 10,
                 batch_size      : int    = 64,
                 gamma           : float  = 0.95,
                 optimizer       : str    = "sgd",
                 learning_rate   : float  = 0.001,
                 L2_penalty      : float  = 0.,
                 momentum        : float  = 0.,   #ignored if optimizer is Adam or Adamax
                 seed            : int    = 23,
                 penalty_type    : str    = "none",
                ):
        super(DQNAgent, self).__init__(optimizer     = optimizer, 
                                       learning_rate = learning_rate, 
                                       L2_penalty    = L2_penalty, 
                                       momentum      = momentum,
                                      ) 
        self.save_hyperparameters() # save hyperparameters when checkpointing
        
        ##### init objects and variables #####
        # initialize env
        self.env = gym.make(env_name)
        self.env.seed(seed)
        
        # initialize buffer for frames
        self.sequence_lenght = 4
        self.sequence = deque(maxlen=self.sequence_lenght)
        
        # initialize parameters
        self.target_sync_rate = target_sync_rate
        self.gamma            = gamma
        self.N_episodes       = N_episodes
        self.loss_fn          = nn.SmoothL1Loss()    #Huber loss
        self.penalty_type     = penalty_type
        
        # reset environment
        self.reset_env()
        self.episode_id           = 0
        self.episode_step_id      = 0
        self.episode_reward       = 0
        self.episode_score        = 0
        self.final_episode_reward = 0
        
        # initialize memory 
        self.memory = memory_class(mem_capacity)
        self.batch_size = batch_size
        
        # initialize behaviour 
        self.behaviour = behaviour_class(N_episodes, behaviour_params)
        
        # initialize policy network
        self.state_space_dim  = self.observation.size()   
        action_space_dim = self.env.action_space.n
        self.policy_net  = policy_class(self.state_space_dim, action_space_dim, policy_params)
        
        # initialize target network
        self.target_net = policy_class(self.state_space_dim, action_space_dim, policy_params)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # teacher model and parameters (teacher can be set by the user with the method "set_teacher")
        self.teacher_net = None
        self.assisted_frac = 0.6          # fraction of episodes to keep teacher assistance on
        self.assisted_init_prob = 0.6     # initial probability for teacher intervention
        
        
    def set_teacher(self, teacher_net):
        self.teacher_net = teacher_net    # model pretrained with env variables
        return
        
    @torch.no_grad()
    def reset_env(self):
        
        self.state_var = self.env.reset()   
        
        # initialize variables
        screen = self.env.render(mode='rgb_array')
        frame, _, _  = self._preprocess_input(screen)
        
        # initialize the observation to the proper size, repeating the initial frame
        for ii in range(self.sequence_lenght):
            self.sequence.append(frame)
        self.observation = torch.cat(list(self.sequence))           

        return
        
    @torch.no_grad()
    def fill_memory(self, steps=10):
        # store some initial experiences into memory
        print("Filling memory with initial random steps...")
        for step in range(steps):
            self.play_step(temperature=1e3)  # high temperature means random action
        
        # reset environment
        self.reset_env()
        self.episode_id      = 0
        self.episode_step_id = 0
        self.episode_score   = 0
        return

    
    @torch.no_grad()
    def play_step(self, temperature):
        
        # select action
        action, net_out = self.behaviour.choose_action(self.policy_net, self.observation, temperature)
        
        # execute action to env
        self.state_var, reward, done, info = self.env.step(action)
        self.episode_step_id += 1
        
        # increase score (CartPole)
        self.episode_score += reward
           
        screen = self.env.render(mode='rgb_array')
        new_frame, displacement, pole_top_displ = self._preprocess_input(screen)
        self.sequence.append(new_frame)
        new_observation = torch.cat(list(self.sequence))
        
        # tweak the reward using the new state observation
        reward = self.reward_correction(reward, displacement, pole_top_displ)        
        
        if self.episode_step_id > self.sequence_lenght:
            # store step in memory
            self.memory.push( self.observation, action, new_observation, reward, done )

        self.observation = new_observation
        if done:
            self.reset_env()
        return reward, done
    
    @torch.no_grad()
    def teacher_step(self):
        
        # select action
        action, net_out = self.behaviour.choose_optimal_action(self.teacher_net, self.state_var)
        
        # execute action to env
        self.state_var, reward, done, info = self.env.step(action)
        self.episode_step_id += 1
        
        # increase score (CartPole)
        self.episode_score += reward
           
        screen = self.env.render(mode='rgb_array')
        new_frame, displacement, pole_top_displ = self._preprocess_input(screen)
        self.sequence.append(new_frame)
        new_observation = torch.cat(list(self.sequence))
        
        # tweak the reward using the new state observation
        reward = self.reward_correction(reward, displacement, pole_top_displ)        

        if self.episode_step_id > self.sequence_lenght:
            # store step in memory
            self.memory.push( self.observation, action, new_observation, reward, done )

        self.observation = new_observation
        if done:
            self.reset_env()
        return reward, done
    
    @torch.no_grad()
    def reward_correction(self, reward, displacement, pole_top_displ): #CartPole
        
        if self.penalty_type == "pixels":
            # displacement
            pos_weight = 0.01        
            pos_penalty = pos_weight*(displacement**2)
            # angle
            angle_weight = 0.01     
            angle_penalty = angle_weight*(pole_top_displ**2)
        elif self.penalty_type == "state":
            pos_weight = 1.
            pos_penalty = pos_weight*np.abs(self.state_var[0])
            angle_weight = 1.
            angle_penalty = angle_weight*np.abs(self.state_var[2])
        elif self.penalty_type == "none":
            return reward
        else:
            print(f"Unknown penalty type ({self.penalty_type}). Set to 'none'.")
            self.penalty_type = "none"
            return reward
        
        return reward - (pos_penalty + angle_penalty)
        

    def forward(self, x):
        # run policy over state screens
        screens, _, _, _ = x        
        net_out = self.policy_net(screens)
        return net_out 
        
    def compute_loss(self, batch):

        # Split elements of the batch
        states, actions, next_states, rewards, dones = batch
  
        # Compute all the Q values (forward pass)
        self.policy_net.train()
        q_values = self.policy_net(states)
        # Select the proper Q value for the corresponding action taken Q(s_t, a)
        state_action_values = q_values.gather(1, actions.unsqueeze(1))

        # Compute the value function of the next states using the target network 
        with torch.no_grad():
            self.target_net.eval()
            next_state_max_q_values = self.target_net(next_states).max(dim=1)[0]
            # next_state_max_q_values are set to zero for ended episodes
            next_state_max_q_values[dones] = 0.0
            next_state_max_q_values = next_state_max_q_values.detach()

        # Compute the expected Q values
        expected_state_action_values = rewards + (next_state_max_q_values * self.gamma)
        expected_state_action_values = expected_state_action_values.unsqueeze(1) # Set the required tensor shape

        # Compute the loss
        return self.loss_fn(state_action_values, expected_state_action_values) 
    
    def training_step(self, batch):
        
        # retrieve episode temperature
        temperature = self.behaviour.exploration_profile[int(self.episode_id)]
        
        # step on environment
        if (self.teacher_net is not None) and (self.episode_id < self.N_episodes*self.assisted_frac):
            assisted_episodes = self.N_episodes*self.assisted_frac
            if self.assisted_init_prob*(assisted_episodes-self.episode_id)/assisted_episodes > random.random(): 
                reward, done = self.teacher_step()
                self.episode_reward += reward
            else:
                reward, done = self.play_step(temperature)
                self.episode_reward += reward
        else:
            reward, done = self.play_step(temperature)
            self.episode_reward += reward           
            
        self.log("global_step", self.global_step, on_step=True, on_epoch=False, prog_bar=True)
        self.log("reward"  , self.episode_reward, on_step=True, on_epoch=False, prog_bar=True)
        self.log("episode_id"  , self.episode_id, on_step=True, on_epoch=False, prog_bar=True)
        self.log("score"   ,  self.episode_score, on_step=True, on_epoch=False, prog_bar=True)
        
        ##### actual training #####
        # calculates training loss
        loss = self.compute_loss(batch)
                
        if done:
            # log results
            results = {"episode_id"  : self.episode_id,
                       "temperature" : temperature,
                       "final_loss"  : loss,
                       "final_reward": self.episode_reward,
                       "score"       : self.episode_score,
                      }
            self.log("results", results)

            self.episode_reward  = 0.
            self.episode_score   = 0
            self.episode_step_id = 0
            self.episode_id     += 1
            
        # eventually update the target network
        if self.global_step % self.target_sync_rate == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss
    
    def train_dataloader(self):
        
        dataset    = RLDataset(self.memory, sample_size=self.batch_size*10)
        dataloader = DataLoader(dataset,
                                batch_size = self.batch_size, 
                                shuffle    = False, 
                                pin_memory = True,
                               )
        return dataloader
        
    @torch.no_grad()    
    def _preprocess_input(self, screen):
        """
        Image preprocessing to be used with CartPole gym environment.
        """
        # to tensor
        preprocessed = transforms.functional.to_tensor(np.ascontiguousarray(screen))
        # to grayscale
        preprocessed = transforms.functional.rgb_to_grayscale(preprocessed)
        # cropping (from 400*600 to 150*200 with top-left corner at (168,200) )
        preprocessed = transforms.functional.crop(preprocessed, 168, 200, 150, 200)
        # take top and bottom images
        top, bottom = preprocessed[:, 0:32, :], preprocessed[:, 118:, :]  
        # stack top and bottom together (resulting shape is 64*200)
        preprocessed = torch.cat([top, bottom], 1)
        # finally resize
        preprocessed = transforms.functional.resize(preprocessed, (32,100)) 
            
        # compute penalty terms if required
        displacement   = 0.
        pole_top_displ = 0.
        if self.penalty_type == "pixels":       
            ### compute center of mass of cart to get its x position and its distance from the center
            cart_center  = ndimage.center_of_mass(bottom[0].numpy())[1]
            displacement = bottom.size()[2]//2 - cart_center
            
            ### pole-top displacement from the center of the cart
            pole_top_center = ndimage.center_of_mass(top[0].numpy())[1]
            pole_top_displ  = cart_center - pole_top_center
        
        return preprocessed, displacement, pole_top_displ

    
    @torch.no_grad()
    def run(self, N_iters=10, record=False, video_folder="CartPolePixels/Videos", name_prefix="Video"):
        """
        Test the trained agent with few episodes
        """
        if record:          
            self.env = gym.wrappers.RecordVideo(self.env, 
                                                video_folder    = video_folder, 
                                                name_prefix     = name_prefix,
                                                episode_trigger = lambda idx: True,
                                               ) 
        res_list = []
        with tqdm(range(N_iters), desc="Progress", postfix={"score":self.episode_score}) as pbar:
            for num in pbar:
                done = False
                while not done:
                    # select action
                    action, net_out = self.behaviour.choose_optimal_action(self.policy_net, self.observation)
                    
                    # execute action to env
                    self.state_var, reward, done, info = self.env.step(action)
                    self.episode_step_id += 1
                    
                    # increase score (CartPole)
                    self.episode_score += 1
                    pbar.set_postfix({"score":self.episode_score})
                    
                    screen = self.env.render(mode='rgb_array')
                    new_frame, displacement, pole_top_displ = self._preprocess_input(screen)
                    self.sequence.append(new_frame)
                    self.observation = torch.cat(list(self.sequence))
                    
                    # save image comparison
                    #self._save_image_comparison(screen, self.observation, self.episode_step_id)
                    
                    # tweak the reward using the new state observation
                    reward = self.reward_correction(reward, displacement, pole_top_displ)        
                    self.episode_reward += reward
                            
                    if done:
                        self.reset_env()
                        
                        # log results
                        results = {"episode_id"  : self.episode_id,
                                   "final_reward": self.episode_reward,
                                   "score"       : self.episode_score,
                                  }
                        res_list.append(results)

                        self.episode_reward  = 0.
                        self.episode_score   = 0
                        self.episode_step_id = 0
                        self.episode_id += 1
            
        return res_list 
    
    
    def _save_image_comparison(self, screen, observation, step_id):

        gs_kw = dict(width_ratios=[2, 1,1], height_ratios=[1, 1])
        fig, axd = plt.subplot_mosaic([['left','right1','right2'],  
                                       ['left','right3','right4']],
                                      gridspec_kw=gs_kw,
                                      figsize=(12, 4),
                                     )
        fig.suptitle("Original rendered image    |    Preprocessed frames           ", fontsize=16)
        
        # plot original screen
        axd["left"].imshow( screen, interpolation='none')
        
        # plot preprocessed frames
        for idx,frame in enumerate(observation):
            axd["right"+str(idx+1)].imshow( frame.cpu().unsqueeze(dim=2).numpy() ,interpolation='none', cmap="gray")
            axd["right"+str(idx+1)].set_title(f"Frame {idx+1}")
        plt.tight_layout()
        
        plt.savefig(f"CartPolePixels/frames/img_{step_id:03d}.png")
        plt.close()
        return
    
    
    
###### hyper-parameters space class -------------------------------------------------------------------------
class DQNAgentHPS(BaseHPS):
    """ 
    Class that receives in input the hyper-parameters space as a dict and provide a sampling function to be
    used within a Optuna study. This class is for the DQNAgent model.
    """        
    def __init__(self, hp_space):
        
        super().__init__(hp_space)
        self.model_class = DQNAgent
    
    
    def _sample_model_params(self, trial):       
        
        # memory capacity 
        mem_capacity = trial.suggest_int("mem_capacity", 
                                         self.hp_space["mem_capacity_range"][0],
                                         self.hp_space["mem_capacity_range"][1],
                                         step = self.hp_space["mem_capacity_range"][2],
                                        )
        # target_sync_rate
        target_sync_rate = trial.suggest_int("target_sync_rate",
                                             self.hp_space["target_sync_rate_range"][0],
                                             self.hp_space["target_sync_rate_range"][1],
                                             step = self.hp_space["target_sync_rate_range"][2],
                                            )
        # batch size
        batch_size = trial.suggest_int("batch_size",
                                       self.hp_space["batch_size_range"][0],
                                       self.hp_space["batch_size_range"][1],
                                       step = self.hp_space["batch_size_range"][2],
                                      )
        # gamma
        gamma = trial.suggest_float("gamma",
                                    self.hp_space["gamma_range"][0],
                                    self.hp_space["gamma_range"][1],
                                   )
        # network params TODO
        
        # behaviour params TODO
        
        # build model hyperparameters dictionary
        params = {"mem_capacity"    : mem_capacity,
                  "target_sync_rate": target_sync_rate,
                  "batch_size"      : batch_size,
                  "gamma"           : gamma,
                  "network_params"  : None,
                  "behaviour_params": None,
                 } 
        
        return params
        
        
                 
