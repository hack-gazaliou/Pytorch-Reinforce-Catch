import numpy as np
import random as rd
import torch
import torch.nn as nn            # Pour construire les couches du réseau (Linear, ReLU)  # ReLu
import torch.optim as optim      #Adam
from torch.distributions import Categorical
from engine import engine

class agent(nn.Module) :
    
    @staticmethod
    def mlp(sizes, activation, output_activation=nn.Identity):
        layers = []
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        return nn.Sequential(*layers)
        
    def __init__(self):
        super().__init__()
        self.hidden_sizes = [64, 64]
        self.n_acts = 5
        self.obs_dim = 9
        self.batch_size = 32 #to change if necessary
        self.policy_net = self.mlp(sizes=[self.obs_dim]+self.hidden_sizes+[self.n_acts], activation=nn.ReLU)
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.gamma = 0.99
    
    def get_policy(self,obs):
        logits = self.policy_net(obs)
        return Categorical(logits=logits)

    def get_action(self,obs):
        return self.get_policy(obs).sample().tolist()

    def compute_loss(self, obs, act, weights):
        logp = self.get_policy(obs).log_prob(act)
        return (-(logp * weights)).mean()

    def train_one_epoch(self):
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []       
        batch_lens = [] 
        
        obs = engine.reset()       
        done = False
        visualize = True            
        ep_rews = []

        finished_rendering_this_epoch = False
        
        while True:
            # rendering
            if (not finished_rendering_this_epoch) and visualize:
                visualize = True
                
            batch_obs.append(obs.copy())

            act = self.get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done = engine.step(act)
            
            batch_acts.append(act)
            ep_rews.append(rew)
            
            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                G = 0
                returns = []
                for r in reversed(ep_rews):
                    G = r + self.gamma * G
                    returns.insert(0, G)
                batch_weights += returns

                obs, done, ep_rews = engine.reset(), False, []
                finished_rendering_this_epoch = True

                if len(batch_rets) > self.batch_size:
                    break
            
        self.optimizer.zero_grad()
        batch_loss = self.compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32), 
                                       act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                       weights=torch.as_tensor(batch_weights, dtype=torch.float32))

        batch_loss.backward()
        self.optimizer.step()
        return batch_loss