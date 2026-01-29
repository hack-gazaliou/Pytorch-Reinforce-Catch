import numpy as np
import random as rd
import torch
import torch.nn as nn            # Pour construire les couches du réseau (Linear, ReLU)
import torch.nn.functional as F   # Pour les fonctions d'activation (softmax, relu)
import torch.optim as optim      # Pour l'optimiseur (Adam ou SGD)
from torch.distributions import Categorical
import engine

hidden_sizes = [64, 64]
n_acts = 5
obs_dim = 9
batch_size = 32 #to change if necessary
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

#policy network
policy_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts], activation=nn.ReLU)

def get_policy(obs):
    logits = policy_net(obs)
    return Categorical(logits=logits)

def get_action(obs):
    return get_policy(obs).sample().tolist()

def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return torch.as_tensor(-(logp * weights)).mean()

def train_one_epoch():
    batch_obs = []
    batch_acts = []
    batch_weights = []
    batch_rets = []       
    batch_lens = [] 
     
    obs = engine.reset()       
    done = False            
    ep_rews = []

    finished_rendering_this_epoch = False
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=policy_net)
    
    while True:
        # rendering
        if (not finished_rendering_this_epoch) and visualize:
            visualize = True
            
        batch_obs.append(obs.copy())

        act = get_action(torch.as_tensor(obs, dtype=torch.float32))
        obs, rew, done = engine.step(act)
        
        batch_acts.append(act)
        ep_rews.append(rew)
        
        if done:
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # the weight for each logprob(a|s) is R(tau)
            batch_weights += [ep_ret] * ep_len

            obs, done, ep_rews = engine.reset(), False, []
            finished_rendering_this_epoch = True

            if len(batch_obs) > batch_size:
                break
        
    optimizer.zero_grad()
    batch_loss = torch.as_tensor(compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32), 
                              act=torch.as_tensor(batch_acts, dtype=torch.int32),
                              weights=torch.as_tensor(batch_weights, dtype=torch.float32)))

    batch_loss.backward()
    optimizer.step()
    return batch_loss