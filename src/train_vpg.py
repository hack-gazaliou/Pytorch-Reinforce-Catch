from VPGAgent import VPGAgent
from engine import engine
import torch
import torch.nn as nn            # Pour construire les couches du réseau (Linear, ReLU)  # ReLu
import torch.optim as optim      #Adam
from torch.distributions import Categorical

def train_one_epoch(age : VPGAgent, eng : engine):
    batch_obs = []
    batch_acts = []
    batch_weights = []
    batch_rets = []       
    batch_lens = [] 
    ep_rews = []
    done = False
    visualize = True    
    finished_rendering_this_epoch = False
    obs = eng.reset()       
        
    while True:
        if (not finished_rendering_this_epoch) and visualize: # rendering
            visualize = True
            
        batch_obs.append(obs.copy())
        act = age.get_action(torch.as_tensor(obs, dtype=torch.float32))
        obs, rew, done = eng.step(act)
        batch_acts.append(act)
        ep_rews.append(rew)
        
        if done:
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            G = 0
            returns = []
            for r in reversed(ep_rews):
                G = r + age.gamma * G
                returns.insert(0, G)
            batch_weights += returns
            obs, done, ep_rews = eng.reset(), False, []
            finished_rendering_this_epoch = True

            if len(batch_rets) > age.batch_size:
                break 
    age.optimizer.zero_grad()
    batch_loss = age.compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32), 
                                    act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                    weights=torch.as_tensor(batch_weights, dtype=torch.float32))
    batch_loss.backward()
    age.optimizer.step()
    return batch_loss