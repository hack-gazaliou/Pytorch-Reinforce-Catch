from VPGAgent import VPGAgent
from engine import engine
import torch
import  numpy as np

def train_one_epoch(age : VPGAgent, eng : engine):
    batch_obs = []
    batch_acts = []
    batch_weights = []
    batch_rets = []       
    batch_lens = [] 
    ep_rews = []
    done = False
    obs = eng.reset()       
        
    while True: #on récolte la data sur cette epoch
        
        batch_obs.append(obs.copy())
        with torch.no_grad():
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

            if len(batch_rets) > age.batch_size:
                break 
    rets = np.array(batch_weights)
    rets = (rets - rets.mean()) / (rets.std() + 1e-8) # Standardization
    batch_weights = rets.tolist()
    age.optimizer.zero_grad() #on efface les anciens gradients de .data
    obs_tensor = torch.as_tensor(np.array(batch_obs), dtype=torch.float32)
    batch_loss = age.compute_loss(
        obs=obs_tensor, 
        act=torch.as_tensor(batch_acts, dtype=torch.int32),
        weights=torch.as_tensor(batch_weights, dtype=torch.float32)
    )
    batch_loss.backward() #on calcule le gradient
    age.optimizer.step() #on modifie les poids du reseau
    avg_ret = np.mean(batch_rets)
    std_ret = np.std(batch_rets)
    avg_len = np.mean(batch_lens)
    
    return batch_loss.item(), avg_ret, std_ret, avg_len