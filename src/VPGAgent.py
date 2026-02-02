
import torch.nn as nn            # Pour construire les couches du réseau (Linear, ReLU)  # ReLu
import torch.optim as optim      #Adam
from torch.distributions import Categorical
import numpy as np

class VPGAgent(nn.Module) :
    
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
        self.obs_dim = 26 #4*4 for fruits (x, y, type, flag) + paddle_x + lives/max_lives
        self.batch_size = 32 #to change if necessary
        self.policy_net = self.mlp(sizes=[self.obs_dim]+self.hidden_sizes+[self.n_acts], activation=nn.ReLU)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.005)
        self.gamma = 0.99
    
    def get_policy(self,obs):
        logits = self.policy_net(obs)
        return Categorical(logits=logits)

    def get_action(self, obs): #0.1 greedy to force exploration (in response to coward strategy)
        if np.random.rand() < 0.1: 
            return np.random.randint(0, self.n_acts)    
        return self.get_policy(obs).sample().item()

    def compute_loss(self, obs, act, weights):
        logp = self.get_policy(obs).log_prob(act)
        return (-(logp * weights)).mean()
    