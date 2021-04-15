import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class CriticNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, seed, fc1_units=400, fc2_units=300):
        ''' 
        state_dim (int): State space dimension 
        action_dim (int): Action space dimension
        seed (int): Random seed
        fcX_units (int): No. of hidden layers units
        '''
        super(CriticNetwork, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_dim, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.init_parameters()

    def init_parameters(self):
        """ Initialize network weights. """
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, seed, fc1_units=400, fc2_units=300):
        ''' Initialize parameters of model and build its.
        Parameters:
        ===========
        state_dim (int): State space dimension 
        action_dim (int): Action space dimension
        seed (int): Random seed
        fcX_units (int): No. of hidden layers units
        '''
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_dim)
        self.init_parameters()

    def init_parameters(self):
        """ Initialize network weights. """
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu((self.fc1(state)))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x