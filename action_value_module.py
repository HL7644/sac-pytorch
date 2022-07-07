import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torchvision.datasets as dsets
import numpy as np

import collections

import gym

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Action_Value_Module(nn.Module): #returns action value
  def __init__(self, o_dim, a_dim):
    super(Action_Value_Module, self).__init__()
    #initializing as TD3 methods
    self.linear1=nn.Linear(o_dim+a_dim, 256).to(device)
    nn.init.uniform_(self.linear1.weight, -1/np.sqrt(o_dim+a_dim), 1/np.sqrt(o_dim+a_dim))
    nn.init.zeros_(self.linear1.bias)
    self.linear2=nn.Linear(256,256).to(device)
    nn.init.uniform_(self.linear1.weight, -1/np.sqrt(256), 1/np.sqrt(256))
    nn.init.zeros_(self.linear1.bias)
    self.linear3=nn.Linear(256, 1).to(device)
    nn.init.uniform_(self.linear1.weight, -(3e-3), 3e-3)
    nn.init.zeros_(self.linear1.bias)
  
  def forward(self, observation, action):
    relu=nn.ReLU()
    fv=torch.cat((observation, action), dim=0).to(device)
    layers=nn.Sequential(self.linear1, relu, self.linear2, relu, self.linear3)
    Q=layers(fv)
    return Q