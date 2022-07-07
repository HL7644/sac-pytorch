import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torchvision.datasets as dsets
import numpy as np
from torch.distributions.normal import Normal

import collections

import gym

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mean_Module(nn.Module): #get mean for action distribution
  def __init__(self, o_dim, a_dim):
    super(Mean_Module, self).__init__()
    self.linear1=nn.Linear(o_dim, 256).to(device)
    self.linear2=nn.Linear(256,256).to(device)
    self.lienar3=nn.Linear(256, a_dim).to(device)
  
  def element_init(self):
    for element in self.children():
      if isinstance(element, nn.Linear):
        nn.init.xavier_uniform_(element.weight)
        nn.init.zeros_(element.bias)
  
  def forward(self, observation):
    relu=nn.ReLU()
    layers=nn.Sequential(self.linear1, relu, self.linear2, relu, self.lienar3)
    mean=layers(observation)
    return mean

class Log_Std_Module(nn.Module): #get log-std for action distribution
  def __init__(self, o_dim,  a_dim):
    super(Log_Std_Module, self).__init__()
    self.linear1=nn.Linear(o_dim, 256).to(device)
    self.linear2=nn.Linear(256,256).to(device)
    self.lienar3=nn.Linear(256, a_dim).to(device)
    self.element_init()
  
  def element_init(self):
    for element in self.children():
      if isinstance(element, nn.Linear):
        nn.init.xavier_uniform_(element.weight)
        nn.init.zeros_(element.bias)
      
  def forward(self, observation):
    relu=nn.ReLU()
    layers=nn.Sequential(self.linear1, relu, self.linear2, relu, self.lienar3)
    lstd=layers(observation)
    return lstd

class DGP(nn.Module):
  def __init__(self, o_dim, a_dim):
    super(DGP, self).__init__()
    self.o_dim=o_dim
    self.a_dim=a_dim
    self.mm=Mean_Module(o_dim, a_dim)
    self.lsm=Log_Std_Module(o_dim, a_dim)
  
  def forward(self, observation):
    #get squashed action from Normal dist.
    mean=self.mm(observation)
    lstd=self.lsm(observation)
    std=torch.exp(lstd).to(device)
    dist=Normal(mean, std)
    return dist
