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

class State_Value_Module(nn.Module): #returns state value
  def __init__(self, o_dim):
    super(State_Value_Module, self).__init__()
    #initializing as TD3
    self.linear1=nn.Linear(o_dim, 256).to(device)
    nn.init.uniform_(self.linear1.weight, -1/np.sqrt(o_dim), 1/np.sqrt(o_dim))
    nn.init.zeros_(self.linear1.bias)
    self.linear2=nn.Linear(256,256).to(device)
    nn.init.uniform_(self.linear1.weight, -1/np.sqrt(256), 1/np.sqrt(256))
    nn.init.zeros_(self.linear1.bias)
    self.linear3=nn.Linear(256, 1).to(device)
    nn.init.uniform_(self.linear1.weight, -(3e-3), 3e-3)
    nn.init.zeros_(self.linear1.bias)
    self.w_sizes, self.b_sizes=self.get_parameter_sizes()
  
  def get_parameter_sizes(self): #initialize linear layers in this part
    w_sizes=[]
    b_sizes=[]
    for element in self.children():
      if isinstance(element, nn.Linear):
        w_s=element.weight.size()
        b_s=element.bias.size()
        w_sizes.append(w_s)
        b_sizes.append(b_s)
    return w_sizes, b_sizes
  
  def forward(self, observation):
    relu=nn.ReLU()
    layers=nn.Sequential(self.linear1, relu, self.linear2, relu, self.linear3)
    sv=layers(observation)
    return sv
  
  def vectorize_parameters(self):
    parameter_vector=torch.Tensor([]).to(device)
    for param in self.parameters():
      p=param.reshape(-1,1)
      parameter_vector=torch.cat((parameter_vector, p), dim=0)
    return parameter_vector.squeeze(dim=1)
  
  def inherit_parameters(self, parameter_vector):
    #size must be identical, input in vectorized form
    vector_idx=0
    #extract weight, bias data
    weights=[]
    biases=[]
    for sz_idx, w_size in enumerate(self.w_sizes):
      w_length=w_size[0]*w_size[1]
      weight=parameter_vector[vector_idx:vector_idx+w_length]
      weight=weight.reshape(w_size[0], w_size[1])
      weights.append(weight)
      vector_idx=vector_idx+w_length
      b_length=self.b_sizes[sz_idx][0]
      bias=parameter_vector[vector_idx:vector_idx+b_length]
      bias=bias.reshape(-1)
      biases.append(bias)
      vector_idx=vector_idx+b_length
    #overwrite parameters
    linear_idx=0
    for element in self.children():
      if isinstance(element, nn.Linear):
        element.weight=nn.Parameter(weights[linear_idx])
        element.bias=nn.Parameter(biases[linear_idx])
        linear_idx+=1
    return