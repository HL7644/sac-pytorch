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

from policy_module import *
from state_value_module import *
from action_value_module import *

class Obs_Wrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super(Obs_Wrapper, self).__init__(env)
  
  def observation(self, observation):
    obs=torch.FloatTensor(observation).to(device) #to tensor
    return obs

class Replay_Buffer(torch.utils.data.Dataset):
  def __init__(self, max_length):
    super(Replay_Buffer, self).__init__()
    self.ep_steps=[]
    self._max_length=max_length
    self._length=0
  
  def sample_batch(self, batch_size):
    batch=[]
    batch_idx=np.random.choice(self._length, batch_size)
    for idx in batch_idx:
      batch.append(self.ep_steps[idx])
    return batch
  
  def add_item(self, ep_step):
    if self._length>=self._max_length:
      #remove earliest element
      self.ep_steps.pop(0)
      self._length=self._length-1
    #add element
    self.ep_steps.append(ep_step)
    self._length=self._length+1
    return
  
  def __getitem__(self, idx):
    return self.ep_steps[idx]
  
  def __len__(self):
    return len(self.ep_steps)

class SAC_Agent():
  def __init__(self, env, test_env, gamma):
    self.env=env
    self.test_env=test_env
    self.gamma=gamma

    self.o_dim=self.env.observation_space.shape[0]
    self.a_dim=self.env.action_space.shape[0]
    self.a_low=torch.FloatTensor(self.env.action_space.low).to(device)
    self.a_high=torch.FloatTensor(self.env.action_space.high).to(device)

    self.svm=State_Value_Module(self.o_dim)
    svm_param=self.svm.vectorize_parameters()
    self.target_svm=State_Value_Module(self.o_dim)
    self.target_svm.inherit_parameters(svm_param)
    self.qm1=Action_Value_Module(self.o_dim, self.a_dim)
    self.qm2=Action_Value_Module(self.o_dim, self.a_dim)
    self.pm=DGP(self.o_dim, self.a_dim)
  
  def get_squashed_logprob(self, dist, action):
    #tanh squashing
    jac=autograd.functional.jacobian(torch.tanh, action)
    det_jac=torch.linalg.det(jac)
    logp=dist.log_prob(action)
    sq_logp=logp-torch.log(det_jac)
    return sq_logp
  
  def get_action(self, observation, a_mode): #used for env. stepping
    if a_mode=='random':
      action=self.env.action_space.sample()
      action=torch.FloatTensor(action).to(device)
      return action
    elif a_mode=='test':
      action=self.pm.mm(observation) #use mean action for testing
      return action
    else:
      #return sampled unsquashed action
      dist=self.pm(observation) #distribution based on gaussian
      action=dist.sample()
      return action

Ep_Step=collections.namedtuple('Ep_Step', field_names=['obs', 'action','reward','obs_f','termin_signal'])

class SAC():
  def __init__(self, agent):
    self.agent=agent
    self.replay_buffer=Replay_Buffer(max_length=1000000)
  
  def check_performance(self): #on max entropy setting
    avg_return=0
    avg_len_ep=0
    avg_entropy=0
    for n in range(1,10+1):
      acc_rew=0
      len_ep=1
      acc_entropy=0
      obs=self.agent.test_env.reset()
      while True:
        dist=self.agent.pm(obs)
        action=self.agent.get_action(obs, a_mode='test').detach().cpu().numpy()
        entropy=dist.entropy()
        obs_f, reward, termin_signal, _=self.agent.test_env.step(action)
        acc_rew=acc_rew+reward
        acc_entropy=acc_entropy+entropy
        
        len_ep+=1
        obs=obs_f
        if termin_signal:
          break
      avg_return=avg_return+(acc_rew-avg_return)/n
      avg_len_ep=avg_len_ep+(len_ep-avg_len_ep)/n
      avg_entropy=avg_entropy+(acc_entropy-avg_entropy)/n
    return avg_return, avg_len_ep, avg_entropy
  
  def get_state_value_loss(self, batch_data, alpha):
    sv_loss=torch.FloatTensor([0]).to(device)
    batch_size=len(batch_data)
    for ep_step in batch_data:
      obs=ep_step.obs
      V=self.agent.svm(obs)

      #sample new action (unsquashed)
      dist=self.agent.pm(obs)
      action=dist.sample()
      logp=dist.log_prob(action).detach()
      #create target for max. entropy objective: choose min Q of 2 qm's
      Q1=self.agent.qm1(obs, action)
      Q2=self.agent.qm2(obs, action)
      min_Q=min(Q1,Q2)
      target=min_Q-alpha*logp
      sv_loss=sv_loss+(target-V)**2
    sv_loss=sv_loss/batch_size #MSE Loss
    return sv_loss
  
  def get_action_value_loss(self, batch_data, index):
    if index==1:
      qm=self.agent.qm1
    else:
      qm=self.agent.qm2

    batch_size=len(batch_data)
    q_loss=torch.FloatTensor([0]).to(device)
    for ep_step in batch_data:
      obs=ep_step.obs
      action=torch.FloatTensor(ep_step.action).to(device)
      reward=ep_step.reward
      obs_f=ep_step.obs_f
      #target computation
      target_V_f=self.agent.target_svm(obs_f)
      target=reward+self.agent.gamma*target_V_f
      Q=qm(obs, action)
      q_loss=q_loss+(target-Q)**2
    q_loss=q_loss/batch_size
    return q_loss
  
  def get_policy_loss(self, batch_data, alpha):
    policy_loss=torch.FloatTensor([0]).to(device)
    batch_size=len(batch_data)
    for ep_step in batch_data:
      obs=ep_step.obs
      #sample squashed action w.r.t current policy
      dist=self.agent.pm(obs)
      action=dist.sample()
      sq_action=torch.tanh(action)
      sq_logp=self.agent.get_squashed_logprob(dist, action)
      Q1=self.agent.qm1(obs, sq_action)
      Q2=self.agent.qm2(obs, sq_action)
      min_Q=min(Q1, Q2)
      policy_loss=policy_loss+(alpha*sq_logp-min_Q)
    policy_loss=policy_loss/batch_size
    return policy_loss

  def train(self, batch_size, n_epochs, steps_per_epoch, start_after, update_after, update_every, temp_param, v_lr, q_lr, p_lr, polyak):
    #optimimizers
    sv_optim=optim.Adam(self.agent.svm.parameters(), lr=v_lr)
    #2 qm's to prevent overestimation bias
    q1_optim=optim.Adam(self.agent.qm1.parameters(), lr=q_lr)
    q2_optim=optim.Adam(self.agent.qm2.parameters(), lr=q_lr)
    policy_optim=optim.Adam(self.agent.pm.parameters(), lr=p_lr)

    step=1
    a_mode='random'
    update=False
    obs=self.agent.env.reset()

    for epoch in range(1, n_epochs+1):
      while True:
        if step>start_after:
          a_mode='policy'
        if step>update_after:
          update=True
        
        action=self.agent.get_action(obs, a_mode)
        action=torch.clamp(action, self.agent.a_low, self.agent.a_high).detach().cpu().numpy() #clamp action for progression
        obs_f, reward, termin_signal, _=self.agent.env.step(action)
        if termin_signal and obs_f[0]>0.45:
          rts=1 #real termin signal comparing horizon and real termination
        else:
          rts=0
        ep_step=Ep_Step(obs, action, reward, obs_f, rts)
        self.replay_buffer.add_item(ep_step)

        if update and step%update_every==0:
          for u_step in range(1, update_every+1):
            #sample batch at every update step -> prevents overfitting -> robust
            batch_data=self.replay_buffer.sample_batch(batch_size) 

            #take single step of update
            sv_loss=self.get_state_value_loss(batch_data, temp_param)

            sv_optim.zero_grad()
            sv_loss.backward()
            sv_optim.step()

            q1_loss=self.get_action_value_loss(batch_data, index=1)

            q1_optim.zero_grad()
            q1_loss.backward()
            q1_optim.step()

            q2_loss=self.get_action_value_loss(batch_data, index=2)

            q2_optim.zero_grad()
            q2_loss.backward()
            q2_optim.step()

            policy_loss=self.get_policy_loss(batch_data, temp_param)

            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()
          #calibrate svm parameters
          svm_param=self.agent.svm.vectorize_parameters()
          target_svm_param=self.agent.target_svm.vectorize_parameters()
          #polyak averaging
          new_svm_param=polyak*target_svm_param+(1-polyak)*svm_param
          self.agent.target_svm.inherit_parameters(new_svm_param)
        obs=obs_f
        step+=1
        if step%steps_per_epoch==1:
          break
      #check performance at end of epoch
      avg_return, avg_len_ep, avg_entropy=self.check_performance()
      print("Epoch: {:d}, Avg_Return: {:.3f}, Avg_Len_Ep: {:.3f}, Avg_Entropy: {:.3f}".format(epoch, avg_return, avg_len_ep, avg_entropy.item()))
    return

env=Obs_Wrapper(gym.make('MountainCarContinuous-v0'))
test_env=Obs_Wrapper(gym.make('MountainCarContinuous-v0'))
agent=SAC_Agent(env, test_env, 0.99)
sac=SAC(agent)
sac.train(batch_size=64, n_epochs=100, steps_per_epoch=4000, start_after=10000, update_after=1000, update_every=50, 
          temp_param=0.2, v_lr=3e-4, q_lr=3e-4, p_lr=3e-4, polyak=0.995)