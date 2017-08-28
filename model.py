import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from helper import *
from gridworld_goals import *

dtype = torch.FloatTensor


class DFP_Network(nn.Module):
    def __init__(self, batch_size,
                 offsets=[1, 2, 4, 8, 16, 32],
                 a_size=6,
                 num_measurements=2,
                 observation_h_size=128,
                 measurements_h_size=64,
                 goal_h_size=64,
                 is_master=False):
        '''
        PyTorch impementation of the network used to predict the measurements
        :param a_size: action space
        :param offsets: number of offset
        :param num_measurements: number of measurements
        :param is_master: true iff it is the shared master memory
        '''
        super(DFP_Network, self).__init__()
        self.batch_size = batch_size
        self.num_measurements = num_measurements
        self.action_size = a_size
        self.num_offsets = len(offsets)

        self.hidden_o = nn.ELU(nn.Linear(batch_size, observation_h_size))       # predict the observation -> [batch_size,128]
        self.hidden_m = nn.ELU(nn.Linear(batch_size, measurements_h_size))      # predict the measurements -> [batch_size,64]
        self.hidden_g = nn.ELU(nn.Linear(batch_size, goal_h_size))              # predict the goal -> [batch_size,64]

        j_h_size = (observation_h_size + measurements_h_size + goal_h_size)
        inner_h_size = (a_size * num_offsets * num_measurements)

        self.hidden_j = nn.ELU(nn.Linear(batch_size, j_h_size))                 # predict the joint-vectorial input -> [?,256]
        self.hidden_e = nn.Linear(j_h_size, inner_h_size)                       # predict the future measurements over all potential actions form the joint vector
        self.hidden_a = nn.Linear(j_h_size, inner_h_size)                       # predict the action-conditional differences {Ai(j)}, which are then combined to produce the final prediction for each action
        self.is_master = is_master
        if self.is_master:
            self.episodes = 0


    def should_stop(self, fn_stop=lambda: False):
        '''
        check if the master have to stop
        :param fn_stop: stoping condition
        :return:
        '''
        if self.is_master and not fn_stop:
            return False
        elif self.is_master and fn_stop:
            return True
        else:
            raise PermissionError("it is not a master")

    def forward(self, observation, measurements, goals, temperature):
        '''
        forward pass of DFP network
        :param observation: current state of the env shape=[batch_size, 5, 5, 3]
        :param measurements: current state of the measurements shape=[batch_size, num_measurements]
        :param goals: current state of the goal shape=[batch_size, num_measurements]
        :param temperature: parameter used to determine how spread out we want our action distribution to be
        :param actions:
        :return:
        '''
        self.temperature = temperature

        observation_flat = observation.view(self.batch_size, sum(observation.size()[1:]))
        # hidden input for the expectaion and action prediction
        hidden_input = torch.cat([self.hidden_o(observation_flat), self.hidden_m(measurements), self.hidden_g(goals)], dim=1)
        hidden_j = self.hidden_j(hidden_input)

        # average of the future measurements over all potential actions
        expectation = self.hidden_e(hidden_j)
        advantages = self.hidden_a(hidden_j)
        advantages = advantages - torch.mean(advantages, dim=1, keepdim=True)

        prediction = expectation + advantages
        # Reshape the predictions to be  [measurements x actions x offsets]
        self.prediction = prediction.view(self.batch_size, self.num_measurements, self.action_size, self.num_offsets)
        self.boltzmann = nn.Softmax(torch.sum(prediction, dim=3)) / self.temperature
        # prediction = nn.Softmax(torch.sum(prediction, dim=3))

        return self.boltzmann

    def loss(self, actions, targets):
        '''
        computation of the loss
        :param actions: actions taken during the sampled step
        :param targets: changes in the measurements
        :return:
        '''
        actions_one_hot = dtype(np.eye(self.action_size)[actions])
        # Select the predictions relevant to the chosen action.
        pred_action = torch.sum(self.prediction * actions_one_hot.view([-1, 1, self.action_size, 1]), dim=2)

        # Loss function
        loss = nn.MSELoss(pred_action, targets)

        # Sparsity of the action distribution
        entropy = -torch.sum(self.boltzmann * torch.log(self.boltzmann + 1e-7))
        return loss, entropy