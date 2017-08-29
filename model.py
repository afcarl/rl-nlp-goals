import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from helper import *
from model_helper import *
from gridworld_goals import *

dtype = torch.FloatTensor


class DFP_Network(nn.Module):
    def __init__(self, observation_i_size,
                 a_size=4,
                 num_offset=6,
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
        self.num_measurements = num_measurements
        self.action_size = a_size
        self.num_offsets = num_offset
        self.observation_i_size = observation_i_size

        self.hidden_o = FullyConnected(observation_i_size, observation_h_size, activation_fn=F.elu, dtype=dtype)       # predict the observation -> [batch_size,128]
        self.hidden_m = FullyConnected(num_measurements, measurements_h_size, activation_fn=F.elu, dtype=dtype)      # predict the measurements -> [batch_size,64]
        self.hidden_g = FullyConnected(num_measurements, goal_h_size, activation_fn=F.elu, dtype=dtype)              # predict the goal -> [batch_size,64]

        j_h_size = (observation_h_size + measurements_h_size + goal_h_size)
        f_h_size = (self.num_offsets * self.num_measurements)
        wf_h_size = (self.action_size * self.num_offsets * self.num_measurements)

        self.hidden_j = FullyConnected(j_h_size, j_h_size, activation_fn=F.elu, dtype=dtype)                   # predict the joint-vectorial input -> [?,256]
        self.hidden_e = FullyConnected(j_h_size, f_h_size, dtype=dtype)                           # predict the future measurements over all potential actions form the joint vector
        self.hidden_a = FullyConnected(j_h_size, wf_h_size, dtype=dtype)                          # predict the action-conditional differences {Ai(j)}, which are then combined to produce the final prediction for each action

        self.apply(weights_init)


        self.is_master = is_master
        if self.is_master:
            self.episodes = 0


    def should_stop(self, fn_stop=False):
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
        # temperature = dtype(temperature)
        observation_flat = Variable(dtype(observation.reshape(-1, self.observation_i_size)))
        measurements = Variable(dtype(measurements))
        goals = Variable(dtype(goals))

        # hidden input for the expectaion and action prediction
        hidden_input = torch.cat([self.hidden_o(observation_flat),
                                  self.hidden_m(measurements),
                                  self.hidden_g(goals)], dim=1)

        hidden_j = F.elu(self.hidden_j(hidden_input))

        # average of the future measurements over all potential actions
        self.expectation = self.hidden_e(hidden_j)
        self.expectation = self.expectation.repeat(1, self.action_size)

        self.advantages = self.hidden_a(hidden_j)
        self.advantages = self.advantages - torch.mean(self.advantages, dim=1, keepdim=True)

        self.prediction = self.expectation + self.advantages
        # Reshape the predictions to be  [measurements x actions x offsets]
        self.prediction = self.prediction.view(-1, self.num_measurements, self.action_size, self.num_offsets)
        self.boltzmann = softmax(torch.sum(self.prediction, dim=3)/temperature, axis=-1)
        # prediction = nn.Softmax(torch.sum(prediction, dim=3))

        return self.boltzmann

    def loss(self, actions, targets):
        '''
        computation of the loss
        :param actions: actions taken during the sampled step
        :param targets: changes in the measurements
        :return:
        '''
        self.actions_one_hot = Variable(dtype(np.eye(self.action_size)[actions.astype(np.int32)]))
        self.targets = Variable(dtype(targets))
        # Select the predictions relevant to the chosen action.
        self.pred_action = torch.sum(self.prediction * self.actions_one_hot.view([-1, 1, self.action_size, 1]), dim=2)


        self.dps_loss = F.mse_loss(self.pred_action, self.targets, size_average=True)

        # Sparsity of the action distribution
        self.entropy = -torch.sum(self.boltzmann * torch.log(self.boltzmann + 1e-7))
        return self.dps_loss, self.entropy