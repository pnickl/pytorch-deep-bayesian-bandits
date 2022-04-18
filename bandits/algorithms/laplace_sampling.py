"""Thompson Sampling with Laplace posterior approximation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bandits.core.contextual_dataset import ContextualDataset
from bandits.algorithms.neural_bandit_model import NeuralBanditModel #KronBanditModel
from laplace_dataset import LaplaceDataset, get_dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from laplace import Laplace
from copy import deepcopy

class LaplaceSampling():

    def __init__(self, name, hparams, optimizer='RMS'):

        self.random = False

        if name == 'FullLaplace':
            self.hessian_structure = 'full'
        elif name == 'KronLaplace':
            self.hessian_structure = 'kron'
        elif name == 'DiagLaplace':
            self.hessian_structure = 'diag'
        elif name == 'Random':
            # random action selection
            self.hessian_structure = 'diag'
            self.random = True
        else:
            print('Invalid name of algorithm')

        self.name = name
        self.hparams = hparams

        # Laplace and NN Update Frequency
        self.update_freq_post = hparams.update_freq_post
        self.update_freq_nn = hparams.training_freq_network

        self.t = 0
        self.optimizer_n = optimizer

        self.num_epochs = hparams.training_epochs
        self.data_h = ContextualDataset(hparams.context_dim,
                                        hparams.num_actions,
                                        intercept=False)

        self.bnn = NeuralBanditModel(optimizer, hparams, '{}-bnn'.format(name))

    def action(self, context):
        """Samples weights from posterior, and chooses best action accordingly."""

        # Round robin until each action has been selected "initial_pulls" times
        if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
            return self.t % self.hparams.num_actions

        if not self.random:
            X_test = context.reshape((1, self.hparams.context_dim))
            # vals2, f_var = self.la._glm_predictive_distribution(torch.from_numpy(X_test).float())
            vals = self.la.predictive_samples(torch.from_numpy(X_test).float(), n_samples=10)
            action = np.argmax(torch.mean(vals, axis=0))
        else:
            weights = torch.tensor([0.5, 0.5], dtype=torch.float)  # create a tensor of weights
            action = torch.multinomial(weights, 1, replacement=True)[0]

        return action

    def update(self, context, action, reward):
        """Updates the Laplace posteiror."""

        # add data
        self.t += 1
        self.data_h.add(context, action, reward)

        # Retrain the network on the original data (data_h)
        if self.t % self.update_freq_nn == 0:

            if self.hparams.reset_lr:
                self.bnn.assign_lr()
            self.bnn.train(self.data_h, self.num_epochs)

        # Update the Laplace approximation
        if self.t == self.hparams.num_actions * self.hparams.initial_pulls or self.t % self.update_freq_post == 0:
            self.la = Laplace(self.bnn.net, 'regression', hessian_structure=self.hessian_structure,
                              subset_of_weights='all')

            z, y, _ = self.data_h.get_data_with_weights()
            X_train, y_train, ds_train = get_dataset(z, y)
            train_loader = DataLoader(ds_train, batch_size=128, shuffle=False)
            self.la.fit(train_loader)


# class LaplaceSamplingKron():
#
#     def __init__(self, name, hparams, optimizer='RMS'):
#
#         self.random = False
#
#         if name == 'FullLaplace':
#             self.hessian_structure = 'full'
#         elif name == 'KronLaplace':
#             self.hessian_structure = 'kron'
#         elif name == 'DiagLaplace':
#             self.hessian_structure = 'diag'
#         elif name == 'Random':
#             # random action selection
#             self.hessian_structure = 'diag'
#             self.random = True
#         else:
#             print('Invalid name of algorithm')
#
#         self.name = name
#         self.hparams = hparams
#
#         # Laplace and NN Update Frequency
#         self.update_freq_post = hparams.update_freq_post
#         self.update_freq_nn = hparams.training_freq_network
#
#         self.t = 0
#         self.optimizer_n = optimizer
#
#         self.num_epochs = hparams.training_epochs
#         self.data_h = ContextualDataset(hparams.context_dim,
#                                         hparams.num_actions,
#                                         intercept=False)
#
#         self.bnn = NeuralBanditModel(optimizer, hparams, '{}-bnn'.format(name))
#
#     def action(self, context):
#         """Samples weights from posterior, and chooses best action accordingly."""
#
#         # Round robin until each action has been selected "initial_pulls" times
#         if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
#             return self.t % self.hparams.num_actions
#
#         if not self.random:
#             X_test = context.reshape((1, self.hparams.context_dim))
#             # vals2, f_var = self.la._glm_predictive_distribution(torch.from_numpy(X_test).float())
#             vals = self.la.predictive_samples(torch.from_numpy(X_test).float(), n_samples=1)
#             action = np.argmax(torch.mean(vals, axis=0))
#         else:
#             weights = torch.tensor([0.5, 0.5], dtype=torch.float)  # create a tensor of weights
#             action = torch.multinomial(weights, 1, replacement=True)[0]
#
#         return action
#
#     def update(self, context, action, reward):
#         """Updates the Laplace posteiror."""
#
#         # add data
#         self.t += 1
#         self.data_h.add(context, action, reward)
#
#         # Retrain the network on the original data (data_h)
#         if self.t % self.update_freq_nn == 0:
#
#             if self.hparams.reset_lr:
#                 self.bnn.assign_lr()
#             self.bnn.train(self.data_h, self.num_epochs)
#
#         # Update the Laplace approximation
#         if self.t == self.hparams.num_actions * self.hparams.initial_pulls or self.t % self.update_freq_post == 0:
#             self.la = Laplace(self.bnn.net, 'regression', hessian_structure=self.hessian_structure,
#                               subset_of_weights='all')
#
#             z, y, _ = self.data_h.get_data_with_weights()
#             X_train, y_train, ds_train = get_dataset(z, y)
#             train_loader = DataLoader(ds_train, batch_size=128, shuffle=False)
#             self.la.fit(train_loader)

