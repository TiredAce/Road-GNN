import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import sys
import argparse
import os
import heapq
import math
import tools as tl
from sklearn.metrics import f1_score
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MLPModel(nn.Module):
    
    def __init__(self,
                 layer_units,
                 output_units,
                 dropout_prob = 0.6,
                 L2 = 0.01,
                 activation = lambda x:x,
                 residual = False, 
                 batch_normalization = 'ZCA',
                 ZCA_iteration = 10,
                 ZCA_lam = 0.5,
                 ZCA_temp = 10,
                 seed = 1):
        
        super(MLPModel, self).__init__()

        torch.manual_seed(seed)
        
        self.output_units = output_units
        self.residual = residual
        self.ZCA_iteration = ZCA_iteration
        self.batch_normalization = batch_normalization
        self.lam = ZCA_lam
        self.ZCA_temp = ZCA_temp
        
        self.dense_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        self.ZCA_mean_list = []
        self.ZCA_C_list = []
        
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = activation
        
        self.ZCA_mean_list.append(torch.zeros(1, layer_units[0]))
        self.ZCA_C_list.append(torch.eye(layer_units[0]))
        
        dense_layer = nn.Linear(layer_units[0], layer_units[0], bias=False)
        self.dense_layers.append(dense_layer)
        
        for l in range(1, len(layer_units)):
            
            self.ZCA_mean_list.append(torch.zeros(1, layer_units[l]))
            self.ZCA_C_list.append(torch.eye(layer_units[l]))
            
            dense_layer = nn.Linear(layer_units[l - 1], layer_units[l], bias=False)
            self.dense_layers.append(dense_layer)
            
            if residual:
                residual_layer = nn.Linear(layer_units[l - 1], layer_units[l], bias=False)
                self.residual_layers.append(residual_layer)
                
        self.ZCA_mean_list.append(torch.zeros(1, output_units))
        self.ZCA_C_list.append(torch.eye(output_units))
                
        dense_layer = nn.Linear(layer_units[-1], output_units, bias=False)
        self.dense_layers.append(dense_layer)
        
        
    def forward(self, x, training=True):
        
        deep_copy_list = []
        
        x = self.dense_layers[0](x)
        
        if self.batch_normalization == 'ZCA':
            if training:
                X, ZCA_mean, ZCA_C = tl.Newton_ZCA_for_features(x, temp = self.ZCA_temp, T = self.ZCA_iteration)
                self.ZCA_mean_list[0] = (1 - self.lam) * self.ZCA_mean_list[0] + self.lam * ZCA_mean
                self.ZCA_C_list[0] = (1 - self.lam) * self.ZCA_C_list[0] + self.lam * ZCA_C
            
            else:
                x = torch.matmul(x - self.ZCA_mean_list[0], self.ZCA_C_list[0])
                
        elif self.batch_normalization == 'Schur_ZCA':
            if training:
                x, Z_mean, ZCA_C = tl.Schur_Newton_ZCA_for_features(x, temp = self.ZCA_temp, T = self.ZCA_iteration)

                self.ZCA_mean_list[0] = (1 - self.lam) * self.ZCA_mean_list[0] + self.lam * ZCA_mean
                self.ZCA_C_list[0] = (1 - self.lam) * self.ZCA_C_list[0] + self.lam * ZCA_C
            
            else:
                x = torch.matmul(x - self.ZCA_mean_list[0], self.ZCA_C_list[0])
                
        elif self.batch_normalization == 'BN':
            x = nn.BatchNorm1d(x.size(1))(x)
        
        x = self.activation(x)
        
        x = self.dropout(x)
        
        deep_copy_list.append(x.clone())
        
        for l in range(1, len(self.dense_layers) - 1):
            
            h = x.clone()
            
            h = self.dense_layers[l](h)
            
            if self.residual:
                h = h + self.residual_layers[l-1](x)
                
            if self.batch_normalization == 'ZCA':
                if training:
                    X, ZCA_mean, ZCA_C = tl.Newton_ZCA_for_features(x, temp = self.ZCA_temp, T = self.ZCA_iteration)
                    self.ZCA_mean_list[l] = (1 - self.lam) * self.ZCA_mean_list[l] + self.lam * ZCA_mean
                    self.ZCA_C_list[l] = (1 - self.lam) * self.ZCA_C_list[l] + self.lam * ZCA_C
            
                else:
                    h = torch.matmul(h - self.ZCA_mean_list[l], self.ZCA_C_list[l])
                    
            elif self.batch_normalization == 'Schur_ZCA':
                if training:
                    x, Z_mean, ZCA_C = tl.Schur_Newton_ZCA_for_features(x, temp = self.ZCA_temp, T = self.ZCA_iteration)

                    self.ZCA_mean_list[l] = (1 - self.lam) * self.ZCA_mean_list[l] + self.lam * ZCA_mean
                    self.ZCA_C_list[l] = (1 - self.lam) * self.ZCA_C_list[l] + self.lam * ZCA_C
            
                else:
                    h = torch.matmul(h - self.ZCA_mean_list[l], self.ZCA_C_list[l])
                    
            elif self.batch_normalization == 'BN':
                h = nn.BatchNorm1d(h.size(1))(h)
                
            h = self.activation(h)
            
            x = self.dropout(h)
            
            deep_copy_list.append(x.clone())
        
        x = self.dense_layers[-1](x)
        
        if self.batch_normalization == 'ZCA':
            if training:
                X, ZCA_mean, ZCA_C = tl.Newton_ZCA_for_features(x, temp = self.ZCA_temp, T = self.ZCA_iteration)
                self.ZCA_mean_list[-1] = (1 - self.lam) * self.ZCA_mean_list[-1] + self.lam * ZCA_mean
                self.ZCA_C_list[-1] = (1 - self.lam) * self.ZCA_C_list[-1] + self.lam * ZCA_C
            
            else:
                x = torch.matmul(x - self.ZCA_mean_list[-1], self.ZCA_C_list[-1])
                
        elif self.batch_normalization == 'Schur_ZCA':
            if training:
                x, Z_mean, ZCA_C = tl.Schur_Newton_ZCA_for_features(x, temp = self.ZCA_temp, T = self.ZCA_iteration)

                self.ZCA_mean_list[-1] = (1 - self.lam) * self.ZCA_mean_list[-1] + self.lam * ZCA_mean
                self.ZCA_C_list[-1] = (1 - self.lam) * self.ZCA_C_list[-1] + self.lam * ZCA_C
            
            else:
                x = torch.matmul(x - self.ZCA_mean_list[-1], self.ZCA_C_list[-1])
                
        elif self.batch_normalization == 'BN':
            x = nn.BatchNorm1d(x.size(1))(x)
            
        x = self.activation(x)
        
        x = self.dropout(x)
        
        deep_copy_list.append(x)
            
        return deep_copy_list



def unsupervised_learning(features,
                          anchor_indexes,
                          augmented_indexes,
                          model,
                          contrast_loss,
                          batch_size=64,
                          lam=0.005,
                          temperature=1.0,
                          num_epochs=5000,
                          learning_rate=0.001,
                          alpha=1.0,
                          beta=1.0,
                          L2=0.01):
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2)
        
    train_loss_results = []
    
    feat_ds = TensorDataset(torch.tensor(anchor_indexes), torch.tensor(augmented_indexes))
    feat_dl = DataLoader(feat_ds, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
                
        loss_value_list = []
                
        for anc_ind, aug_ind in feat_dl:
        
            optimizer.zero_grad()
            
            y_pred_list = model(features[anc_ind])
            true_pred_list = model(features[aug_ind])
                
            loss_value = 0.0
                
            for y_pred, true_pred in zip(y_pred_list, true_pred_list):
                y_pred = y_pred / (torch.norm(y_pred, dim=1, keepdim=True) + 1e-8)
                true_pred = true_pred / (torch.norm(true_pred, dim=1, keepdim=True) + 1e-8)
                
                if contrast_loss == 'distance':
                    loss_value += tl.loss_dot_product_v2(y_pred=y_pred, true_pred=true_pred, temperature=temperature)
                        
                elif contrast_loss == 'signal_distance':
                    loss_value += tl.loss_dot_product_v2(y_pred=y_pred, true_pred=true_pred, axis=0, temperature=temperature)
                        
                elif contrast_loss == 'contrastive':
                    loss_value += tl.loss_dot_product_v3(y_pred=y_pred, true_pred=true_pred, temperature=temperature)
                        
                elif contrast_loss == 'signal_contrastive':
                    loss_value += tl.loss_dot_product_v3(y_pred=y_pred, true_pred=true_pred, axis=0, temperature=temperature)
                        
                elif contrast_loss == 'distance+auto-correlation':
                    loss_value += (alpha * tl.loss_dot_product_v2(y_pred=y_pred, true_pred=true_pred, temperature=temperature) + \
                                   beta * tl.auto_correlation(y_pred=y_pred, lam=lam) + tl.auto_correlation(y_pred=true_pred, lam=lam))
                            
                elif contrast_loss == 'signal_distance+auto-correlation':
                    loss_value += (alpha * tl.loss_dot_product_v2(y_pred=y_pred, true_pred=true_pred, axis=0, temperature=temperature) + \
                                   beta * tl.auto_correlation(y_pred=y_pred, lam=lam) + tl.auto_correlation(y_pred=true_pred, lam=lam))
                            
                elif contrast_loss == 'cross-correlation+auto-correlation':
                    loss_value += (alpha * tl.cross_correlation(y_pred=y_pred, true_pred=true_pred, lam=lam) + \
                                   beta * tl.auto_correlation(y_pred=y_pred, lam=lam) + tl.auto_correlation(y_pred=true_pred, lam=lam))
                            
                elif contrast_loss == 'cross-correlation':
                    loss_value += tl.cross_correlation(y_pred=y_pred, true_pred=true_pred, lam=lam)
                
            loss_value.backward()
            optimizer.step()
            
            loss_value_list.append(loss_value.item())
        
        loss = np.mean(loss_value_list)    
        train_loss_results.append(loss)
            
        print('Epoch {:03d}: Loss: {:.5f}'.format(epoch + 1, loss))
        
    logits = model(features, training = False)
    
    return logits[0]



parser = argparse.ArgumentParser()

parser.add_argument(
        '-d', 
        '--dataset',
        type = str,
        default = 'Cora')

parser.add_argument(
        '--alpha',
        type = float,
        default = 1.0)

parser.add_argument(
        '--beta',
        type = float,
        default = 1.0)

parser.add_argument(
        '-cl', 
        '--contrast-loss',
        type = str,
        default = 'distance')

parser.add_argument(
        '-la',
        '--lam',
        type = float,
        default = 0.0)


parser.add_argument(
        '-t',
        '--temperature',
        type = float,
        default = 1.0)

parser.add_argument(
        '-r',
        '--num-run',
        type = int,
        default = 1)

parser.add_argument(
        '-bs',
        '--batch-size',
        type = int,
        default = 2048) #65536

parser.add_argument(
        '-zi',
        '--ZCA-iteration',
        type = int,
        default = 4)

parser.add_argument(
        '-zt',
        '--ZCA-temp',
        type = float,
        default = 0.05)

parser.add_argument(
        '-zl',
        '--ZCA-lam',
        type = float,
        default = 0.0)

parser.add_argument(
        '-bn', 
        '--batch-normalization',
        type = str,
        default = 'Schur_ZCA') # None, BN, ZCA, Schur_ZCA

parser.add_argument(
        '-pe',
        '--pretext-num-epochs',
        type = int,
        default = 70)

parser.add_argument(
        '-de',
        '--downstream-num-epochs',
        type = int,
        default = 3000)

parser.add_argument(
        '-pu',
        '--pretext-hidden-units',
        nargs='+', 
        type=int,
        default = [512])

parser.add_argument(
        '-pt',
        '--pretext-output-units',
        type = int,
        default = 256)

parser.add_argument(
        '-po',
        '--pretext-dropout',
        type = float,
        default = 0.)

parser.add_argument(
        '-do',
        '--downstream-dropout',
        type = float,
        default = 0.5)

parser.add_argument(
        '-pL',
        '--pretext-L2',
        type = float,
        default = 0.)

parser.add_argument(
        '-dL',
        '--downstream-L2',
        type = float,
        default = 0.001)

parser.add_argument(
        '-pl',
        '--pretext-learning-rate',
        type = float,
        default = 0.001)

parser.add_argument(
        '-dl',
        '--downstream-learning-rate',
        type = float,
        default = 0.001)

parser.add_argument(
        '-y',
        '--early-stopping',
        type = int,
        default = 200)

args = parser.parse_args()

"""
Input Data

anchor_indexes:
augmented_indexes:
features

"""