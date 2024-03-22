import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import tools as tl



class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__()
        self.weight = tl.glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs
    
class VGAE(nn.Module):
    def __init__(self, 
                 adj,
                 input_dim,
                 hidden1_dim,
                 hidden2_dim,
                 ):
        super(VGAE, self).__init__()
        self.adj = adj
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.base_gcn = GraphConvSparse(input_dim, hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x:x)
        self.gcn_logstddev = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x:x)
    
    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.hidden2_dim)
        sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
        return sampled_z
    
    def forward(self, X):
        Z = self.encode(X)
        A_pred = tl.dot_product_decode(Z)
        return A_pred