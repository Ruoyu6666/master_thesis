# import dependencies
import os
import argparse
import math
import time
from filelock import FileLock
from typing import Dict

import numpy as np
import tempfile
import importlib
import numbers
import pickle
import scipy.sparse as sp
from scipy.sparse import linalg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch.autograd import Variable
from torch.utils.data import random_split

import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler, AsyncHyperBandScheduler

##### From layer.py #####
class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncwl, vw->ncvl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)

class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)  # A + I
        d = adj.sum(1)                                   # D0

        h = x
        out = [h]

        a = adj / d.view(-1, 1)                          # D0(-1)(A + I)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho

class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x

class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat = None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]             # dimensions
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim) # E1 (num_nodes, node_dim)
            self.emb2 = nn.Embedding(nnodes, dim) # E2 (num_nodes, node_dim)
            self.lin1 = nn.Linear(dim, dim)       # Theta1 (node_dim, node_dim) 
            self.lin2 = nn.Linear(dim, dim)       # Theta2 (node_dim, node_dim) 

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        # Get A
        a = torch.mm(nodevec1, nodevec2.transpose(1,0)) - torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1
        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        # Get A
        a = torch.mm(nodevec1, nodevec2.transpose(1,0)) - torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj

class hypergraph_constructor(nn.Module):
    def __init__(self, nnodes, nhedges, k, n_dim, he_dim, dim, 
                 device, alpha=3, feat_node=None, feat_hedges=None):
        super(hypergraph_constructor, self).__init__()
        
        self.nnodes = nnodes
        self.nhedges = nhedges
        self.n_dim = n_dim              # node feature dimension
        self.he_dim = he_dim            # hyperedge feature dimension
        self.dim = dim
        
        self.device = device
        self.k = k                      # subgraph size
        self.alpha = alpha
        self.feat_node = feat_node      # predefined node feature
        self.feat_hedges = feat_hedges  # predefined hyperedge feature

        if feat_node is None:
            self.embn = nn.Embedding(nnodes, n_dim)    # E_n: node embedding        (num_nodes, node_dim)
        if feat_hedges is None:
            self.embhe = nn.Embedding(nhedges, he_dim) # E_he: hyperedges embedding (num_hyperedges, hyperedge_dim)
        
        self.lin1 = nn.Linear(n_dim, dim)              # parameter: (node_dim, dim)
        self.lin2 = nn.Linear(he_dim, dim)             # parameter: (hyperedge_dim, dim)
 
    def forward(self, idx):
        if self.feat_node is None:
            nodevec1 = self.embn(idx)                  # (num_nodes, node_dim)
        if self.feat_hedges is None:
            nodevec2 = self.embhe(torch.tensor(range(self.nhedges)).to(self.device))   # (num_hyperedges, hyperedge_dim)
        else:
            nodevec1 = self.feat_node[idx,:]
            nodevec2 = self.feat_hedges[idx,:]

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1)) # E_n *  lin1 -> num_nodes * dim
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2)) # E_he * lin2 -> num_hyperedges * dim

        # Get H
        h0 = torch.mm(nodevec1, nodevec2.transpose(1,0))      # (num_nodes, num_hyperedges)
        H = F.relu(torch.tanh(self.alpha * h0))               # (num_nodes, num_hyperedges)
        adj = torch.mm(H, H.transpose(1,0))                   # H * H^T
        
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1,t1, s1.fill_(1))
        adj = adj * mask
        
        return adj

class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
    

### From net.py ###
class gthnet(nn.Module):
    
    def __init__(self, gcn_true, buildA_true, buildH_true, gcn_depth, 
                 num_nodes, num_hedges, device = "cuda" ,
                 predefined_A=None, static_feat=None, feat_node=None, feat_hedges=None,
                 dropout=0.3, subgraph_size=20, node_dim=40, hedge_dim=20, dim=40,
                 dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, 
                 seq_length=24, in_dim=2, out_dim=24,
                 layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):

        super(gthnet, self).__init__()
        
        self.gcn_true = gcn_true          # if there's gc_module
        self.buildA_true = buildA_true    # if build A 
        self.buildH_true = buildH_true    # if build H

        self.num_nodes = num_nodes
        self.num_hedges = num_hedges
        self.predefined_A = predefined_A
        self.static_feat = static_feat
        self.feat_node = feat_node
        self.feat_hedges = feat_hedges

        self.dropout = dropout
        self.seq_length = seq_length
        self.layers = layers
        
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        # self.start_conv
        self.start_conv = nn.Conv2d(in_channels = in_dim, out_channels = residual_channels, kernel_size = (1, 1))
       
        # self.gc: adj
        if self.buildA_true:
            self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)
        # self.hgc: adj
        if self.buildH_true:
            self.hgc = hypergraph_constructor(num_nodes, num_hedges, subgraph_size, node_dim, hedge_dim, dim, 
                                              device, alpha=tanhalpha, feat_node=feat_node, feat_hedges=feat_hedges)
        
        # self.receptive_field
        kernel_size = 7              # what is kernel_size ???
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        # rf_size_i, rf_size_j
        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1, layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)
                
                # filter_convs and gate_convs for TC
                # residual_convs, skip_convs
                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation)) # what IS dilated_inception ???
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels, kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,out_channels=skip_channels, kernel_size=(1, self.receptive_field-rf_size_j+1)))
                
                # gconv1, gconv2
                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                
                # self.norm
                if self.seq_length > self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential


        # end_conv_1, end_conv_2
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,  out_channels=out_dim, kernel_size=(1,1), bias=True)
        # skip0, skipE
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, 
                                   kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)
    
    
    ##### forward #####
    def forward(self, input, idx=None):

        seq_len = input.size(3) # batch_size * input_dim * num_nodes * seq_in_len
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field-self.seq_length, 0, 0, 0))
        
        # adp
        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            elif self.buildH_true:
                if idx is None:
                    adp = self.hgc(self.idx)
                else:
                    adp = self.hgc(idx)
            else:
                adp = self.predefined_A

        # start conv
        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training = self.training))

        for i in range(self.layers):
            residual = x
            
            # TC module
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training = self.training)
            
            # GC module
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)
            
            # Add residual connections
            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
    


# From trainer.py
class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)

        # for param in self.params:
        #     grad_norm += math.pow(param.grad.data.norm(), 2)
        #
        # grad_norm = math.sqrt(grad_norm)
        # if grad_norm > 0:
        #     shrinkage = self.max_grad_norm / grad_norm
        # else:
        #     shrinkage = 1.
        #
        # for param in self.params:
        #     if shrinkage < 1:
        #         param.grad.data.mul_(shrinkage)
        self.optimizer.step()
        return  grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()



##### Loss Functions #####
def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae, mape, rmse


##### Data Loader #####
class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, normalize = 2):
        self.P = window
        self.h = horizon
        
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        
        self.normalize = 2
        self.scale = np.ones(self.m) #self.m: number of nodes
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)# [5261, 321]

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

    def _normalized(self, normalize):
        if (normalize == 0):
            self.dat = self.rawdat
        
        # normalized by the maximum value of entire matrix.
        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))


    def _split(self, train, valid, test):#self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        train_set = range(self.P + self.h - 1, train) # (24*7 + 3 - 1, n * 0.6)
        valid_set = range(train, valid)# n * 0.6, n * 0.8
        test_set = range(valid, self.n)# n * 0.8, n
        
        self.train = self._batchify(train_set, self.h)# [15612, 168, 321], [15612, 321]
        self.valid = self._batchify(valid_set, self.h)# [5261, 168, 321], [5261, 321]
        self.test = self._batchify(test_set, self.h)  # [5261, 168, 321], [5261, 321]


    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m)) 
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1 # self.h:3
            start = end - self.P          # self.P:24*7
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :]) #([15612, 168, 321])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])   #([15612, 321])
        return [X, Y]


    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs) #15612
        if shuffle:
            index = torch.randperm(length) #Returns a random permutation of integers from 0 to length - 1.
        else:
            index = torch.LongTensor(range(length))
        
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device) #[4, 168, 321]
            Y = Y.to(self.device) #[4, 321]
            yield Variable(X), Variable(Y)
            start_idx += batch_size




parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')

parser.add_argument('--data', type=str, default='/home/guo/ma-thesis-ruoyu-code/MTHGNN/data/others/traffic.txt', help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='./model.pt', help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device',type=str,default='cuda',help='')

parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
parser.add_argument('--buildA_true', type=bool, default=False, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--buildH_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')

##### config #####
parser.add_argument('--num_nodes', type=int, default=862, help='number of nodes/variables')
parser.add_argument('--num_hyperedge', type=int, default=20, help='number of hyperedges/variables')
parser.add_argument('--dropout',type=float, default=0.3, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=20, help='k')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--hedge_dim',type=int,default=20, help='dim of hyperedge feature')
parser.add_argument('--dim',type=int,default=40, help='dim')

parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')
parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=16, help='residual channels')
parser.add_argument('--skip_channels', type=int,default=32, help='skip channels')
parser.add_argument('--end_channels', type=int, default=64, help='end channels')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=24, help='input sequence length')
parser.add_argument('--seq_out_len',type=int, default=1, help='output sequence length')
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--layers',type=int, default=3, help='number of layers')

parser.add_argument('--batch_size', type=int, default=4, help='batch size')

### config ###
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00005,help='weight decay rate')

parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')
parser.add_argument('--epochs', type=int, default=30, help='')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=100,help='step_size')


args = parser.parse_args()
#device = torch.device(args.device)
torch.set_num_threads(3)



# Train Function
def train_func(data, X, Y, model, criterion, optim, batch_size):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            id = torch.tensor(id).to(device).long()
            tx = X[:, :, id, :]
            ty = Y[:, id]
            output = model(tx, id)
            output = torch.squeeze(output)
            scale = data.scale.expand(output.size(0), data.m)
            scale = scale[:,id]
            loss = criterion(output * scale, ty * scale)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * data.m)
            grad_norm = optim.step()

        if iter%100==0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0) * data.m)))
        iter += 1
    return total_loss / n_samples


# Evaluate Function
def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):

    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim = 1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        if len(output.shape)==1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m)

    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return rse, rae, correlation, predict, Ytest



### Config & Train ###
def train_mthgnn_single_step(config):
    
    should_checkpoint = config.get("should_checkpoint", False)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Data Loader
    Data = DataLoaderS(args.data,
                       0.6, 0.2, device, args.horizon, args.seq_in_len, args.normalize)
    
    # Model
    model = gthnet(True, args.buildA_true, args.buildH_true, args.gcn_depth,
                   args.num_nodes, config["num_hedges"], device,
                   predefined_A=None, static_feat=None, feat_node=None, feat_hedges=None,
                   dropout=args.dropout, subgraph_size=config["subgraph_size"],
                   node_dim=args.node_dim, hedge_dim=args.hedge_dim, dim=args.dim, 
                   dilation_exponential=args.dilation_exponential,
                   conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                   skip_channels=args.skip_channels, end_channels= args.end_channels,
                   seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                   layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, 
                   layer_norm_affline=False)
    
    model = model.to(device)
    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
    )


    print(args)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)
    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)

    print('begin training')
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train_func(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
        val_loss, val_rae, val_corr, _, _ = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size)
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr), flush=True)
        # Save the model if the validation loss is the best we've seen so far.
        metrics = {
            "rse":val_loss.numpy(),
            "corr": val_corr
            }

        if should_checkpoint:
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(model.state_dict(), os.path.join(tempdir, "model.pt"))
                train.report(metrics, checkpoint=Checkpoint.from_directory(tempdir))
        else:
            train.report(metrics)


if __name__ == "__main__":

    ray.init(num_cpus=2)

    # for early stopping
    sched = AsyncHyperBandScheduler()

    resources_per_trial = {"cpu": 2, "gpu": 2}  # set this for GPUs
    tuner = tune.Tuner(
        tune.with_resources(train_mthgnn_single_step, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="rse",
            mode="min",
            scheduler=sched,
            num_samples=9,
        ),
        run_config=train.RunConfig(
            name="exp",
            stop={
                "rse": 0.001,
                "training_iteration": 100,
            },
        ),
        param_space={
        "num_hedges":  tune.choice([20, 30, 40, 50]),
        "subgraph_size": tune.choice([30, 40, 50])
        },
    )
    results = tuner.fit()

    print("Best config is:", results.get_best_result().config)

    assert not results.errors