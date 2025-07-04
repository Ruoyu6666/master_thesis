from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F

# nconv
# linear
# prop
# mixprop
# dialted_1D
# dialted_inception
# graph constructor
# graph directed
# graph undirected
# hypergraph constructor
# Layer Norm


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncwl, vw->ncvl', (x, A))
        return x.contiguous()




class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), 
                                   padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)
    



# Input (N, C_in, L_in), Output (N, C_out, L_out)
class linear1(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear1,self).__init__()
        self.mlp = torch.nn.Conv1d(c_in, c_out, kernel_size=1, padding=0, stride=1, bias=bias)
    def forward(self,x):
        return self.mlp(x)





class prop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device) # torch.eye: Returns a 2-D tensor with ones on the diagonal and zeros elsewhere
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        ho = self.mlp(h)
        return ho





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
    



class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        self.tconv = nn.Conv2d(cin, cout, (1,7), dilation=(1, dilation_factor))

    def forward(self,input):
        x = self.tconv(input)
        return x



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



class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj



class graph_directed(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

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

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
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