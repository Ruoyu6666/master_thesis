import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import linear1, prop, mixprop, dilated_inception,graph_constructor, hypergraph_constructor, LayerNorm


### Basic Model ###
class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, 
                 dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1,
                 conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, 
                 seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,out_channels=end_channels,kernel_size=(1,1),bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,out_channels=out_dim, kernel_size=(1,1),bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to(device)


    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
    

    
class gthnet(nn.Module):
    
    def __init__(self, gcn_true, buildA_true, buildH_true, gcn_depth, num_nodes, num_hedges, 
                 device = "cuda", predefined_A=None, static_feat=None, feat_node=None, feat_hedges=None,
                 dropout=0.3, subgraph_size=20, node_dim=40, hedge_dim=20, dim=40, dilation_exponential=1, 
                 conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, 
                 seq_length=24, in_dim=2, out_dim=24, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):

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
                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels, kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,out_channels=skip_channels, kernel_size=(1, self.receptive_field-rf_size_j+1)))
                
                # gconv1, gconv2
                if self.gcn_true:
                    self.gconv1.append(prop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(prop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                
                # self.norm
                if self.seq_length > self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        # end_conv_1, end_conv_2
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels,kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,  out_channels=out_dim, kernel_size=(1, 1), bias=True)
       
        # skip0, skipE
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)
    

    ######## forward ########
    def forward(self, input, idx=None):

        seq_len = input.size(3) # input: batch_size * input_dim * num_nodes * seq_in_len
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field-self.seq_length, 0, 0, 0)) # B * input_dim * num_nodes * receptive_field
        
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
        x = self.start_conv(input) # B * residual_channel * num_nodes * receptive_field
        skip = self.skip0(F.dropout(input, self.dropout, training = self.training)) # B * skip_channels * num_nodes * receptive_field
        
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
            # x: B * residual_channel * num_nodes * receptive_field

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
    


#########################
### Probalistic Model ###
#########################

class gthnet_prob(nn.Module):
    
    def __init__(self, gcn_true, buildA_true, buildH_true, gcn_depth, 
                 num_nodes, num_hedges, device = "cuda" ,
                 predefined_A=None, static_feat=None, feat_node=None, feat_hedges=None,
                 dropout=0.3, subgraph_size=20, node_dim=40, hedge_dim=20, dim=40,
                 dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, 
                 seq_length=24, in_dim=2, out_dim=24, 
                 layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True,
                 num_samples=30):

        super(gthnet_prob, self).__init__()
        
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.buildH_true = buildH_true

        self.num_nodes = num_nodes
        self.num_hedges = num_hedges
        self.predefined_A = predefined_A
        self.static_feat = static_feat
        self.feat_node = feat_node
        self.feat_hedges = feat_hedges

        self.dropout = dropout
        self.seq_length = seq_length
        self.layers = layers

        self.num_samples= num_samples
        
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels = in_dim, out_channels = residual_channels, kernel_size = (1, 1))
       
        if self.buildA_true:
            self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)
        if self.buildH_true:
            self.hgc = hypergraph_constructor(num_nodes, num_hedges, subgraph_size, node_dim, hedge_dim, dim, 
                                              device, alpha=tanhalpha, feat_node=feat_node, feat_hedges=feat_hedges)
        
        # self.receptive_field
        kernel_size = 7              # what is kernel_size ???
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1


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
                
                
                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation)) # what IS dilated_inception ???
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels, kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels,
                                                     kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,out_channels=skip_channels,
                                                     kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                
                if self.seq_length > self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential


        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels,kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,  out_channels=out_dim, kernel_size=(1,1), bias=True)

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)

        # TODO  Probabilistic Layer, single step prediction output after squeeze [batch_size, num_nodes]
        self.prob = ProbLinearV2(self.num_nodes, self.num_nodes, device)
    

    ######## forward ########
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

        ########################
        ### probability test ###
        ########################
        
        x1 = torch.squeeze(x) # batch_size * num_nodes
        mu, sigma = self.prob(x1)
        #preds = torch.stack([self.prob(x1) for i in range(self.num_samples)])
        #preds = torch.reshape(preds, (input.size(0), self.num_samples, -1))  #(B, n_sample, V)
        #return x1, preds
        return mu, sigma
    






class btm(nn.Module):
    def __init__(self, gcn_true, buildA_true, buildH_true, gcn_depth, 
                 num_nodes, num_hedges, device="cuda",
                 predefined_A = None, static_feat = None, feat_node=None, feat_hedges=None,
                 dropout = 0.3, subgraph_size = 20, node_dim = 40, hedge_dim = 20, dim = 40, 
                 dilation_exponential=2, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, 
                 seq_length=24, in_dim1=2, in_dim2=6, out_dim=24, 
                 layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        
        super(btm, self).__init__()
        
        self.gcn_true = gcn_true          # if there's gc_module
        self.buildA_true = buildA_true    # if build A 
        self.buildH_true = buildH_true    # if build H

        self.num_nodes = num_nodes
        self.num_hedges = num_hedges
        self.predefined_A = predefined_A
        self.dropout = dropout
        self.seq_length = seq_length
        self.layers = layers

        ### Load Forecasting ###
        self.filter_convsL = nn.ModuleList()
        self.gate_convsL = nn.ModuleList()
        self.residual_convsL = nn.ModuleList()
        self.skip_convsL = nn.ModuleList()
        self.gconv1L = nn.ModuleList()
        self.gconv2L = nn.ModuleList()
        self.normL = nn.ModuleList()

        ### PV Forecasting ###
        self.filter_convsPV = nn.ModuleList()
        self.gate_convsPV = nn.ModuleList()
        self.residual_convsPV = nn.ModuleList()
        self.skip_convsPV = nn.ModuleList()
        self.gconv1PV = nn.ModuleList()
        self.gconv2PV = nn.ModuleList()
        self.normPV = nn.ModuleList()
        

        # self.hgc: adj
        if self.buildA_true:
            self.gcL  = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)
            self.gcPV = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)
        if self.buildH_true:
            self.hgcL  = hypergraph_constructor(num_nodes, num_hedges, subgraph_size, node_dim, hedge_dim, dim, 
                                                device, alpha=tanhalpha, feat_node=feat_node, feat_hedges=feat_hedges)
            self.hgcPV = hypergraph_constructor(num_nodes, num_hedges, subgraph_size, node_dim, hedge_dim, dim, 
                                                device, alpha=tanhalpha, feat_node=feat_node, feat_hedges=feat_hedges)

        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        # start_conv
        self.start_convL  = nn.Conv2d(in_channels = in_dim1, out_channels = residual_channels, kernel_size = (1, 1))
        
        self.start_convPV = nn.Conv2d(in_channels = in_dim2, out_channels = residual_channels, kernel_size = (1, 1))


        ##### Load #####
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
                self.filter_convsL.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convsL.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convsL.append(nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels, kernel_size=(1, 1)))
                
                if self.seq_length > self.receptive_field:
                    self.skip_convsL.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels,
                                                      kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convsL.append(nn.Conv2d(in_channels=conv_channels,out_channels=skip_channels,
                                                      kernel_size=(1, self.receptive_field-rf_size_j+1)))
                # gconv1, gconv2
                if self.gcn_true:
                    self.gconv1L.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2L.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                
                # self.norm
                if self.seq_length > self.receptive_field:
                    self.normL.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.normL.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                new_dilation *= dilation_exponential
        
        # end_conv_1, end_conv_2
        self.end_conv_1L = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2L = nn.Conv2d(in_channels=end_channels,  out_channels=out_dim, kernel_size=(1,1), bias=True)
        
        # skip0, skipE
        if self.seq_length > self.receptive_field:
            self.skip0L = nn.Conv2d(in_channels=in_dim1, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipEL = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)
        else:
            self.skip0L = nn.Conv2d(in_channels=in_dim1, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipEL = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        ##### PV #####
        for i in range(1):
            if dilation_exponential > 1:   # what is rf_size ??? receptive_field size?
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
                self.filter_convsPV.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation)) # what IS dilated_inception ???
                self.gate_convsPV.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convsPV.append(nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels, kernel_size=(1, 1)))
                
                if self.seq_length>self.receptive_field:
                    self.skip_convsPV.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels,
                                                       kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convsPV.append(nn.Conv2d(in_channels=conv_channels,out_channels=skip_channels,
                                                       kernel_size=(1, self.receptive_field-rf_size_j+1)))
                # gconv1, gconv2
                if self.gcn_true:
                    self.gconv1PV.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2PV.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                
                # self.norm
                if self.seq_length > self.receptive_field:
                    self.normPV.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.normPV.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                new_dilation *= dilation_exponential
        
        # end_conv_1, end_conv_2
        self.end_conv_1PV = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2PV = nn.Conv2d(in_channels=end_channels,  out_channels=out_dim, kernel_size=(1,1), bias=True)
        
        # skip0, skipE
        if self.seq_length > self.receptive_field:
            self.skip0PV = nn.Conv2d(in_channels=in_dim2, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipEPV = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)
        else:
            self.skip0PV = nn.Conv2d(in_channels=in_dim2, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipEPV = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        # After Fusion 
        # Predict Load
        self.L1L = linear1(out_dim, out_dim)
        #self.L2L = linear1(skip_channels, out_dim)
        
        # Predict PV 
        self.L1PV = linear1(out_dim, out_dim)
        #self.L2PV = linear1(end_channels, skip_channels)
        #self.L3PV = linear1(skip_channels, out_dim)
        
        self.idx = torch.arange(self.num_nodes).to(device)



    def forward(self, input1, input2, idx=None):
        
        seq_len = input1.size(3)         # batch_size * input_dim * num_nodes * seq_in_len
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length < self.receptive_field:
            input1 = nn.functional.pad(input1, (self.receptive_field-self.seq_length, 0, 0, 0))
            input2 = nn.functional.pad(input2, (self.receptive_field-self.seq_length, 0, 0, 0))
        
        if self.buildA_true:
            if idx is None:
                adpL  = self.gcL(self.idx)
                adpPV = self.gcPV(self.idx)
            else:
                adpL  = self.gcL(idx)
                adpPV = self.gcPV(idx)
        
        if self.buildH_true:
            if idx is None:
                adpL  = self.hgcL(self.idx)
                adpPV = self.hgcPV(self.idx)
            else:
                adpL = self.hgcL(idx)
                adpPV = self.hgcPV(idx)

       
        ##### Load #####
        xL = self.start_convL(input1)
        skipL = self.skip0L(F.dropout(input1, self.dropout, training = self.training))
        
        for i in range(self.layers):
            residualL = xL
            
            # TC module
            filterL = self.filter_convsL[i](xL)
            filterL = torch.tanh(filterL)
            gateL = self.gate_convsL[i](xL)
            gateL = torch.sigmoid(gateL)
            xL = filterL * gateL
            xL = F.dropout(xL, self.dropout, training = self.training)
            
            # GC module
            sL = xL
            sL = self.skip_convsL[i](sL)
            skipL = sL + skipL
            if self.gcn_true:
                xL = self.gconv1L[i](xL, adpL) + self.gconv2L[i](xL, adpL.transpose(1,0))
            else:
                xL = self.residual_convsL[i](xL)
            
            # plus residual connections
            xL = xL + residualL[:, :, :, -xL.size(3):]
            if idx is None:
                xL = self.normL[i](xL, self.idx)
            else:
                xL = self.normL[i](xL, idx)

        skipL = self.skipEL(xL) + skipL
        xL = F.relu(skipL)
        xL = F.relu(self.end_conv_1L(xL))
        xL = self.end_conv_2L(xL) # ([4, 24, 300, 1])

        

        ##### PV #####
        xPV = self.start_convPV(input2)
        skipPV = self.skip0PV(F.dropout(input2, self.dropout, training = self.training))
        
        for i in range(self.layers):
            residualPV = xPV
            
            # TC module
            filterPV = self.filter_convsPV[i](xPV)
            filterPV = torch.tanh(filterPV)
            gatePV = self.gate_convsPV[i](xPV)
            gatePV = torch.sigmoid(gatePV)
            xPV = filterPV * gatePV
            xPV = F.dropout(xPV, self.dropout, training = self.training)
            
            # GC module
            sPV = xPV
            sPV = self.skip_convsPV[i](sPV)
            skipPV = sPV + skipPV
            if self.gcn_true:
                xPV = self.gconv1PV[i](xPV, adpPV) + self.gconv2PV[i](xPV, adpPV.transpose(1,0))
            else:
                xPV = self.residual_convsPV[i](xPV)
            
            # plus residual connections
            xPV = xPV + residualPV[:, :, :, -xPV.size(3):]
            if idx is None:
                xPV = self.normPV[i](xPV, self.idx)
            else:
                xPV = self.normPV[i](xPV, idx)

        skipPV = self.skipEPV(xPV) + skipPV
        xPV = F.relu(skipPV)
        xPV = F.relu(self.end_conv_1PV(xPV))
        xPV = self.end_conv_2PV(xPV)        # ([4, 24, 300, 1])
        
        #XF = torch.cat((xL, xPV), dim=1)   # ([4, 48, 300, 1])
        #XF = torch.squeeze(XF)
        
        xL = torch.squeeze(xL, 3)
        xPV = torch.squeeze(xPV, 3)
        
        xF = F.relu(xL + xPV)
        xL = F.relu(self.L1L(xF))
        xPV = F.relu(self.L1PV(xF))

        return xL, xPV