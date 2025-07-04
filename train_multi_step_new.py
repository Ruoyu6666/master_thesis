import argparse
import math
import time

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import importlib

from net import gthnet
from util import DataLoaderM_new, masked_mae, masked_mape, metric
from trainer import Optim



def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    train_loss = []
    total_loss = 0
    n_samples = 0
    iter = 0
    
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1) # a dimension of size one inserted at dim = 1.
        X = X.transpose(2, 3)         #  batch_size * 1 * num_nodes * seq_in_len

        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes)) # return a permuted range.
        num_sub = int(args.num_nodes / args.num_split) # = num_nodes when num_split = 1

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            id = torch.tensor(id).to(device).long() # ([1413])
            tx = X[:, :, id, :]                     # ([4, 1, 1413, 168])
            ty = Y[:, :, id]
            output = model(tx, id)                  # ([4, 24, 1413, 1])
            output = torch.squeeze(output)          # ([4, 24, 1413])
            
            scale = data.scale.expand(output.size(0), output.size(1), data.m)
            scale = scale[:, :, id]
            loss = criterion(output * scale, ty * scale)
            metrics = metric(output, ty)
            loss.backward()
            
            total_loss += loss.item()
            n_samples += (output.size(0)*output.size(1)*data.m)
            grad_norm = optim.step()
            train_loss.append(metrics[0]*output.size(1))

        if iter%100==0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0) *output.size(1) * data.m)))
        iter += 1
    #return total_loss / n_samples
    return np.mean(train_loss)




def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    valid_loss = []
    valid_rmse = []
    valid_mape = []

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim = 1) # Insert a dimension of size one at the specified position
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)           # [4, 24, num_nodes, 1]
        output = torch.squeeze(output)  # [4, 24, num_nodes]
         
        if len(output.size()) < 3:
            output = output.unsqueeze(dim=0)
        if len(Y.size()) < 3:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        predict = torch.cat((predict, output))
        test = torch.cat((test, Y))

        metrics = metric(output, Y)
        valid_mape0 = masked_mape(torch.sum(output, 1), torch.sum(Y, 1), 0.0).item()
        valid_loss.append(metrics[0])
        valid_mape.append(valid_mape0)
        #valid_mape.append(metrics[1])
        valid_rmse.append(metrics[2])

        scale = data.scale.expand(output.size(0), output.size(1), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) *output.size(1) * data.m)
        loss = total_loss_l1 / n_samples

    mae = np.mean(valid_loss)
    rmse = np.mean(valid_rmse)
    mape = np.mean(valid_mape)

    return mae, rmse, mape, loss


#############################
### Define ArgumentParser ###
#############################
parser = argparse.ArgumentParser(description='PyTorch Time Series Forecasting')
parser.add_argument('--data', type=str, default='./data/btm/pv.csv', help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='./model/model-pv-mthgnn.pt', help='path to save the final model')
parser.add_argument('--L1Loss', type=bool, default=True)
#####
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--epochs',type=int, default=30, help='')
parser.add_argument('--num_split', type=int, default=1,  help='number of splits for graphs')
parser.add_argument('--step_size', type=int, default=100,help='step_size')

# Model
parser.add_argument('--device', type=str,default='cuda', help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=False, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--buildH_true', type=bool, default=True, help='whether to construct adaptive incidence matrix')
parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
parser.add_argument('--num_nodes', type=int, default=300, help='number of nodes/variables')
parser.add_argument('--num_hyperedge', type=int, default=30, help='number of hyperedges/variables')
parser.add_argument('--dropout',type=float,default=0.3, help='dropout rate')
#####
parser.add_argument('--subgraph_size',type=int,default=20, help='k')
parser.add_argument('--node_dim',type=int,default=40, help='dim of node feature')
parser.add_argument('--hedge_dim',type=int,default=20, help='dim of hyperedge feature')
parser.add_argument('--dim',type=int,default=40, help='dim')

parser.add_argument('--dilation_exponential',type=int, default=2,help='dilation exponential')
parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=16, help='residual channels')
parser.add_argument('--skip_channels', type=int,default=32, help='skip channels')
parser.add_argument('--end_channels', type=int, default=64, help='end channels')

parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=24, help='input sequence length')
parser.add_argument('--seq_out_len',type=int, default=24, help='output sequence length')
parser.add_argument('--layers',type=int, default=3, help='number of layers')
parser.add_argument('--propalpha',type=float, default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float, default=3,help='tanh alpha')

# DataLoader
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--horizon', type=int, default=3)

# trainer: Optim
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--optim', type=str, default='adam')


args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)

def main():

    Data = DataLoaderM_new(args.data, 0.7, 0.15, device, args.horizon,
                       args.seq_in_len, args.seq_out_len, args.normalize)

    model = gthnet(args.gcn_true, args.buildA_true, args.buildH_true, args.gcn_depth, 
                   args.num_nodes, args.num_hyperedge, device,
                   dropout=args.dropout, subgraph_size=args.subgraph_size, 
                   node_dim=args.node_dim, hedge_dim=args.hedge_dim, dim = args.dim, 
                   dilation_exponential=args.dilation_exponential,
                   conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                   skip_channels=args.skip_channels, end_channels= args.end_channels,
                   seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                   layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, 
                   layer_norm_affline=False)
    
    model = model.to(device)

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


    best_val = 10000000
    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
    )
    
    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
            val_mae, val_rmse, val_mape, loss = evaluate(Data, Data.valid[0], Data.valid[1], model, 
                                                         evaluateL2, evaluateL1, args.batch_size)
            

            print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid_loss {:5.4f} | valid mae {:5.4f} | valid rmse {:5.4f} | valid mape  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, loss, val_mae, val_rmse, val_mape), flush=True)
            # Save the model if the validation loss is the best we've seen so far.

            if val_mae < best_val:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val = val_mae
            if epoch % 5 == 0:
                test_acc, test_rae, test_corr, _ = evaluate(Data, Data.test[0], Data.test[1], model, 
                                                            evaluateL2, evaluateL1, args.batch_size)
                print("test mae {:5.4f} | test rmse {:5.4f} | test mape {:5.4f}".format(test_acc, test_rae, test_corr), flush=True)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    vtest_acc, vtest_rae, vtest_corr, _ = evaluate(Data, Data.valid[0], Data.valid[1], model, 
                                                   evaluateL2, evaluateL1, args.batch_size)
    test_acc, test_rae, test_corr, _ = evaluate(Data, Data.test[0], Data.test[1], model, 
                                                evaluateL2, evaluateL1, args.batch_size)
    print("final test mae {:5.4f} | test rmse {:5.4f} | test mape {:5.4f}".format(test_acc, test_rae, test_corr))
    return vtest_acc, vtest_rae, vtest_corr, test_acc, test_rae, test_corr

if __name__ == "__main__":
    vacc = []
    vrae = []
    vcorr = []
    acc = []
    rae = []
    corr = []
    for i in range(1):
        val_acc, val_rae, val_corr, test_acc, test_rae, test_corr = main()
        vacc.append(val_acc)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        acc.append(test_acc)
        rae.append(test_rae)
        corr.append(test_corr)
    print('\n\n')
    print('10 runs average')
    print('\n\n')
    print("valid\trse\trae\tcorr")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae), np.mean(vcorr)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae), np.std(vcorr)))
    print('\n\n')
    print("test\trse\trae\tcorr")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae), np.mean(corr)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae), np.std(corr)))