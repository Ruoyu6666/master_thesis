import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable



def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))



class StandardScaler():
    """Standardize the input"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data

############
### Loss ###
############
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


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
        )


def pinball_loss(target, predictions, quantiles= [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]):
    """
    Calculates pinball loss or quantile loss against the specified quantiles

    target: torch.tensor. The true values of the target variable. Dimensions are (sample number, timestep, 1).
    predictions: torch.tensor. The predicted values for each quantile. Dimensions are (sample number, timestep, quantile).
    quantiles: List[float] Quantiles that we are estimating for
        
    Returns: float> The total quantile loss (the lower the better)
    """

    errors = target - predictions
    upper = quantiles * errors
    lower = (quantiles - 1) * errors
    loss = torch.sum(torch.max(upper, lower), dim=2)

    return torch.mean(loss)




def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))

# CPRS
def calc_quantile_CRPS(target, forecast, eval_points):
    # target = target * scaler + mean_scaler
    # forecast = forecast * scaler + mean_scaler
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    
    return CRPS.item() / len(quantiles)


# MIS
def MIS(target: np.ndarray, lower_quantile: np.ndarray,
        upper_quantile: np.ndarray, alpha: float,) -> float:
    r"""mean interval score
    Implementation comes form glounts.evalution metrics
    .. math:: msis = mean(U - L + 2/alpha * (L-Y) * I[Y<L] + 2/alpha * (Y-U) * I[Y>U])
    """
    numerator = np.mean(upper_quantile - lower_quantile
                        + 2.0 /alpha * (lower_quantile - target) * (target<lower_quantile)
                        + 2.0 /alpha * (target - upper_quantile) * (target>upper_quantile))
    return numerator


def calc_mis(target, forecast, alpha = 0.05):
    """ target:   (B, T, V),
        forecast: (B, n_sample, T, V)"""
    return MIS(target = target.cpu().numpy(),
               lower_quantile = torch.quantile(forecast, alpha / 2, dim=1).cpu().numpy(),
               upper_quantile = torch.quantile(forecast, 1.0 - alpha / 2, dim=1).cpu().numpy(),
               alpha = alpha)

##############
### Loader ###
##############

### For BTM
class DataLoaderM_btm(object):
    # train, valid: ratios of training set and validation set
    # test = 1 - train - valid
    def __init__(self, file1, file2, fileL, filePV, train, valid, 
                 device, horizon, window, w, normalize = 2):
        self.P = window
        self.h = horizon
        self.W = w

        self.dat1 = np.load(file1)                    # input1: n * m * in_dim1
        self.dat2 = np.load(file2)                    # input2: n * m * in_dim2
                
        finL = open(fileL)                               # Load data
        self.rawdatL = np.loadtxt(finL, delimiter=',')   # n * m: num_timestamps * num_nodes
        finPV = open(filePV)                             # PV data
        self.rawdatPV = np.loadtxt(finPV, delimiter=',') # n * m

        # Store normalized data
        self.datL  = np.zeros(self.rawdatL.shape)        # n * m
        self.datPV = np.zeros(self.rawdatPV.shape)       # n * m

        # Get shape parameters, d, n (num of timestamps), m (number of nodes)
        self.n, self.m = self.datL.shape
        self.d1 = self.dat1.shape[2]
        self.d2 = self.dat2.shape[2]
        
        self.scaleL =  np.ones(self.m)
        self.scalePV = np.ones(self.m)

        # normalize data
        self._normalized(normalize)
    
        self._split(int(train * self.n), int((train + valid) * self.n), self.n) 

        self.scaleL = torch.from_numpy(self.scaleL).float()
        self.scaleL = self.scaleL.to(device)
        self.scaleL = Variable(self.scaleL)

        self.scalePV = torch.from_numpy(self.scalePV).float()
        self.scalePV = self.scalePV.to(device)
        self.scalePV = Variable(self.scalePV)

        self.device = device


    def _normalized(self, normalize):
        for i in range(self.m):
            self.scaleL[i]  = np.max(np.abs(self.rawdatL[:, i]))
            self.scalePV[i] = np.max(np.abs(self.rawdatPV[:, i]))
            
            if self.scaleL[i] > 0:
                self.datL[:, i]  = self.rawdatL[:, i]  / np.max(np.abs(self.rawdatL[:, i]))
            if self.scalePV[i] > 0:
                self.datPV[:, i] = self.rawdatPV[:, i] / np.max(np.abs(self.rawdatPV[:, i]))

        
    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train) # (seq_in_len + 3 - 1, n * 0.6)
        valid_set = range(train, valid)               # (n * 0.6, n * 0.8)
        test_set =  range(valid, self.n)              # (n * 0.8, n)

        self.train = self._batchify(train_set, self.h)# [train_set, seq_in_lenth, num_node], [5076, 24, 300][5076, 24, 300]
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    
    def _batchify(self, idx_set, horizon):
        
        n = len(idx_set) - self.W
        X1  = torch.zeros((n, self.P, self.m, self.d1)) # input 1
        X2  = torch.zeros((n, self.P, self.m, self.d2)) # input 2
        YL  = torch.zeros((n, self.W, self.m))          # Predicted Load
        YPV = torch.zeros((n, self.W, self.m))          # Predicted PV

        for i in range(n):
            end = idx_set[i] - self.h + 1 # self.h:24
            start = end - self.P          # self.P:24

            X1[i, :, :, :] = torch.from_numpy(self.dat1[start:end, :, :])
            X2[i, :, :, :] = torch.from_numpy(self.dat2[start:end, :, :])
            YL[i, :, :]    = torch.from_numpy(self.datL [idx_set[i]:idx_set[i] + self.W, :])
            YPV[i, :, :]   = torch.from_numpy(self.datPV[idx_set[i]:idx_set[i] + self.W, :])
        
        return [X1, X2, YL, YPV]
    
    
    def get_batches(self, inputs, inputsW, targetsYL, targetsYPV, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length) #Returns a random permutation of integers from 0 to length - 1.
        else:
            index = torch.LongTensor(range(length))
          
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X1 = inputs[excerpt]
            X2 = inputsW[excerpt]
            YL = targetsYL[excerpt]
            YPV = targetsYPV[excerpt]


            X1 = X1.to(self.device)
            X2 = X2.to(self.device)
            YL = YL.to(self.device)
            YPV = YPV.to(self.device)
            yield Variable(X1), Variable(X2), Variable(YL), Variable(YPV)
            start_idx += batch_size


class DataLoaderM_new(object):
    def __init__(self, file_name, train, valid, device, horizon, window, w, normalize = 2):
        self.P = window                  # seq_in_len
        self.h = horizon                 # horizon
        self.W = w                       # seq_out_len
        
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape  # self.n: number of timestamps

        self.normalize = 2
        self.scale = np.ones(self.m)     # self.m: num_nodes 
        self._normalized(normalize)      
        self._split(int(train * self.n), int((train + valid) * self.n), self.n) 

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.test[1].size(1), self.m) # [1733, 24, 1413]
        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device
    

    def _normalized(self, normalize):
        if (normalize == 0):
            self.dat = self.rawdat
        
        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))


    def _split(self, train, valid, test):
        train_set = range(self.P + self.h - 1, train) # (seq_in_len + 3 - 1, n * 0.6)
        valid_set = range(train, valid)               # (n * 0.6, n * 0.8)
        test_set = range(valid, self.n)               # (n * 0.8, n)
        
        self.train = self._batchify(train_set, self.h)# [train_set, seq_in_lenth, num_node], [5076, 168, 1413][5076, 24, 1413]
        self.valid = self._batchify(valid_set, self.h)# [1733, 168, 1413], [1733, 24, 1413]
        self.test = self._batchify(test_set, self.h)  # [1733, 168, 1413], [1733, 24, 1413]


    def _batchify(self, idx_set, horizon):
        n = len(idx_set)-self.W
        X = torch.zeros((n, self.P, self.m))          # self.P: seq_in_len
        Y = torch.zeros((n, self.W, self.m))          # self.W: seq_out_len
        
        for i in range(n):
            end = idx_set[i] - self.h + 1 # self.h (horizon): 3
            start = end - self.P          # self.P (seq_in_len)
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])                      # ([index, seq_in_len, num_nodes])
            Y[i, :, :] = torch.from_numpy(self.dat[idx_set[i]:idx_set[i] + self.W, :]) # ([index, seq_out_len, num_nodes])
        return [X, Y]


    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
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
            X = X.to(self.device) #[4, 168, 1413]
            Y = Y.to(self.device) #[4, 24, 1413]
            yield Variable(X), Variable(Y)
            start_idx += batch_size



class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()




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